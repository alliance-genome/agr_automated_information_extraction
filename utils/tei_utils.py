import logging
import os
import re
import html

from grobid_client.models import TextWithRefs
from grobid_client.types import TEI

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    text = html.unescape(text)
    # replace only angle‐brackets and square‐brackets; keep parentheses for e.g. “HT115(DE3)”
    text = re.sub(r"[<>\[\]]+", " ", text)
    # collapse any run of whitespace into a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_sentences_from_tei_section(section):
    """
    Extract cleaned sentences from a TEI section, ensuring each ends with a period.
    Returns (sentences, num_errors).
    """
    sentences = []
    num_errors = 0
    for paragraph in section.paragraphs:
        paras = [paragraph] if isinstance(paragraph, TextWithRefs) else paragraph
        for sentence in paras:
            try:
                clean = re.sub(r'<[^<]+>', '', sentence.text or '').strip()
                if clean:
                    sentences.append(clean)
            except Exception:
                num_errors += 1

    # Ensure each sentence ends with a period
    sentences = [s if s.endswith('.') else f"{s}." for s in sentences if s.strip()]
    return sentences, num_errors


def get_fulltext_from_tei(tei_obj):
    """
    Build full text by concatenating all section sentences with spaces.
    """
    sentences = []
    for section in tei_obj.sections:
        sec_sentences, _ = get_sentences_from_tei_section(section)
        sentences.extend(sec_sentences)
    return ' '.join(sentences)


class AllianceTEI:
    """
    Wrapper around a GROBID-parsed TEI document to extract title, abstract, and full text,
    including table cells and formulas from raw XML.
    """
    def __init__(self):
        self.tei_obj = None
        self.raw_xml = None

    def load_from_file(self, file_path: str):
        # Cache raw XML for formula and cell extraction
        with open(file_path, 'rb') as f:
            raw_bytes = f.read()
            try:
                self.raw_xml = raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                self.raw_xml = raw_bytes.decode('latin-1', errors='replace')

        # Parse TEI content via GROBID
        with open(file_path, 'rb') as f:
            self.tei_obj = TEI.parse(f, figures=True)

    def get_title(self) -> str:
        """
        Return the document title, normalized (always a string).
        """
        if not self.tei_obj:
            return ""
        title = self.tei_obj.title or ""
        return _normalize_text(title)

    def get_abstract(self) -> str:
        """
        Return the abstract text from the first 'abstract' section, normalized.
        """
        if not self.tei_obj:
            return ""
        abstract_parts = []
        for section in self.tei_obj.sections:
            if section.name and section.name.lower() == 'abstract':
                for paragraph in section.paragraphs:
                    paras = [paragraph] if isinstance(paragraph, TextWithRefs) else paragraph
                    for sentence in paras:
                        cleaned = re.sub(r'<[^<]+>', '', sentence.text or '')
                        abstract_parts.append(cleaned)
                break
        abstract = ' '.join(abstract_parts).strip()
        return _normalize_text(abstract)

    def get_fulltext(self, include_attributes: bool = False) -> str:  # noqa: C901
        """
        Return the full text of the document, including sentences, table cells,
        formulas, figure/table captions, list items, and notes/footnotes,
        normalized and space-separated.
        """
        if not self.tei_obj:
            return ""

        # 1) Base sentences from TEI structure
        text = get_fulltext_from_tei(self.tei_obj)

        if not self.raw_xml:
            return _normalize_text(text)

        def _clean_block(s: str) -> str:
            s = re.sub(r'<[^<]+>', '', s or '').strip()
            if not s:
                return ''
            # ensure terminal period for sentence boundary stability
            return s if s.endswith('.') else (s + '.')

        rx = re.compile  # shorthand
        xml = self.raw_xml

        # 2) Table cells
        cells = rx(r'<cell[^>]*>(.*?)</cell>', re.DOTALL).findall(xml)
        for block in cells:
            cleaned = _clean_block(block)
            if cleaned:
                text += ' ' + cleaned

        # 3) Formulas
        formulas = rx(r'<formula[^>]*>(.*?)</formula>', re.DOTALL).findall(xml)
        for block in formulas:
            cleaned = _clean_block(block)
            if cleaned:
                text += ' ' + cleaned

        # 4) Figure captions/descriptions
        #    - <figDesc> ... </figDesc>
        #    - <figure> ... <head>Caption</head> ... </figure>
        fig_descs = rx(r'<figDesc[^>]*>(.*?)</figDesc>', re.DOTALL).findall(xml)
        for block in fig_descs:
            cleaned = _clean_block(block)
            if cleaned:
                text += ' ' + cleaned

        fig_heads = rx(r'<figure[^>]*>.*?<head[^>]*>(.*?)</head>.*?</figure>', re.DOTALL).findall(xml)
        for block in fig_heads:
            cleaned = _clean_block(block)
            if cleaned:
                text += ' ' + cleaned

        # 5) Table captions/heads
        tbl_heads = rx(r'<table[^>]*>.*?<head[^>]*>(.*?)</head>.*?</table>', re.DOTALL).findall(xml)
        for block in tbl_heads:
            cleaned = _clean_block(block)
            if cleaned:
                text += ' ' + cleaned

        # 6) List items (e.g., <list> … <item>Text</item> … </list>)
        # list_items = rx(r'<list[^>]*>.*?(?:<item[^>]*>(.*?)</item>)+.*?</list>', re.DOTALL).findall(xml)
        # re.findall with a single capture returns only the LAST capture per list,
        # so also directly capture all items:
        all_items = rx(r'<item[^>]*>(.*?)</item>', re.DOTALL).findall(xml)
        for block in all_items:
            cleaned = _clean_block(block)
            if cleaned:
                text += ' ' + cleaned

        # 7) Notes / footnotes (often carry species mentions)
        notes = rx(r'<note[^>]*>(.*?)</note>', re.DOTALL).findall(xml)
        for block in notes:
            cleaned = _clean_block(block)
            if cleaned:
                text += ' ' + cleaned

        # 8) Section heads (titles inside body divs)
        sec_heads = rx(r'<div[^>]*>.*?<head[^>]*>(.*?)</head>', re.DOTALL).findall(xml)
        for block in sec_heads:
            cleaned = _clean_block(block)
            if cleaned:
                text += ' ' + cleaned

        # ------------------------------------------------------------------
        # 9) Entity-unifying mode: scan all text + attributes for entity-like
        #    tokens and append them once, so attributes and text are treated
        #    the same by downstream code.
        # ------------------------------------------------------------------
        if include_attributes:

            def _looks_like_entity(tok: str) -> bool:
                tok = tok.strip()
                if len(tok) < 2:
                    return False

                has_letter = any(c.isalpha() for c in tok)
                has_digit = any(c.isdigit() for c in tok)
                if not (has_letter and has_digit):
                    return False

                lowered = tok.lower()
                bad_prefixes = ("fig", "sup", "sub", "sec", "eq")
                if lowered.startswith(bad_prefixes):
                    return False

                if lowered.startswith(("lt", "gt", "amp")):
                    return False

                if re.fullmatch(r"[ivx]+", lowered):
                    return False

                return True

            entity_tokens: set[str] = set()
            text_tokens: set[str] = set()  # NEW: track tokens seen in visible text

            # 9a) Scan visible text from the entire TEI (tags stripped)
            plain_all = re.sub(r'<[^>]+>', ' ', xml)
            for tok in re.findall(
                r'[A-Za-z][A-Za-z0-9_.-]*\d+[A-Za-z0-9_.-]*',
                plain_all
            ):
                if _looks_like_entity(tok):
                    text_tokens.add(tok)
                    entity_tokens.add(tok)

            # 9b) Scan attribute values (xml:id, id, n, target, ref)
            attr_values = re.findall(
                r'\b(?:xml:id|id|n|target|ref)\s*=\s*"([^"]+)"',
                xml,
                flags=re.IGNORECASE
            )
            for val in attr_values:
                for tok in re.findall(
                    r'[A-Za-z][A-Za-z0-9_.-]*\d+[A-Za-z0-9_.-]*',
                    val
                ):
                    if not _looks_like_entity(tok):
                        continue

                    # If we've already seen this token in real text, keep it
                    if tok in text_tokens:
                        entity_tokens.add(tok)
                        continue

                    t_lower = tok.lower()

                    # NEW: drop attribute-only bibliography IDs like b1001, b290, etc.
                    if re.fullmatch(r"b\d{2,4}", t_lower):
                        continue

                    entity_tokens.add(tok)

            if entity_tokens:
                text += ' ' + '. '.join(sorted(entity_tokens)) + '.'

        return _normalize_text(text)

    def get_sentences(self):
        """
        Return a list of all sentences in the document, normalized.
        """
        if not self.tei_obj:
            return []
        sentences = []
        for section in self.tei_obj.sections:
            sec_sentences, _ = get_sentences_from_tei_section(section)
            sentences.extend(sec_sentences)
        return [_normalize_text(s) for s in sentences]


def convert_all_tei_files_in_dir_to_txt(dir_path: str):
    """
    Convert each .tei in a directory to a .txt containing its full text.
    """
    for fname in os.listdir(dir_path):
        if not fname.endswith('.tei'):
            continue
        tei_file = os.path.join(dir_path, fname)
        try:
            alliance = AllianceTEI()
            alliance.load_from_file(tei_file)
            article_text = alliance.get_fulltext()
            txt_path = tei_file.replace('.tei', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as out:
                out.write(article_text)
        except Exception as e:
            logger.error(f"Error parsing TEI file {tei_file}: {e}")
