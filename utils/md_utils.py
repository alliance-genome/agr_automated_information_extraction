import glob
import logging
import os
import re

from agr_abc_document_parsers import (
    Document,
    extract_abstract_text,
    extract_plain_text,
    extract_sentences,
    load_document_with_supplements,
    read_markdown,
    strip_markdown_formatting,
)

logger = logging.getLogger(__name__)


# Entity-token regex + filter ported from utils/tei_utils.py for the
# "include_attributes" allele-extraction path. We mine the entire MD
# (including references) for letter+digit identifiers that look like
# biological entity names — replicates the spirit of the old TEI
# attribute mining, minus the XML-attribute-only IDs which don't exist
# in Markdown. The XML-attribute-junk filters (b#### bib IDs, e### page
# codes inside biblScope) are dropped because their TEI-only triggers
# don't apply.
_ENTITY_TOKEN_RE = re.compile(r'[A-Za-z][A-Za-z0-9_.\-]*\d+[A-Za-z0-9_.\-]*')
_ROMAN_NUMERAL_RE = re.compile(r'^[ivx]+$')
_BAD_ENTITY_PREFIXES = ('fig', 'sup', 'sub', 'sec', 'eq', 'lt', 'gt', 'amp')


def _looks_like_entity(tok: str) -> bool:
    tok = tok.strip()
    if len(tok) < 2:
        return False
    has_letter = any(c.isalpha() for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    if not (has_letter and has_digit):
        return False
    lowered = tok.lower()
    if lowered.startswith(_BAD_ENTITY_PREFIXES):
        return False
    if _ROMAN_NUMERAL_RE.fullmatch(lowered):
        return False
    return True


def _mine_entity_tokens(text: str) -> list[str]:
    """Return sorted, deduped entity-like tokens scraped from ``text``."""
    tokens = {tok for tok in _ENTITY_TOKEN_RE.findall(text or "") if _looks_like_entity(tok)}
    return sorted(tokens)


def _read_md_text(path: str) -> str:
    with open(path, "rb") as f:
        raw_bytes = f.read()
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="replace")


def _discover_supplement_paths(main_path: str) -> list[str]:
    """Return sibling ``<basename>.supp_*.md`` files for a given main MD path,
    sorted by the numeric ``N`` in ``supp_<N>``.
    """
    directory = os.path.dirname(main_path) or "."
    base = os.path.basename(main_path)
    if not base.endswith(".md"):
        return []
    stem = base[: -len(".md")]
    pattern = os.path.join(directory, f"{stem}.supp_*.md")

    def _key(p: str) -> tuple[int, str]:
        name = os.path.basename(p)
        try:
            num = int(name[len(stem) + len(".supp_"):-len(".md")])
        except ValueError:
            num = 10**9
        return (num, name)

    return sorted(glob.glob(pattern), key=_key)


class AllianceMarkdown:
    """Wrapper around an ABC-format Markdown document.

    Provides the same surface as ``AllianceTEI`` (``load_from_file``,
    ``get_title``, ``get_abstract``, ``get_fulltext``, ``get_sentences``)
    so it can be used as a drop-in replacement for downstream pipelines
    that previously consumed TEI.

    The underlying parser is :func:`agr_abc_document_parsers.read_markdown`,
    which builds a ``Document`` model from the ABC Markdown format. Plain
    text generation goes through :func:`extract_plain_text` whose defaults
    already exclude the reference list — matching the existing TEI
    behaviour.

    Supplements are off by default. Call ``load_from_file(path,
    include_supplements=True)`` (or pass an explicit ``supplement_paths``
    list) to also pull sibling ``<basename>.supp_*.md`` files into the
    Document; they are then included in :meth:`get_fulltext` and
    :meth:`get_sentences` output.
    """

    def __init__(self):
        self.doc: Document | None = None
        self.raw_md: str | None = None
        self.has_supplements: bool = False

    def load_from_file(self, file_path: str, include_supplements: bool = False,
                       supplement_paths: list[str] | None = None):
        """Load a main MD file. When ``include_supplements`` is True (or
        ``supplement_paths`` is provided), also load supplemental MD files
        and merge them into the resulting Document.
        """
        self.raw_md = _read_md_text(file_path)

        paths = list(supplement_paths) if supplement_paths is not None else (
            _discover_supplement_paths(file_path) if include_supplements else []
        )

        if paths:
            supp_texts = [_read_md_text(p) for p in paths]
            self.doc = load_document_with_supplements(self.raw_md, supp_texts)
            self.has_supplements = True
        else:
            self.doc = read_markdown(self.raw_md)
            self.has_supplements = False

    def get_title(self) -> str:
        if not self.doc or not self.doc.title:
            return ""
        return strip_markdown_formatting(self.doc.title).strip()

    def get_abstract(self) -> str:
        if not self.doc:
            return ""
        return extract_abstract_text(self.doc).strip()

    def get_fulltext(self, include_attributes: bool = False, **kwargs) -> str:
        """Return the document fulltext.

        Any keyword argument accepted by
        :func:`agr_abc_document_parsers.extract_plain_text` is forwarded as-is,
        so callers can toggle every section the library knows about
        (``include_authors``, ``include_correspondence``, ``include_metadata``,
        ``include_abstract``, ``include_keywords``, ``include_body``,
        ``include_acknowledgments``, ``include_funding``, ``include_author_notes``,
        ``include_competing_interests``, ``include_data_availability``,
        ``include_back_matter``, ``include_references``, ``include_supplements``,
        ``include_sub_articles``).

        Defaults match the library: title is always present; abstract, body,
        acknowledgments, funding, author-notes, competing-interests,
        data-availability, back-matter and supplements are on; authors,
        correspondence, metadata, keywords, references and sub-articles are off.
        ``include_supplements`` defaults to ``self.has_supplements`` unless the
        caller passes it explicitly.

        ``include_attributes=True`` additionally appends entity-like tokens
        (letter+digit identifiers) mined from the entire Markdown — body plus
        references — replicating the allele-extraction TEI behaviour. This is
        only useful for the allele topic where downstream NER benefits from
        seeing identifiers that may live in citation entries. Note that
        switching it on forces ``include_references=True`` for the underlying
        regex sweep, but the formatted reference list is *not* added to the
        textual output unless the caller also passes ``include_references=True``
        explicitly.
        """
        if not self.doc:
            return ""
        kwargs.setdefault("include_supplements", self.has_supplements)
        text = extract_plain_text(self.doc, **kwargs).strip()
        if include_attributes:
            kwargs_for_mining = dict(kwargs)
            kwargs_for_mining["include_references"] = True
            mined_source = extract_plain_text(self.doc, **kwargs_for_mining)
            tokens = _mine_entity_tokens(mined_source)
            if tokens:
                appended = ". ".join(tokens) + "."
                text = (text + " " + appended).strip() if text else appended
        return text

    def get_sentences(self, include_supplements: bool | None = None) -> list[str]:
        """Return sentences from the document.

        Mirrors the underlying library, which only exposes
        ``include_supplements`` for sentence splitting (it always uses the
        full default body for the underlying text). Pass an explicit value
        to override; otherwise defaults to ``self.has_supplements``.

        Callers who need sentences with custom section toggles should split
        the output of :meth:`get_fulltext` directly.
        """
        if not self.doc:
            return []
        if include_supplements is None:
            include_supplements = self.has_supplements
        return extract_sentences(self.doc, include_supplements=include_supplements)


def convert_all_md_files_in_dir_to_txt(dir_path: str, include_supplements: bool = False):
    """Convert every main ``.md`` file in ``dir_path`` to a sibling ``.txt``
    containing its fulltext.

    ``.supp_*.md`` files are skipped here — they're picked up automatically
    when ``include_supplements`` is True.
    """
    for fname in os.listdir(dir_path):
        if not fname.endswith(".md"):
            continue
        # Skip supplement files; they're consumed alongside their main file.
        stem = fname[: -len(".md")]
        if ".supp_" in stem:
            continue
        md_file = os.path.join(dir_path, fname)
        try:
            md = AllianceMarkdown()
            md.load_from_file(md_file, include_supplements=include_supplements)
            article_text = md.get_fulltext()
            txt_path = md_file[: -len(".md")] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as out:
                out.write(article_text)
        except Exception as e:
            logger.error(f"Error parsing Markdown file {md_file}: {e}")
