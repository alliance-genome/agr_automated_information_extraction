import logging
import os
import re

from grobid_client.models import TextWithRefs
from grobid_client.types import TEI


logger = logging.getLogger(__name__)


class AllianceTEI:
    def __init__(self):
        self.tei_obj = None
        self.raw_xml = None           # <-- store raw XML for later <formula> extraction

    def load_from_file(self, file_path):
        # Read entire file as bytes/string so that we can extract <formula> blocks later
        with open(file_path, "rb") as file_stream:
            raw_bytes = file_stream.read()
            try:
                self.raw_xml = raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                # if not UTF-8, fall back to latin-1 or replace errors
                self.raw_xml = raw_bytes.decode('latin-1', errors='replace')

        # Now parse with grobid's TEI parser
        # (Note: we reopen in binary mode for TEI.parse, since parse expects a file‐like in bytes)
        with open(file_path, "rb") as file_stream:
            self.tei_obj = TEI.parse(file_stream, figures=True)

    def get_title(self):
        if self.tei_obj is None:
            return None
        return self.tei_obj.title

    def get_abstract(self):
        if self.tei_obj is None:
            return None

        abstract = ""
        for section in self.tei_obj.sections:
            if section.name.lower() == "abstract":
                for paragraph in section.paragraphs:
                    if isinstance(paragraph, TextWithRefs):
                        paragraph = [paragraph]
                    for sentence in paragraph:
                        # strip any inline tags (e.g. <hi>, <formula>, etc.)
                        cleaned = re.sub(r'<[^<]+>', '', sentence.text)
                        abstract += cleaned + " "
                return abstract
        return abstract

    def get_fulltext(self):
        if self.tei_obj is None:
            return None
        # Build the “sentences” version of full text:
        base_fulltext = get_fulltext_from_tei(self.tei_obj)

        # Now also pull out every <formula>…</formula> occurrence from the raw XML,
        # strip tags inside it, and append to the end.
        if self.raw_xml:
            # Find all <formula ...> inner content blocks (DOTALL so newlines are OK)
            formula_blocks = re.findall(r'<formula[^>]*>(.*?)</formula>', self.raw_xml, flags=re.DOTALL)
            for block in formula_blocks:
                # Strip any tags inside <formula> (e.g. if there are nested <hi> or <xref>)
                cleaned = re.sub(r'<[^<]+>', '', block).strip()
                if cleaned:
                    # Append a period if missing, so it doesn't run into the previous sentence
                    if not cleaned.endswith("."):
                        cleaned = cleaned + "."
                    base_fulltext += " " + cleaned

        return base_fulltext

    def get_sentences(self):
        if self.tei_obj is None:
            return None

        sentences = []
        for section in self.tei_obj.sections:
            sec_sentences, _ = get_sentences_from_tei_section(section)
            sentences.extend(sec_sentences)
        return sentences


def get_sentences_from_tei_section(section):
    sentences = []
    num_errors = 0
    for paragraph in section.paragraphs:
        if isinstance(paragraph, TextWithRefs):
            paragraph = [paragraph]
        for sentence in paragraph:
            try:
                # Just strip tags and keep any non‐empty text
                clean = re.sub(r'<[^<]+>', '', sentence.text).strip()
                if clean:
                    sentences.append(clean)
            except Exception:
                num_errors += 1

    # Guarantee each sentence ends with a period
    sentences = [
        s if s.endswith(".") else f"{s}."
        for s in sentences
        if s.strip()
    ]
    return sentences, num_errors


def get_fulltext_from_tei(tei_object):
    sentences = []
    for section in tei_object.sections:
        sec_sentences, _ = get_sentences_from_tei_section(section)
        sentences.extend(sec_sentences)
    # Join with a single space between sentences
    return " ".join(sentences)


def convert_all_tei_files_in_dir_to_txt(dir_path):
    file_paths = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.endswith(".tei")
    ]

    for tei_file in file_paths:
        try:
            # Load and parse
            alliance = AllianceTEI()
            alliance.load_from_file(tei_file)

            # Build combined fulltext (including <formula> content at the end)
            article_text = alliance.get_fulltext()

            # Write out to .txt
            txt_path = tei_file.replace(".tei", ".txt")
            with open(txt_path, "w", encoding="utf-8") as text_file:
                text_file.write(article_text)

        except Exception as e:
            logger.error(f"Error parsing TEI file {tei_file}: {e}")
        # Optionally remove the original .tei if desired:
        # os.remove(tei_file)
