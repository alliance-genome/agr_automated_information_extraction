import logging
import os
import re

from grobid_client.models import TextWithRefs
from grobid_client.types import TEI


logger = logging.getLogger(__name__)


class AllianceTEI:
    def __init__(self):
        self.tei_obj = None

    def load_from_file(self, file_path):
        with open(file_path, "rb") as file_stream:
            self.tei_obj = TEI.parse(file_stream, figures=True)

    def get_title(self):
        if self.tei_obj is None:
            return None
        else:
            return self.tei_obj.title

    def get_abstract(self):
        if self.tei_obj is None:
            return None
        else:
            return self.tei_obj.abstract

    def get_fulltext(self):
        if self.tei_obj is None:
            return None
        else:
            return get_fulltext_from_tei(self.tei_obj)

    def get_sentences(self):
        if self.tei_obj is None:
            return None
        else:
            sentences = []
            for section in self.tei_obj.sections:
                sec_sentences, _ = get_sentences_from_tei_section(section)
                sentences.extend(sec_sentences)
            return sentences


def get_sentences_from_tei_section(section):
    sentences = []
    num_errors = 0  # Initialize error count
    for paragraph in section.paragraphs:
        if isinstance(paragraph, TextWithRefs):
            paragraph = [paragraph]
        for sentence in paragraph:
            try:
                if not sentence.text.isdigit() and not (
                    len(section.paragraphs) == 3
                    and section.paragraphs[0][0].text in ['\n', ' ']
                    and section.paragraphs[-1][0].text in ['\n', ' ']
                ):
                    sentences.append(re.sub('<[^<]+>', '', sentence.text))
            except Exception:
                num_errors += 1
    sentences = [sentence if sentence.endswith(".") else f"{sentence}." for sentence in sentences]
    return sentences, num_errors


def get_fulltext_from_tei(tei_object):
    sentences = []
    for section in tei_object.sections:
        sec_sentences, _ = get_sentences_from_tei_section(section)
        sentences.extend(sec_sentences)
    fulltext = " ".join(sentences)
    return fulltext


def convert_all_tei_files_in_dir_to_txt(dir_path):
    file_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".tei")]
    for tei_file in file_paths:
        try:
            with open(tei_file, "rb") as file_stream:
                article = TEI.parse(file_stream, figures=True)
                article_text = get_fulltext_from_tei(article)
                with open(tei_file.replace(".tei", ".txt"), "w") as text_file:
                    text_file.write(article_text)
        except Exception as e:
            logger.error(f"Error parsing TEI file {tei_file}: {e}")
        #os.remove(tei_file)
