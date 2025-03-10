import re

from grobid_client.models import TextWithRefs


def get_sentences_from_tei_section(section):
    sentences = []
    num_errors = 0  # Initialize error count
    for paragraph in section.paragraphs:
        if isinstance(paragraph, TextWithRefs):
            paragraph = [paragraph]
        for sentence in paragraph:
            try:
                if not sentence.text.isdigit() and not (
                        len(section.paragraphs) == 3 and
                        section.paragraphs[0][0].text in ['\n', ' '] and
                        section.paragraphs[-1][0].text in ['\n', ' ']
                ):
                    sentences.append(re.sub('<[^<]+>', '', sentence.text))
            except Exception as e:
                num_errors += 1
    sentences = [sentence if sentence.endswith(".") else f"{sentence}." for sentence in sentences]
    return sentences, num_errors


def convert_tei_to_text(tei_object):
    sentences = []
    for section in tei_object.sections:
        sec_sentences, _ = get_sentences_from_tei_section(section)
        sentences.extend(sec_sentences)
    abstract = ""
    for section in tei_object.sections:
        if section.name == "ABSTRACT":
            abs_sentences, _ = get_sentences_from_tei_section(section)
            abstract = " ".join(abs_sentences)
            break
    return f"{tei_object.title}\n\n{abstract}\n\n{sentences}"
