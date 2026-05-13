import glob
import os
import logging
from typing import Tuple, List

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

from utils.md_utils import AllianceMarkdown

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

logger = logging.getLogger(__name__)


def get_documents(input_docs_dir: str) -> List[Tuple[str, str, str, str]]:
    """Load every parseable Markdown document in ``input_docs_dir`` as
    ``(file_path, fulltext, title, abstract)``.

    Only ``.md`` files are read. Markdown files are produced upstream by
    :func:`utils.abc_utils.download_md_files_for_references`, which fetches
    the converted main MD from ABC, falls back to a TEI→MD library
    conversion, and (optionally) requests on-demand server-side conversion
    via the ``/reference/referencefile/conversion_request`` endpoint when
    no MD or TEI is yet available in ABC.
    """
    documents: List[Tuple[str, str, str, str]] = []
    for file_path in glob.glob(os.path.join(input_docs_dir, "*")):
        if not file_path.endswith(".md"):
            continue
        # Skip supplement files that share the dir; they're consumed via
        # AllianceMarkdown.load_from_file(..., include_supplements=True) on
        # the main file when callers opt in.
        if ".supp_" in os.path.basename(file_path):
            continue
        try:
            md = AllianceMarkdown()
            md.load_from_file(file_path)
            documents.append((file_path, md.get_fulltext(), md.get_title(), md.get_abstract()))
        except Exception as e:
            logger.warning(f"Failed to parse Markdown {file_path}: {e}")
    return documents


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)
