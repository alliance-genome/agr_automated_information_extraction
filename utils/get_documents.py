import glob
import os
import logging
from pathlib import Path
from typing import Tuple, List
from grobid_client import Client
from grobid_client.api.pdf import process_fulltext_document
from grobid_client.models import Article, ProcessForm
from grobid_client.types import TEI, File

from utils.tei_utils import get_sentences_from_tei_section

logger = logging.getLogger(__name__)


def get_documents(input_docs_dir: str) -> List[Tuple[str, str, str, str]]:
    documents = []
    client = None
    for file_path in glob.glob(os.path.join(input_docs_dir, "*")):
        num_errors = 0
        file_obj = Path(file_path)
        if file_path.endswith(".tei") or file_path.endswith(".pdf"):
            with file_obj.open("rb") as fin:
                if file_path.endswith(".pdf"):
                    if client is None:
                        client = Client(base_url=os.environ.get("GROBID_API_URL"), timeout=1000, verify_ssl=False)
                    logger.info("Started pdf to TEI conversion")
                    form = ProcessForm(
                        segment_sentences="1",
                        input_=File(file_name=file_obj.name, payload=fin, mime_type="application/pdf"))
                    r = process_fulltext_document.sync_detailed(client=client, multipart_data=form)
                    file_stream = r.content
                else:
                    file_stream = fin
                try:
                    article: Article = TEI.parse(file_stream, figures=True)
                except Exception:
                    num_errors += 1
                    continue
                sentences = []
                for section in article.sections:
                    sec_sentences, sec_num_errors = get_sentences_from_tei_section(section)
                    sentences.extend(sec_sentences)
                    num_errors += sec_num_errors
                abstract = ""
                for section in article.sections:
                    if section.name == "ABSTRACT":
                        abs_sentences, num_errors = get_sentences_from_tei_section(section)
                        abstract = " ".join(abs_sentences)
                        break
                documents.append((file_path, " ".join(sentences), article.title, abstract))
        if num_errors > 0:
            logger.debug(f"Couldn't read {str(num_errors)} sentence(s) from {str(file_path)}")
    return documents
