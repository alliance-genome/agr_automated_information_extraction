"""Tests for get_documents text assembly (section selection).

Keywords/metadata are opt-in and default OFF so the fulltext stays
byte-identical to what production models were trained on.
"""

import os
import tempfile
from unittest.mock import patch

from utils import get_documents


def _run(tmp, **kwargs):
    with open(os.path.join(tmp, "AGRKB_101000000000001.md"), "w") as fh:
        fh.write("# doc")
    return get_documents.get_documents(tmp, **kwargs)


@patch("utils.get_documents.AllianceMarkdown")
def test_default_excludes_keywords_and_metadata(mock_md_cls):
    inst = mock_md_cls.return_value
    inst.get_fulltext.return_value = "body"
    with tempfile.TemporaryDirectory() as tmp:
        _run(tmp)
    inst.get_fulltext.assert_called_once_with(include_keywords=False, include_metadata=False)


@patch("utils.get_documents.AllianceMarkdown")
def test_opt_in_includes_keywords_and_metadata(mock_md_cls):
    inst = mock_md_cls.return_value
    inst.get_fulltext.return_value = "body"
    with tempfile.TemporaryDirectory() as tmp:
        _run(tmp, include_keywords=True, include_metadata=True)
    inst.get_fulltext.assert_called_once_with(include_keywords=True, include_metadata=True)
