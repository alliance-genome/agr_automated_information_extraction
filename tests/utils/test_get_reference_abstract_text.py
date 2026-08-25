"""The BoW-only abstract profile reads its text from the reference record rather
than from an embedding parquet (SCRUM-5764)."""
from unittest.mock import patch

from utils import abc_utils


@patch("utils.abc_utils.get_reference_title_and_abstract",
       return_value=("Probe synthesis", "We made a probe."))
def test_returns_the_shared_chunk_text_convention(_mock_fetch):
    # Must be byte-identical to what the embedding profiles put in the parquet's
    # `content` column, since both feed the same exact-token BoW hasher.
    assert abc_utils.get_reference_abstract_text("AGRKB:1") == "Probe synthesis\n\nWe made a probe."


@patch("utils.abc_utils.get_reference_title_and_abstract", return_value=("", ""))
def test_none_when_the_reference_has_no_title_or_abstract(_mock_fetch):
    # A reference with no text cannot be classified from its abstract at all, so it
    # must come back unavailable and be failed like a missing MD, not scored on "".
    assert abc_utils.get_reference_abstract_text("AGRKB:1") is None


@patch("utils.abc_utils.get_reference_title_and_abstract", return_value=("Title only", ""))
def test_title_alone_is_still_usable(_mock_fetch):
    assert abc_utils.get_reference_abstract_text("AGRKB:1") == "Title only"


@patch("utils.abc_utils.get_reference_title_and_abstract",
       side_effect=RuntimeError("boom"))
def test_none_rather_than_raising_when_the_lookup_fails(_mock_fetch):
    # get_reference_title_and_abstract raises on a non-200; this wrapper must be
    # total like get_reference_embedding so one bad reference cannot abort a batch.
    assert abc_utils.get_reference_abstract_text("AGRKB:1") is None
