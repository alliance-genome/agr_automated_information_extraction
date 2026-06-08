"""Tests for the training-set MD downloader.

When building a training set we must NOT trigger on-demand server-side
conversion: a reference with no MD/TEI means its upload/conversion is not
complete, and it should simply be excluded from the training set.
"""

import os
from unittest.mock import patch

from agr_dataset_manager import dataset_downloader


@patch("agr_dataset_manager.dataset_downloader.download_md_files_for_references")
def test_training_download_does_not_request_on_demand_conversion(mock_dl, tmp_path):
    os.makedirs(tmp_path / "positive", exist_ok=True)
    os.makedirs(tmp_path / "negative", exist_ok=True)

    dataset_downloader.download_md_files_from_abc_or_convert_pdf(
        ["AGRKB:1"], ["AGRKB:2"], str(tmp_path), "WB",
    )

    assert mock_dl.call_count == 2
    for call in mock_dl.call_args_list:
        assert call.kwargs.get("request_conversion") is False
