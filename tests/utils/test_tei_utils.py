"""SCRUM-6826: utils.tei_utils parses TEI through grobid_client's model classes only (not its HTTP
client), which lets grobid-client be installed without its pinned httpx==0.23.0. This guards that
the parsing still works with the HTTP stack it is installed next to."""
import os

from utils.tei_utils import AllianceTEI, convert_all_tei_files_in_dir_to_txt

SAMPLE = os.path.join(os.path.dirname(__file__), "data", "sample.tei")


def test_title_abstract_and_fulltext_are_parsed():
    tei = AllianceTEI()
    tei.load_from_file(SAMPLE)
    assert tei.get_title() == "Wingless signalling in the Drosophila wing disc"
    assert "wg mutants in the wing disc" in tei.get_abstract()
    fulltext = tei.get_fulltext()
    assert "Wingless (wg) patterns the wing." in fulltext
    assert "Loss of wg reduced the wing margin." in fulltext


def test_directory_conversion_writes_txt(tmp_path):
    target = tmp_path / "AGRKB_101.tei"
    target.write_bytes(open(SAMPLE, "rb").read())
    convert_all_tei_files_in_dir_to_txt(str(tmp_path))
    assert "Loss of wg reduced the wing margin." in (tmp_path / "AGRKB_101.txt").read_text()
