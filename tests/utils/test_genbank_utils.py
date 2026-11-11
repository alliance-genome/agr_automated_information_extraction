import json
from unittest.mock import MagicMock, patch

import pytest

from utils.genbank_utils import (
    AccessionHit,
    NcbiAccessionResolver,
    accession_db,
    find_accessions,
    parse_elink_linksets,
)


@pytest.mark.parametrize("text, expected", [
    ("GenBank accession no. AF123456.1", "AF123456"),
    ("isoform a, NP_497054.2)", "NP_497054"),
    ("(refseq ID: NM_014671.1) showing", "NM_014671"),
    ("U12345", "U12345"),
    ("under accession number LC424332.", "LC424332"),
    ("CAA12345.1, AAO16387.1", "CAA12345"),
    ("GLB-6 (CCD65109)", "CCD65109"),
    ("| *unc-22* | **AF123456** |", "AF123456"),
    ("daf-2 _NM_001256799.2_", "NM_001256799"),
    ("PP738530 for RNA2", "PP738530"),
    ("AAAA01000001", "AAAA01000001"),
])
def test_find_accessions_matches_accession_formats(text, expected):
    assert expected in find_accessions(text)


@pytest.mark.parametrize("text", [
    "grant R01GM123456",            # letters glued to a grant prefix
    "NIH P40OD010440",
    "ZDB-GENE-000101-1",
    "pNM_0012345",                  # RefSeq prefix inside a word
    "NM_0012345678901",             # too many digits
    "UniProt P12345 and Q9Y2X3",    # UniProt, not INSDC
    "GEO GSE128746, GSM3684366",    # GEO series and samples
    "SRA SRR1234567",
    "C. briggsae CBG12345 and PPA01234",  # WormBase non-elegans gene IDs
    "Strain name: BKK00100; Gene deletion: dacA",  # B. subtilis knockout strain
    "WGS project MOLV00000000.1",   # WGS master record
    "NC_003279.8 chromosome I",     # chromosome-level RefSeq
    "WP_000000001",                 # prokaryotic protein, no gene
    "strain N2",
    "p.A123456",                    # protein change notation
])
def test_find_accessions_rejects_non_accessions(text):
    assert find_accessions(text) == {}


def test_find_accessions_strips_version_counts_and_context():
    text = ("Sequences were deposited in GenBank (AF123456.1). " + "x " * 200
            + "AF123456 again and BC058269 without any keyword nearby.")
    hits = find_accessions(text)
    assert hits["AF123456"].count == 2
    assert hits["AF123456"].has_context is True
    assert hits["BC058269"].has_context is False


@pytest.mark.parametrize("accession, db", [
    ("NM_013646", "nuccore"), ("NR_001234", "nuccore"), ("NG_012345", "nuccore"),
    ("NP_497054", "protein"), ("XP_005161234", "protein"),
    ("AF123456", "nuccore"), ("U12345", "nuccore"), ("AAAA01000001", "nuccore"),
    ("CAA12345", "protein"), ("AAO1638700", "protein"),
])
def test_accession_db(accession, db):
    assert accession_db(accession) == db


def test_parse_elink_linksets_matches_by_echoed_id_and_position():
    accessions = ["AK159011", "ZZ999999", "BX571857", "NM_013646"]
    linksets = [
        {"ids": ["AK159011.1"], "linksetdbs": [{"dbto": "gene", "links": ["225998"]}]},
        {"ids": [], "linksetdbs": [{"dbto": "gene", "ERROR": "exception"}], "ERROR": "BLOB ID IS NOT IMPLEMENTED"},
        {"ids": ["BX571857.1"]},
        {"ids": ["NM_013646.3"], "linksetdbs": [{"dbto": "gene", "links": ["19883", "19883"]}]},
    ]
    assert parse_elink_linksets(accessions, linksets) == {
        "AK159011": [225998], "ZZ999999": [], "BX571857": [], "NM_013646": [19883]}


def test_parse_elink_linksets_leaves_unmatched_out_when_counts_differ():
    accessions = ["AK159011", "ZZ999999"]
    linksets = [{"ids": ["AK159011.1"], "linksetdbs": [{"dbto": "gene", "links": ["225998"]}]}]
    assert parse_elink_linksets(accessions, linksets) == {"AK159011": [225998]}


def _elink_response(linksets):
    response = MagicMock()
    response.text = json.dumps({"linksets": linksets})
    response.raise_for_status.return_value = None
    return response


@patch("utils.genbank_utils.time.sleep")
@patch("utils.genbank_utils.requests.post")
def test_resolver_routes_by_db_sends_one_id_per_accession_and_caches(mock_post, _sleep, tmp_path):
    def fake_post(url, data, timeout):
        params = dict((k, v) for k, v in data if k != "id")
        ids = [v for k, v in data if k == "id"]
        if params["dbfrom"] == "nuccore":
            assert ids == ["NM_013646"]
            return _elink_response([{"ids": ["NM_013646.3"], "linksetdbs": [{"dbto": "gene", "links": ["19883"]}]}])
        assert ids == ["NP_497054"]
        return _elink_response([{"ids": ["NP_497054.2"], "linksetdbs": [{"dbto": "gene", "links": ["175410"]}]}])

    mock_post.side_effect = fake_post
    cache_path = str(tmp_path / "cache.json")
    resolver = NcbiAccessionResolver(cache_path=cache_path, api_key=None)
    hits = [AccessionHit("NM_013646", "nuccore"), AccessionHit("NP_497054", "protein")]
    assert resolver.resolve(hits) == {"NM_013646": [19883], "NP_497054": [175410]}
    resolver.save_cache()
    assert mock_post.call_count == 2

    mock_post.reset_mock()
    cached = NcbiAccessionResolver(cache_path=cache_path, api_key=None)
    assert cached.resolve(hits) == {"NM_013646": [19883], "NP_497054": [175410]}
    mock_post.assert_not_called()


@patch("utils.genbank_utils.time.sleep")
@patch("utils.genbank_utils.requests.post")
def test_resolver_does_not_cache_failed_requests(mock_post, _sleep, tmp_path):
    import requests
    mock_post.side_effect = requests.ConnectionError("down")
    resolver = NcbiAccessionResolver(cache_path=str(tmp_path / "cache.json"), api_key=None, retries=2)
    assert resolver.resolve([AccessionHit("NM_013646", "nuccore")]) == {"NM_013646": []}
    assert resolver._cached("NM_013646") is None


@patch("utils.genbank_utils.time.sleep")
@patch("utils.genbank_utils.requests.post")
def test_resolver_splits_a_failing_batch_to_isolate_the_bad_accession(mock_post, _sleep, tmp_path):
    import requests

    def fake_post(url, data, timeout):
        ids = [v for k, v in data if k == "id"]
        if "BAD00001" in ids:
            raise requests.ConnectionError("Connection broken: InvalidChunkLength")
        return _elink_response([{"ids": [f"{i}.1"], "linksetdbs": [{"dbto": "gene", "links": ["1"]}]}
                                for i in ids])

    mock_post.side_effect = fake_post
    resolver = NcbiAccessionResolver(cache_path=str(tmp_path / "cache.json"), api_key=None, retries=1)
    accessions = ["AF000001", "AF000002", "BAD00001", "AF000003"]
    result = resolver.resolve([AccessionHit(a, "nuccore") for a in accessions])
    assert result == {"AF000001": [1], "AF000002": [1], "BAD00001": [], "AF000003": [1]}
    assert resolver._cached("BAD00001") is None
    assert resolver._cached("AF000003") == [1]
