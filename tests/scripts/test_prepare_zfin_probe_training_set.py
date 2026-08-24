import csv
import json
from unittest.mock import patch

from scripts import prepare_zfin_probe_training_set as prep


# Mirrors the real sheet byte-for-byte: positives read "Positive " — capital P AND
# a trailing space — which the trainer's exact `== "positive"` comparison rejects.
# Built by joining so the meaningful trailing space survives lint and review.
CSV = "\n".join([
    "AGRKBID,XREF,Classification",
    ",ZDB-PUB-1,Positive ",
    ",ZDB-PUB-1,Positive ",
    ",ZDB-PUB-2,Negative",
    ",ZDB-PUB-3,Negative",
    ",ZDB-PUB-4,Positive ",
]) + "\n"


def test_normalizes_labels_dedups_and_drops_unusable_rows(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(CSV)

    # ZDB-PUB-3 does not resolve; ZDB-PUB-4 resolves but has no abstract.
    # Keys carry the ZFIN prefix because that is what reaches the ABC endpoint.
    curies = {"ZFIN:ZDB-PUB-1": "AGRKB:1", "ZFIN:ZDB-PUB-2": "AGRKB:2",
              "ZFIN:ZDB-PUB-3": None, "ZFIN:ZDB-PUB-4": "AGRKB:4"}
    abstracts = {"AGRKB:1": ("Title one", "Abstract one."),
                 "AGRKB:2": ("Title two", "Abstract two."),
                 "AGRKB:4": ("Title four", "")}

    with patch.object(prep, "get_curie_from_xref", side_effect=lambda x: curies[x]), \
         patch.object(prep, "get_reference_title_and_abstract",
                      side_effect=lambda c: abstracts[c]):
        report = prep.prepare(str(csv_path), str(tmp_path))

    records = json.loads((tmp_path / "labelled_abstracts.json").read_text())
    # One positive (deduped) + one negative survive.
    assert [(r["curie"], r["label"]) for r in records] == [
        ("AGRKB:1", "positive"), ("AGRKB:2", "negative")]
    # Labels are lowercased and stripped — the trainer compares them exactly.
    assert all(r["label"] in ("positive", "negative") for r in records)
    assert report["rows_read"] == 5
    assert report["duplicates_dropped"] == 1
    assert report["unresolved_xrefs"] == 1
    assert report["missing_abstracts"] == 1
    assert report["usable"] == 2
    assert report["positives"] == 1
    assert report["negatives"] == 1
    assert (tmp_path / "coverage_report.txt").exists()


def test_bare_xrefs_are_mod_prefixed_before_resolving(tmp_path):
    # The ABC's by_cross_reference endpoint needs the MOD-prefixed xref
    # ("ZFIN:ZDB-PUB-..."); the bare ZDB-PUB id 404s for every row. The sheet
    # ships bare ids, so prefixing is what makes resolution work at all.
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(
        "AGRKBID,XREF,Classification\n"
        ",ZDB-PUB-1,Positive \n"
        ",ZFIN:ZDB-PUB-2,Negative\n")

    with patch.object(prep, "get_curie_from_xref",
                      side_effect=["AGRKB:1", "AGRKB:2"]) as mock_resolve, \
         patch.object(prep, "get_reference_title_and_abstract", return_value=("T", "A")):
        prep.prepare(str(csv_path), str(tmp_path))

    # Bare id gets the prefix; an already-prefixed xref is passed through as-is.
    assert [call.args[0] for call in mock_resolve.call_args_list] == [
        "ZFIN:ZDB-PUB-1", "ZFIN:ZDB-PUB-2"]
    # The unprefixed id is what is recorded, since that is what the sheet says.
    records = json.loads((tmp_path / "labelled_abstracts.json").read_text())
    assert [r["xref"] for r in records] == ["ZDB-PUB-1", "ZFIN:ZDB-PUB-2"]


def test_mod_prefix_is_configurable(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text("AGRKBID,XREF,Classification\n,WBPaper1,Positive \n")

    with patch.object(prep, "get_curie_from_xref", return_value="AGRKB:1") as mock_resolve, \
         patch.object(prep, "get_reference_title_and_abstract", return_value=("T", "A")):
        prep.prepare(str(csv_path), str(tmp_path), mod_prefix="WB")

    assert mock_resolve.call_args[0][0] == "WB:WBPaper1"


def test_uses_the_sheets_agrkbid_when_present_instead_of_resolving(tmp_path):
    # The sheet has an AGRKBID column; when it is filled there is no reason to
    # spend an API round-trip resolving the xref.
    csv_path = tmp_path / "in.csv"
    csv_path.write_text("AGRKBID,XREF,Classification\nAGRKB:99,ZDB-PUB-9,Positive \n")

    with patch.object(prep, "get_curie_from_xref") as mock_resolve, \
         patch.object(prep, "get_reference_title_and_abstract",
                      return_value=("T", "A")):
        prep.prepare(str(csv_path), str(tmp_path))

    mock_resolve.assert_not_called()
    records = json.loads((tmp_path / "labelled_abstracts.json").read_text())
    assert records[0]["curie"] == "AGRKB:99"


def test_a_lookup_failure_does_not_abort_the_run(tmp_path):
    # get_reference_title_and_abstract raises (rather than returning None) on a
    # non-200; one bad reference must not lose the other 949.
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(
        "AGRKBID,XREF,Classification\n"
        "AGRKB:1,ZDB-PUB-1,Positive \n"
        "AGRKB:2,ZDB-PUB-2,Negative\n")

    def flaky(curie):
        if curie == "AGRKB:1":
            raise RuntimeError("boom")
        return ("Title two", "Abstract two.")

    with patch.object(prep, "get_reference_title_and_abstract", side_effect=flaky):
        report = prep.prepare(str(csv_path), str(tmp_path))

    records = json.loads((tmp_path / "labelled_abstracts.json").read_text())
    assert [r["curie"] for r in records] == ["AGRKB:2"]
    assert report["lookup_errors"] == 1
    assert report["usable"] == 1


def test_unparseable_labels_are_counted_not_guessed(tmp_path):
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(
        "AGRKBID,XREF,Classification\n"
        "AGRKB:1,ZDB-PUB-1,Maybe\n"
        "AGRKB:2,ZDB-PUB-2,\n"
        "AGRKB:3,,Positive \n")

    with patch.object(prep, "get_reference_title_and_abstract", return_value=("T", "A")):
        report = prep.prepare(str(csv_path), str(tmp_path))

    assert report["unparseable_labels"] == 3
    assert report["usable"] == 0


def test_writes_an_upload_ready_csv_that_sidesteps_the_xref_lookup(tmp_path):
    # dataset_upload_from_csv.py only calls get_curie_from_xref when AGRKBID is
    # blank, and that call 404s on a bare ZDB-PUB id. Emitting the resolved curie
    # means the upload never takes that path — and never re-resolves 950 xrefs.
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(
        "AGRKBID,XREF,Classification\n"
        ",ZDB-PUB-1,Positive \n"
        ",ZDB-PUB-2,Negative\n")

    with patch.object(prep, "get_curie_from_xref", side_effect=["AGRKB:1", "AGRKB:2"]), \
         patch.object(prep, "get_reference_title_and_abstract", return_value=("T", "A")):
        prep.prepare(str(csv_path), str(tmp_path))

    rows = list(csv.DictReader((tmp_path / "upload_ready.csv").open()))
    assert [r["AGRKBID"] for r in rows] == ["AGRKB:1", "AGRKB:2"]
    # Labels lowercased/stripped — the trainer compares == "positive" exactly and
    # add_entry_to_dataset passes the value straight through.
    assert [r["Classification"] for r in rows] == ["positive", "negative"]
    # Xrefs carry the MOD prefix too, so the fallback path also works if AGRKBID
    # is ever cleared.
    assert [r["XREF"] for r in rows] == ["ZFIN:ZDB-PUB-1", "ZFIN:ZDB-PUB-2"]


def test_upload_ready_csv_only_contains_usable_references(tmp_path):
    # A reference that never resolved has no curie to upload; including it would
    # reintroduce the blank-AGRKBID path this file exists to avoid.
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(
        "AGRKBID,XREF,Classification\n"
        ",ZDB-PUB-1,Positive \n"
        ",ZDB-PUB-2,Negative\n")

    with patch.object(prep, "get_curie_from_xref", side_effect=["AGRKB:1", None]), \
         patch.object(prep, "get_reference_title_and_abstract", return_value=("T", "A")):
        prep.prepare(str(csv_path), str(tmp_path))

    rows = list(csv.DictReader((tmp_path / "upload_ready.csv").open()))
    assert [r["AGRKBID"] for r in rows] == ["AGRKB:1"]
    assert all(r["AGRKBID"] for r in rows)


def test_captures_the_unnamed_curator_annotation_column(tmp_path):
    # The real export has a fourth column with an EMPTY header carrying Ceri's
    # 'string "probe" in abstract' marker on the hard negatives. Those are exactly
    # the references a keyword rule gets wrong, so the note must survive to the
    # evaluation rather than being dropped as an unrecognised column.
    csv_path = tmp_path / "in.csv"
    csv_path.write_text(
        "AGRKBID,XREF,Classification,\n"
        'AGRKB:1,ZDB-PUB-1,Negative,"string ""probe"" in abstract"\n'
        "AGRKB:2,ZDB-PUB-2,Negative,\n")

    with patch.object(prep, "get_reference_title_and_abstract", return_value=("T", "A")):
        report = prep.prepare(str(csv_path), str(tmp_path))

    records = json.loads((tmp_path / "labelled_abstracts.json").read_text())
    assert records[0]["note"] == 'string "probe" in abstract'
    assert records[1]["note"] == ""
    assert report["annotated_hard_negatives"] == 1
