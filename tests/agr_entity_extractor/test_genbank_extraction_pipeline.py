import io
from unittest.mock import MagicMock, patch

from agr_entity_extractor import agr_genbank_extraction_pipeline as pipeline
from agr_entity_extractor.agr_genbank_extraction_pipeline import (
    ReferenceHits,
    assign_genes,
    collect_reference_hits,
    format_note,
)
from utils.genbank_utils import AccessionHit


def test_collect_reference_hits_merges_sources_and_skips_large_supplements():
    big_table = " ".join(f"NM_{100000 + i}" for i in range(6))
    texts = [
        ("main", "GenBank AF123456 and NM_013646."),
        ("supp_1", "Table: NM_013646, AK159011"),
        ("supp_2", big_table),
    ]
    ref_hits = collect_reference_hits(texts, max_accessions_per_supplement=5)
    assert set(ref_hits.hits) == {"AF123456", "NM_013646", "AK159011"}
    assert ref_hits.hits["NM_013646"].count == 2
    assert ref_hits.sources["NM_013646"] == {"main", "supp_1"}
    assert ref_hits.skipped_supplements == [("supp_2", 6)]


def test_collect_reference_hits_never_caps_the_main_text():
    main = " ".join(f"NM_{100000 + i}" for i in range(6))
    ref_hits = collect_reference_hits([("main", main)], max_accessions_per_supplement=5)
    assert len(ref_hits.hits) == 6


def _ref_hits(*accessions):
    ref_hits = ReferenceHits()
    for accession in accessions:
        ref_hits.hits[accession] = AccessionHit(accession, "nuccore", count=1)
        ref_hits.sources[accession].add("main")
    return ref_hits


def test_assign_genes_keeps_single_gene_accessions_in_the_mod():
    ref_hits = _ref_hits("NM_013646", "AK159011", "AC123456", "AF123456", "BX571857")
    accession_genes = {
        "NM_013646": [19883],          # Rora
        "AK159011": [19883],           # same gene, second accession
        "AC123456": [1, 2, 3],         # BAC clone: several genes
        "AF123456": [769693],          # chicken gene, not an Alliance gene for this MOD
        "BX571857": [],                # bacterial genome, no gene link
    }
    ncbi_to_alliance = {19883: ["MGI:104661"], 1: ["MGI:1"]}
    assert assign_genes(ref_hits, accession_genes, ncbi_to_alliance) == {
        "MGI:104661": ["AK159011", "NM_013646"]}


def test_assign_genes_max_genes_per_accession_is_configurable():
    ref_hits = _ref_hits("XM_000001")
    genes = assign_genes(ref_hits, {"XM_000001": [1, 2]}, {1: ["WB:WBGene1"], 2: ["WB:WBGene2"]},
                         max_genes_per_accession=2)
    assert genes == {"WB:WBGene1": ["XM_000001"], "WB:WBGene2": ["XM_000001"]}


def test_format_note_says_where_each_accession_was_found():
    sources = {"NM_013646": {"main"}, "AK159011": {"supp_1"}, "AF123456": {"main", "supp_2"}}
    assert format_note(["AF123456", "AK159011", "NM_013646"], sources) == (
        "GenBank/RefSeq accession: AF123456 (main text and supplement), AK159011 (supplement), "
        "NM_013646 (main text)")


def _write(path, text):
    path.write_text(text, encoding="utf-8")


@patch.object(pipeline, "send_entity_tag_to_abc", return_value=True)
@patch.object(pipeline, "get_alliance_genes_for_ncbi_gene_ids")
@patch.object(pipeline, "read_reference_texts")
def test_process_references_sends_one_tag_per_gene(mock_texts, mock_lookup, mock_send):
    mock_texts.side_effect = lambda md_dir, curie: {
        "AGRKB:1": [("main", "GenBank AF123456"), ("supp_1", "NM_013646 AK159011")],
        "AGRKB:2": [("main", "no accessions here")],
    }[curie]
    resolver = MagicMock()
    resolver.resolve.return_value = {"AF123456": [769693], "NM_013646": [19883], "AK159011": [19883]}
    mock_lookup.return_value = {19883: ["MGI:104661"]}
    hits_fh = io.StringIO()

    n = pipeline.process_references("MGI", ["AGRKB:1", "AGRKB:2"], "/unused", resolver,
                                    hits_fh=hits_fh, tet_source_id=7)

    assert n == 1
    mock_lookup.assert_called_once_with({769693, 19883}, taxon_curie="NCBITaxon:10090")
    mock_send.assert_called_once()
    kwargs = mock_send.call_args.kwargs
    assert kwargs["reference_curie"] == "AGRKB:1"
    assert kwargs["entity"] == "MGI:104661"
    assert kwargs["topic"] == kwargs["entity_type"] == "ATP:0000005"
    assert kwargs["species"] == "NCBITaxon:10090"
    assert kwargs["tet_source_id"] == 7
    assert kwargs["note"] == "GenBank/RefSeq accession: AK159011 (supplement), NM_013646 (supplement)"
    assert "negated" not in kwargs
    rows = [line.split("\t") for line in hits_fh.getvalue().splitlines()]
    assert [r[2] for r in rows] == ["AF123456", "AK159011", "NM_013646"]
    assert rows[0][7] == ""            # chicken gene has no MGI match
    assert rows[2][7] == "MGI:104661"


@patch.object(pipeline, "send_entity_tag_to_abc")
@patch.object(pipeline, "get_alliance_genes_for_ncbi_gene_ids", return_value={19883: ["MGI:104661"]})
def test_process_references_test_mode_reads_md_dir_and_sends_nothing(_lookup, mock_send, tmp_path):
    _write(tmp_path / "AGRKB_1.md", "# Title\n\nThe Rora transcript (GenBank NM_013646.3).\n")
    _write(tmp_path / "AGRKB_1.supp_1.md", "| Gene | Accession |\n|---|---|\n| Rora | AK159011 |\n")
    resolver = MagicMock()
    resolver.resolve.return_value = {"NM_013646": [19883], "AK159011": [19883]}
    test_fh = io.StringIO()

    with patch.object(pipeline, "AllianceMarkdown") as mock_md_cls:
        def load(path):
            mock_md_cls.return_value.get_fulltext.return_value = open(path, encoding="utf-8").read()
        mock_md_cls.return_value.load_from_file.side_effect = load
        n = pipeline.process_references("MGI", ["AGRKB:1"], str(tmp_path), resolver, test_fh=test_fh)

    assert n == 1
    mock_send.assert_not_called()
    assert test_fh.getvalue() == (
        "AGRKB:1\tMGI:104661\tGenBank/RefSeq accession: AK159011 (supplement), NM_013646 (main text)\n")
