from unittest.mock import patch

from utils import ateam_utils


@patch.object(ateam_utils, "_query_ncbi_gene_xrefs")
def test_ncbi_gene_ids_map_to_alliance_genes_in_the_taxon(mock_query):
    mock_query.return_value = [
        ("NCBI_Gene:19883", "MGI:104661", "NCBITaxon:10090"),
        ("NCBI_Gene:19883", "MGI:104661", "NCBITaxon:10090"),  # duplicate row
        ("NCBI_Gene:175410", "WB:WBGene00000001", "NCBITaxon:6239"),
    ]
    result = ateam_utils.get_alliance_genes_for_ncbi_gene_ids([19883, 175410, 769693],
                                                              taxon_curie="NCBITaxon:10090")
    mock_query.assert_called_once_with(["NCBI_Gene:175410", "NCBI_Gene:19883", "NCBI_Gene:769693"])
    assert result == {19883: ["MGI:104661"]}


@patch.object(ateam_utils, "_query_ncbi_gene_xrefs")
def test_ncbi_gene_ids_without_taxon_keep_every_match(mock_query):
    mock_query.return_value = [
        ("NCBI_Gene:19883", "MGI:104661", "NCBITaxon:10090"),
        ("NCBI_Gene:175410", "WB:WBGene00000001", "NCBITaxon:6239"),
    ]
    assert ateam_utils.get_alliance_genes_for_ncbi_gene_ids([19883, 175410]) == {
        19883: ["MGI:104661"], 175410: ["WB:WBGene00000001"]}


@patch.object(ateam_utils, "_query_ncbi_gene_xrefs")
def test_no_ncbi_gene_ids_skips_the_query(mock_query):
    assert ateam_utils.get_alliance_genes_for_ncbi_gene_ids([]) == {}
    mock_query.assert_not_called()
