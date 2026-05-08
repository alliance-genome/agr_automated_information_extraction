import os
import tempfile
import textwrap

from utils.md_utils import AllianceMarkdown


SAMPLE_MD = textwrap.dedent(
    """\
    # On the role of daf-16 in *C. elegans*

    **Journal:** Cell

    **DOI:** 10.1234/abc

    ## Abstract

    This paper studies the role of *daf-16* in **longevity**.

    ## Introduction

    The dauer pathway involves daf-2 and daf-16. See Figure 1.

    **Figure 1.** Pathway diagram.

    ## Results

    daf-16 mutants are short-lived. See Table 1.

    | Strain | Lifespan |
    | --- | --- |
    | WT | 20 |
    | daf-16 | 12 |

    **Table 1.** Lifespan data.

    ## References

    1. Doe, J. (2020) The pathway. *Nature*, 1(1), 1-2. doi:10.1/foo
    2. Smith, K. (2019) Another paper. *Cell*, 5, 100-110.
    """
)


def _write_md(content: str) -> str:
    fh = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
    try:
        fh.write(content)
    finally:
        fh.close()
    return fh.name


def test_load_from_file_populates_document():
    path = _write_md(SAMPLE_MD)
    try:
        md = AllianceMarkdown()
        md.load_from_file(path)
        assert md.doc is not None
        assert md.raw_md is not None
    finally:
        os.unlink(path)


def test_get_title_strips_markdown_formatting():
    path = _write_md(SAMPLE_MD)
    try:
        md = AllianceMarkdown()
        md.load_from_file(path)
        # Italics in the title should be stripped.
        assert md.get_title() == "On the role of daf-16 in C. elegans"
    finally:
        os.unlink(path)


def test_get_abstract_returns_plain_text():
    path = _write_md(SAMPLE_MD)
    try:
        md = AllianceMarkdown()
        md.load_from_file(path)
        abstract = md.get_abstract()
        # Bold/italic markers must be gone.
        assert "**" not in abstract
        assert "*daf-16*" not in abstract
        assert "daf-16" in abstract
        assert "longevity" in abstract
    finally:
        os.unlink(path)


def test_get_fulltext_excludes_references():
    path = _write_md(SAMPLE_MD)
    try:
        md = AllianceMarkdown()
        md.load_from_file(path)
        ft = md.get_fulltext()
        # Body content should be present.
        assert "daf-16" in ft
        assert "daf-2" in ft
        assert "Lifespan" in ft  # table caption
        assert "WT" in ft        # table cell
        # References must not leak in.
        assert "Doe, J." not in ft
        assert "Smith, K." not in ft
    finally:
        os.unlink(path)


def test_get_fulltext_excludes_supplements():
    main_md = textwrap.dedent(
        """\
        # Main Paper

        ## Abstract

        Body of the main paper.
        """
    )
    supplement_md = textwrap.dedent(
        """\
        SHOULD_NOT_APPEAR

        Some supplemental content.
        """
    )
    # Splice the supplement at the end of the main markdown the way the
    # md_reader recognises it (currently the parser treats trailing content
    # before sub-articles as part of the main doc; the supplements list is
    # populated only when callers pass them via load_document_with_supplements,
    # which we do not use). We still assert the AllianceMarkdown wrapper
    # never includes ``doc.supplements`` content.
    path = _write_md(main_md)
    try:
        md = AllianceMarkdown()
        md.load_from_file(path)
        # Manually attach a supplement Document to mimic what the loader
        # would do if the caller asked for it.
        from agr_abc_document_parsers import read_markdown
        md.doc.supplements.append(read_markdown(supplement_md))
        ft = md.get_fulltext()
        assert "SHOULD_NOT_APPEAR" not in ft
        assert "Body of the main paper." in ft
    finally:
        os.unlink(path)


def test_get_sentences_returns_list_of_strings():
    path = _write_md(SAMPLE_MD)
    try:
        md = AllianceMarkdown()
        md.load_from_file(path)
        sentences = md.get_sentences()
        assert isinstance(sentences, list)
        assert all(isinstance(s, str) for s in sentences)
        assert any("daf-16 mutants are short-lived" in s for s in sentences)
    finally:
        os.unlink(path)


def test_methods_handle_unloaded_state():
    md = AllianceMarkdown()
    assert md.get_title() == ""
    assert md.get_abstract() == ""
    assert md.get_fulltext() == ""
    assert md.get_sentences() == []


def test_load_from_file_handles_non_utf8_bytes():
    # Latin-1 byte sequence that is not valid UTF-8.
    raw = "# Café\n\n## Abstract\n\nBody.\n".encode("latin-1")
    fh = tempfile.NamedTemporaryFile(suffix=".md", delete=False)
    try:
        fh.write(raw)
        fh.close()
        md = AllianceMarkdown()
        md.load_from_file(fh.name)
        # No exception; raw_md is decoded somehow and parsing succeeds.
        assert md.raw_md is not None
        assert md.doc is not None
    finally:
        os.unlink(fh.name)


def test_load_with_supplements_auto_discovers_siblings(tmp_path):
    """When include_supplements=True, sibling ``<basename>.supp_<N>.md`` files
    are picked up and merged into the Document; their text appears in
    get_fulltext output."""
    main = tmp_path / "AGRKB_1.md"
    main.write_text("# Main paper\n\n## Abstract\n\nMain abstract.\n", encoding="utf-8")
    (tmp_path / "AGRKB_1.supp_1.md").write_text(
        "## S1\n\nSUPP_ONE_BODY\n", encoding="utf-8")
    (tmp_path / "AGRKB_1.supp_2.md").write_text(
        "## S2\n\nSUPP_TWO_BODY\n", encoding="utf-8")
    # Unrelated MD file in same dir — must NOT be loaded as a supplement.
    (tmp_path / "AGRKB_2.md").write_text("# Other paper\n", encoding="utf-8")

    md = AllianceMarkdown()
    md.load_from_file(str(main), include_supplements=True)

    assert md.has_supplements is True
    assert len(md.doc.supplements) == 2
    ft = md.get_fulltext()
    assert "Main paper" in ft
    assert "Main abstract" in ft
    assert "SUPP_ONE_BODY" in ft
    assert "SUPP_TWO_BODY" in ft
    assert "Other paper" not in ft


def test_load_with_supplements_default_is_off(tmp_path):
    """Without include_supplements, sibling supp_*.md files are ignored."""
    main = tmp_path / "AGRKB_1.md"
    main.write_text("# Main paper\n\n## Abstract\n\nMain abstract.\n", encoding="utf-8")
    (tmp_path / "AGRKB_1.supp_1.md").write_text(
        "## S1\n\nSUPP_ONE_BODY\n", encoding="utf-8")

    md = AllianceMarkdown()
    md.load_from_file(str(main))

    assert md.has_supplements is False
    assert md.doc.supplements == []
    ft = md.get_fulltext()
    assert "Main paper" in ft
    assert "SUPP_ONE_BODY" not in ft


def test_explicit_supplement_paths(tmp_path):
    """An explicit list of supplement_paths overrides auto-discovery."""
    main = tmp_path / "AGRKB_1.md"
    main.write_text("# Main paper\n\n## Abstract\n\nMain abstract.\n", encoding="utf-8")
    explicit = tmp_path / "elsewhere.md"
    explicit.write_text("## S\n\nEXPLICIT_BODY\n", encoding="utf-8")
    # A discoverable supp file that should be ignored because we passed an
    # explicit list.
    (tmp_path / "AGRKB_1.supp_1.md").write_text(
        "## S1\n\nDISCOVERABLE_BODY\n", encoding="utf-8")

    md = AllianceMarkdown()
    md.load_from_file(str(main), supplement_paths=[str(explicit)])

    assert md.has_supplements is True
    ft = md.get_fulltext()
    assert "EXPLICIT_BODY" in ft
    assert "DISCOVERABLE_BODY" not in ft


def test_convert_all_md_files_in_dir_skips_supplements(tmp_path):
    """The dir->txt converter must not produce a separate .txt for each
    supplement; supplements are consumed alongside their main file."""
    from utils.md_utils import convert_all_md_files_in_dir_to_txt

    (tmp_path / "AGRKB_1.md").write_text(
        "# Main\n\n## Abstract\n\nMain body.\n", encoding="utf-8")
    (tmp_path / "AGRKB_1.supp_1.md").write_text(
        "## S\n\nSupp body.\n", encoding="utf-8")

    convert_all_md_files_in_dir_to_txt(str(tmp_path))

    files = sorted(p.name for p in tmp_path.iterdir())
    assert "AGRKB_1.txt" in files
    # No txt for the supplement.
    assert "AGRKB_1.supp_1.txt" not in files
