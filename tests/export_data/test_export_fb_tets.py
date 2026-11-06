"""SCRUM-6669: the weekly FB export reports a negated "genetic data" (ATP:0000402) tag as nocur.
SCRUM-6670: it reads topic_entity_tag only; the old manual_indexing_tag rows are backfilled."""
from datetime import datetime
from unittest.mock import patch

from export_data import export_fb_tets, export_fb_tets_using_score

DATE = datetime(2026, 10, 30)


def _dump(module, tet_rows):
    written = {}
    with patch.object(module, "get_data", return_value=tet_rows), \
            patch.object(module, "write_to_top", side_effect=lambda path, text: written.update({path: text})):
        module.dump_tet()
    return written


def test_negated_genetic_data_tag_is_a_positive_nocur_flag():
    written = _dump(export_fb_tets, [(DATE, "ATP:0000402", "HIGH", "PMID:12345")])
    assert written == {'/curation_status/textmining_positive_ABC.txt': "261030\t\t12345\tnocur:high\tcam\n"}


def test_genetic_data_tag_with_its_score_in_the_score_export():
    written = _dump(export_fb_tets_using_score, [(DATE, "ATP:0000402", 0.91, "PMID:12345", "not_validated")])
    assert written == {'/curation_status/textmining_positive_ABC_using_score.txt':
                       "261030\t\t12345\tnocur:0.91\tcam\tnot_validated\n"}


def test_only_negated_genetic_data_tags_are_selected():
    captured = []

    class _Session:
        def execute(self, query):
            captured.append(str(query))
            return self

        def fetchall(self):
            return []

    for module in (export_fb_tets, export_fb_tets_using_score):
        with patch.object(module, "create_postgres_session", return_value=_Session()):
            module.get_data()
    assert len(captured) == 2
    assert all("AND (tet.topic != 'ATP:0000402' OR tet.negated)" in query for query in captured)
    # SCRUM-6670: no manual_indexing_tag query any more
    assert not any("manual_indexing_tag" in query for query in captured)


def test_files_are_only_written_when_fb_svn_commit_is_true(tmp_path, monkeypatch, capsys):
    """Only the production GoCD pipeline sets FB_SVN_COMMIT=true. Anything else prints instead."""
    for module in (export_fb_tets, export_fb_tets_using_score):
        target = tmp_path / f"{module.__name__}.txt"
        target.write_text("old\n")

        monkeypatch.delenv("FB_SVN_COMMIT", raising=False)
        module.write_to_top(str(target), "new\n")
        assert target.read_text() == "old\n"
        assert "is not written. It would add:\nnew\n" in capsys.readouterr().out

        monkeypatch.setenv("FB_SVN_COMMIT", "false")
        module.write_to_top(str(target), "new\n")
        assert target.read_text() == "old\n"

        monkeypatch.setenv("FB_SVN_COMMIT", "true")
        module.write_to_top(str(target), "new\n")
        assert target.read_text() == "new\nold\n"
