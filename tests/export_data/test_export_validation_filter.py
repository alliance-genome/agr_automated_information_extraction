"""The weekly FB exports leave out only what a curator marked wrong.

An unvalidated row has a NULL validation, and in SQL `NULL != 'x'` is not true, so a `!=` filter
dropped every unvalidated row: from 2026-07-30 no "no genetic data" (manual_indexing_tag) row reached
FB. IS DISTINCT FROM keeps NULL. The two tables use different values for "wrong".
"""
from unittest.mock import patch

from export_data import export_fb_tets, export_fb_tets_using_score


def _queries(module):
    captured = {}

    class _Session:
        def execute(self, query):
            captured.setdefault('sql', []).append(str(query))
            return self

        def fetchall(self):
            return []

    with patch.object(module, "create_postgres_session", return_value=_Session()):
        module.get_data('tet')
        module.get_data('mi')
    return captured['sql']


def test_both_exports_keep_unvalidated_rows():
    for module in (export_fb_tets, export_fb_tets_using_score):
        tet_sql, mit_sql = _queries(module)
        assert "tet.validation_by_professional_biocurator IS DISTINCT FROM 'validated_wrong'" in tet_sql
        assert "mit.validation_by_biocurator IS DISTINCT FROM 'wrong'" in mit_sql
        assert "!= 'validated_wrong'" not in tet_sql + mit_sql
