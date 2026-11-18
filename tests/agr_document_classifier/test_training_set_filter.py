"""SCRUM-6482: keep only the papers the ABC sends to classification when building a training set."""
from argparse import Namespace
from unittest.mock import patch

from agr_document_classifier import agr_document_classifier_trainer as trainer

META = {
    "AGRKB:1": {"date": "2010-05-01", "category": "Research_Article", "mod_reference_types": [("WB", "Experimental")]},
    "AGRKB:2": {"date": "2010-05-01", "category": "Review_Article", "mod_reference_types": [("WB", "Experimental")]},
    "AGRKB:3": {"date": "2010-05-01", "category": "Research_Article", "mod_reference_types": [("WB", "Review")]},
    "AGRKB:4": {"date": "2001-05-01", "category": "Research_Article", "mod_reference_types": [("WB", "Experimental")]},
    "AGRKB:5": {"date": None, "category": "Research_Article", "mod_reference_types": [("WB", "Experimental")]},
    "AGRKB:6": {"date": "2012", "category": "research_article", "mod_reference_types": [("FB", "Experimental")]},
}
TRAINING_SET = {"data_training": {"AGRKB:1": "positive", "AGRKB:2": "positive", "AGRKB:3": "negative",
                                  "AGRKB:4": "negative", "AGRKB:5": "negative", "AGRKB:6": "negative"}}


def _args(**kwargs):
    defaults = {"mod_train": "WB", "filter_date_before": None, "require_category": None,
                "require_mod_reference_type": None}
    defaults.update(kwargs)
    return Namespace(**defaults)


def _filter(args):
    with patch.object(trainer, "get_reference_metadata", side_effect=lambda c: META[c]) as lookup:
        result = trainer.get_filtered_training_curies(args, TRAINING_SET)
    return result, lookup


def test_no_filters_keeps_everything_without_lookups():
    result, lookup = _filter(_args())
    assert result == {"positive": ["AGRKB:1", "AGRKB:2"], "negative": ["AGRKB:3", "AGRKB:4", "AGRKB:5", "AGRKB:6"]}
    lookup.assert_not_called()


def test_date_filter_keeps_unparseable_dates():
    result, _ = _filter(_args(filter_date_before="2005-01-01"))
    assert result == {"positive": ["AGRKB:1", "AGRKB:2"], "negative": ["AGRKB:3", "AGRKB:5", "AGRKB:6"]}


def test_eligibility_filters_category_and_mod_reference_type():
    result, _ = _filter(_args(filter_date_before="2005-01-01", require_category="Research_Article",
                              require_mod_reference_type="Experimental"))
    assert result == {"positive": ["AGRKB:1"], "negative": ["AGRKB:5"]}


def test_unreachable_reference_is_dropped_by_eligibility_filters():
    with patch.object(trainer, "get_reference_metadata", return_value=None):
        result = trainer.get_filtered_training_curies(_args(require_category="Research_Article"), TRAINING_SET)
    assert result == {"positive": [], "negative": []}


def test_category_matches_api_enum_value():
    meta = {"date": "2010", "category": "research_article", "mod_reference_types": [("WB", "Experimental")]}
    with patch.object(trainer, "get_reference_metadata", return_value=meta):
        result = trainer.get_filtered_training_curies(
            _args(require_category="Research_Article", require_mod_reference_type="Experimental"), TRAINING_SET)
    assert len(result["positive"]) == 2 and len(result["negative"]) == 4


def test_metadata_is_cached_between_runs(tmp_path):
    args = _args(filter_date_before="2005-01-01", require_category="Research_Article",
                 require_mod_reference_type="Experimental")
    args.embedding_cache_dir = str(tmp_path)
    first, lookup1 = _filter(args)
    assert lookup1.call_count == 6
    second, lookup2 = _filter(args)
    lookup2.assert_not_called()
    assert first == second == {"positive": ["AGRKB:1"], "negative": ["AGRKB:5"]}
