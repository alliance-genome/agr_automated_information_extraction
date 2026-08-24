from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from scripts import evaluate_probe_classifiers as ev
from scripts import generate_abstract_embeddings as gen


def _records():
    return [{"curie": f"AGRKB:{i}", "label": "positive" if i % 2 else "negative",
             "title": f"Title {i}", "abstract": f"Abstract number {i} about probes."}
            for i in range(1, 7)]


def _parquets(tmp_path, records):
    for i, record in enumerate(records):
        with patch.object(gen, "embed_texts", return_value=[[float(i), 1.0, 0.0]]):
            gen.generate([record], str(tmp_path))


def test_embedding_bow_matrix_is_sparse_and_wider_than_bow_alone(tmp_path):
    records = _records()
    _parquets(tmp_path, records)

    X_both, y_both, kept_both = ev.build_matrix(
        records, str(tmp_path), use_embedding=True, use_bow=True)
    X_bow, y_bow, kept_bow = ev.build_matrix(
        records, str(tmp_path), use_embedding=False, use_bow=True)

    assert sp.issparse(X_both) and sp.issparse(X_bow)
    assert X_both.shape[0] == X_bow.shape[0] == len(records)
    # The dense embedding block adds exactly its own width (3 here) on top of BoW.
    assert X_both.shape[1] == X_bow.shape[1] + 3
    # _records() starts at i=1, so odd i (1,3,5) are the positives.
    assert y_both == y_bow == [1, 0, 1, 0, 1, 0]
    assert kept_both == kept_bow == records


def test_build_matrix_skips_references_with_no_parquet_and_reports_kept(tmp_path):
    records = _records()
    _parquets(tmp_path, records[:4])   # last two have no parquet
    X, y, kept = ev.build_matrix(records, str(tmp_path), use_embedding=True, use_bow=True)
    assert X.shape[0] == 4
    assert len(y) == 4
    # Row i must correspond to kept[i], or the LLM arm would score different
    # references than the supervised arms.
    assert [r["curie"] for r in kept] == ["AGRKB:1", "AGRKB:2", "AGRKB:3", "AGRKB:4"]


def test_holdout_indices_reproduce_the_trainer_internal_split():
    # _select_and_fit_model splits 80/20 stratified with random_state=42. The
    # arms are only comparable if we can reproduce exactly that test set.
    from sklearn.model_selection import train_test_split as sk_split

    y = np.array([0, 1] * 25)
    train_idx, test_idx = ev.holdout_indices(y)
    expected_train, expected_test = sk_split(
        np.arange(len(y)), test_size=0.20, stratify=y, random_state=42)
    np.testing.assert_array_equal(train_idx, expected_train)
    np.testing.assert_array_equal(test_idx, expected_test)


def test_bow_only_can_be_built_from_records_without_any_parquet(tmp_path):
    # The BoW block hashes abstract_chunk_text(title, abstract), which is exactly
    # what the parquet's `content` column holds — so the pure-ML baseline needs no
    # embeddings, no OpenAI key, and no parquet directory at all.
    records = _records()
    X, y, kept = ev.build_matrix(records, None, use_embedding=False, use_bow=True)
    assert X.shape[0] == len(records)
    assert kept == records

    # ...and it must be identical to going through the parquets.
    _parquets(tmp_path, records)
    X_parquet, _y, _kept = ev.build_matrix(
        records, str(tmp_path), use_embedding=False, use_bow=True)
    assert (X != X_parquet).nnz == 0


def test_build_matrix_rejects_embeddings_without_a_parquet_dir():
    # Embeddings can only come from the parquets; silently returning a BoW-only
    # matrix here would mislabel the arm in the results table.
    try:
        ev.build_matrix(_records(), None, use_embedding=True, use_bow=True)
    except ValueError as exc:
        assert "parquet_dir" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_out_of_fold_gives_every_record_a_held_out_probability():
    # The single 80/20 split leaves most hard negatives in the training set, so
    # the hard-negative number has to come from out-of-fold predictions covering
    # all of them.
    records = [{"curie": f"AGRKB:{i}",
                "label": "positive" if i % 2 else "negative",
                "title": f"Title {i}",
                "abstract": ("probe synthesis characterization dye" if i % 2
                             else "gene expression phenotype heart development")}
               for i in range(1, 41)]
    # LogisticRegression rather than the LGBM default: LightGBM's min_child_samples
    # floor stops it splitting a 30-row training fold at all, which would make
    # every probability 0.5 for reasons that have nothing to do with this code.
    y, y_prob, kept = ev.out_of_fold_probabilities(
        records, None, use_embedding=False, use_bow=True, n_splits=4,
        model_name="LogisticRegression")
    assert len(y) == len(y_prob) == len(kept) == 40
    # Every entry was actually written by some fold (default array is zeros, so
    # assert the two classes separate rather than just "not all zero").
    assert y_prob[y == 1].mean() > y_prob[y == 0].mean()


def test_build_matrix_requires_at_least_one_feature_block():
    try:
        ev.build_matrix([], "/nonexistent", use_embedding=False, use_bow=False)
    except ValueError as exc:
        assert "use_embedding" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_scores_reports_recall_at_each_precision_target():
    # A perfectly separable ranking must reach full recall at every precision
    # target; this is the number the ZFIN auto-tagging gate is chosen from.
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.01, 0.02, 0.03, 0.04, 0.96, 0.97, 0.98, 0.99])
    y_pred = (y_prob >= 0.5).astype(int)
    scores = ev._scores(y_true, y_pred, y_prob)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["n_test"] == 8
    assert scores["n_positive_test"] == 4
    for target in (90, 95, 99):
        assert scores[f"recall_at_precision_{target}"]["recall"] == 1.0


def test_scores_reports_none_when_a_precision_target_is_unreachable():
    # An uninformative ranking cannot hit 99% precision; the report must say so
    # rather than silently omitting the row or inventing a threshold.
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.5, 0.5, 0.5, 0.5])
    y_pred = np.array([1, 1, 1, 1])
    scores = ev._scores(y_true, y_pred, y_prob)
    assert scores["recall_at_precision_99"] is None


def test_hard_negative_breakdown_scores_the_annotated_subset():
    # Ceri annotated the negatives whose abstract contains the word "probe".
    # Those are the ones a keyword rule gets wrong, so they get their own number.
    records = [{"curie": "AGRKB:1", "label": "negative", "note": 'string "probe" in abstract'},
               {"curie": "AGRKB:2", "label": "negative", "note": 'string "probe" in abstract'},
               {"curie": "AGRKB:3", "label": "negative", "note": ""},
               {"curie": "AGRKB:4", "label": "positive", "note": ""}]
    # Model calls the first hard negative wrong, the second right.
    y_pred = np.array([1, 0, 0, 1])
    breakdown = ev.hard_negative_breakdown(records, y_pred)
    assert breakdown["n_hard_negatives"] == 2
    assert breakdown["false_positives"] == 1
    assert breakdown["specificity"] == 0.5


def test_hard_negative_breakdown_is_none_without_annotations():
    records = [{"curie": "AGRKB:1", "label": "negative", "note": ""}]
    assert ev.hard_negative_breakdown(records, np.array([0])) is None


def test_llm_arm_parses_structured_verdicts():
    responses = ['{"label": "positive", "confidence": 0.91}',
                 '{"label": "negative", "confidence": 0.12}']
    with patch.object(ev, "_chat_json", side_effect=responses):
        first = ev.classify_with_llm(
            {"title": "Synthesis of a probe", "abstract": "We characterize a new dye."},
            few_shot=[], model="gpt-5.4-nano")
        second = ev.classify_with_llm(
            {"title": "Gene X in heart development", "abstract": "We used a probe to stain."},
            few_shot=[], model="gpt-5.4-nano")
    assert first == (1, 0.91)
    assert second == (0, 0.12)


def test_llm_arm_treats_unparseable_output_as_negative_low_confidence():
    # A malformed verdict must not crash a 950-abstract run, and must not be
    # silently counted as a positive.
    with patch.object(ev, "_chat_json", return_value="not json at all"):
        assert ev.classify_with_llm({"title": "T", "abstract": "A"},
                                    few_shot=[], model="gpt-5.4-nano") == (0, 0.0)


def test_llm_arm_survives_a_transport_failure():
    # One API error in a 240-abstract run must not lose the other 239.
    with patch.object(ev, "_chat_json", side_effect=RuntimeError("rate limited")):
        assert ev.classify_with_llm({"title": "T", "abstract": "A"},
                                    few_shot=[], model="gpt-5.4-nano") == (0, 0.0)


def test_llm_prompt_includes_few_shot_examples_and_the_target():
    prompt = ev._prompt_for(
        {"title": "Target title", "abstract": "Target abstract."},
        [{"title": "Shot title", "abstract": "Shot abstract.", "label": "positive"}])
    assert "Shot title" in prompt and '"label": "positive"' in prompt
    # The reference under test comes last, with an open Answer: for the model.
    assert prompt.rstrip().endswith("Answer:")
    assert "Target abstract." in prompt


def test_evaluate_scores_every_arm_on_the_same_shared_holdout(tmp_path):
    # The whole comparison is only valid if the arms see one identical test set.
    records = [{"curie": f"AGRKB:{i}", "label": "positive" if i % 2 else "negative",
                "title": f"Title {i}", "abstract": f"Abstract {i} about probes."}
               for i in range(1, 21)]
    parquet_dir = tmp_path / "parquets"
    for i, record in enumerate(records):
        with patch.object(gen, "embed_texts", return_value=[[float(i), 1.0, 0.0]]):
            gen.generate([record], str(parquet_dir))

    seen_by_llm = []

    def fake_llm(test_records, few_shot, model):
        seen_by_llm.extend(r["curie"] for r in test_records)
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "average_precision": 1.0,
                "n_test": len(test_records), "recall_at_precision_95": None}

    with patch.object(ev, "run_supervised",
                      return_value={"precision": 0.8, "recall": 0.7, "f1": 0.75,
                                    "average_precision": 0.8,
                                    "recall_at_precision_95": None}), \
         patch.object(ev, "run_llm", side_effect=fake_llm):
        results = ev.evaluate(records, str(parquet_dir), str(tmp_path / "out"))

    y = np.array([int(r["label"] == "positive") for r in records])
    _train_idx, test_idx = ev.holdout_indices(y)
    expected_test = [records[i]["curie"] for i in test_idx]
    assert seen_by_llm == expected_test
    assert results["n_test"] == len(expected_test)
    assert set(results["arms"]) == {"embedding+bow", "bow_only", "llm"}
    assert (tmp_path / "out" / "results.json").exists()
    assert (tmp_path / "out" / "results.md").exists()


def test_evaluate_can_skip_the_llm_arm(tmp_path):
    # The supervised arms must be runnable with no OPENAI_API_KEY at all.
    records = [{"curie": f"AGRKB:{i}", "label": "positive" if i % 2 else "negative",
                "title": f"Title {i}", "abstract": f"Abstract {i}."}
               for i in range(1, 11)]
    parquet_dir = tmp_path / "parquets"
    for i, record in enumerate(records):
        with patch.object(gen, "embed_texts", return_value=[[float(i), 1.0, 0.0]]):
            gen.generate([record], str(parquet_dir))

    with patch.object(ev, "run_supervised",
                      return_value={"precision": 0.8, "recall": 0.7, "f1": 0.75,
                                    "average_precision": 0.8,
                                    "recall_at_precision_95": None}), \
         patch.object(ev, "run_llm") as mock_llm:
        results = ev.evaluate(records, str(parquet_dir), str(tmp_path / "out"), skip_llm=True)

    mock_llm.assert_not_called()
    assert set(results["arms"]) == {"embedding+bow", "bow_only"}


def test_run_llm_ranks_confident_negatives_below_unsure_positives():
    # The PR curve needs a single monotonic score. A confident negative must
    # rank below an unsure positive, or average_precision is meaningless.
    records = [{"curie": "AGRKB:1", "label": "positive", "title": "T", "abstract": "A"},
               {"curie": "AGRKB:2", "label": "negative", "title": "T", "abstract": "A"}]
    with patch.object(ev, "classify_with_llm", side_effect=[(1, 0.55), (0, 0.99)]):
        result = ev.run_llm(records, few_shot=[], model="gpt-5.4-nano")
    assert result["n_test"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["model"] == "gpt-5.4-nano"
