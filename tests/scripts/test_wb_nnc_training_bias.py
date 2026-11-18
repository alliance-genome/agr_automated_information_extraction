from scripts.wb_nnc_training_bias import NO_BUCKET, Paper, bucket_stats, format_table, subsample_size


def _papers():
    papers = []
    # corpus: 10 HIGH, 90 NEG; training: 8 HIGH (6 positive), 9 NEG (1 positive)
    for i in range(10):
        cls = None if i >= 8 else ("positive" if i < 6 else "negative")
        papers.append(Paper(f"AGRKB:H{i}", "HIGH", cls, 2010))
    for i in range(90):
        cls = None if i >= 9 else ("positive" if i == 0 else "negative")
        papers.append(Paper(f"AGRKB:N{i}", "NEG", cls, 2010))
    # a training paper NNC never scored, and an old paper the date filter drops
    papers.append(Paper("AGRKB:X", NO_BUCKET, "negative", 2012))
    papers.append(Paper("AGRKB:OLD", "HIGH", "positive", 1999))
    return papers


def test_bucket_stats():
    stats = {s.bucket: s for s in bucket_stats(_papers(), year_from=2005)}
    high, neg = stats["HIGH"], stats["NEG"]
    assert (high.corpus, high.training, high.positive, high.negative) == (10, 8, 6, 2)
    assert (neg.corpus, neg.training, neg.positive, neg.negative) == (90, 9, 1, 8)
    assert high.sampled_pct == 80.0 and neg.sampled_pct == 10.0
    assert high.corpus_mix_pct == 10.0 and neg.corpus_mix_pct == 90.0
    assert high.training_mix_pct == 47.1 and neg.training_mix_pct == 52.9
    # weight = corpus mix / training mix
    assert high.weight == 0.21 and neg.weight == 1.7
    # unscored training papers are reported but left out of the mixes
    assert stats[NO_BUCKET].training == 1 and stats[NO_BUCKET].corpus_mix_pct is None
    assert "MEDIUM" in stats and stats["MEDIUM"].corpus == 0


def test_date_filter():
    assert {s.bucket: s for s in bucket_stats(_papers())}["HIGH"].corpus == 11
    assert {s.bucket: s for s in bucket_stats(_papers(), year_from=2005)}["HIGH"].corpus == 10


def test_subsample_size():
    # NEG limits it: 9 NEG at a 90% share -> 10 papers: 1 HIGH, 9 NEG
    size, per_bucket = subsample_size(bucket_stats(_papers(), year_from=2005))
    assert per_bucket["HIGH"] == 1 and per_bucket["NEG"] == 9
    assert size == 10


def test_format_table():
    text = format_table("ATP:0000061", (65, 2), bucket_stats(_papers(), year_from=2005))
    assert "## ATP:0000061 (dataset 65, version 2)" in text
    assert "| HIGH | 10 | 8 | 6 | 2 | 80.0 | 10.0 | 47.1 | 75.0 | 0.21 |" in text
    assert "keeps 10 of 17 papers" in text
