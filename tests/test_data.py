from ragcmp.data import build_corpus, gold_titles, paragraph_text, sample_indices


def _ex(titles, sents, sup):
    return {"context": {"title": titles, "sentences": sents}, "supporting_facts": {"title": sup, "sent_id": [0] * len(sup)}}


def test_sampling_is_deterministic_and_disjoint():
    a = sample_indices(1000, 50, 300, seed=42)
    b = sample_indices(1000, 50, 300, seed=42)
    assert a == b
    dev, test = a
    assert len(dev) == 50 and len(test) == 300
    assert not set(dev) & set(test)


def test_gold_titles_dedup_preserves_order():
    ex = _ex([], [], ["B", "A", "B", "A"])
    assert gold_titles(ex) == ["B", "A"]


def test_corpus_pools_and_dedups_by_title():
    e1 = _ex(["A", "B"], [["a1. ", "a2."], ["b1."]], ["A"])
    e2 = _ex(["B", "C"], [["b-other."], ["c1."]], ["C"])
    corpus = build_corpus([e1, e2])
    assert [c["id"] for c in corpus] == ["A", "B", "C"]
    assert corpus[1]["text"] == paragraph_text("B", ["b1."])  # first occurrence wins
    assert corpus[0]["text"] == "A: a1. a2."
