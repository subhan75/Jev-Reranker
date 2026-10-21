import math

import pytest

from ragcmp.metrics import (
    all_gold_at_k, exact_match, f1_score, mcnemar, mrr, ndcg_at_k, normalize_answer,
    paired_bootstrap, recall_at_k,
)

RANKED = ["x", "g1", "y", "z", "g2", "w"]
GOLD = ["g1", "g2"]


def test_recall():
    assert recall_at_k(RANKED, GOLD, 1) == 0.0
    assert recall_at_k(RANKED, GOLD, 3) == 0.5
    assert recall_at_k(RANKED, GOLD, 5) == 1.0


def test_all_gold():
    assert all_gold_at_k(RANKED, GOLD, 4) == 0.0
    assert all_gold_at_k(RANKED, GOLD, 5) == 1.0


def test_mrr():
    assert mrr(RANKED, GOLD) == 0.5
    assert mrr(["a", "b"], GOLD) == 0.0


def test_ndcg_hand_computed():
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert ndcg_at_k(RANKED, GOLD, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["g1", "g2"], GOLD, 5) == pytest.approx(1.0)


def test_normalize_and_em():
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert exact_match("the Eiffel tower", "Eiffel Tower.") == 1.0
    assert exact_match("Paris", "London") == 0.0


def test_f1():
    assert f1_score("Barack Obama", "Obama") == pytest.approx(2 * 0.5 * 1 / 1.5)
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes", "Yes.") == 1.0
    assert f1_score("no answer here", "no") == 0.0


def test_bootstrap_detects_clear_gain():
    a, b = [1.0] * 80 + [0.0] * 20, [0.0] * 80 + [0.0] * 20
    r = paired_bootstrap(a, b, n=2000)
    assert r["delta"] == pytest.approx(0.8)
    assert r["ci_low"] > 0.6 and r["p"] < 0.01


def test_bootstrap_identical_is_zero():
    r = paired_bootstrap([1, 0, 1, 0], [1, 0, 1, 0], n=500)
    assert r["delta"] == 0 and r["ci_low"] == 0 and r["ci_high"] == 0


def test_mcnemar():
    r = mcnemar([1, 1, 1, 1, 0], [0, 0, 0, 0, 0])
    assert (r["a_only"], r["b_only"]) == (4, 0)
    assert r["p"] == pytest.approx(0.125)
