from ragcmp.cache import Cache, make_key


def test_make_key_stable_and_order_insensitive_for_dicts():
    assert make_key("ns", {"a": 1, "b": 2}) == make_key("ns", {"b": 2, "a": 1})
    assert make_key("ns", "x") != make_key("other", "x")


def test_get_or_compute_caches(tmp_path):
    c = Cache(tmp_path / "c.sqlite")
    calls = []
    v1, cached1 = c.get_or_compute("ns", "p", lambda: calls.append(1) or {"v": 1})
    v2, cached2 = c.get_or_compute("ns", "p", lambda: calls.append(1) or {"v": 2})
    assert v1 == v2 == {"v": 1}
    assert (cached1, cached2) == (False, True)
    assert len(calls) == 1
    c.close()
