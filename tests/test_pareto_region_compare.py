"""Tests for urban_micro.pareto_region_compare: pairwise cross-region
permutation comparisons within Pareto-pick levels."""

import json

import pytest

from urban_micro import pareto_region_compare

_GREEN = ("green",)
_ROOF = ("roof",)
_MATERIAL = ("material",)


def _row(r, w, t, p, u, e=0.0, v=0.0):
    return (r, w, t, p, u, e, v)


def _payload(out):
    return json.loads(out)


def test_empty_rows_yields_empty_groups():
    out = pareto_region_compare([])
    assert out == '{"alpha":0.050000,"groups":[]}\n'
    assert list(_payload(out).keys()) == ["alpha", "groups"]


def test_single_region_pick_omitted():
    out = pareto_region_compare([
        _row("a", "w1", 0, _GREEN, 1.0),
        _row("a", "w2", 0, _GREEN, 2.0),
    ])
    assert _payload(out)["groups"] == []


def test_basic_pair_keys_and_order():
    out = pareto_region_compare([
        _row("a", "w1", 0, _GREEN, 1.0),
        _row("a", "w2", 0, _GREEN, 3.0),
        _row("b", "w3", 0, _GREEN, 5.0),
    ])
    assert out.endswith("\n")
    assert not out.endswith("\n\n")
    data = _payload(out)
    group = data["groups"][0]
    assert list(group.keys()) == ["pick", "tests"]
    assert group["pick"] == ["green"]
    assert [t["metric"] for t in group["tests"]] == ["uhi", "energy", "vent"]
    test = group["tests"][0]
    assert list(test.keys()) == [
        "a", "b", "metric", "n_a", "n_b", "diff", "p", "q", "reject", "rank",
    ]
    assert test["a"] == "a"
    assert test["b"] == "b"
    assert test["n_a"] == 2
    assert test["n_b"] == 1
    assert test["diff"] == -3.0
    # C(3,2) = 3 allocations; the two extreme splits ({1,3} and {3,5})
    # give |diff| = 3, so p = 2/3.
    assert test["p"] == pytest.approx(2 / 3, abs=1e-6)
    assert test["rank"] == 1
    assert isinstance(test["reject"], bool)
    assert isinstance(test["n_a"], int)
    assert isinstance(test["rank"], int)


def test_window_averaging_precedes_pooling():
    # Two t rows for one (a, w1, green) window average to 2.0; that one
    # window mean is compared against b's single window mean 5.0.
    out = pareto_region_compare([
        _row("a", "w1", 0, _GREEN, 0.0),
        _row("a", "w1", 1, _GREEN, 4.0),
        _row("b", "w2", 0, _GREEN, 5.0),
    ])
    test = _payload(out)["groups"][0]["tests"][0]
    assert test["n_a"] == 1
    assert test["n_b"] == 1
    assert test["diff"] == -3.0
    # 1v1: the two allocations give diff -3 and +3, both counted.
    assert test["p"] == 1.0


def test_exact_two_vs_two_enumeration():
    # a means [0, 2], b means [5, 5]; pooled [0, 2, 5, 5], C(4,2) = 6.
    # diff = mean_a - mean_b = 1 - 5 = -4. Allocations whose a-sum is
    # 2 ({0,2}, diff -4) or 10 ({5,5}, diff +4) hit: 2 of 6.
    out = pareto_region_compare([
        _row("a", "w1", 0, _GREEN, 0.0),
        _row("a", "w2", 0, _GREEN, 2.0),
        _row("b", "w1", 0, _GREEN, 5.0),
        _row("b", "w2", 0, _GREEN, 5.0),
    ])
    test = _payload(out)["groups"][0]["tests"][0]
    assert test["p"] == pytest.approx(2 / 6, abs=1e-6)
    # Global BH with the two flat metrics at p = 1.0 lifts q to 1.0.
    assert test["q"] == 1.0


def test_global_benjamini_hochberg_across_tests():
    # One pair, three metrics; only uhi separates (3v2, smallest possible
    # two-sided p = 2/10 = 0.1). The other two p-values are 1.0, so the
    # global BH q for uhi is N * 0.1 / 1 = 0.3.
    out = pareto_region_compare([
        _row("a", "w1", 0, _GREEN, 0.0),
        _row("a", "w2", 0, _GREEN, 0.0),
        _row("a", "w3", 0, _GREEN, 0.0),
        _row("b", "w1", 0, _GREEN, 100.0),
        _row("b", "w2", 0, _GREEN, 100.0),
    ])
    tests = _payload(out)["groups"][0]["tests"]
    uhi = tests[0]
    assert uhi["p"] == pytest.approx(0.1, abs=1e-6)
    assert uhi["q"] == pytest.approx(0.3, abs=1e-6)
    assert uhi["reject"] is False
    assert pareto_region_compare(
        [
            _row("a", "w1", 0, _GREEN, 0.0),
            _row("a", "w2", 0, _GREEN, 0.0),
            _row("a", "w3", 0, _GREEN, 0.0),
            _row("b", "w1", 0, _GREEN, 100.0),
            _row("b", "w2", 0, _GREEN, 100.0),
        ],
        alpha=0.5,
    ).count('"reject":true') == 1


def test_pair_ordering_and_per_pick_metric_ranks():
    out = pareto_region_compare([
        _row("a", "w1", 0, _GREEN, 1.0),
        _row("b", "w1", 0, _GREEN, 2.0),
        _row("c", "w1", 0, _GREEN, 10.0),
    ])
    tests = _payload(out)["groups"][0]["tests"]
    # Three metric blocks, each holding the three pairs.
    assert [t["metric"] for t in tests] == ["uhi"] * 3 + ["energy"] * 3 + [
        "vent"
    ] * 3
    uhi = tests[:3]
    # Nothing rejects; ranks follow descending |diff|: a-c (9), b-c (8),
    # a-b (1).
    assert [(t["a"], t["b"]) for t in uhi] == [
        ("a", "c"), ("b", "c"), ("a", "b"),
    ]
    assert [t["rank"] for t in uhi] == [1, 2, 3]


def test_groups_sort_by_ascending_pick():
    rows = [
        _row("a", "w1", 0, _ROOF, 1.0),
        _row("b", "w1", 0, _ROOF, 4.0),
        _row("a", "w1", 1, _MATERIAL, 1.0),
        _row("b", "w1", 1, _MATERIAL, 4.0),
        _row("z", "w1", 0, ("green", "roof"), 9.0),
    ]
    data = _payload(pareto_region_compare(rows))
    # Tuple lexicographic order puts ("material",) before ("roof",).
    assert [g["pick"] for g in data["groups"]] == [
        ["material"], ["roof"],
    ]


def test_negative_zero_normalized():
    out = pareto_region_compare([
        _row("a", "w1", 0, _GREEN, 2.0),
        _row("b", "w1", 0, _GREEN, 2.0),
    ])
    assert '"diff":0.000000' in out
    assert "-0.000000" not in out


def test_pooled_window_limit():
    rows = [_row("a", f"w{i}", 0, _GREEN, float(i)) for i in range(6)]
    rows += [_row("b", f"x{i}", 0, _GREEN, float(i)) for i in range(6)]
    with pytest.raises(ValueError):
        pareto_region_compare(rows)
    # Exactly at the 10-window boundary (6 + 4) is allowed.
    assert pareto_region_compare(rows[:-2])


def test_validation_matches_pareto_heterogeneity():
    with pytest.raises(TypeError):
        pareto_region_compare("not-a-list")
    with pytest.raises(ValueError):
        pareto_region_compare([], alpha=0.0)
    with pytest.raises(ValueError):
        pareto_region_compare([], alpha=1.5)
    with pytest.raises(ValueError):
        pareto_region_compare([("a", "w1", 0, _GREEN, 1.0, 0.0)])
    with pytest.raises(ValueError):
        pareto_region_compare([
            _row("a", "w1", 0, _GREEN, 1.0),
            _row("a", "w1", 0, _GREEN, 1.0),
        ])
