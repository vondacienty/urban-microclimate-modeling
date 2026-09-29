"""Tests for urban_micro.region_attr: Pearson attribution of the uhi
items of a canonical pareto_region_stability report to
weather/morph/cover factor differences with an n! positional
permutation p-value, a pooled Benjamini-Hochberg pass and per-pick
ranks."""

import json
import math
from itertools import permutations

import pytest

from urban_micro import (
    pareto_region_compare,
    pareto_region_stability,
    region_attr,
)
from urban_micro import uhi as _uhi

_GREEN = ("green",)
_ROOF = ("roof",)
_FACTORS = ("weather", "morph", "cover")


def _row(r, w, p, u, e=0.0, v=0.0):
    return (r, w, 0, p, u, e, v)


def _report(periods, *, alpha=0.05):
    return pareto_region_stability(
        {name: pareto_region_compare(rows, alpha=alpha)
         for name, rows in periods.items()},
    )


def _payload(out):
    return json.loads(out)


def _period(prefix, values, p=_GREEN):
    return [
        _row(region, f"{prefix}-{region}", p, float(value))
        for region, value in values.items()
    ]


def _reference_effect_p(pairs, y_values, drivers, factor_index):
    """Exact float reference for the Pearson effect and permutation p."""
    xs = [
        drivers[a][factor_index] - drivers[b][factor_index]
        for a, b in pairs
    ]
    ys = list(y_values)
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n
    xc = [x - x_bar for x in xs]
    yc = [y - y_bar for y in ys]
    sxx = sum(dx * dx for dx in xc)
    syy = sum(dy * dy for dy in yc)
    sxy = sum(dx * dy for dx, dy in zip(xc, yc))
    if sxx == 0 or syy == 0:
        return 0.0, 1.0
    effect = sxy / (sxx * syy) ** 0.5
    tail = 0
    for permuted in permutations(yc):
        psxy = sum(dx * dy for dx, dy in zip(xc, permuted))
        psyy = sum(dy * dy for dy in permuted)
        if psxy * psxy * syy >= sxy * sxy * psyy:
            tail += 1
    return effect, tail / math.factorial(n)


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 9.0, "c": 20.0}),
        "t2": _period("2", {"a": 1.0, "b": 9.0, "c": 20.0}),
    }, alpha=1.0)
    drivers = {
        "a": (0.0, 0.0, 1.0),
        "b": (5.0, 2.0, 3.0),
        "c": (10.0, 8.0, 9.0),
    }
    out = region_attr(report, drivers)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    data = _payload(out)
    assert list(data) == ["alpha", "groups"]
    assert data["alpha"] == 1.0
    group = data["groups"][0]
    assert list(group) == ["pick", "items"]
    assert group["pick"] == ["green"]
    assert len(group["items"]) == 3
    for item in group["items"]:
        assert list(item) == [
            "factor", "n", "effect", "p", "q", "reject", "rank",
        ]
        assert item["n"] == 3
        assert isinstance(item["n"], int) and not isinstance(item["n"], bool)
        assert isinstance(item["rank"], int) and not isinstance(item["rank"], bool)
        assert isinstance(item["reject"], bool)
        assert item["factor"] in _FACTORS


def test_effect_and_permutation_p_match_reference():
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 9.0, "c": 20.0}),
        "t2": _period("2", {"a": 2.0, "b": 8.0, "c": 19.0}),
    }, alpha=1.0)
    drivers = {
        "a": (0.0, 0.0, 1.0),
        "b": (5.0, 2.0, 3.0),
        "c": (10.0, 8.25, 9.0),
    }
    stability = _payload(report)["groups"][0]["items"]
    # The first (uhi) block holds one item per pair in canonical order.
    pairs = [(item["a"], item["b"]) for item in stability[:3]]
    y_values = [
        item["mean"] * item["consistency"] * item["significant"]
        for item in stability[:3]
    ]
    items = _payload(region_attr(report, drivers))["groups"][0]["items"]
    for factor_index, factor in enumerate(_FACTORS):
        effect, p_value = _reference_effect_p(
            pairs, y_values, drivers, factor_index
        )
        item = next(item for item in items if item["factor"] == factor)
        assert item["effect"] == round(effect, 6)
        assert item["p"] == round(p_value, 6)


def test_only_uhi_block_feeds_y():
    # Energy/vent differ across regions while uhi is flat: the stability
    # energy/vent blocks carry nonzero means that must be ignored.
    t1 = [
        _row("a", "a1", _GREEN, 5.0, 0.0, 0.0),
        _row("b", "b1", _GREEN, 5.0, 9.0, 20.0),
    ]
    t2 = [
        _row("a", "a2", _GREEN, 5.0, 20.0, 9.0),
        _row("b", "b2", _GREEN, 5.0, 0.0, 0.0),
    ]
    report = _report({"t1": t1, "t2": t2}, alpha=1.0)
    drivers = {"a": (0.0, 0.0, 0.0), "b": (1.0, 2.0, 3.0)}
    items = _payload(region_attr(report, drivers))["groups"][0]["items"]
    # UHI mean/consistency make y == 0 everywhere -> zero variance.
    for item in items:
        assert item["effect"] == 0.0
        assert item["p"] == 1.0
        assert item["n"] == 1


def test_y_is_mean_times_consistency_times_significant():
    # Two periods agreeing in sign reject in both at alpha 1; the y
    # values therefore equal the pair means and the reference computed
    # the same way must match.
    values1 = {"a": 0.0, "b": 4.0, "c": 10.0}
    values2 = {"a": 0.0, "b": 8.0, "c": 2.0}
    report = _report({
        "t1": _period("1", values1),
        "t2": _period("2", values2),
    }, alpha=1.0)
    drivers = {
        "a": (1.0, 9.0, 0.0),
        "b": (4.0, 1.0, 2.0),
        "c": (6.0, 2.0, 8.0),
    }
    stability = _payload(report)["groups"][0]["items"][:3]
    pairs = [(item["a"], item["b"]) for item in stability]
    y_values = [
        item["mean"] * item["consistency"] * item["significant"]
        for item in stability
    ]
    items = _payload(region_attr(report, drivers))["groups"][0]["items"]
    for factor_index, factor in enumerate(_FACTORS):
        effect, p_value = _reference_effect_p(
            pairs, y_values, drivers, factor_index
        )
        item = next(item for item in items if item["factor"] == factor)
        assert item["effect"] == round(effect, 6)
        assert item["p"] == round(p_value, 6)


def test_zero_variance_gives_effect_zero_p_one():
    report = _report({
        "t1": _period("1", {"a": 2.0, "b": 2.0}),
        "t2": _period("2", {"a": 2.0, "b": 2.0}),
    })
    drivers = {"a": (0.0, 0.0, 0.0), "b": (1.0, 2.0, 3.0)}
    items = _payload(region_attr(report, drivers))["groups"][0]["items"]
    for item in items:
        assert item["effect"] == 0.0
        assert item["p"] == 1.0
        assert item["q"] == 1.0
        assert item["reject"] is False


def test_constant_factor_gives_zero_effect_p_one():
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 9.0, "c": 20.0}),
        "t2": _period("2", {"a": 1.0, "b": 9.0, "c": 20.0}),
    }, alpha=1.0)
    drivers = {
        "a": (3.0, 7.0, 2.0),
        "b": (3.0, 7.0, 2.0),
        "c": (3.0, 7.0, 2.0),
    }
    items = _payload(region_attr(report, drivers))["groups"][0]["items"]
    for item in items:
        assert item["effect"] == 0.0
        assert item["p"] == 1.0


def test_global_bh_and_reject_threshold():
    # Two picks, three regions each and all three metrics fully
    # separated (exact compare p 1/3 -> shared compare BH rejects at
    # alpha 0.5), so the stability reports carry significant == 1 and
    # region_attr runs one six-test BH pass.
    def period_rows(prefix, pick):
        return [
            _row("a", f"{prefix}-a1", pick, 1.0, 1.0, 1.0),
            _row("a", f"{prefix}-a2", pick, 1.0, 1.0, 1.0),
            _row("b", f"{prefix}-b1", pick, 9.0, 9.0, 9.0),
            _row("b", f"{prefix}-b2", pick, 9.0, 9.0, 9.0),
            _row("c", f"{prefix}-c1", pick, 20.0, 20.0, 20.0),
            _row("c", f"{prefix}-c2", pick, 20.0, 20.0, 20.0),
        ]

    period = period_rows("1", _GREEN) + period_rows("2", _ROOF)
    report = _report({"t1": period, "t2": period}, alpha=0.5)
    drivers = {
        "a": (0.0, 0.0, 1.0),
        "b": (5.0, 2.0, 3.0),
        "c": (10.0, 8.25, 9.0),
    }
    data = _payload(report)
    out = _payload(region_attr(report, drivers))

    p_values = {}
    for group in data["groups"]:
        stability = group["items"]
        count = len(stability) // 3
        pairs = [(item["a"], item["b"]) for item in stability[:count]]
        y_values = [
            item["mean"] * item["consistency"] * item["significant"]
            for item in stability[:count]
        ]
        for factor_index, factor in enumerate(_FACTORS):
            _effect, p_value = _reference_effect_p(
                pairs, y_values, drivers, factor_index
            )
            p_values[(tuple(group["pick"]), factor)] = p_value

    ordered = sorted(p_values, key=lambda key: (p_values[key], key[0], key[1]))
    total = len(ordered)
    expected_q = {}
    running = 1.0
    for j in range(total, 0, -1):
        key = ordered[j - 1]
        running = min(running, total * p_values[key] / j)
        expected_q[key] = min(1.0, running)

    assert len(out["groups"]) == 2
    for group in out["groups"]:
        pick = tuple(group["pick"])
        for item in group["items"]:
            key = (pick, item["factor"])
            assert item["q"] == round(expected_q[key], 6)
            assert item["reject"] is (expected_q[key] <= 0.5)


def test_ranks_rejected_first_then_effect_desc_then_factor_order():
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 9.0, "c": 20.0}),
        "t2": _period("2", {"a": 1.0, "b": 9.0, "c": 20.0}),
    }, alpha=1.0)
    drivers = {
        "a": (0.0, 0.0, 1.0),
        "b": (5.0, 2.0, 3.0),
        "c": (10.0, 8.01, 9.0),
    }
    items = _payload(region_attr(report, drivers))["groups"][0]["items"]
    # Everything rejects at alpha 1; effects are all positive and ordered
    # weather > cover > morph for these drivers.
    assert [item["factor"] for item in items] == [
        "weather", "cover", "morph",
    ]
    assert [item["rank"] for item in items] == [1, 2, 3]
    effects = [item["effect"] for item in items]
    assert effects == sorted(effects, reverse=True)


def test_non_rejected_sort_after_rejected_with_factor_order_ties():
    report = _report({
        "t1": _period("1", {"a": 2.0, "b": 2.0}),
        "t2": _period("2", {"a": 2.0, "b": 2.0}),
    })
    drivers = {"a": (0.0, 0.0, 0.0), "b": (1.0, 2.0, 3.0)}
    items = _payload(region_attr(report, drivers))["groups"][0]["items"]
    assert [item["factor"] for item in items] == [
        "weather", "morph", "cover",
    ]
    assert [item["rank"] for item in items] == [1, 2, 3]


def test_groups_sort_by_ascending_pick():
    rows = [
        _row("a", "g1", _GREEN, 1.0), _row("b", "g2", _GREEN, 4.0),
        _row("a", "r1", _ROOF, 1.0), _row("b", "r2", _ROOF, 4.0),
    ]
    report = _report({"t1": rows, "t2": rows}, alpha=1.0)
    drivers = {"a": (0.0, 0.0, 0.0), "b": (1.0, 1.0, 1.0)}
    groups = _payload(region_attr(report, drivers))["groups"]
    assert [group["pick"] for group in groups] == [["green"], ["roof"]]
    for group in groups:
        assert [item["rank"] for item in group["items"]] == [1, 2, 3]


def test_negative_zero_normalized():
    report = _report({
        "t1": _period("1", {"a": 2.0, "b": 2.0}),
        "t2": _period("2", {"a": 2.0, "b": 2.0}),
    })
    out = region_attr(
        report, {"a": (1.0, 1.0, 1.0), "b": (0.0, 0.0, 0.0)}
    )
    assert "-0.000000" not in out
    assert '"effect":0.000000' in out


def test_empty_report_yields_empty_groups():
    empty = _report({"t1": [], "t2": []})
    assert region_attr(empty, {}) == '{"alpha":0.050000,"groups":[]}\n'


def test_integer_factor_values_accepted():
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 9.0, "c": 20.0}),
        "t2": _period("2", {"a": 1.0, "b": 9.0, "c": 20.0}),
    }, alpha=1.0)
    ints = {"a": (0, 0, 1), "b": (5, 2, 3), "c": (10, 8, 9)}
    floats = {
        key: tuple(float(value) for value in entry)
        for key, entry in ints.items()
    }
    assert region_attr(report, ints) == region_attr(report, floats)


# --- validation -------------------------------------------------------------

def test_more_than_eight_pairs_value_error():
    regions = list("abcdefghi")
    report = _report({
        "t1": _period("1", {r: float(i) for i, r in enumerate(regions)}),
        "t2": _period("2", {r: float(i) for i, r in enumerate(regions)}),
    }, alpha=1.0)
    drivers = {r: (float(i), 0.0, 0.0) for i, r in enumerate(regions)}
    with pytest.raises(ValueError):
        region_attr(report, drivers)


def test_non_string_report_type_error():
    for bad in (1, None, [], {}, True, ()):
        with pytest.raises(TypeError):
            region_attr(bad, {})


def test_non_dict_drivers_type_error():
    empty = _report({"t1": [], "t2": []})
    for bad in ([], None, 1, True, (), "drivers"):
        with pytest.raises(TypeError):
            region_attr(empty, bad)


def test_empty_report_accepts_only_empty_drivers():
    empty = _report({"t1": [], "t2": []})
    with pytest.raises(ValueError):
        region_attr(empty, {"a": (1.0, 2.0, 3.0)})


def test_drivers_keys_must_match_report_regions():
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 4.0}),
        "t2": _period("2", {"a": 1.0, "b": 4.0}),
    })
    with pytest.raises(ValueError):
        region_attr(report, {})
    with pytest.raises(ValueError):
        region_attr(report, {"a": (0.0, 0.0, 0.0)})
    with pytest.raises(ValueError):
        region_attr(
            report,
            {"a": (0.0, 0.0, 0.0), "b": (1.0, 1.0, 1.0),
             "c": (2.0, 2.0, 2.0)},
        )


def test_driver_tuple_shape_value_error():
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 4.0}),
        "t2": _period("2", {"a": 1.0, "b": 4.0}),
    })
    base = {"a": (0.0, 0.0, 0.0), "b": (1.0, 1.0, 1.0)}
    for mutated in (
        {**base, "a": [0.0, 0.0, 0.0]},
        {**base, "a": (0.0, 0.0)},
        {**base, "a": (0.0, 0.0, 0.0, 0.0)},
    ):
        with pytest.raises(ValueError):
            region_attr(report, mutated)


def test_driver_values_must_be_finite_non_bool_numbers():
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 4.0}),
        "t2": _period("2", {"a": 1.0, "b": 4.0}),
    })
    base = {"a": (0.0, 0.0, 0.0), "b": (1.0, 1.0, 1.0)}
    for bad_value in (True, False, "1.0", None, float("nan"),
                      float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            region_attr(report, {**base, "a": (bad_value, 0.0, 0.0)})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.rstrip("\n"),
        lambda s: s + "\n",
        lambda s: '{"groups":[],"alpha":0.050000}\n',
        lambda s: s.replace('"direction":"down"', '"direction":"up"', 1),
        lambda s: s.replace('"direction":"flat"', '"direction":"up"', 1),
        lambda s: s.replace('"rank":1', '"rank":2', 1),
        lambda s: s.replace('"alpha":1.000000,', '"alpha":1.0,', 1),
        lambda s: "not json\n",
    ],
)
def test_non_canonical_report_value_error(mutate):
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 9.0, "c": 20.0}),
        "t2": _period("2", {"a": 1.0, "b": 9.0, "c": 20.0}),
    }, alpha=1.0)
    drivers = {
        "a": (0.0, 0.0, 1.0),
        "b": (5.0, 2.0, 3.0),
        "c": (10.0, 8.0, 9.0),
    }
    with pytest.raises(ValueError):
        region_attr(mutate(report), drivers)


def test_duplicate_pair_in_block_rejected():
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 9.0, "c": 20.0}),
        "t2": _period("2", {"a": 1.0, "b": 9.0, "c": 20.0}),
    }, alpha=1.0)
    mutated = report.replace(
        '{"a":"b","b":"c","mean"', '{"a":"a","b":"c","mean"', 1
    )
    drivers = {
        "a": (0.0, 0.0, 1.0),
        "b": (5.0, 2.0, 3.0),
        "c": (10.0, 8.0, 9.0),
    }
    with pytest.raises(ValueError):
        region_attr(mutated, drivers)


def test_divergent_pair_sets_across_blocks_rejected():
    # Drop the third (vent) block: only uhi/energy items remain.
    report = _report({
        "t1": _period("1", {"a": 1.0, "b": 9.0, "c": 20.0}),
        "t2": _period("2", {"a": 1.0, "b": 9.0, "c": 20.0}),
    }, alpha=1.0)
    payload = _payload(report)
    payload["groups"][0]["items"] = payload["groups"][0]["items"][:6]
    drivers = {
        "a": (0.0, 0.0, 1.0),
        "b": (5.0, 2.0, 3.0),
        "c": (10.0, 8.0, 9.0),
    }
    with pytest.raises(ValueError):
        region_attr(
            json.dumps(payload, separators=(",", ":")) + "\n", drivers
        )


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.region_attr is urban_micro.region_attr
    assert "region_attr" in urban_micro.__all__
