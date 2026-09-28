"""Tests for urban_micro.driver_attr: Pearson attribution of kind_impact
UHI changes to station/lst/morph/cover factors with an n! positional
permutation p-value, a pooled Benjamini-Hochberg pass and one picked
factor per by dimension."""

import json
import math
from decimal import Decimal
from fractions import Fraction
from itertools import permutations

import pytest

from urban_micro import driver_attr, kind_impact
from urban_micro import uhi as _uhi

_KINDS = ("green", "roof", "material")
_FACTORS = ("station", "lst", "morph", "cover")


# --- report / data builders -------------------------------------------------

def _portfolio_group(by, pick=("green",), budget=3.0, cost=1.0,
                     remaining=2.0, score=2.0):
    pick = tuple(sorted(pick))
    skip = [kind for kind in ("green", "material", "roof") if kind not in pick]
    skip_token = "[" + ",".join(
        '{"kind":' + json.dumps(kind) + ',"reason":"a"}' for kind in skip
    ) + "]"
    return (
        '{"by":' + json.dumps(by)
        + f',"budget":{budget:.6f},"cost":{cost:.6f},"remaining":{remaining:.6f},'
        + f'"score":{score:.6f},"pick":'
        + json.dumps(list(pick), separators=(",", ":"))
        + ',"skip":' + skip_token + "}"
    )


def _portfolio_report(*groups, alpha=0.05):
    return (
        f'{{"alpha":{alpha:.6f},"groups":[' + ",".join(groups) + "]}\n"
    )


def _panel(base, green=(0, 0, 0)):
    return (
        tuple(base),
        {
            "green": tuple(green),
            "roof": (0, 0, 0),
            "material": (0, 0, 0),
        },
    )


def _impact(by_to_changes, alpha=0.05):
    """Build a canonical kind_impact report: by -> {key: uhi change}."""
    groups = []
    panels = {}
    for by in ("region", "window"):
        if by not in by_to_changes:
            continue
        groups.append(_portfolio_group(by))
        for key, change in by_to_changes[by].items():
            panels[(by, key)] = _panel((10.0, 0.0, 0.0), green=(change, 0.0, 0.0))
    return kind_impact(_portfolio_report(*groups, alpha=alpha), panels)


def _driver_data(by_to_factors):
    """by -> {key: (station, lst, morph, cover)} -> data dict."""
    return {
        (by, key): tuple(values)
        for by, per_key in by_to_factors.items()
        for key, values in per_key.items()
    }


# --- exact Fraction reference ----------------------------------------------

def _reference(impact, data):
    """Mirror driver_attr with exact Fractions; return list of picked
    (by, factor, n, r_float, p Fraction, q Fraction, reject, direction)."""
    payload = json.loads(impact)
    alpha = Fraction(str(payload["alpha"]))
    groups = []
    for group in payload["groups"]:
        by = group["by"]
        keys = [item["key"] for item in group["items"]]
        ys = [Fraction(str(item["uhi"][2])) for item in group["items"]]
        groups.append((by, keys, ys))

    tests = {}
    ordered = []
    for by, keys, ys in groups:
        n = len(keys)
        y_bar = sum(ys, Fraction(0)) / n
        syy = sum((y - y_bar) ** 2 for y in ys)
        for factor_index, factor in enumerate(_FACTORS):
            xs = [
                Fraction(str(data[(by, key)][factor_index])) for key in keys
            ]
            x_bar = sum(xs, Fraction(0)) / n
            sxx = sum((x - x_bar) ** 2 for x in xs)
            sxy = sum(
                (x - x_bar) * (y - y_bar)
                for x, y in zip(xs, ys)
            )
            if sxx == 0 or syy == 0:
                r_float = 0.0
                r_squared = Fraction(0)
                p = Fraction(1)
            else:
                r_float = float(sxy) / math.sqrt(float(sxx * syy))
                r_squared = sxy * sxy / (sxx * syy)
                x_centered = [x - x_bar for x in xs]
                y_centered = [y - y_bar for y in ys]
                total = math.factorial(n)
                tail = 0
                for permuted in permutations(y_centered):
                    permuted_sxy = sum(
                        dx * dy for dx, dy in zip(x_centered, permuted)
                    )
                    permuted_syy = sum(dy * dy for dy in permuted)
                    if (
                        permuted_sxy * permuted_sxy * syy
                        >= sxy * sxy * permuted_syy
                    ):
                        tail += 1
                p = Fraction(tail, total)
            tests[(by, factor)] = (n, r_float, r_squared, p)
            ordered.append(
                (p, 0 if by == "region" else 1, factor_index, by, factor)
            )

    ordered.sort(key=lambda entry: entry[:3])
    tested = len(ordered)
    q_values = {}
    running = Fraction(1)
    for rank in range(tested, 0, -1):
        p, _bo, _fi, by, factor = ordered[rank - 1]
        running = min(running, Fraction(tested) * p / rank)
        q_values[(by, factor)] = min(Fraction(1), running)

    picked_rows = []
    for by, keys, ys in groups:
        def _key(factor):
            _n, _r, r_squared, _p = tests[(by, factor)]
            return (
                0 if q_values[(by, factor)] <= alpha else 1,
                -r_squared,
                _FACTORS.index(factor),
            )

        factor = min(_FACTORS, key=_key)
        n, r_float, _r_squared, p = tests[(by, factor)]
        q = q_values[(by, factor)]
        reject = q <= alpha
        if r_float < 0:
            direction = "down"
        elif r_float > 0:
            direction = "up"
        else:
            direction = "flat"
        picked_rows.append(
            (by, factor, n, r_float, p, q, reject, direction)
        )
    return picked_rows


def _assert_matches_reference(impact, data):
    out = driver_attr(impact, data)
    payload = json.loads(out)
    expected = _reference(impact, data)
    assert len(payload["items"]) == len(expected)
    for item, (by, factor, n, r_float, p, q, reject, direction) in zip(
        payload["items"], expected
    ):
        assert item["by"] == by
        assert item["factor"] == factor
        assert item["n"] == n and isinstance(item["n"], int)
        assert item["reject"] is reject
        assert item["direction"] == direction
        assert abs(item["effect"] - r_float) < 5.1e-7
        assert abs(item["p"] - float(p)) < 5.1e-7
        assert abs(item["q"] - float(q)) < 5.1e-7
    return out


# --- datasets ---------------------------------------------------------------

def _data_n3():
    changes = {"region": {"a": -1, "b": -2, "c": -3}}
    factors = {
        "region": {
            "a": (1.0, 10.0, 0.5, 0.2),
            "b": (2.0, 20.0, 0.5, 0.9),
            "c": (3.0, 30.0, 0.5, 0.1),
        }
    }
    return changes, factors


def _data_n6():
    changes = {"region": {f"k{i}": -float(i) for i in range(1, 7)}}
    factors = {
        "region": {
            # station and morph are perfectly (negatively) correlated with
            # y; lst is one adjacent swap away; cover wanders.
            "k1": (1.0, 2.0, 1.0, 3.0),
            "k2": (2.0, 1.0, 2.0, 1.0),
            "k3": (3.0, 3.0, 3.0, 4.0),
            "k4": (4.0, 4.0, 4.0, 1.0),
            "k5": (5.0, 5.0, 5.0, 5.0),
            "k6": (6.0, 6.0, 6.0, 9.0),
        }
    }
    return changes, factors


def _data_two_bys():
    changes = {
        "region": {"a": 1, "b": -1},
        "window": {"w1": 0, "w2": 1, "w3": -2},
    }
    factors = {
        "region": {
            "a": (1.0, 0.0, 0.1, 5.0),
            "b": (2.0, 5.0, 0.9, 1.0),
        },
        "window": {
            "w1": (1.0, 0.0, 0.2, 0.4),
            "w2": (2.0, 1.0, 0.5, 0.3),
            "w3": (3.0, 4.0, 0.8, 0.9),
        },
    }
    return changes, factors


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order_n3():
    changes, factors = _data_n3()
    impact = _impact(changes)
    out = driver_attr(impact, _driver_data(factors))
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "items"]
    assert list(payload["items"][0]) == [
        "by", "factor", "n", "effect", "p", "q", "reject", "direction",
    ]


def test_n3_perfect_correlation_tokens():
    changes, factors = _data_n3()
    out = driver_attr(_impact(changes), _driver_data(factors))
    # station and lst tie at r = -1; station wins on factor order.
    item = json.loads(out)["items"][0]
    assert item == {
        "by": "region",
        "factor": "station",
        "n": 3,
        "effect": -1.0,
        "p": 0.333333,
        "q": 0.666667,
        "reject": False,
        "direction": "down",
    }
    assert '"n":3' in out
    assert '"effect":-1.000000' in out
    assert '"p":0.333333' in out
    assert '"q":0.666667' in out
    assert '"reject":false' in out
    assert '"direction":"down"' in out


def test_n6_perfect_driver_rejects_under_bh():
    changes, factors = _data_n6()
    out = _assert_matches_reference(_impact(changes), _driver_data(factors))
    item = json.loads(out)["items"][0]
    # p = 2 / 6! for a perfect ordering; station and morph tie at that p
    # (ranks 1 and 2) so the BH running minimum leaves the rank-1 item
    # q = 4 * 2 / (6! * 2) = 4 / 6!.
    assert item["factor"] == "station"
    assert item["effect"] == -1.0
    assert abs(item["p"] - 2 / 720) < 5.1e-7
    assert abs(item["q"] - 4 / 720) < 5.1e-7
    assert item["reject"] is True
    assert item["direction"] == "down"


def test_constant_y_is_flat_with_r_zero_p_one():
    changes = {"region": {"a": 2, "b": 2, "c": 2}}
    factors = {
        "region": {
            "a": (1.0, 9.0, 0.2, 0.1),
            "b": (2.0, 2.0, 0.3, 0.8),
            "c": (3.0, 7.0, 0.9, 0.4),
        }
    }
    out = driver_attr(_impact(changes), _driver_data(factors))
    item = json.loads(out)["items"][0]
    assert item == {
        "by": "region",
        "factor": "station",
        "n": 3,
        "effect": 0.0,
        "p": 1.0,
        "q": 1.0,
        "reject": False,
        "direction": "flat",
    }
    assert '"effect":0.000000' in out
    assert "-0.000000" not in out


def test_constant_factor_gets_r_zero_p_one():
    changes = {"region": {"a": -1, "b": 0, "c": 1}}
    factors = {
        "region": {
            # station constant -> (r, p) = (0, 1); lst perfectly tracks y.
            "a": (5.0, 1.0, 0.2, 0.1),
            "b": (5.0, 2.0, 0.3, 0.8),
            "c": (5.0, 3.0, 0.9, 0.4),
        }
    }
    out = _assert_matches_reference(_impact(changes), _driver_data(factors))
    picked = json.loads(out)["items"][0]
    assert picked["factor"] == "lst"
    assert picked["effect"] == 1.0


def test_single_panel_is_flat():
    changes = {"region": {"a": -3}}
    factors = {"region": {"a": (1.0, 2.0, 0.3, 0.4)}}
    out = driver_attr(_impact(changes), _driver_data(factors))
    item = json.loads(out)["items"][0]
    assert item["n"] == 1
    assert item["effect"] == 0.0
    assert item["p"] == 1.0
    assert item["q"] == 1.0
    assert item["reject"] is False
    assert item["direction"] == "flat"


def test_items_region_before_window():
    changes, factors = _data_two_bys()
    out = _assert_matches_reference(_impact(changes), _driver_data(factors))
    bys = [item["by"] for item in json.loads(out)["items"]]
    assert bys == ["region", "window"]


def test_matches_fraction_reference_on_all_datasets():
    for changes, factors in (
        _data_n3(),
        _data_n6(),
        _data_two_bys(),
        (
            {"window": {f"w{i}": ((i * 7) % 5) - 2 for i in range(5)}},
            {
                "window": {
                    f"w{i}": (
                        float((i * 3) % 7),
                        float((i * 5) % 11),
                        float(i % 2),
                        float((i + 1) * (i + 2) % 13),
                    )
                    for i in range(5)
                }
            },
        ),
    ):
        _assert_matches_reference(_impact(changes), _driver_data(factors))


def test_duplicate_y_values_counted_in_permutations():
    # y has a repeated value; the brute-force p accounts for duplicate
    # permutations of the positions of y.
    changes = {"region": {"a": -1, "b": -1, "c": 2}}
    factors = {
        "region": {
            "a": (1.0, 0.0, 0.0, 0.0),
            "b": (2.0, 0.0, 0.0, 0.0),
            "c": (3.0, 0.0, 0.0, 0.0),
        }
    }
    out = _assert_matches_reference(_impact(changes), _driver_data(factors))
    item = json.loads(out)["items"][0]
    assert item["factor"] == "station"
    assert item["direction"] == "up"


def test_integer_factor_values_accepted():
    changes = {"region": {"a": -1, "b": 0, "c": 1}}
    factors = {
        "region": {
            "a": (1, 2, 0, 3),
            "b": (2, 4, 1, 3),
            "c": (3, 6, 2, 3),
        }
    }
    out = driver_attr(_impact(changes), _driver_data(factors))
    item = json.loads(out)["items"][0]
    assert item["effect"] == 1.0
    assert item["direction"] == "up"


def test_alpha_threshold_changes_rejection():
    # n = 6 perfect driver: q = 1/90.
    changes, factors = _data_n6()
    strict = _impact(changes, alpha=0.001)
    item = json.loads(driver_attr(strict, _driver_data(factors)))["items"][0]
    assert item["reject"] is False
    loose = _impact(changes, alpha=0.02)
    item = json.loads(driver_attr(loose, _driver_data(factors)))["items"][0]
    assert item["reject"] is True


def test_no_negative_zero_anywhere():
    changes, factors = _data_two_bys()
    out = driver_attr(_impact(changes), _driver_data(factors))
    assert "-0.000000" not in out


# --- empty report -----------------------------------------------------------

def test_empty_report_only_with_empty_data():
    empty = '{"alpha":0.050000,"groups":[]}\n'
    assert driver_attr(empty, {}) == '{"alpha":0.050000,"items":[]}\n'
    with pytest.raises(ValueError):
        driver_attr(empty, {("region", "a"): (1.0, 2.0, 0.3, 0.4)})


# --- validation -------------------------------------------------------------

def test_type_errors():
    empty = '{"alpha":0.050000,"groups":[]}\n'
    with pytest.raises(TypeError):
        driver_attr(1, {})
    with pytest.raises(TypeError):
        driver_attr(None, {})
    with pytest.raises(TypeError):
        driver_attr(empty, [])


def test_data_keys_must_match_panels():
    changes, factors = _data_n3()
    impact = _impact(changes)
    good = _driver_data(factors)
    missing = dict(good)
    del missing[("region", "c")]
    extra = dict(good)
    extra[("region", "zzz")] = (1.0, 2.0, 0.3, 0.4)
    wrong_by = dict(good)
    del wrong_by[("region", "c")]
    wrong_by[("window", "c")] = (1.0, 2.0, 0.3, 0.4)
    for bad in (missing, extra, wrong_by, {}):
        with pytest.raises(ValueError):
            driver_attr(impact, bad)


def test_data_keys_must_be_two_tuples():
    changes, factors = _data_n3()
    impact = _impact(changes)
    bad = {f"region-{key}": values for key, values in factors["region"].items()}
    with pytest.raises(ValueError):
        driver_attr(impact, bad)
    with pytest.raises(ValueError):
        driver_attr(impact, {("region",): (1.0, 2.0, 0.3, 0.4)})


def test_data_values_must_be_four_tuples():
    changes, factors = _data_n3()
    impact = _impact(changes)
    for bad_value in (
        [1.0, 2.0, 0.5, 0.2],
        (1.0, 2.0, 0.5),
        (1.0, 2.0, 0.5, 0.2, 1.0),
    ):
        bad = {
            ("region", key): (
                bad_value if key == "a" else _driver_data(factors)[("region", key)]
            )
            for key in ("a", "b", "c")
        }
        with pytest.raises(ValueError):
            driver_attr(impact, bad)


@pytest.mark.parametrize("bad", [
    True, False, float("inf"), -float("inf"), float("nan"), "1.0", None,
])
def test_factor_numbers_rejected(bad):
    changes, factors = _data_n3()
    impact = _impact(changes)
    data = _driver_data(factors)
    data[("region", "a")] = (bad, 10.0, 0.5, 0.2)
    with pytest.raises(ValueError):
        driver_attr(impact, data)


def test_more_than_eight_panels_rejected():
    changes = {"region": {f"k{i}": -float(i) for i in range(9)}}
    factors = {
        "region": {
            f"k{i}": (float(i), 2 * float(i), 0.1 * i, 0.2 * i)
            for i in range(9)
        }
    }
    impact = _impact(changes)
    with pytest.raises(ValueError, match="8"):
        driver_attr(impact, _driver_data(factors))


def test_eight_panels_accepted():
    changes = {"region": {f"k{i}": -float(i) for i in range(1, 9)}}
    factors = {
        "region": {
            f"k{i}": (float(i), 0.0, 0.0, 0.0) for i in range(1, 9)
        }
    }
    out = driver_attr(_impact(changes), _driver_data(factors))
    item = json.loads(out)["items"][0]
    assert item["n"] == 8
    assert item["factor"] == "station"
    assert item["effect"] == -1.0
    assert abs(item["p"] - 2 / math.factorial(8)) < 5.1e-7
    assert item["reject"] is True


@pytest.mark.parametrize("raw", [
    "",
    "not json",
    '{"alpha":0.050000,"groups":[]}',
    '{"alpha":0.050000,"groups":[]}\n\n',
    ' {"alpha":0.050000,"groups":[]}\n',
    '{"alpha":0.05,"groups":[]}\n',
])
def test_noncanonical_report_raises_value_error(raw):
    with pytest.raises(ValueError):
        driver_attr(raw, {})


def test_portfolio_report_is_not_a_kind_impact_report():
    report = _portfolio_report(_portfolio_group("region"))
    with pytest.raises(ValueError):
        driver_attr(
            report,
            {("region", "a"): (1.0, 2.0, 0.3, 0.4)},
        )


def test_kind_impact_summary_shape_rejected():
    # driver_attr needs the kind_impact (per-panel) shape, not summary.
    summary_like = '{"alpha":0.050000,"items":[]}\n'
    with pytest.raises(ValueError):
        driver_attr(summary_like, {})


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.driver_attr is urban_micro.driver_attr
    assert "driver_attr" in urban_micro.__all__
