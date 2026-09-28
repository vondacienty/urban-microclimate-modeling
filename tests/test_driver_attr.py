"""Tests for urban_micro.driver_attr: per-dimension driver attribution of
kind_impact UHI changes via permutation correlations and BH correction."""

import itertools
import json
import math
from fractions import Fraction

import pytest

from urban_micro import driver_attr, kind_impact
from urban_micro import uhi as _uhi

_FACTORS = ("station", "lst", "morph", "cover")
_KINDS = ("green", "roof", "material")


def _portfolio_group(by, pick, cost, remaining, budget=3.0, score=2.0):
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


def _panel(change):
    return (
        (10.0, 100.0, 5.0),
        {
            "green": (float(change), 0.0, 0.0),
            "roof": (0.0, 0.0, 0.0),
            "material": (0.0, 0.0, 0.0),
        },
    )


def _impact(changes_by, alpha=0.05):
    """Build a canonical kind_impact report; changes_by maps by -> {key: uhi
    change from the picked green kind}."""
    groups = []
    data = {}
    for by in ("region", "window"):
        if by not in changes_by:
            continue
        groups.append(
            _portfolio_group(by, ("green",), 1.0, 2.0)
        )
        for key, change in changes_by[by].items():
            data[(by, key)] = _panel(change)
    return kind_impact(_portfolio_report(*groups, alpha=alpha), data)


def _factors(changes_by, series_by):
    """series_by maps by -> {factor: [x per panel in sorted-key order]}."""
    data = {}
    for by, changes in changes_by.items():
        keys = sorted(changes)
        for index, key in enumerate(keys):
            data[(by, key)] = tuple(
                series_by[by][factor][index] for factor in _FACTORS
            )
    return data


def _brute_force(alpha, changes_by, series_by):
    """Independent Fraction model: returns the expected picked
    ``{by: (factor, r2, p, q, direction)}`` and per-cell stats."""
    cells = {}
    for by, changes in changes_by.items():
        keys = sorted(changes)
        n = len(keys)
        y = [Fraction(str(changes[key])) for key in keys]
        mean_y = sum(y, Fraction(0)) / n
        dy = [value - mean_y for value in y]
        syy = sum((d * d for d in dy), Fraction(0))
        for factor in _FACTORS:
            x = [Fraction(str(v)) for v in series_by[by][factor]]
            mean_x = sum(x, Fraction(0)) / n
            dx = [value - mean_x for value in x]
            sxx = sum((d * d for d in dx), Fraction(0))
            sxy = sum(
                (dx[i] * dy[i] for i in range(n)), Fraction(0)
            )
            if sxx == 0 or syy == 0:
                cells[(by, factor)] = dict(
                    r2=Fraction(0), p=Fraction(1),
                    sign=0,
                )
                continue
            hits = 0
            for perm in itertools.permutations(dy):
                psxy = sum(
                    (dx[i] * perm[i] for i in range(n)), Fraction(0)
                )
                if abs(psxy) >= abs(sxy):
                    hits += 1
            cells[(by, factor)] = dict(
                r2=sxy * sxy / (sxx * syy),
                p=Fraction(hits, math.factorial(n)),
                sign=0 if sxy == 0 else (1 if sxy > 0 else -1),
            )
    ordered = sorted(
        cells,
        key=lambda c: (
            cells[c]["p"],
            0 if c[0] == "region" else 1,
            _FACTORS.index(c[1]),
        ),
    )
    total = len(ordered)
    q_values = {}
    running = Fraction(1)
    for rank in range(total, 0, -1):
        cell = ordered[rank - 1]
        running = min(
            running, Fraction(total) * cells[cell]["p"] / rank
        )
        q_values[cell] = min(Fraction(1), running)
    alpha_value = Fraction(str(alpha))
    picked = {}
    for by in changes_by:
        rejected = [
            (factor, cells[(by, factor)])
            for factor in _FACTORS
            if q_values[(by, factor)] <= alpha_value
        ]
        if not rejected:
            continue
        factor, stats = min(
            rejected,
            key=lambda entry: (-entry[1]["r2"], _FACTORS.index(entry[0])),
        )
        direction = (
            "flat" if stats["sign"] == 0
            else ("up" if stats["sign"] > 0 else "down")
        )
        picked[by] = (
            factor, stats["r2"], stats["p"], q_values[(by, factor)],
            direction,
        )
    return picked


def _perfect(changes, sign=1, noise=0.0, index=0):
    """A factor series perfectly correlated (sign * y) plus a constant
    offset so distinct panels stay distinct."""
    return [sign * value + index * noise + 10.0 for value in changes]


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    changes = {"region": {f"k{i}": v for i, v in enumerate(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    )}}
    series = {
        "region": {
            "station": _perfect([-2, -1, 0, 1, 2, 3]),
            "lst": [0.0] * 6,
            "morph": [0.0] * 6,
            "cover": [0.0] * 6,
        }
    }
    out = driver_attr(_impact(changes), _factors(changes, series))
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "items"]
    assert list(payload["items"][0]) == [
        "by", "factor", "n", "effect", "p", "q", "reject", "direction",
    ]


def test_empty_report_only_with_empty_data():
    empty = '{"alpha":0.050000,"groups":[]}\n'
    assert driver_attr(empty, {}) == '{"alpha":0.050000,"items":[]}\n'
    with pytest.raises(ValueError):
        driver_attr(empty, {("region", "r1"): (1.0, 2.0, 3.0, 4.0)})


def test_n_integer_reject_boolean_and_six_decimals():
    changes = {"region": {f"k{i}": v for i, v in enumerate(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    )}}
    series = {
        "region": {
            "station": _perfect([-2, -1, 0, 1, 2, 3]),
            "lst": [0.0] * 6,
            "morph": [0.0] * 6,
            "cover": [0.0] * 6,
        }
    }
    out = driver_attr(_impact(changes), _factors(changes, series))
    assert '"n":6' in out
    assert '"reject":true' in out
    assert '"effect":1.000000' in out
    assert '"p":0.002778' in out


def test_negative_zero_normalized():
    changes = {"region": {f"k{i}": v for i, v in enumerate(
        [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    )}}
    series = {
        "region": {
            "station": _perfect([-2, -1, 0, 1, 2, 3]),
            "lst": [0.0] * 6,
            "morph": [0.0] * 6,
            "cover": [0.0] * 6,
        }
    }
    out = driver_attr(_impact(changes), _factors(changes, series))
    assert "-0.000000" not in out


# --- statistics against the exact Fraction reference ------------------------

def test_perfect_correlation_permutation_p_and_directions():
    ys = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    changes = {
        "region": {f"r{i}": ys[i] for i in range(6)},
        "window": {f"w{i}": ys[i] for i in range(6)},
    }
    series = {
        "region": {
            "station": _perfect(ys),
            "lst": [0.0] * 6,
            "morph": [0.0] * 6,
            "cover": [0.0] * 6,
        },
        "window": {
            "station": [0.0] * 6,
            "lst": _perfect(ys, sign=-1),
            "morph": [0.0] * 6,
            "cover": [0.0] * 6,
        },
    }
    out = json.loads(
        driver_attr(_impact(changes), _factors(changes, series))
    )
    picked = _brute_force(0.05, changes, series)
    assert [item["by"] for item in out["items"]] == ["region", "window"]
    by_item = {item["by"]: item for item in out["items"]}
    region = by_item["region"]
    assert region["factor"] == "station"
    assert region["direction"] == "up"
    assert region["effect"] == pytest.approx(1.0, abs=1e-6)
    assert region["p"] == pytest.approx(float(picked["region"][2]), abs=6e-7)
    assert region["q"] == pytest.approx(float(picked["region"][3]), abs=6e-7)
    window = by_item["window"]
    assert window["factor"] == "lst"
    assert window["direction"] == "down"
    assert window["effect"] == pytest.approx(-1.0, abs=1e-6)
    # Two equally-smallest p cells among eight: the rank-1 q takes the
    # running minimum from rank 2: q = N * p / 2 with N = 8 tested cells.
    assert region["q"] == pytest.approx(8 * (2 / 720) / 2, abs=6e-7)
    assert window["q"] == pytest.approx(8 * (2 / 720) / 2, abs=6e-7)


def test_brute_force_values_many_configurations():
    # Generic imperfect data; compare every rendered item to the model.
    changes = {"region": {
        "a": -1.5, "b": 0.25, "c": 1.0, "d": -0.75, "e": 2.0,
    }}
    series = {"region": {
        "station": [-1.2, 0.4, 0.8, -0.9, 2.4],
        "lst": [3.0, -1.0, -2.0, 2.5, -3.5],
        "morph": [0.1, 0.1, 0.2, 0.2, 0.3],
        "cover": [5.0, 5.0, 5.0, 5.0, 5.0],
    }}
    out = json.loads(
        driver_attr(_impact(changes), _factors(changes, series))
    )
    picked = _brute_force(0.05, changes, series)
    assert len(out["items"]) == len(picked)
    for item in out["items"]:
        factor, r2, p, q, direction = picked[item["by"]]
        assert item["factor"] == factor
        assert item["direction"] == direction
        assert item["n"] == 5
        assert item["p"] == pytest.approx(float(p), abs=6e-7)
        assert item["q"] == pytest.approx(float(q), abs=6e-7)
        expected_r = float(r2) ** 0.5 * (
            -1.0 if direction == "down" else 1.0
        )
        assert item["effect"] == pytest.approx(expected_r, abs=6e-7)
        assert item["reject"] is True


def test_factor_tie_resolved_by_factor_order():
    # station and lst are perfectly, equally correlated with y; station
    # must win the per-by pick on factor order.
    ys = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    changes = {"region": {f"k{i}": ys[i] for i in range(6)}}
    series = {"region": {
        "station": _perfect(ys),
        "lst": _perfect(ys),
        "morph": [7.0, -3.0, 2.0, -8.0, 4.0, 1.0],
        "cover": [-7.0, 3.0, -2.0, 8.0, -4.0, -1.0],
    }}
    out = json.loads(
        driver_attr(_impact(changes), _factors(changes, series))
    )
    assert len(out["items"]) == 1
    assert out["items"][0]["factor"] == "station"
    assert out["items"][0]["direction"] == "up"


def test_constant_x_or_constant_y_is_not_rejected():
    changes = {"region": {
        "a": -1.0, "b": 0.0, "c": 1.0, "d": 0.5,
    }}
    series = {"region": {
        "station": [0.0, 0.0, 0.0, 0.0],
        "lst": [1.0, 2.0, 3.0, 4.0],
        "morph": [2.0, 4.0, 1.0, 3.0],
        "cover": [0.0, 0.0, 0.0, 0.0],
    }}
    out = json.loads(
        driver_attr(_impact(changes), _factors(changes, series))
    )
    assert all(
        item["factor"] not in ("station", "cover") for item in out["items"]
    )


def test_constant_y_every_factor_gives_no_items():
    changes = {"region": {f"k{i}": 1.0 for i in range(4)}}
    series = {"region": {
        "station": [1.0, 2.0, 3.0, 4.0],
        "lst": [4.0, 3.0, 2.0, 1.0],
        "morph": [1.0, 1.0, 2.0, 2.0],
        "cover": [0.0, 0.0, 0.0, 0.0],
    }}
    assert json.loads(
        driver_attr(_impact(changes), _factors(changes, series))
    )["items"] == []


def test_alpha_one_picks_flat_for_constant_factor():
    changes = {"region": {"a": -1.0, "b": 1.0}}
    series = {"region": {
        "station": [0.0, 0.0],
        "lst": [0.0, 0.0],
        "morph": [0.0, 0.0],
        "cover": [0.0, 0.0],
    }}
    out = json.loads(
        driver_attr(_impact(changes, alpha=1.0), _factors(changes, series))
    )
    assert len(out["items"]) == 1
    item = out["items"][0]
    assert item["factor"] == "station"
    assert item["effect"] == 0.0
    assert item["reject"] is True
    assert item["direction"] == "flat"


# --- selection across dimensions --------------------------------------------

def test_dimension_without_rejection_is_omitted():
    # n = 2 cannot reach significance: every permutation of a two-point y
    # keeps |r| = 1, so p = 1 for every factor.
    changes = {"region": {"a": -1.0, "b": 1.0}}
    series = {"region": {
        "station": [1.0, 2.0],
        "lst": [2.0, 1.0],
        "morph": [1.0, 1.0],
        "cover": [3.0, 0.0],
    }}
    out = json.loads(
        driver_attr(_impact(changes), _factors(changes, series))
    )
    assert out["items"] == []


def test_items_sort_region_before_window():
    ys = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    changes = {
        "window": {f"w{i}": ys[i] for i in range(6)},
        "region": {f"r{i}": ys[i] for i in range(6)},
    }
    series = {
        "region": {
            "station": _perfect(ys), "lst": [0.0] * 6,
            "morph": [0.0] * 6, "cover": [0.0] * 6,
        },
        "window": {
            "station": [0.0] * 6, "lst": _perfect(ys),
            "morph": [0.0] * 6, "cover": [0.0] * 6,
        },
    }
    out = json.loads(
        driver_attr(_impact(changes), _factors(changes, series))
    )
    assert [item["by"] for item in out["items"]] == ["region", "window"]


# --- input validation -------------------------------------------------------

def test_type_errors():
    report = '{"alpha":0.050000,"groups":[]}\n'
    with pytest.raises(TypeError):
        driver_attr(1, {})
    with pytest.raises(TypeError):
        driver_attr(None, {})
    with pytest.raises(TypeError):
        driver_attr(report, [])


def test_too_many_panels_rejected():
    changes = {"region": {f"k{i}": float(i) for i in range(9)}}
    series = {"region": {
        factor: [float(i) for i in range(9)] for factor in _FACTORS
    }}
    with pytest.raises(ValueError, match="8"):
        driver_attr(_impact(changes), _factors(changes, series))


def test_data_key_set_must_match_panels():
    changes = {"region": {"a": -1.0, "b": 1.0}}
    base_series = {"region": {
        "station": [1.0, 2.0], "lst": [2.0, 1.0],
        "morph": [0.0, 0.0], "cover": [1.0, 1.0],
    }}
    report = _impact(changes)
    good = _factors(changes, base_series)
    driver_attr(report, good)
    with pytest.raises(ValueError):
        driver_attr(report, {})
    missing = dict(good)
    del missing[("region", "a")]
    with pytest.raises(ValueError):
        driver_attr(report, missing)
    extra = dict(good)
    extra[("region", "c")] = (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError):
        driver_attr(report, extra)
    wrong_by = {
        ("window", key): value for (_, key), value in good.items()
    }
    with pytest.raises(ValueError):
        driver_attr(report, wrong_by)


def test_duplicate_panel_key_rejected():
    # A dict cannot hold duplicate keys, so emulate a non-hash/mapping
    # oddity through a key of the wrong shape instead.
    changes = {"region": {"a": -1.0, "b": 1.0}}
    report = _impact(changes)
    with pytest.raises(ValueError):
        driver_attr(report, {
            ("region",): (1.0, 2.0, 3.0, 4.0),
        })
    with pytest.raises(ValueError):
        driver_attr(report, {
            ("region", "a", "x"): (1.0, 2.0, 3.0, 4.0),
        })
    with pytest.raises(ValueError):
        driver_attr(report, {
            ("region", ""): (1.0, 2.0, 3.0, 4.0),
        })


@pytest.mark.parametrize("bad", [
    (1.0, 2.0, 3.0),
    (1.0, 2.0, 3.0, 4.0, 5.0),
    [1.0, 2.0, 3.0, 4.0],
    (1, 2, 3, True),
    (1.0, 2.0, False, 4.0),
    (1.0, float("inf"), 3.0, 4.0),
    (1.0, -float("inf"), 3.0, 4.0),
    (1.0, float("nan"), 3.0, 4.0),
    ("1", 2.0, 3.0, 4.0),
    (None, 2.0, 3.0, 4.0),
])
def test_factor_values_rejected(bad):
    changes = {"region": {"a": -1.0, "b": 1.0}}
    report = _impact(changes)
    data = {
        ("region", "a"): bad,
        ("region", "b"): (1.0, 2.0, 3.0, 4.0),
    }
    with pytest.raises(ValueError):
        driver_attr(report, data)


def test_integer_factor_values_accepted():
    changes = {"region": {"a": -1.0, "b": 0.0, "c": 1.0}}
    series = {"region": {
        "station": [-1, 0, 1],
        "lst": [0, 0, 0],
        "morph": [0, 0, 0],
        "cover": [0, 0, 0],
    }}
    out = driver_attr(_impact(changes), _factors(changes, series))
    assert out.endswith("\n")


@pytest.mark.parametrize("raw", [
    "",
    "not json",
    '{"alpha":0.050000,"groups":[]}',
    '{"alpha":0.050000,"groups":[]}\n\n',
    '{"alpha":0.05,"groups":[]}\n',
])
def test_noncanonical_report_raises_value_error(raw):
    with pytest.raises(ValueError):
        driver_attr(raw, {})


def test_portfolio_report_is_not_a_kind_impact_report():
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 1.0, 2.0)
    )
    with pytest.raises(ValueError):
        driver_attr(report, {})


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    assert _uhi.driver_attr is driver_attr
    assert "driver_attr" in __import__("urban_micro").__all__
