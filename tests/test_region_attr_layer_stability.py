"""Tests for urban_micro.region_attr_layer_stability: per-identity
mean/min/max, rejection share and adjacent rejection-flip aggregation
of region_attr_layer_report outputs across named windows."""

import json
from fractions import Fraction

import pytest

import urban_micro
from urban_micro import region_attr, region_attr_layer_report
from urban_micro import region_attr_layer_stability
from urban_micro import pareto_region_compare, pareto_region_stability
from urban_micro import uhi as _uhi

_FACTORS = ("weather", "morph", "cover")


# --- report / driver builders ----------------------------------------------

def _row(r, w, p, u):
    return (r, w, 0, p, u, 0.0, 0.0)


_GREEN = ("green",)
_ROOF = ("roof",)


def _two_pick_rows():
    return [
        _row("a", "a1", _GREEN, 1.0),
        _row("b", "b1", _GREEN, 9.0),
        _row("c", "c1", _GREEN, 3.0),
        _row("a", "x1", _ROOF, 2.0),
        _row("b", "x2", _ROOF, 8.0),
        _row("c", "x3", _ROOF, 6.0),
    ]


_DRIVERS_A = {
    "a": (0.0, 5.0, 9.0),
    "b": (1.0, 5.0, 2.0),
    "c": (3.0, 5.0, 1.0),
}

_DRIVERS_B = {
    "a": (4.0, 2.0, 0.0),
    "b": (1.0, 8.0, 3.0),
    "c": (3.0, 5.0, 6.0),
}

_DRIVERS_C = {
    "a": (2.0, 9.0, 1.0),
    "b": (0.0, 1.0, 7.0),
    "c": (5.0, 4.0, 2.0),
}


def _stability(rows, *, alpha=0.05):
    single = pareto_region_compare(rows, alpha=alpha)
    return pareto_region_stability({"t0": single, "t1": single})


def _layer_report(drivers_a, drivers_b, layers, *, alpha=0.05):
    stability = _stability(_two_pick_rows(), alpha=alpha)
    reports = {
        "L1": region_attr(stability, drivers_a),
        "L2": region_attr(stability, drivers_b),
    }
    return region_attr_layer_report(reports, layers)


_LAYERS = {
    "L1": ("sun", "dense", "asphalt"),
    "L2": ("rain", "sparse", "grass"),
}


def _three_windows(*, alpha=0.05):
    # Same layers (hence identical by/key identities) and the same
    # underlying picks, but distinct driver maps so the per-window means
    # and rejection flags differ.
    pairs = [
        (_DRIVERS_A, _DRIVERS_B),
        (_DRIVERS_B, _DRIVERS_C),
        (_DRIVERS_C, _DRIVERS_A),
    ]
    return {
        name: _layer_report(a, b, _LAYERS, alpha=alpha)
        for name, (a, b) in zip(("w1", "w2", "w3"), pairs)
    }


# --- exact Fraction reference ----------------------------------------------

def _reference(windows):
    parsed = {name: json.loads(raw) for name, raw in windows.items()}
    alpha = Fraction(str(parsed[next(iter(parsed))]["alpha"]))
    samples = {}
    for name in sorted(parsed):
        for group in parsed[name]["groups"]:
            for item in group["items"]:
                identity = (
                    group["by"], group["key"],
                    tuple(item["pick"]), item["factor"],
                )
                samples.setdefault(identity, []).append(
                    (Fraction(str(item["mean"])), item["reject"])
                )
    groups = {}
    for identity, values in samples.items():
        by, key, pick, factor = identity
        n = len(values)
        mean = sum((v for v, _ in values), Fraction(0)) / n
        low = min(v for v, _ in values)
        high = max(v for v, _ in values)
        rejects = [reject for _v, reject in values]
        significant = Fraction(sum(rejects), n)
        changes = sum(
            1 for index in range(n - 1)
            if rejects[index] != rejects[index + 1]
        )
        groups[identity] = (n, mean, low, high, significant, changes)
    return alpha, groups


def _assert_matches_reference(windows):
    out = region_attr_layer_stability(windows)
    payload = json.loads(out)
    alpha, expected = _reference(windows)
    assert payload["alpha"] == float(alpha)
    seen = set()
    for group in payload["groups"]:
        for item in group["items"]:
            identity = (
                group["by"], group["key"],
                tuple(item["pick"]), item["factor"],
            )
            n, mean, low, high, significant, changes = expected[identity]
            seen.add(identity)
            assert item["n"] == n
            assert isinstance(item["n"], int) and not isinstance(item["n"], bool)
            assert item["changes"] == changes
            assert (
                isinstance(item["changes"], int)
                and not isinstance(item["changes"], bool)
            )
            assert abs(item["mean"] - float(mean)) < 5.1e-7
            assert abs(item["low"] - float(low)) < 5.1e-7
            assert abs(item["high"] - float(high)) < 5.1e-7
            assert abs(item["significant"] - float(significant)) < 5.1e-7
    assert seen == set(expected)
    return out


# --- canonical output --------------------------------------------------------

def test_canonical_shape_and_key_order():
    out = _assert_matches_reference(_three_windows())
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    for group in payload["groups"]:
        assert list(group) == ["by", "key", "items"]
        for item in group["items"]:
            assert list(item) == [
                "pick", "factor", "n", "mean", "low", "high",
                "significant", "changes",
            ]


def test_groups_ordered_by_factor_then_key_and_items_by_pick_then_factor():
    windows = _three_windows()
    groups = json.loads(region_attr_layer_stability(windows))["groups"]
    group_keys = [
        (_FACTORS.index(g["by"]), g["key"]) for g in groups
    ]
    assert group_keys == sorted(group_keys)
    for group in groups:
        item_keys = [
            (tuple(item["pick"]), _FACTORS.index(item["factor"]))
            for item in group["items"]
        ]
        assert item_keys == sorted(item_keys)
        for item in group["items"]:
            assert item["pick"] == sorted(item["pick"])


def test_n_is_window_count():
    for count in range(2, 7):
        pairs = [
            (_DRIVERS_A, _DRIVERS_B),
            (_DRIVERS_B, _DRIVERS_C),
            (_DRIVERS_C, _DRIVERS_A),
        ]
        windows = {
            f"w{i}": _layer_report(*(pairs[i % 3]), _LAYERS)
            for i in range(count)
        }
        groups = json.loads(
            region_attr_layer_stability(windows)
        )["groups"]
        assert groups
        for group in groups:
            assert all(item["n"] == count for item in group["items"])


def test_window_order_is_ascending_for_flips():
    # Windows inserted out of lexicographic order; ascending order is
    # a(T), m(F), z(T) -> two adjacent flips.
    crafted = {
        "z": _crafted_window("0.100000", True),
        "a": _crafted_window("0.100000", True),
        "m": _crafted_window("0.100000", False),
    }
    groups = json.loads(
        region_attr_layer_stability(crafted)
    )["groups"]
    for group in groups:
        assert all(item["changes"] == 2 for item in group["items"])
        assert all(
            abs(item["significant"] - 2 / 3) < 5.1e-7
            for item in group["items"]
        )


def test_rejection_share_and_flip_patterns():
    patterns = [
        ([True, True], 1.0, 0),
        ([False, False], 0.0, 0),
        ([True, False], 0.5, 1),
        ([False, True], 0.5, 1),
        ([True, False, True, False], 0.5, 3),
        ([True, True, False, True], 0.75, 2),
    ]
    for rejects, share, changes in patterns:
        windows = {
            f"w{i:02d}": _crafted_window(
                f"{0.1 + 0.01 * i:.6f}", reject
            )
            for i, reject in enumerate(rejects)
        }
        groups = json.loads(
            region_attr_layer_stability(windows)
        )["groups"]
        for group in groups:
            for item in group["items"]:
                assert item["significant"] == share
                assert item["changes"] == changes


def test_mean_low_high_over_ascending_windows():
    means = ["0.300000", "-0.200000", "0.100000"]
    windows = {
        key: _crafted_window(mean, False)
        for key, mean in zip(("w1", "w2", "w3"), means)
    }
    items = json.loads(
        region_attr_layer_stability(windows)
    )["groups"][0]["items"]
    for item in items:
        assert item["low"] == -0.2
        assert item["high"] == 0.3
        assert abs(item["mean"] - (0.3 - 0.2 + 0.1) / 3) < 5.1e-7


def test_no_negative_zero():
    windows = {
        "w1": _crafted_window("0.000003", True),
        "w2": _crafted_window("-0.000003", False),
    }
    out = region_attr_layer_stability(windows)
    assert "-0.000000" not in out


# --- empty reports -----------------------------------------------------------

def test_empty_reports_yield_empty_groups():
    single = pareto_region_compare([], alpha=0.05)
    stability = pareto_region_stability({"t0": single, "t1": single})
    empty = region_attr(stability, {})
    empty_layer = region_attr_layer_report(
        {"L1": empty, "L2": empty},
        {"L1": ("sun", "dense", "asphalt"),
         "L2": ("rain", "sparse", "grass")},
    )
    out = region_attr_layer_stability(
        {"w1": empty_layer, "w2": empty_layer}
    )
    assert out == '{"alpha":0.050000,"groups":[]}\n'


# --- validation --------------------------------------------------------------

def test_type_errors():
    windows = _three_windows()
    with pytest.raises(TypeError):
        region_attr_layer_stability(None)
    with pytest.raises(TypeError):
        region_attr_layer_stability([])
    with pytest.raises(TypeError):
        region_attr_layer_stability(42)


@pytest.mark.parametrize("count", [0, 1, 17])
def test_report_count_bounds(count):
    raw = next(iter(_three_windows().values()))
    bad = {f"w{i}": raw for i in range(count)}
    with pytest.raises(ValueError):
        region_attr_layer_stability(bad)


def test_empty_or_non_string_report_keys_rejected():
    raw = next(iter(_three_windows().values()))
    bad = {"": raw, "w2": raw}
    with pytest.raises(ValueError):
        region_attr_layer_stability(bad)
    for non_string in (1, True):
        with pytest.raises(ValueError):
            region_attr_layer_stability({non_string: raw, "w2": raw})


def test_alpha_mismatch_rejected():
    windows = _three_windows()
    windows["w2"] = _layer_report(
        _DRIVERS_A, _DRIVERS_B, _LAYERS, alpha=0.5
    )
    with pytest.raises(ValueError):
        region_attr_layer_stability(windows)


def test_identity_mismatch_rejected():
    windows = _three_windows()
    other_layers = {
        "L1": ("sun", "dense", "asphalt"),
        "L2": ("rain", "sparse", "gravel"),
    }
    windows["w3"] = _layer_report(_DRIVERS_C, _DRIVERS_A, other_layers)
    with pytest.raises(ValueError):
        region_attr_layer_stability(windows)


def test_empty_mixed_with_nonempty_rejected():
    single = pareto_region_compare([], alpha=0.05)
    stability = pareto_region_stability({"t0": single, "t1": single})
    empty_layer = region_attr_layer_report(
        {"L1": region_attr(stability, {}),
         "L2": region_attr(stability, {})},
        {"L1": ("sun", "dense", "asphalt"),
         "L2": ("rain", "sparse", "grass")},
    )
    windows = _three_windows()
    windows["w1"] = empty_layer
    with pytest.raises(ValueError):
        region_attr_layer_stability(windows)


@pytest.mark.parametrize("mutate", [
    lambda s: s.rstrip("\n"),
    lambda s: s + "\n",
    lambda s: s.replace('"factor":', '"f":', 1),
    lambda s: '{"groups":[],"alpha":0.050000}\n',
    lambda s: s.replace("0.050000", "0.05", 1),
    lambda s: s.replace("weather", "xather", 1),
    lambda s: s.replace('"items":[', '"items": [', 1),
    lambda s: "not json\n",
])
def test_non_canonical_report_value_error(mutate):
    windows = _three_windows()
    name = next(iter(windows))
    windows[name] = mutate(windows[name])
    with pytest.raises(ValueError):
        region_attr_layer_stability(windows)


def test_non_string_value_value_error():
    with pytest.raises(ValueError):
        region_attr_layer_stability({"w1": 42, "w2": 43})


# --- export ------------------------------------------------------------------

def test_exported_from_package_root():
    assert _uhi.region_attr_layer_stability is (
        urban_micro.region_attr_layer_stability
    )
    assert "region_attr_layer_stability" in urban_micro.__all__
    assert "region_attr_layer_stability" in _uhi.__all__


# --- hand-crafted canonical layer-report strings -----------------------------

def _crafted_window(mean: str, reject: bool) -> str:
    """A canonical region_attr_layer_report with one weather/sun group,
    the three factors over pick ``["green"]`` carrying identical mean and
    rejection flags."""
    flag = "true" if reject else "false"
    item_tokens = [
        '{"pick":["green"]'
        f',"factor":{json.dumps(factor)}'
        ',"n":2'
        f',"mean":{mean}'
        ',"p":0.500000'
        ',"q":0.500000'
        f',"reject":{flag}'
        + "}"
        for factor in _FACTORS
    ]
    return (
        '{"alpha":0.050000,"groups":['
        '{"by":"weather","key":"sun","items":['
        + ",".join(item_tokens)
        + "]}]}\n"
    )
