"""Tests for urban_micro.region_attr_layer_stability: aggregation of
region_attr_layer_report outputs across named windows into per-identity
mean/low/high, significance share and adjacent-reject change counts."""

import json
import re
from fractions import Fraction

import pytest

import urban_micro
from urban_micro import pareto_region_compare, pareto_region_stability
from urban_micro import region_attr, region_attr_layer_report
from urban_micro import region_attr_layer_stability
from urban_micro import uhi as _uhi

_FACTORS = ("weather", "morph", "cover")


# --- report builders --------------------------------------------------------

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

_LAYERS = {
    "L1": ("sun", "dense", "asphalt"),
    "L2": ("rain", "sparse", "grass"),
    "L3": ("wind", "loose", "gravel"),
}


def _stability(rows, *, alpha=0.05):
    single = pareto_region_compare(rows, alpha=alpha)
    return pareto_region_stability({"t0": single, "t1": single})


def _region_reports(rows, drivers_by_layer, *, alpha=0.05):
    stability = _stability(rows, alpha=alpha)
    return {
        name: region_attr(stability, drivers)
        for name, drivers in drivers_by_layer.items()
    }


def _canonical_layer_report(drivers, *, alpha=1.0):
    reports = _region_reports(
        _two_pick_rows(),
        {name: drivers for name in _LAYERS},
        alpha=alpha,
    )
    return region_attr_layer_report(reports, _LAYERS)


def _with_rejects(raw, forced):
    """Return a canonical layer report copy with ``reject`` set to the
    given bool for the identities in ``forced`` (identity -> bool).

    Only the targeted ``reject`` tokens are rewritten, so every other
    byte (including the six-decimal number tokens and key order) is
    preserved and the result stays canonical."""
    body = raw[:-1]
    pattern = re.compile(r',"reject":(?:true|false)')
    identities = _identities(raw)
    replacements = {
        index: ',"reject":' + ("true" if forced[identity] else "false")
        for index, identity in enumerate(identities)
        if identity in forced
    }
    counter = 0

    def _swap(match):
        nonlocal counter
        replacement = replacements.get(counter, match.group(0))
        counter += 1
        return replacement

    return pattern.sub(_swap, body) + "\n"


def _identities(raw):
    out = []
    for group in json.loads(raw)["groups"]:
        for item in group["items"]:
            out.append(
                (group["by"], group["key"], tuple(item["pick"]),
                 item["factor"])
            )
    return out


# --- exact Fraction reference ----------------------------------------------

def _reference(windows):
    parsed = {name: json.loads(raw) for name, raw in windows.items()}
    names = sorted(parsed)
    alpha = Fraction(str(parsed[names[0]]["alpha"]))

    series = {}
    for name in names:
        for group in parsed[name]["groups"]:
            for item in group["items"]:
                identity = (
                    group["by"], group["key"], tuple(item["pick"]),
                    item["factor"],
                )
                series.setdefault(identity, []).append(
                    (Fraction(str(item["mean"])), bool(item["reject"]))
                )

    groups = {}
    for identity, samples in series.items():
        by, key, pick, factor = identity
        assert len(samples) == len(names)
        values = [x for x, _b in samples]
        rejects = [b for _x, b in samples]
        n = len(values)
        groups[identity] = {
            "n": n,
            "mean": sum(values, Fraction(0)) / n,
            "low": min(values),
            "high": max(values),
            "significant": Fraction(
                sum(1 for b in rejects if b), n
            ),
            "changes": sum(
                1
                for i in range(n - 1)
                if rejects[i] != rejects[i + 1]
            ),
        }
    return alpha, groups


def _assert_matches_reference(windows):
    out = region_attr_layer_stability(dict(windows))
    expected_alpha, expected = _reference(windows)
    payload = json.loads(out)
    assert Fraction(str(payload["alpha"])) == expected_alpha
    seen = set()
    for group in payload["groups"]:
        for item in group["items"]:
            identity = (
                group["by"], group["key"], tuple(item["pick"]),
                item["factor"],
            )
            exp = expected[identity]
            assert item["n"] == exp["n"]
            assert isinstance(item["n"], int) and not isinstance(item["n"], bool)
            assert item["changes"] == exp["changes"]
            assert isinstance(item["changes"], int)
            assert not isinstance(item["changes"], bool)
            assert abs(float(item["mean"]) - float(exp["mean"])) < 5.1e-7
            assert Fraction(str(item["low"])) == exp["low"]
            assert Fraction(str(item["high"])) == exp["high"]
            assert (
                abs(float(item["significant"]) - float(exp["significant"]))
                < 5.1e-7
            )
            seen.add(identity)
    assert seen == set(expected)
    return out


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    windows = {
        "w2": _canonical_layer_report(_DRIVERS_A),
        "w1": _canonical_layer_report(_DRIVERS_B),
        "w3": _canonical_layer_report(_DRIVERS_C),
    }
    out = _assert_matches_reference(windows)
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
            assert isinstance(item["pick"], list)
            assert item["pick"] == sorted(item["pick"])


def test_n_is_window_count_and_groups_order_by_factor_then_key():
    windows = {
        "w2": _canonical_layer_report(_DRIVERS_A),
        "w1": _canonical_layer_report(_DRIVERS_B),
        "w3": _canonical_layer_report(_DRIVERS_C),
    }
    groups = json.loads(region_attr_layer_stability(windows))["groups"]
    by_keys = [(g["by"], g["key"]) for g in groups]
    expected_order = [
        ("weather", "rain"),
        ("weather", "sun"),
        ("weather", "wind"),
        ("morph", "dense"),
        ("morph", "loose"),
        ("morph", "sparse"),
        ("cover", "asphalt"),
        ("cover", "grass"),
        ("cover", "gravel"),
    ]
    assert by_keys == expected_order
    for group in groups:
        for item in group["items"]:
            assert item["n"] == 3
        keys = [(tuple(it["pick"]), _FACTORS.index(it["factor"]))
                for it in group["items"]]
        assert keys == sorted(keys)


def test_mean_low_high_over_varying_windows():
    windows = {
        "w1": _canonical_layer_report(_DRIVERS_A),
        "w2": _canonical_layer_report(_DRIVERS_B),
        "w3": _canonical_layer_report(_DRIVERS_C),
    }
    _assert_matches_reference(windows)


def test_window_order_is_ascending_key_not_insertion_order():
    raw_a = _canonical_layer_report(_DRIVERS_A)
    raw_b = _canonical_layer_report(_DRIVERS_B)
    # Insertion order deliberately reversed; statistics must not depend
    # on it (both variants share means here, so this guards ordering).
    assert region_attr_layer_stability({"w1": raw_a, "w2": raw_b}) == (
        region_attr_layer_stability({"w2": raw_b, "w1": raw_a})
    )


# --- significant / changes via reject patterns ------------------------------

def test_significant_and_changes_reject_patterns():
    base = _canonical_layer_report(_DRIVERS_A, alpha=0.05)
    identities = _identities(base)
    # With alpha 0.05 and n=3 layers nothing rejects: choose four
    # identities and force explicit 4-window reject patterns.
    a = identities[0]
    b = identities[1]
    c = identities[2]
    d = identities[3]
    # Sorted window order is p0 < q0 < r0 < s0; assign patterns in that
    # order: true, true, false, false etc.
    patterns = {
        a: (True, True, False, False),   # 0.50, 1 change
        b: (True, False, True, False),   # 0.50, 3 changes
        c: (False, False, False, False),  # 0.00, 0 changes
        d: (True, True, True, False),    # 0.75, 1 change
    }
    window_keys = ("p0", "q0", "r0", "s0")
    windows = {}
    for index, name in enumerate(window_keys):
        forced = {identity: pattern[index]
                  for identity, pattern in patterns.items()}
        windows[name] = _with_rejects(base, forced)
    out = region_attr_layer_stability(windows)
    wanted = {a: (0.5, 1), b: (0.5, 3), c: (0.0, 0), d: (0.75, 1)}
    found = {}
    for group in json.loads(out)["groups"]:
        for item in group["items"]:
            identity = (
                group["by"], group["key"], tuple(item["pick"]),
                item["factor"],
            )
            if identity in wanted:
                assert item["n"] == 4
                found[identity] = (item["significant"], item["changes"])
    assert found == wanted
    # Untouched identities stay all-false: significant 0, changes 0.
    for group in json.loads(out)["groups"]:
        for item in group["items"]:
            identity = (
                group["by"], group["key"], tuple(item["pick"]),
                item["factor"],
            )
            if identity not in wanted:
                assert item["significant"] == 0.0
                assert item["changes"] == 0


def test_no_negative_zero():
    raw_a = _canonical_layer_report(_DRIVERS_A, alpha=0.05)
    raw_b = _canonical_layer_report(_DRIVERS_B, alpha=0.05)
    assert "-0.000000" not in region_attr_layer_stability(
        {"w1": raw_a, "w2": raw_b}
    )


# --- empty reports -----------------------------------------------------------

def test_empty_reports_yield_empty_groups():
    single = pareto_region_compare([], alpha=0.05)
    stability = pareto_region_stability({"t0": single, "t1": single})
    empty = region_attr(stability, {})
    layers = {
        "L1": ("sun", "dense", "asphalt"),
        "L2": ("rain", "sparse", "grass"),
    }
    empty_layer = region_attr_layer_report(
        {"L1": empty, "L2": empty}, layers
    )
    assert region_attr_layer_stability(
        {"w1": empty_layer, "w2": empty_layer}
    ) == '{"alpha":0.050000,"groups":[]}\n'


# --- validation --------------------------------------------------------------

def _two_windows():
    raw = _canonical_layer_report(_DRIVERS_A)
    return {"w1": raw, "w2": raw}


def test_type_errors():
    windows = _two_windows()
    with pytest.raises(TypeError):
        region_attr_layer_stability(None)
    with pytest.raises(TypeError):
        region_attr_layer_stability([])
    with pytest.raises(TypeError):
        region_attr_layer_stability(list(windows.values()))


@pytest.mark.parametrize("count", [0, 1, 17])
def test_window_count_bounds(count):
    raw = _canonical_layer_report(_DRIVERS_A)
    bad = {f"w{i}": raw for i in range(count)}
    with pytest.raises(ValueError):
        region_attr_layer_stability(bad)


def test_empty_or_non_string_window_keys_rejected():
    raw = _canonical_layer_report(_DRIVERS_A)
    with pytest.raises(ValueError):
        region_attr_layer_stability({"": raw, "w2": raw})
    # Non-string keys are ValueError, not TypeError.
    try:
        region_attr_layer_stability({1: raw, "w2": raw})
    except ValueError:
        pass
    except TypeError:  # pragma: no cover - explicit contract guard
        raise AssertionError("non-string key must raise ValueError")
    else:  # pragma: no cover
        raise AssertionError("non-string key must raise ValueError")


def test_alpha_mismatch_rejected():
    windows = {
        "w1": _canonical_layer_report(_DRIVERS_A, alpha=1.0),
        "w2": _canonical_layer_report(_DRIVERS_B, alpha=0.05),
    }
    with pytest.raises(ValueError):
        region_attr_layer_stability(windows)


def test_identity_mismatch_rejected():
    # Different layer category strings across windows -> different
    # (by, key) identities.
    rows = _two_pick_rows()
    reports = _region_reports(
        rows, {"L1": _DRIVERS_A, "L2": _DRIVERS_B}, alpha=1.0
    )
    layers_one = {
        "L1": ("sun", "dense", "asphalt"),
        "L2": ("rain", "sparse", "grass"),
    }
    layers_two = {
        "L1": ("moon", "dense", "asphalt"),
        "L2": ("rain", "sparse", "grass"),
    }
    windows = {
        "w1": region_attr_layer_report(reports, layers_one),
        "w2": region_attr_layer_report(reports, layers_two),
    }
    with pytest.raises(ValueError):
        region_attr_layer_stability(windows)


@pytest.mark.parametrize("mutate", [
    lambda s: s.rstrip("\n"),
    lambda s: s + "\n",
    lambda s: s.replace('"mean":', '"m":', 1),
    lambda s: '{"groups":[],"alpha":1.000000}\n',
    lambda s: s.replace("1.000000", "1.0", 1),
    lambda s: "not json\n",
])
def test_non_canonical_window_value_error(mutate):
    windows = _two_windows()
    windows = dict(windows)
    windows["w1"] = mutate(windows["w1"])
    with pytest.raises(ValueError):
        region_attr_layer_stability(windows)


def test_empty_mixed_with_nonempty_rejected():
    single = pareto_region_compare([], alpha=0.05)
    stability = pareto_region_stability({"t0": single, "t1": single})
    empty = region_attr(stability, {})
    empty_layer = region_attr_layer_report(
        {"L1": empty, "L2": empty},
        {"L1": ("sun", "dense", "asphalt"),
         "L2": ("rain", "sparse", "grass")},
    )
    with pytest.raises(ValueError):
        region_attr_layer_stability(
            {"w1": empty_layer, "w2": _canonical_layer_report(_DRIVERS_A)}
        )


# --- export ------------------------------------------------------------------

def test_exported_from_package_root():
    assert _uhi.region_attr_layer_stability is (
        urban_micro.region_attr_layer_stability
    )
    assert "region_attr_layer_stability" in urban_micro.__all__
    assert "region_attr_layer_stability" in _uhi.__all__
