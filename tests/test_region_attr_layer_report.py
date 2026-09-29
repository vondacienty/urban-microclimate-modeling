"""Tests for urban_micro.region_attr_layer_report: sign-flip
aggregation of region_attr effect values across named layers, grouped
by the weather/morph/cover category strings, with a global
Benjamini-Hochberg pass."""

import json
from fractions import Fraction

import pytest

import urban_micro
from urban_micro import pareto_region_compare, pareto_region_stability
from urban_micro import region_attr, region_attr_layer_report
from urban_micro import uhi as _uhi

_FACTORS = ("weather", "morph", "cover")


# --- report / driver builders ----------------------------------------------

def _row(r, w, p, u):
    return (r, w, 0, p, u, 0.0, 0.0)


_GREEN = ("green",)
_ROOF = ("roof",)


def _n3_rows():
    return [
        _row("a", "a1", _GREEN, 0.0),
        _row("b", "b1", _GREEN, 3.0),
        _row("c", "c1", _GREEN, 9.0),
    ]


def _two_pick_rows():
    # Three regions contribute to both picks -> three pairs per pick,
    # keeping the region_attr effects non-degenerate.
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


def _region_reports(rows, drivers_by_layer, *, alpha=0.05):
    stability = _stability(rows, alpha=alpha)
    return {
        name: region_attr(stability, drivers)
        for name, drivers in drivers_by_layer.items()
    }


# --- exact Fraction reference ----------------------------------------------

def _reference(reports, layers):
    parsed = {name: json.loads(raw) for name, raw in reports.items()}
    alpha = Fraction(str(parsed[next(iter(parsed))]["alpha"]))

    # (slot, category, pick, factor index) -> [effect Fraction]
    samples = {}
    for name, payload in parsed.items():
        triple = layers[name]
        for group in payload["groups"]:
            pick = tuple(group["pick"])
            effects = {item["factor"]: item["effect"]
                       for item in group["items"]}
            for slot, category in enumerate(triple):
                for factor_index, factor in enumerate(_FACTORS):
                    key = (slot, category, pick, factor_index)
                    samples.setdefault(key, []).append(
                        Fraction(str(effects[factor]))
                    )

    rows = []
    for (slot, category, pick, factor_index), values in samples.items():
        n = len(values)
        total = sum(values, Fraction(0))
        mean = total / n
        hits = 0
        for mask in range(1 << n):
            signed = Fraction(0)
            for index, value in enumerate(values):
                signed += -value if (mask >> index) & 1 else value
            if abs(signed) >= abs(total):
                hits += 1
        p_value = Fraction(hits, 1 << n)
        rows.append((p_value, slot, category, pick, factor_index, n, mean))

    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
    tested = len(rows)
    q_values = {}
    running = Fraction(1)
    for rank in range(tested, 0, -1):
        p_value, slot, category, pick, factor_index, _n, _mean = rows[rank - 1]
        running = min(running, Fraction(tested) * p_value / rank)
        q_values[(slot, category, pick, factor_index)] = min(
            Fraction(1), running
        )

    groups = []
    for slot in range(3):
        categories = sorted({
            category for s, category, *_ in q_values if s == slot
        })
        for category in categories:
            items = []
            keys = sorted(
                (pick, factor_index)
                for s, cat, pick, factor_index in q_values
                if s == slot and cat == category
            )
            for pick, factor_index in keys:
                match = next(
                    row for row in rows
                    if row[1] == slot and row[2] == category
                    and row[3] == pick and row[4] == factor_index
                )
                p_value, _s, _c, _pk, _fi, n, mean = match
                q_value = q_values[(slot, category, pick, factor_index)]
                items.append(
                    {
                        "pick": list(pick),
                        "factor": _FACTORS[factor_index],
                        "n": n,
                        "mean": float(mean),
                        "p": float(p_value),
                        "q": float(q_value),
                        "reject": q_value <= alpha,
                    }
                )
            groups.append(
                {"by": _FACTORS[slot], "key": category, "items": items}
            )
    return {"alpha": float(alpha), "groups": groups}


def _assert_matches_reference(reports, layers):
    out = region_attr_layer_report(reports, layers)
    payload = json.loads(out)
    expected = _reference(reports, layers)
    assert payload["alpha"] == expected["alpha"]
    assert len(payload["groups"]) == len(expected["groups"])
    for group, exp_group in zip(payload["groups"], expected["groups"]):
        assert group["by"] == exp_group["by"]
        assert group["key"] == exp_group["key"]
        assert len(group["items"]) == len(exp_group["items"])
        for item, exp in zip(group["items"], exp_group["items"]):
            assert list(item) == [
                "pick", "factor", "n", "mean", "p", "q", "reject",
            ]
            assert item["pick"] == exp["pick"]
            assert item["factor"] == exp["factor"]
            assert item["n"] == exp["n"]
            assert isinstance(item["n"], int) and not isinstance(item["n"], bool)
            assert item["reject"] is exp["reject"]
            assert abs(item["mean"] - exp["mean"]) < 5.1e-7
            assert abs(item["p"] - exp["p"]) < 5.1e-7
            assert abs(item["q"] - exp["q"]) < 5.1e-7
    return out


# --- canonical output --------------------------------------------------------

def test_canonical_shape_and_key_order():
    reports = _region_reports(
        _n3_rows(), {"L1": _DRIVERS_A, "L2": _DRIVERS_B}, alpha=1.0
    )
    layers = {
        "L1": ("sun", "dense", "asphalt"),
        "L2": ("rain", "sparse", "grass"),
    }
    out = _assert_matches_reference(reports, layers)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    for group in payload["groups"]:
        assert list(group) == ["by", "key", "items"]
        for item in group["items"]:
            assert list(item) == [
                "pick", "factor", "n", "mean", "p", "q", "reject",
            ]
            assert isinstance(item["reject"], bool)


def test_n_is_report_count_and_groups_ordered_by_layer_then_key():
    reports = _region_reports(
        _n3_rows(),
        {"L1": _DRIVERS_A, "L2": _DRIVERS_B, "L3": _DRIVERS_C},
        alpha=1.0,
    )
    # Categories deliberately inserted out of lexicographic order.
    layers = {
        "L1": ("zzz-sun", "mmm-dense", "bbb-asphalt"),
        "L2": ("aaa-rain", "zzz-sparse", "mmm-grass"),
        "L3": ("mmm-wind", "aaa-loose", "zzz-gravel"),
    }
    out = _assert_matches_reference(reports, layers)
    groups = json.loads(out)["groups"]
    by_keys = [(g["by"], g["key"]) for g in groups]
    expected_order = [
        ("weather", "aaa-rain"),
        ("weather", "mmm-wind"),
        ("weather", "zzz-sun"),
        ("morph", "aaa-loose"),
        ("morph", "mmm-dense"),
        ("morph", "zzz-sparse"),
        ("cover", "bbb-asphalt"),
        ("cover", "mmm-grass"),
        ("cover", "zzz-gravel"),
    ]
    assert by_keys == expected_order
    # Every category string is unique to its layer, so each group
    # pools a single report: n == 1.
    for group in groups:
        assert all(item["n"] == 1 for item in group["items"])


def test_n_counts_layers_sharing_a_category():
    reports = _region_reports(
        _n3_rows(),
        {"L1": _DRIVERS_A, "L2": _DRIVERS_B, "L3": _DRIVERS_C},
        alpha=1.0,
    )
    # All three layers share one weather category; morph/cover differ.
    layers = {
        "L1": ("shared", "dense1", "cover1"),
        "L2": ("shared", "dense2", "cover2"),
        "L3": ("shared", "dense3", "cover3"),
    }
    groups = json.loads(
        region_attr_layer_report(reports, layers)
    )["groups"]
    weather = [g for g in groups if g["by"] == "weather"]
    assert [g["key"] for g in weather] == ["shared"]
    assert all(item["n"] == 3 for item in weather[0]["items"])
    assert all(
        item["n"] == 1
        for group in groups if group["by"] != "weather"
        for item in group["items"]
    )


def test_items_ordered_by_pick_then_factor():
    reports = _region_reports(
        _two_pick_rows(), {"L1": _DRIVERS_A, "L2": _DRIVERS_B}, alpha=1.0
    )
    layers = {
        "L1": ("sun", "dense", "asphalt"),
        "L2": ("rain", "sparse", "grass"),
    }
    out = _assert_matches_reference(reports, layers)
    expected = [
        (("green",), "weather"),
        (("green",), "morph"),
        (("green",), "cover"),
        (("roof",), "weather"),
        (("roof",), "morph"),
        (("roof",), "cover"),
    ]
    for group in json.loads(out)["groups"]:
        keys = [(tuple(item["pick"]), item["factor"])
                for item in group["items"]]
        assert keys == expected


def test_shared_category_collapses_layers_into_one_group():
    # Both layers report the same weather category string, so the
    # weather group holds n=2 per identity while morph/cover stay split.
    reports = _region_reports(
        _n3_rows(), {"L1": _DRIVERS_A, "L2": _DRIVERS_B}, alpha=1.0
    )
    layers = {
        "L1": ("same", "dense", "asphalt"),
        "L2": ("same", "sparse", "grass"),
    }
    out = _assert_matches_reference(reports, layers)
    groups = json.loads(out)["groups"]
    weather = [g for g in groups if g["by"] == "weather"]
    assert [g["key"] for g in weather] == ["same"]
    assert all(item["n"] == 2 for item in weather[0]["items"])
    morph = [g["key"] for g in groups if g["by"] == "morph"]
    assert morph == ["dense", "sparse"]


def test_alpha_levels():
    for alpha in (0.05, 0.5, 1.0):
        reports = _region_reports(
            _n3_rows(),
            {"L1": _DRIVERS_A, "L2": _DRIVERS_B},
            alpha=alpha,
        )
        layers = {
            "L1": ("sun", "dense", "asphalt"),
            "L2": ("rain", "sparse", "grass"),
        }
        _assert_matches_reference(reports, layers)


def test_no_negative_zero():
    reports = _region_reports(
        _two_pick_rows(), {"L1": _DRIVERS_A, "L2": _DRIVERS_B}
    )
    layers = {
        "L1": ("sun", "dense", "asphalt"),
        "L2": ("rain", "sparse", "grass"),
    }
    assert "-0.000000" not in region_attr_layer_report(reports, layers)


# --- empty reports -----------------------------------------------------------

def test_empty_reports_yield_empty_groups():
    single = pareto_region_compare([], alpha=0.05)
    stability = pareto_region_stability({"t0": single, "t1": single})
    empty = region_attr(stability, {})
    reports = {"L1": empty, "L2": empty}
    layers = {
        "L1": ("sun", "dense", "asphalt"),
        "L2": ("rain", "sparse", "grass"),
    }
    assert region_attr_layer_report(reports, layers) == (
        '{"alpha":0.050000,"groups":[]}\n'
    )


# --- validation --------------------------------------------------------------

def _two_reports(alpha=0.05):
    return _region_reports(
        _two_pick_rows(),
        {"L1": _DRIVERS_A, "L2": _DRIVERS_B},
        alpha=alpha,
    )


def _two_layers():
    return {
        "L1": ("sun", "dense", "asphalt"),
        "L2": ("rain", "sparse", "grass"),
    }


def test_type_errors():
    reports = _two_reports()
    layers = _two_layers()
    with pytest.raises(TypeError):
        region_attr_layer_report(None, layers)
    with pytest.raises(TypeError):
        region_attr_layer_report([], layers)
    with pytest.raises(TypeError):
        region_attr_layer_report(reports, None)
    with pytest.raises(TypeError):
        region_attr_layer_report(reports, [])


@pytest.mark.parametrize("count", [0, 1, 17])
def test_report_count_bounds(count):
    pair = _two_reports()
    values = list(pair.values())
    keys = [f"L{i}" for i in range(count)]
    bad_reports = {key: values[i % 2] for i, key in enumerate(keys)}
    bad_layers = {key: (f"w{i}", f"m{i}", f"c{i}")
                  for i, key in enumerate(keys)}
    with pytest.raises(ValueError):
        region_attr_layer_report(bad_reports, bad_layers)


def test_empty_or_non_string_report_keys_rejected():
    reports = _two_reports()
    layers = _two_layers()
    bad_reports = dict(reports)
    bad_reports[""] = next(iter(reports.values()))
    bad_layers = dict(layers)
    bad_layers[""] = ("w", "m", "c")
    with pytest.raises(ValueError):
        region_attr_layer_report(bad_reports, bad_layers)


def test_layers_keys_must_match_reports():
    reports = _two_reports()
    with pytest.raises(ValueError):
        region_attr_layer_report(
            reports, {"L1": ("w", "m", "c"), "ZZZ": ("w", "m", "c")}
        )
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, {"L1": ("w", "m", "c")})


@pytest.mark.parametrize("bad", [
    ["w", "m", "c"],
    ("w", "m"),
    ("w", "m", "c", "x"),
    ("", "m", "c"),
    ("w", "", "c"),
    ("w", "m", ""),
    ("w", 1, "c"),
    ("w", "m", None),
])
def test_layer_values_must_be_three_nonempty_strings(bad):
    reports = _two_reports()
    layers = _two_layers()
    layers["L1"] = bad
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, layers)


def test_alpha_mismatch_rejected():
    stable = _stability(_two_pick_rows(), alpha=0.05)
    reports = {
        "L1": region_attr(stable, _DRIVERS_A),
        "L2": region_attr(
            _stability(_two_pick_rows(), alpha=0.5), _DRIVERS_B
        ),
    }
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, _two_layers())


def test_identity_mismatch_rejected():
    # L2 carries only the green pick while L1 carries green and roof;
    # both share regions a, b so the driver keys stay valid.
    two = _two_pick_rows()
    green_only = [row for row in two if row[3] == _GREEN]
    reports = {
        "L1": region_attr(_stability(two), _DRIVERS_A),
        "L2": region_attr(_stability(green_only), _DRIVERS_B),
    }
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, _two_layers())


@pytest.mark.parametrize("mutate", [
    lambda s: s.rstrip("\n"),
    lambda s: s + "\n",
    lambda s: s.replace('"effect":', '"e":', 1),
    lambda s: '{"groups":[],"alpha":0.050000}\n',
    lambda s: s.replace("0.050000", "0.05", 1),
    lambda s: "not json\n",
])
def test_non_canonical_report_value_error(mutate):
    reports = _two_reports()
    reports = dict(reports)
    reports["L1"] = mutate(reports["L1"])
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, _two_layers())


def test_empty_mixed_with_nonempty_rejected():
    single = pareto_region_compare([], alpha=0.05)
    empty_stable = pareto_region_stability({"t0": single, "t1": single})
    empty = region_attr(empty_stable, {})
    reports = {"L1": empty, "L2": next(iter(_two_reports().values()))}
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, _two_layers())


# --- export ------------------------------------------------------------------

def test_exported_from_package_root():
    assert _uhi.region_attr_layer_report is (
        urban_micro.region_attr_layer_report
    )
    assert "region_attr_layer_report" in urban_micro.__all__
    assert "region_attr_layer_report" in _uhi.__all__
