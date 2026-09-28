"""Tests for urban_micro.driver_attr_stability_summary: per-factor
stable / conflict / cross_zero verdicts over a canonical
driver_attr_stability report."""

import json

import pytest

from urban_micro import driver_attr_stability, driver_attr_stability_summary
from urban_micro import uhi as _uhi

_FACTORS = ("station", "lst", "morph", "cover")
_ITEM_KEYS = [
    "factor", "direction", "frequency", "consistency", "significant",
    "low", "high", "stable", "conflict", "cross_zero",
]


# --- report builders --------------------------------------------------------

def _da_item(by, factor, effect, reject, direction, n=3,
             p="0.333333", q="0.666667", alpha=0.05):
    """Hand-build one canonical driver_attr report string."""
    reject_token = "true" if reject else "false"
    return (
        f'{{"alpha":{alpha:.6f},"items":[{{"by":{json.dumps(by)},'
        f'"factor":{json.dumps(factor)},"n":{n},"effect":{effect:.6f},'
        f'"p":{p},"q":{q},"reject":{reject_token},'
        f'"direction":{json.dumps(direction)}}}]}}\n'
    )


def _stability_item_token(factor, n, frequency, direction, consistency,
                          low, high, significant):
    """One canonical driver_attr_stability item object token."""
    return (
        '{"factor":' + json.dumps(factor)
        + f',"n":{n},"frequency":{frequency:.6f},'
        + '"direction":' + json.dumps(direction)
        + f',"consistency":{consistency:.6f},"low":{low:.6f},'
        + f'"high":{high:.6f},"significant":{significant:.6f}}}'
    )


def _stability_report(*group_tokens, alpha=0.05):
    return (
        f'{{"alpha":{alpha:.6f},"groups":[' + ",".join(group_tokens) + "]}\n"
    )


def _region_group(*item_tokens):
    return '{"by":"region","items":[' + ",".join(item_tokens) + "]}"


def _four_items(station=(0, 0.0, "flat", 0.0, 0.0, 0.0, 0.0),
                lst=None, morph=None, cover=None):
    """Build the four canonical items; each tuple is
    ``(n, frequency, direction, consistency, low, high, significant)``.
    Unspecified factors are empty (n == 0)."""
    empty = (0, 0.0, "flat", 0.0, 0.0, 0.0, 0.0)
    rows = [station, lst or empty, morph or empty, cover or empty]
    return [_stability_item_token(factor, *row)
            for factor, row in zip(_FACTORS, rows)]


def _items_by_factor(out):
    groups = json.loads(out)["groups"]
    return {item["factor"]: item for item in groups[0]["items"]}


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    report = _stability_report(_region_group(*_four_items()))
    out = driver_attr_stability_summary(report)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    assert list(payload["groups"][0]) == ["by", "items"]
    group = payload["groups"][0]
    assert [item["factor"] for item in group["items"]] == list(_FACTORS)
    for item in group["items"]:
        assert list(item) == _ITEM_KEYS
    # n is deliberately not carried into the summary.
    assert "n" not in group["items"][0]


def test_region_before_window_order():
    region = _region_group(*_four_items())
    window = '{"by":"window","items":[' + ",".join(_four_items()) + "]}"
    report = _stability_report(region, window)
    out = driver_attr_stability_summary(report)
    assert [g["by"] for g in json.loads(out)["groups"]] == ["region", "window"]


def test_empty_groups_round_trip():
    report = '{"alpha":0.050000,"groups":[]}\n'
    assert driver_attr_stability_summary(report) == report


def test_stable_requires_every_condition():
    # Picked in every report, unanimous down, half significant, entirely
    # negative: meets the default thresholds.
    out = driver_attr_stability_summary(
        _stability_report(_region_group(*_four_items(station=(
            4, 1.0, "down", 1.0, -1.0, -0.1, 0.5
        ))))
    )
    item = _items_by_factor(out)["station"]
    assert item["stable"] is True
    assert item["conflict"] is False
    assert item["cross_zero"] is False


def test_n_zero_is_never_stable_conflicting_or_crossing():
    out = driver_attr_stability_summary(
        _stability_report(_region_group(*_four_items()))
    )
    for item in json.loads(out)["groups"][0]["items"]:
        assert item["stable"] is False
        assert item["conflict"] is False
        assert item["cross_zero"] is False


def test_conflict_when_split_direction():
    out = driver_attr_stability_summary(
        _stability_report(_region_group(*_four_items(cover=(
            2, 1.0, "down", 0.5, -0.3, 0.3, 1.0
        ))))
    )
    item = _items_by_factor(out)["cover"]
    assert item["conflict"] is True
    assert item["cross_zero"] is True
    assert item["stable"] is False


def test_unanimous_nonzero_interval_does_not_conflict():
    out = driver_attr_stability_summary(
        _stability_report(_region_group(*_four_items(lst=(
            3, 1.0, "up", 1.0, 0.1, 0.4, 0.0
        ))))
    )
    item = _items_by_factor(out)["lst"]
    assert item["conflict"] is False
    assert item["cross_zero"] is False


def test_cross_zero_endpoints_inclusive():
    # low == 0 crosses zero even when every effect is non-negative.
    out = driver_attr_stability_summary(
        _stability_report(_region_group(*_four_items(morph=(
            2, 1.0, "up", 1.0, 0.0, 0.4, 1.0
        ))))
    )
    item = _items_by_factor(out)["morph"]
    assert item["cross_zero"] is True
    assert item["stable"] is False
    # high == 0 likewise.
    out = driver_attr_stability_summary(
        _stability_report(_region_group(*_four_items(morph=(
            2, 1.0, "down", 1.0, -0.4, 0.0, 1.0
        ))))
    )
    assert _items_by_factor(out)["morph"]["cross_zero"] is True


def test_flat_direction_never_stable():
    out = driver_attr_stability_summary(
        _stability_report(_region_group(*_four_items(station=(
            4, 1.0, "flat", 1.0, 0.0, 0.0, 1.0
        ))))
    )
    item = _items_by_factor(out)["station"]
    assert item["direction"] == "flat"
    assert item["cross_zero"] is True
    assert item["stable"] is False


def test_threshold_boundaries_are_inclusive():
    # Exactly at frequency 0.5, consistency 0.75, significant 0.5.
    row = (4, 0.5, "up", 0.75, 0.1, 0.9, 0.5)
    report = _stability_report(_region_group(*_four_items(lst=row)))
    item = _items_by_factor(driver_attr_stability_summary(report))["lst"]
    assert item["stable"] is True
    # Any threshold one notch above its metric flips stable off.
    assert _items_by_factor(
        driver_attr_stability_summary(report, min_frequency=0.51)
    )["lst"]["stable"] is False
    assert _items_by_factor(
        driver_attr_stability_summary(report, min_consistency=0.76)
    )["lst"]["stable"] is False
    assert _items_by_factor(
        driver_attr_stability_summary(report, min_significant=0.51)
    )["lst"]["stable"] is False


def test_thresholds_compare_on_decimal_not_float():
    # frequency = 1/3 serialized as 0.333333; threshold 0.333333 must be
    # met by the parsed six-decimal token even though 1/3 > 0.333333.
    row = (1, 1 / 3, "up", 1.0, 0.1, 0.1, 1.0)
    report = _stability_report(_region_group(*_four_items(lst=row)))
    item = _items_by_factor(
        driver_attr_stability_summary(report, min_frequency=0.333333)
    )["lst"]
    assert item["stable"] is True
    item = _items_by_factor(
        driver_attr_stability_summary(report, min_frequency=0.333334)
    )["lst"]
    assert item["stable"] is False


def test_zero_and_one_thresholds_accepted():
    report = _stability_report(_region_group(*_four_items()))
    out = driver_attr_stability_summary(
        report, min_frequency=0, min_consistency=0, min_significant=0
    )
    assert out.endswith("\n")
    # With every threshold 0 a unanimous non-flat, non-crossing n>0 item is
    # stable; flat n>0 zero-interval items still cross zero.
    out = driver_attr_stability_summary(
        _stability_report(_region_group(*_four_items(cover=(
            2, 1.0, "up", 0.5, 0.1, 0.2, 0.0
        )))),
        min_frequency=0, min_consistency=0, min_significant=0,
    )
    assert _items_by_factor(out)["cover"]["stable"] is True


def test_six_decimals_and_no_negative_zero():
    out = driver_attr_stability_summary(
        _stability_report(_region_group(*_four_items()))
    )
    assert "-0.000000" not in out
    assert '"frequency":0.000000' in out
    assert '"stable":false' in out and '"conflict":false' in out
    assert '"cross_zero":false' in out


def test_accepts_genuine_stability_report():
    raw = driver_attr_stability(
        {
            "r1": _da_item("region", "station", -1.0, True, "down"),
            "r2": _da_item("region", "station", -0.5, True, "down"),
        }
    )
    out = driver_attr_stability_summary(raw)
    item = _items_by_factor(out)["station"]
    assert item["direction"] == "down"
    assert item["stable"] is True
    assert item["frequency"] == 1.0


# --- validation -------------------------------------------------------------

def test_non_string_report_type_error():
    for bad in (1, None, [], {}, True):
        with pytest.raises(TypeError):
            driver_attr_stability_summary(bad)


def test_thresholds_are_keyword_only():
    report = _stability_report(_region_group(*_four_items()))
    with pytest.raises(TypeError):
        driver_attr_stability_summary(report, 0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_frequency": -0.01},
        {"min_frequency": 1.01},
        {"min_consistency": -1},
        {"min_consistency": 2},
        {"min_significant": -1.0},
        {"min_significant": 1.0001},
    ],
)
def test_out_of_range_thresholds_value_error(kwargs):
    report = _stability_report(_region_group(*_four_items()))
    with pytest.raises(ValueError):
        driver_attr_stability_summary(report, **kwargs)


@pytest.mark.parametrize(
    "value",
    [True, False, "0.5", None, 0j, float("nan"), float("inf"),
     float("-inf"), [0.5], {"x": 0.5}],
)
def test_bad_threshold_types_value_error(value):
    report = _stability_report(_region_group(*_four_items()))
    for name in ("min_frequency", "min_consistency", "min_significant"):
        with pytest.raises(ValueError):
            driver_attr_stability_summary(report, **{name: value})


def test_threshold_validation_precedes_report_parsing():
    # A malformed report alongside a bad threshold still reports the
    # threshold ValueError (thresholds are checked while reading args).
    with pytest.raises(ValueError):
        driver_attr_stability_summary("not a report\n", min_frequency=2)


def test_malformed_report_value_error():
    good = _stability_report(_region_group(*_four_items()))
    for bad in (
        "",
        "not json\n",
        good.rstrip("\n"),
        good + "\n",
        '{"alpha":0.05,"groups":[]}\n',
        '{"groups":[],"alpha":0.050000}\n',
        '{"alpha":0.050000}\n',
    ):
        with pytest.raises(ValueError):
            driver_attr_stability_summary(bad)


def test_reordered_item_keys_rejected():
    token = (
        '{"factor":"station","n":0,"frequency":0.000000,'
        '"direction":"flat","consistency":0.000000,"low":0.000000,'
        '"high":0.000000,"significant":0.000000}'
    )
    swapped = token.replace(
        '"frequency":0.000000,"direction":"flat"',
        '"direction":"flat","frequency":0.000000',
    )
    assert swapped != token
    bad = _stability_report(_region_group(swapped, *_four_items()[1:]))
    with pytest.raises(ValueError):
        driver_attr_stability_summary(bad)


def test_wrong_factor_set_rejected():
    # Three items instead of the canonical four.
    tokens = _four_items()[:3]
    report = _stability_report(_region_group(*tokens))
    with pytest.raises(ValueError):
        driver_attr_stability_summary(report)


def test_duplicate_factor_rejected():
    tokens = _four_items()
    tokens[1] = tokens[0]  # lst slot repeats station
    report = _stability_report(_region_group(*tokens))
    with pytest.raises(ValueError):
        driver_attr_stability_summary(report)


def test_negative_zero_token_rejected():
    token = (
        '{"factor":"station","n":1,"frequency":-0.000000,'
        '"direction":"flat","consistency":0.000000,"low":0.000000,'
        '"high":0.000000,"significant":0.000000}'
    )
    report = _stability_report(_region_group(token, *_four_items()[1:]))
    with pytest.raises(ValueError):
        driver_attr_stability_summary(report)


def test_driver_attr_report_shape_rejected():
    # A driver_attr (alpha, items) report is not a stability report.
    da = _da_item("region", "station", -1.0, False, "down")
    with pytest.raises(ValueError):
        driver_attr_stability_summary(da)


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert (
        _uhi.driver_attr_stability_summary
        is urban_micro.driver_attr_stability_summary
    )
    assert "driver_attr_stability_summary" in urban_micro.__all__
