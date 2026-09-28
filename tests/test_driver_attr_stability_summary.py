"""Tests for urban_micro.driver_attr_stability_summary: thresholded
stability and interval flags re-rendered from one canonical
driver_attr_stability report."""

import json
from decimal import Decimal

import pytest

from urban_micro import driver_attr_stability, driver_attr_stability_summary
from urban_micro import uhi as _uhi

_FACTORS = ("station", "lst", "morph", "cover")


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


def _stab_item_token(factor, n, frequency, direction, consistency,
                     low, high, significant):
    """One canonical driver_attr_stability item object token."""
    return (
        f'{{"factor":{json.dumps(factor)},"n":{n},"frequency":{frequency:.6f},'
        f'"direction":{json.dumps(direction)},"consistency":{consistency:.6f},'
        f'"low":{low:.6f},"high":{high:.6f},'
        f'"significant":{significant:.6f}}}'
    )


def _zero_item_tokens():
    return [
        _stab_item_token(f, 0, 0.0, "flat", 0.0, 0.0, 0.0, 0.0)
        for f in _FACTORS
    ]


def _stab_group(by, *item_tokens):
    return (
        '{"by":' + json.dumps(by)
        + ',"items":[' + ",".join(item_tokens) + "]}"
    )


def _stab_report(*group_tokens, alpha=0.05):
    return (
        f'{{"alpha":{alpha:.6f},"groups":['
        + ",".join(group_tokens) + "]}\n"
    )


def _region_report(*item_tokens, alpha=0.05):
    return _stab_report(_stab_group("region", *item_tokens), alpha=alpha)


def _summary_items(out):
    groups = json.loads(out)["groups"]
    return {
        group["by"]: {item["factor"]: item for item in group["items"]}
        for group in groups
    }


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    report = _region_report(
        _stab_item_token("station", 2, 1.0, "down", 1.0, -1.0, -0.5, 0.0),
        *_zero_item_tokens()[1:],
    )
    out = driver_attr_stability_summary(report)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    group = payload["groups"][0]
    assert list(group) == ["by", "items"]
    assert [item["factor"] for item in group["items"]] == list(_FACTORS)
    for item in group["items"]:
        assert list(item) == [
            "factor", "direction", "frequency", "consistency",
            "significant", "low", "high", "stable", "conflict",
            "cross_zero",
        ]
        assert "n" not in item
    assert '"n"' not in out


def test_groups_region_before_window_each_with_four_items():
    station = _stab_item_token(
        "station", 2, 1.0, "down", 1.0, -1.0, -0.5, 1.0
    )
    lst = _stab_item_token("lst", 2, 1.0, "up", 1.0, 0.5, 1.0, 1.0)
    report = _stab_report(
        _stab_group("region", station, *_zero_item_tokens()[1:]),
        _stab_group("window", *_zero_item_tokens()[:1], lst,
                    *_zero_item_tokens()[2:]),
    )
    out = driver_attr_stability_summary(report)
    groups = json.loads(out)["groups"]
    assert [group["by"] for group in groups] == ["region", "window"]
    for group in groups:
        assert [item["factor"] for item in group["items"]] == list(_FACTORS)


def test_zero_pick_item_all_flags_false():
    # n == 0: cross_zero would hold on 0 <= 0 <= 0 and consistency < 1
    # would hold, but both flags require n > 0.
    report = _region_report(*_zero_item_tokens())
    items = _summary_items(driver_attr_stability_summary(report))["region"]
    for factor in _FACTORS:
        item = items[factor]
        assert item["direction"] == "flat"
        assert item["stable"] is False
        assert item["conflict"] is False
        assert item["cross_zero"] is False


def test_conflict_requires_mixed_directions():
    mixed = _region_report(
        _stab_item_token("station", 4, 1.0, "up", 0.75, -0.3, 0.4, 0.5),
        *_zero_item_tokens()[1:],
    )
    item = _summary_items(driver_attr_stability_summary(mixed))["region"][
        "station"
    ]
    assert item["conflict"] is True

    unanimous = _region_report(
        _stab_item_token("station", 2, 1.0, "up", 1.0, 0.1, 0.4, 0.5),
        *_zero_item_tokens()[1:],
    )
    item = _summary_items(driver_attr_stability_summary(unanimous))[
        "region"
    ]["station"]
    assert item["conflict"] is False


def test_cross_zero_boundaries():
    def flag(low, high):
        report = _region_report(
            _stab_item_token("station", 1, 0.5, "up", 1.0, low, high, 0.0),
            *_zero_item_tokens()[1:],
        )
        return _summary_items(driver_attr_stability_summary(report))[
            "region"
        ]["station"]["cross_zero"]

    assert flag(-0.5, 0.5) is True
    assert flag(0.0, 0.0) is True
    assert flag(0.0, 1.0) is True
    assert flag(-1.0, 0.0) is True
    assert flag(0.1, 1.0) is False
    assert flag(-1.0, -0.1) is False


def test_stable_passes_at_default_threshold_boundaries():
    # frequency 0.5 and significant 0.5 sit on their thresholds; the
    # unanimous pick keeps consistency 1 so conflict stays false.
    report = _region_report(
        _stab_item_token("station", 2, 0.5, "up", 1.0, 0.1, 0.2, 0.5),
        *_zero_item_tokens()[1:],
    )
    item = _summary_items(driver_attr_stability_summary(report))["region"][
        "station"
    ]
    assert item["stable"] is True
    assert item["conflict"] is False
    assert item["cross_zero"] is False

    # consistency exactly at 0.75 still passes its threshold even though
    # it simultaneously flags conflict (0.75 < 1).
    report = _region_report(
        _stab_item_token("station", 4, 0.5, "up", 0.75, 0.1, 0.2, 0.5),
        *_zero_item_tokens()[1:],
    )
    item = _summary_items(driver_attr_stability_summary(report))["region"][
        "station"
    ]
    assert item["stable"] is True
    assert item["conflict"] is True


def test_stable_fails_just_below_default_thresholds():
    cases = (
        (0.499999, 0.75, 0.5),
        (0.5, 0.749999, 0.5),
        (0.5, 0.75, 0.499999),
    )
    for frequency, consistency, significant in cases:
        report = _region_report(
            _stab_item_token(
                "station", 2, frequency, "up", consistency,
                0.1, 0.2, significant
            ),
            *_zero_item_tokens()[1:],
        )
        item = _summary_items(driver_attr_stability_summary(report))[
            "region"
        ]["station"]
        assert item["stable"] is False


def test_stable_requires_non_flat_direction():
    report = _region_report(
        _stab_item_token("station", 2, 1.0, "flat", 1.0, 0.1, 0.2, 1.0),
        *_zero_item_tokens()[1:],
    )
    item = _summary_items(driver_attr_stability_summary(report))["region"][
        "station"
    ]
    assert item["stable"] is False
    assert item["cross_zero"] is False


def test_stable_requires_no_cross_zero():
    report = _region_report(
        _stab_item_token("station", 2, 1.0, "up", 1.0, -0.1, 0.2, 1.0),
        *_zero_item_tokens()[1:],
    )
    item = _summary_items(driver_attr_stability_summary(report))["region"][
        "station"
    ]
    assert item["stable"] is False
    assert item["cross_zero"] is True
    assert item["conflict"] is False


def test_custom_thresholds_gate_stable():
    def with_thresholds(**thresholds):
        report = _region_report(
            _stab_item_token(
                "station", 2, 0.8, "down", 0.8, -0.4, -0.1, 0.2
            ),
            *_zero_item_tokens()[1:],
        )
        return _summary_items(
            driver_attr_stability_summary(report, **thresholds)
        )["region"]["station"]["stable"]

    assert with_thresholds(
        min_frequency=0.8, min_consistency=0.8, min_significant=0.2
    ) is True
    assert with_thresholds(min_frequency=0.81) is False
    assert with_thresholds(min_consistency=0.81) is False
    assert with_thresholds(min_significant=0.21) is False
    # Zero thresholds are inclusive.
    report = _region_report(
        _stab_item_token("station", 2, 0.0, "up", 0.0, 0.1, 0.2, 0.0),
        *_zero_item_tokens()[1:],
    )
    item = _summary_items(
        driver_attr_stability_summary(
            report, min_frequency=0, min_consistency=0, min_significant=0
        )
    )["region"]["station"]
    assert item["stable"] is True
    # Integer thresholds are accepted and compared as Decimal(str).
    item = _summary_items(
        driver_attr_stability_summary(
            report, min_frequency=1, min_consistency=1, min_significant=1
        )
    )["region"]["station"]
    assert item["stable"] is False


def test_six_decimal_tokens_no_negative_zero():
    report = _region_report(
        _stab_item_token("station", 2, 2 / 3, "down", 1.0, -1.0, -0.5, 0.5),
        _stab_item_token("lst", 1, 1 / 3, "up", 1.0, 1.0, 1.0, 0.0),
        *_zero_item_tokens()[2:],
    )
    out = driver_attr_stability_summary(report)
    assert '"frequency":0.666667' in out
    assert '"frequency":0.333333' in out
    assert '"consistency":1.000000' in out
    assert '"low":0.000000,"high":0.000000' in out
    assert "-0.000000" not in out


def test_end_to_end_from_genuine_stability_report():
    stability = driver_attr_stability(
        {
            "r1": _da_item("region", "station", -1.0, False, "down"),
            "r2": _da_item("region", "station", -0.5, False, "down"),
            "r3": _da_item("region", "lst", 1.0, False, "up"),
        }
    )
    items = _summary_items(driver_attr_stability_summary(stability))[
        "region"
    ]
    station = items["station"]
    assert station["frequency"] == pytest.approx(2 / 3, abs=5.1e-7)
    assert station["direction"] == "down"
    assert station["consistency"] == 1.0
    assert station["significant"] == 0.0
    assert station["low"] == -1.0
    assert station["high"] == -0.5
    # significant 0 fails the default 0.5 threshold.
    assert station["stable"] is False
    assert station["conflict"] is False
    assert station["cross_zero"] is False
    assert items["lst"]["stable"] is False
    # Lowering min_significant makes the unanimous station pick stable.
    items = _summary_items(
        driver_attr_stability_summary(stability, min_significant=0.0)
    )["region"]
    assert items["station"]["stable"] is True
    assert items["lst"]["stable"] is False


# --- validation -------------------------------------------------------------

def test_non_str_report_type_error():
    for bad in (1, None, [], {}, b"x\n"):
        with pytest.raises(TypeError):
            driver_attr_stability_summary(bad)


def test_thresholds_must_be_unit_interval_numbers():
    report = _region_report(*_zero_item_tokens())

    class FloatSubclass(float):
        pass

    for bad in (
        True,
        False,
        "0.5",
        None,
        Decimal("0.5"),
        0.5 + 0j,
        float("nan"),
        float("inf"),
        -float("inf"),
        -0.1,
        1.1,
        2,
        -1,
    ):
        with pytest.raises(ValueError):
            driver_attr_stability_summary(report, min_frequency=bad)
        with pytest.raises(ValueError):
            driver_attr_stability_summary(report, min_consistency=bad)
        with pytest.raises(ValueError):
            driver_attr_stability_summary(report, min_significant=bad)
    # Endpoints (int and float) are accepted; a float subclass is a float.
    driver_attr_stability_summary(
        report,
        min_frequency=0,
        min_consistency=FloatSubclass(1.0),
        min_significant=0.0,
    )


def test_thresholds_are_keyword_only():
    report = _region_report(*_zero_item_tokens())
    with pytest.raises(TypeError):
        driver_attr_stability_summary(report, 0.5, 0.75, 0.5)


def test_report_must_be_canonical_stability_output():
    good = _region_report(
        _stab_item_token("station", 2, 1.0, "down", 1.0, -1.0, -0.5, 0.0),
        *_zero_item_tokens()[1:],
    )
    # A driver_attr report has the wrong shape.
    with pytest.raises(ValueError):
        driver_attr_stability_summary(
            _da_item("region", "station", -1.0, False, "down")
        )
    for raw in ("", "not json\n", good.rstrip("\n"), good + "\n"):
        with pytest.raises(ValueError):
            driver_attr_stability_summary(raw)
    # The summary output is not itself a stability report.
    with pytest.raises(ValueError):
        driver_attr_stability_summary(
            driver_attr_stability_summary(good)
        )


def test_canonical_byte_for_byte_required():
    good = _region_report(
        _stab_item_token("station", 2, 1.0, "down", 1.0, -1.0, -0.5, 0.0),
        *_zero_item_tokens()[1:],
    )
    # Reordered item keys.
    reordered = good.replace(
        '"n":2,"frequency":1.000000',
        '"frequency":1.000000,"n":2',
    )
    assert reordered != good
    with pytest.raises(ValueError):
        driver_attr_stability_summary(reordered)
    # Negative zero never serializes.
    neg_zero = good.replace('"low":-1.000000', '"low":-0.000000')
    with pytest.raises(ValueError):
        driver_attr_stability_summary(neg_zero)
    # Numbers must carry six decimals.
    short = good.replace('"frequency":1.000000', '"frequency":1')
    with pytest.raises(ValueError):
        driver_attr_stability_summary(short)
    # Unknown direction / factor.
    bad_direction = good.replace('"direction":"down"', '"direction":"side"')
    with pytest.raises(ValueError):
        driver_attr_stability_summary(bad_direction)
    bad_factor = good.replace('"factor":"station"', '"factor":"wind"')
    with pytest.raises(ValueError):
        driver_attr_stability_summary(bad_factor)
    # Negative n.
    bad_n = _region_report(
        _stab_item_token("station", -1, 0.0, "flat", 0.0, 0.0, 0.0, 0.0),
        *_zero_item_tokens()[1:],
    )
    assert '"n":-1' in bad_n
    with pytest.raises(ValueError):
        driver_attr_stability_summary(bad_n)


def test_alpha_range_enforced():
    report = _region_report(*_zero_item_tokens(), alpha=0.0)
    with pytest.raises(ValueError):
        driver_attr_stability_summary(report)
    report = _region_report(*_zero_item_tokens(), alpha=1.000001)
    with pytest.raises(ValueError):
        driver_attr_stability_summary(report)


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert (
        _uhi.driver_attr_stability_summary
        is urban_micro.driver_attr_stability_summary
    )
    assert "driver_attr_stability_summary" in urban_micro.__all__
