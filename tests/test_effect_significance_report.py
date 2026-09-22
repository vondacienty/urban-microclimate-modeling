"""Tests for urban_micro.effect_significance_report."""

import json
import math

import pytest

from urban_micro import effect_significance_report


def test_empty_details_gives_empty_groups():
    report = effect_significance_report([], by="time")
    assert report == '{"by":"time","minutes":60,"alpha":0.050000,"groups":[]}'
    assert json.loads(report) == {
        "by": "time",
        "minutes": 60,
        "alpha": 0.05,
        "groups": [],
    }


def test_time_buckets_z_and_significance():
    # bucket 0: deltas 1, 3 -> mu = 2, se = 1, z = 2,
    # p = erfc(2 / sqrt(2)) = 0.04550026389635842...
    details = [
        (0, "a", 1.0, 2.0, 1.0, 0.5, 0.25, 0.25),
        (3599, "b", 3.0, 6.0, 3.0, 1.5, 0.75, 0.75),
        (3600, "a", 0.0, -1.0, -1.0, 0.0, -1.0, 0.0),
    ]
    report = effect_significance_report(details, by="time", minutes=60)
    assert report == (
        '{"by":"time","minutes":60,"alpha":0.050000,"groups":['
        '{"key":0,"n":2,"delta":2.000000,"se":1.000000,"z":2.000000,'
        '"p":0.045500,"significant":true},'
        '{"key":3600,"n":1,"delta":-1.000000,"se":0.000000,"z":0.000000,'
        '"p":1.000000,"significant":false}]}'
    )


def test_cell_grouping_sorted_string_keys():
    details = [
        (0, "b", 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        (1, "a", 0.0, 2.0, 2.0, 0.0, 0.0, 0.0),
        (2, "b", 0.0, 3.0, 3.0, 0.0, 0.0, 0.0),
    ]
    report = effect_significance_report(details, by="cell", alpha=1.0)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == ["a", "b"]
    # cell "a": single row -> se = 0, z = 0, p = 1, significant at alpha = 1
    assert groups[0]["se"] == 0.0
    assert groups[0]["z"] == 0.0
    assert groups[0]["p"] == 1.0
    assert groups[0]["significant"] is True
    # cell "b": deltas 1, 3 -> mu = 2, se = 1, z = 2
    assert groups[1]["delta"] == 2.0
    assert groups[1]["se"] == 1.0
    assert groups[1]["z"] == 2.0
    assert groups[1]["p"] == pytest.approx(math.erfc(2 / math.sqrt(2)), abs=1e-6)


def test_se_uses_sample_denominator():
    # deltas 1, 2, 3 -> mu = 2, ss = 2, se = sqrt(2 / (3 * 2)) = sqrt(1/3)
    details = [
        (0, "a", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (1, "b", 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        (2, "c", 0.0, 0.0, 3.0, 0.0, 0.0, 0.0),
    ]
    report = effect_significance_report(details, by="time")
    group = json.loads(report)["groups"][0]
    se = (1 / 3) ** 0.5
    assert group["se"] == pytest.approx(se, abs=1e-6)
    assert group["z"] == pytest.approx(2 / se, abs=1e-6)
    assert group["p"] == pytest.approx(math.erfc((2 / se) / math.sqrt(2)), abs=1e-6)


def test_negative_zero_normalized():
    report = effect_significance_report(
        [(0, "a", -0.0, -0.0, -0.0, -0.0, -0.0, -0.0)], by="time"
    )
    assert "-0.000000" not in report
    assert report == (
        '{"by":"time","minutes":60,"alpha":0.050000,"groups":['
        '{"key":0,"n":1,"delta":0.000000,"se":0.000000,"z":0.000000,'
        '"p":1.000000,"significant":false}]}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_significance_report(
        [(0, "céll", 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)], by="cell"
    )
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    report = effect_significance_report(
        [(0, "a", 1.0, 2.0, 1.0, 0.0, 0.0, 0.0)], by="time", minutes=30, alpha=0.1
    )
    assert list(json.loads(report)) == ["by", "minutes", "alpha", "groups"]
    assert list(json.loads(report)["groups"][0]) == [
        "key", "n", "delta", "se", "z", "p", "significant",
    ]
    assert '"minutes":30' in report
    assert '"alpha":0.100000' in report


def test_minutes_bucket_floor():
    details = [
        (0, "a", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (1799, "b", 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        (1800, "c", 0.0, 0.0, 4.0, 0.0, 0.0, 0.0),
    ]
    report = effect_significance_report(details, by="time", minutes=30)
    assert [g["key"] for g in json.loads(report)["groups"]] == [0, 1800]


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_significance_report("not a list", by="time")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"by": "row"},
        {"by": "time", "minutes": 0},
        {"by": "time", "minutes": 7},
        {"by": "time", "minutes": True},
        {"by": "time", "alpha": 0.0},
        {"by": "time", "alpha": 1.5},
        {"by": "time", "alpha": True},
        {"by": "time", "alpha": float("nan")},
        {"by": "time", "alpha": "0.05"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_significance_report([], **kwargs)


@pytest.mark.parametrize(
    "row",
    [
        [0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # not a tuple
        (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0),  # wrong length
        (True, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # bool timestamp
        (-1, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # negative timestamp
        (0, "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # empty cell id
        (0, "a", 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0),  # non-finite
        (0, "a", 0.0, 0.0, True, 0.0, 0.0, 0.0),  # bool value
    ],
)
def test_invalid_rows_raise_value_error(row):
    with pytest.raises(ValueError):
        effect_significance_report([row], by="time")


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        effect_significance_report([row, row], by="cell")
