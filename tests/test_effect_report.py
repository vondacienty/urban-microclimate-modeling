"""Tests for urban_micro.effect_report."""

import json

import pytest

from urban_micro import effect_report


def test_empty_details_gives_empty_groups():
    report = effect_report([], by="time")
    assert report == '{"by":"time","minutes":60,"z":1.960000,"groups":[]}'
    assert json.loads(report) == {"by": "time", "minutes": 60, "z": 1.96, "groups": []}


def test_time_buckets_and_interval():
    details = [
        (0, "a", 1.0, 2.0, 1.0, 0.5, 0.25, 0.25),
        (3599, "b", 3.0, 6.0, 3.0, 1.5, 0.75, 0.75),
        (3600, "a", 0.0, -1.0, -1.0, 0.0, -1.0, 0.0),
    ]
    report = effect_report(details, by="time", minutes=60)
    assert report == (
        '{"by":"time","minutes":60,"z":1.960000,"groups":['
        '{"key":0,"n":2,"base":2.000000,"post":4.000000,"delta":2.000000,'
        '"cg":1.000000,"cr":0.500000,"cm":0.500000,"p":0.500000,'
        '"se":1.000000,"lower":0.040000,"upper":3.960000},'
        '{"key":3600,"n":1,"base":0.000000,"post":-1.000000,"delta":-1.000000,'
        '"cg":0.000000,"cr":-1.000000,"cm":0.000000,"p":1.000000,'
        '"se":0.000000,"lower":-1.000000,"upper":-1.000000}]}'
    )


def test_cell_grouping_sorted_string_keys():
    details = [
        (0, "b", 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        (1, "a", 0.0, 2.0, 2.0, 0.0, 0.0, 0.0),
        (2, "b", 0.0, 3.0, 3.0, 0.0, 0.0, 0.0),
    ]
    report = effect_report(details, by="cell", z=0.0)
    assert report == (
        '{"by":"cell","minutes":60,"z":0.000000,"groups":['
        '{"key":"a","n":1,"base":0.000000,"post":2.000000,"delta":2.000000,'
        '"cg":0.000000,"cr":0.000000,"cm":0.000000,"p":1.000000,'
        '"se":0.000000,"lower":2.000000,"upper":2.000000},'
        '{"key":"b","n":2,"base":0.000000,"post":2.000000,"delta":2.000000,'
        '"cg":0.000000,"cr":0.000000,"cm":0.000000,"p":0.500000,'
        '"se":1.000000,"lower":2.000000,"upper":2.000000}]}'
    )


def test_se_uses_sample_denominator():
    # deltas 1, 2, 3 -> mu = 2, ss = 2, se = sqrt(2 / (3 * 2)) = sqrt(1/3)
    details = [
        (0, "a", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (1, "b", 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        (2, "c", 0.0, 0.0, 3.0, 0.0, 0.0, 0.0),
    ]
    report = effect_report(details, by="time", z=1.0)
    group = json.loads(report)["groups"][0]
    assert group["se"] == pytest.approx((1 / 3) ** 0.5, abs=1e-6)
    assert group["lower"] == pytest.approx(2 - (1 / 3) ** 0.5, abs=1e-6)
    assert group["upper"] == pytest.approx(2 + (1 / 3) ** 0.5, abs=1e-6)


def test_negative_zero_normalized():
    report = effect_report(
        [(0, "a", -0.0, -0.0, -0.0, -0.0, -0.0, -0.0)], by="time"
    )
    assert "-0.000000" not in report
    assert report == (
        '{"by":"time","minutes":60,"z":1.960000,"groups":['
        '{"key":0,"n":1,"base":0.000000,"post":0.000000,"delta":0.000000,'
        '"cg":0.000000,"cr":0.000000,"cm":0.000000,"p":1.000000,'
        '"se":0.000000,"lower":0.000000,"upper":0.000000}]}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_report(
        [(0, "céll", 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)], by="cell"
    )
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    report = effect_report(
        [(0, "a", 1.0, 2.0, 1.0, 0.0, 0.0, 0.0)], by="time", minutes=30, z=2.5
    )
    assert list(json.loads(report)) == ["by", "minutes", "z", "groups"]
    assert list(json.loads(report)["groups"][0]) == [
        "key", "n", "base", "post", "delta", "cg", "cr", "cm",
        "p", "se", "lower", "upper",
    ]
    assert '"minutes":30' in report
    assert '"z":2.500000' in report


def test_minutes_bucket_floor():
    details = [
        (0, "a", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (1799, "b", 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        (1800, "c", 0.0, 0.0, 4.0, 0.0, 0.0, 0.0),
    ]
    report = effect_report(details, by="time", minutes=30)
    assert [g["key"] for g in json.loads(report)["groups"]] == [0, 1800]


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_report("not a list", by="time")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"by": "row"},
        {"by": "time", "minutes": 0},
        {"by": "time", "minutes": 7},
        {"by": "time", "minutes": True},
        {"by": "time", "z": -1.0},
        {"by": "time", "z": True},
        {"by": "time", "z": float("nan")},
        {"by": "time", "z": "1.96"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_report([], **kwargs)


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
        effect_report([row], by="time")


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        effect_report([row, row], by="cell")
