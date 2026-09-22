"""Tests for urban_micro.effect_bootstrap_report."""

import json

import pytest

from urban_micro import effect_bootstrap_report


def test_empty_details_gives_empty_groups():
    report = effect_bootstrap_report([])
    assert report == '{"confidence":0.950000,"groups":[]}'
    assert json.loads(report) == {"confidence": 0.95, "groups": []}


def test_single_row_group():
    report = effect_bootstrap_report(
        [(5, "a", 1.0, 2.0, 3.25, 0.0, 0.0, 0.0)]
    )
    assert report == (
        '{"confidence":0.950000,"groups":['
        '{"key":"a","n":1,"delta":3.250000,"lower":3.250000,"upper":3.250000}]}'
    )


def test_groups_by_cell_only_and_sorted_ascending():
    # timestamps never participate in the grouping
    details = [
        (9, "b", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (0, "a", 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        (4, "b", 0.0, 0.0, 3.0, 0.0, 0.0, 0.0),
    ]
    report = effect_bootstrap_report(details)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == ["a", "b"]
    assert groups[0]["n"] == 1
    assert groups[1]["n"] == 2
    # deltas 1, 3 -> mean 2; sorted bootstrap means [1, 2, 2, 3],
    # r = 0.075 -> lower 1.075, upper 2.925
    assert groups[1] == {
        "key": "b",
        "n": 2,
        "delta": 2.0,
        "lower": 1.075,
        "upper": 2.925,
    }


def test_n3_interpolation_and_delta_mean():
    details = [
        (0, "a", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (1, "a", 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        (2, "a", 0.0, 0.0, 3.0, 0.0, 0.0, 0.0),
    ]
    report = effect_bootstrap_report(details)
    group = json.loads(report)["groups"][0]
    # r = 26 * 0.025 = 0.65: 1 + 0.65 * (4/3 - 1) = 1.216666...
    assert group["delta"] == 2.0
    assert report.split('"lower":')[1].split(",")[0] == "1.216667"
    assert report.split('"upper":')[1].split("}")[0] == "2.783333"


def test_integral_position_takes_that_value():
    # n = 3: 27 resamples; confidence 11/13 gives q = 1/13 and r = 26/13 = 2
    details = [
        (0, "a", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (1, "a", 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        (2, "a", 0.0, 0.0, 4.0, 0.0, 0.0, 0.0),
    ]
    report = effect_bootstrap_report(details, confidence=11 / 13)
    group = json.loads(report)["groups"][0]
    # sorted means: positions 0 -> 1, 1..3 -> 4/3, 4..9 -> 5/3 ...
    assert group["lower"] == pytest.approx(4 / 3, abs=1e-6)
    assert group["upper"] == pytest.approx(10 / 3, abs=1e-6)


def test_duplicate_deltas_do_not_break_multiplicity():
    details = [
        (0, "a", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (1, "a", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
        (2, "a", 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
    ]
    report = effect_bootstrap_report(details)
    group = json.loads(report)["groups"][0]
    assert group["delta"] == pytest.approx(4 / 3, abs=1e-6)
    assert group["lower"] <= group["upper"]


def test_negative_zero_normalized():
    report = effect_bootstrap_report(
        [(0, "a", -0.0, -0.0, -0.0, -0.0, -0.0, -0.0)]
    )
    assert "-0.000000" not in report
    assert report == (
        '{"confidence":0.950000,"groups":['
        '{"key":"a","n":1,"delta":0.000000,"lower":0.000000,"upper":0.000000}]}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_bootstrap_report(
        [(0, "céll", 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)]
    )
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    report = effect_bootstrap_report(
        [(0, "a", 1.0, 2.0, 1.0, 0.0, 0.0, 0.0)], confidence=0.1
    )
    assert list(json.loads(report)) == ["confidence", "groups"]
    assert list(json.loads(report)["groups"][0]) == [
        "key", "n", "delta", "lower", "upper",
    ]
    assert '"confidence":0.100000' in report


def test_max_group_size_is_eight():
    details = [
        (i, "a", 0.0, 0.0, float(i), 0.0, 0.0, 0.0) for i in range(9)
    ]
    with pytest.raises(ValueError):
        effect_bootstrap_report(details)


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_bootstrap_report("not a list")


@pytest.mark.parametrize(
    "confidence",
    [0.0, 1.0, -0.1, 1.5, True, False, float("nan"), float("inf"), "0.95"],
)
def test_invalid_confidence_raises_value_error(confidence):
    with pytest.raises(ValueError):
        effect_bootstrap_report([], confidence=confidence)


@pytest.mark.parametrize(
    "row",
    [
        [0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # not a tuple
        (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0),  # wrong length
        (True, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # bool timestamp
        (-1, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # negative timestamp
        (0, "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # empty cell id
        (0, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # non-string cell id
        (0, "a", True, 0.0, 0.0, 0.0, 0.0, 0.0),  # bool numeric field
        (0, "a", 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0),  # non-finite
        (0, "a", 0.0, 0.0, float("nan"), 0.0, 0.0, 0.0),
        (0, "a", "x", 0.0, 0.0, 0.0, 0.0, 0.0),  # non-number
    ],
)
def test_invalid_rows_raise_value_error(row):
    with pytest.raises(ValueError):
        effect_bootstrap_report([row])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        effect_bootstrap_report([row, row])


def test_same_cell_different_timestamps_share_group():
    report = effect_bootstrap_report(
        [
            (0, "a", 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            (7, "a", 0.0, 0.0, 3.0, 0.0, 0.0, 0.0),
        ]
    )
    assert json.loads(report)["groups"] == [
        {"key": "a", "n": 2, "delta": 2.0, "lower": 1.075, "upper": 2.925}
    ]
