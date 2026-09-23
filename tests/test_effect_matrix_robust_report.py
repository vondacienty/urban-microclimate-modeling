"""Tests for urban_micro.effect_matrix_robust_report."""

import json
import math

import pytest

from urban_micro import effect_matrix_robust_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def test_empty_details_gives_empty_groups():
    report = effect_matrix_robust_report([])
    assert report == '{"minutes":60,"z":1.960000,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "z": 1.96, "groups": []}


def test_odd_count_median_mad_and_se():
    # Deltas 1..5: median 3, absolute deviations sorted [0,1,1,2,2],
    # MAD 1, se = 1.4826 / sqrt(5).
    details = [_row(t, "a", float(t + 1)) for t in range(5)]
    cell = json.loads(effect_matrix_robust_report(details))["groups"][0]["cells"][0]
    se = 1.4826 / math.sqrt(5)
    assert cell["n"] == 5
    assert cell["median"] == 3.0
    assert cell["mad"] == 1.0
    assert cell["se"] == pytest.approx(se, abs=1e-6)
    assert cell["lower"] == pytest.approx(3 - 1.96 * se, abs=1e-6)
    assert cell["upper"] == pytest.approx(3 + 1.96 * se, abs=1e-6)


def test_even_count_interpolated_median_and_mad():
    # Deltas [1, 2, 3, 10]: median 2.5; deviations sorted
    # [0.5, 0.5, 1.5, 7.5], MAD = (0.5 + 1.5) / 2 = 1.0.
    details = [_row(t, "a", d) for t, d in enumerate((1.0, 2.0, 3.0, 10.0))]
    cell = json.loads(effect_matrix_robust_report(details))["groups"][0]["cells"][0]
    assert cell["median"] == 2.5
    assert cell["mad"] == 1.0
    assert cell["se"] == pytest.approx(1.4826 / 2, abs=1e-6)


def test_single_row_has_zero_mad_se():
    report = effect_matrix_robust_report([_row(0, "a", 7.0)])
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell == {
        "key": "a",
        "n": 1,
        "median": 7.0,
        "mad": 0.0,
        "se": 0.0,
        "lower": 7.0,
        "upper": 7.0,
    }


def test_buckets_floor_and_cells_sorted_within_bucket():
    details = [
        _row(0, "b", 1.0),
        _row(1799, "a", 1.0),
        _row(1800, "a", 1.0),
        _row(1800, "b", 1.0),
    ]
    report = effect_matrix_robust_report(details, minutes=30)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [0, 1800]
    assert [cell["key"] for cell in groups[0]["cells"]] == ["a", "b"]
    assert [cell["n"] for cell in groups[0]["cells"]] == [1, 1]
    assert [cell["key"] for cell in groups[1]["cells"]] == ["a", "b"]


def test_same_cell_rows_aggregate_within_bucket():
    details = [_row(0, "a", 1.0), _row(5, "a", 5.0)]
    cell = json.loads(effect_matrix_robust_report(details))["groups"][0]["cells"][0]
    assert cell["n"] == 2
    assert cell["median"] == 3.0
    assert cell["mad"] == 2.0


def test_negative_zero_normalized():
    report = effect_matrix_robust_report([_row(0, "a", -0.0)])
    assert "-0.000000" not in report
    assert report == (
        '{"minutes":60,"z":1.960000,"groups":['
        '{"key":0,"cells":['
        '{"key":"a","n":1,"median":0.000000,"mad":0.000000,"se":0.000000,'
        '"lower":0.000000,"upper":0.000000}]}]}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_matrix_robust_report([_row(0, "céll", 1.0)])
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    report = effect_matrix_robust_report([_row(0, "a", 1.0)], minutes=30, z=2.0)
    assert list(json.loads(report)) == ["minutes", "z", "groups"]
    assert list(json.loads(report)["groups"][0]) == ["key", "cells"]
    assert list(json.loads(report)["groups"][0]["cells"][0]) == [
        "key", "n", "median", "mad", "se", "lower", "upper",
    ]
    assert '"minutes":30' in report
    assert '"z":2.000000' in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_robust_report("not a list")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes": 0},
        {"minutes": 7},
        {"minutes": 1441},
        {"minutes": True},
        {"z": -0.01},
        {"z": True},
        {"z": float("nan")},
        {"z": float("inf")},
        {"z": "1.96"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_matrix_robust_report([], **kwargs)


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
        effect_matrix_robust_report([row])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 0.0)
    with pytest.raises(ValueError):
        effect_matrix_robust_report([row, row])
