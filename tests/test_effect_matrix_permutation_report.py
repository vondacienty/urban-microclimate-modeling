"""Tests for urban_micro.effect_matrix_permutation_report."""

import json

import pytest

from urban_micro import effect_matrix_permutation_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def test_empty_details_gives_empty_groups():
    report = effect_matrix_permutation_report([])
    assert report == '{"minutes":60,"z":1.960000,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "z": 1.96, "groups": []}


def test_single_row_gives_p_one_and_zero_se():
    report = effect_matrix_permutation_report([_row(0, "a", 2.0)])
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell == {
        "key": "a",
        "n": 1,
        "delta": 2.0,
        "p": 1.0,
        "se": 0.0,
        "lower": 2.0,
        "upper": 2.0,
    }


def test_sign_flip_p_exact():
    # d = [3, 1]: sign sums are +/-4, +/-2; only +/-4 reach |total| = 4
    # -> 2 of 4 sign vectors, p = 0.5.
    report = effect_matrix_permutation_report([_row(0, "a", 3.0), _row(5, "a", 1.0)])
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["n"] == 2
    assert cell["delta"] == 2.0
    assert cell["p"] == 0.5


def test_three_equal_positive_deltas_p():
    # All sign vectors reach |sum| = 3 only when all signs agree -> 2/8.
    report = effect_matrix_permutation_report(
        [_row(0, "a", 1.0), _row(1, "a", 1.0), _row(2, "a", 1.0)]
    )
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["p"] == 0.25


def test_zero_deltas_give_p_one():
    report = effect_matrix_permutation_report([_row(0, "a", 0.0), _row(1, "a", 0.0)])
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["delta"] == 0.0
    assert cell["p"] == 1.0


def test_jackknife_se_and_interval():
    # d = [3, 1]: leave-one means [1, 3], mu_bar = 2,
    # se = sqrt((1/2) * ((1-2)**2 + (3-2)**2)) = 1.
    report = effect_matrix_permutation_report(
        [_row(0, "a", 3.0), _row(5, "a", 1.0)], z=1.96
    )
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["se"] == 1.0
    assert cell["lower"] == pytest.approx(0.04)
    assert cell["upper"] == pytest.approx(3.96)


def test_buckets_floor_and_cells_sorted_within_bucket():
    details = [
        _row(0, "b", 1.0),
        _row(1800, "a", 1.0),
        _row(1800, "b", 1.0),
        _row(0, "a", 1.0),
    ]
    report = effect_matrix_permutation_report(details, minutes=30)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [0, 1800]
    assert [cell["key"] for cell in groups[0]["cells"]] == ["a", "b"]
    assert [cell["key"] for cell in groups[1]["cells"]] == ["a", "b"]


def test_same_cell_rows_aggregate_within_bucket():
    details = [_row(0, "a", 1.0, base=2.0), _row(5, "a", 3.0, base=4.0)]
    report = effect_matrix_permutation_report(details)
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["n"] == 2
    assert cell["delta"] == 2.0
    assert cell["p"] == 0.5


def test_more_than_16_rows_per_cell_raises_value_error():
    with pytest.raises(ValueError):
        effect_matrix_permutation_report([_row(i, "a", 1.0) for i in range(17)])


def test_16_rows_per_cell_allowed():
    report = effect_matrix_permutation_report([_row(i, "a", 1.0) for i in range(16)])
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["n"] == 16
    # Only the all-plus / all-minus sign vectors reach |sum| = 16:
    # exact p = 2/65536, rendered to six decimals.
    assert cell["p"] == 0.000031
    assert '"p":0.000031' in report


def test_16_limit_is_per_bucket_cell():
    # 17 rows of the same cell spread across buckets is fine.
    report = effect_matrix_permutation_report(
        [_row(i * 3600, "a", 1.0) for i in range(17)], minutes=1
    )
    groups = json.loads(report)["groups"]
    assert len(groups) == 17


def test_negative_zero_normalized():
    report = effect_matrix_permutation_report([_row(0, "a", -0.0)])
    assert "-0.000000" not in report
    assert report == (
        '{"minutes":60,"z":1.960000,"groups":['
        '{"key":0,"cells":['
        '{"key":"a","n":1,"delta":0.000000,"p":1.000000,"se":0.000000,'
        '"lower":0.000000,"upper":0.000000}]}]}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_matrix_permutation_report([_row(0, "céll", 1.0)])
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    report = effect_matrix_permutation_report([_row(0, "a", 1.0)], minutes=30, z=2.0)
    assert list(json.loads(report)) == ["minutes", "z", "groups"]
    assert list(json.loads(report)["groups"][0]) == ["key", "cells"]
    assert list(json.loads(report)["groups"][0]["cells"][0]) == [
        "key", "n", "delta", "p", "se", "lower", "upper",
    ]
    assert '"minutes":30' in report
    assert '"z":2.000000' in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_permutation_report("not a list")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes": 0},
        {"minutes": 7},
        {"minutes": 1441},
        {"minutes": True},
        {"z": -0.0000001},
        {"z": True},
        {"z": float("nan")},
        {"z": float("inf")},
        {"z": "1.96"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_matrix_permutation_report([], **kwargs)


@pytest.mark.parametrize("minutes", [1, 30, 60, 1440])
def test_minutes_boundaries(minutes):
    report = effect_matrix_permutation_report([], minutes=minutes)
    assert json.loads(report)["minutes"] == minutes


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
        effect_matrix_permutation_report([row])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 0.0)
    with pytest.raises(ValueError):
        effect_matrix_permutation_report([row, row])
