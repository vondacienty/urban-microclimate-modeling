"""Tests for urban_micro.effect_matrix_trimmed_report."""

import json

import pytest

from urban_micro import effect_matrix_trimmed_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def test_empty_details_gives_empty_groups():
    report = effect_matrix_trimmed_report([])
    assert report == '{"minutes":60,"trim":0.100000,"z":1.960000,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "trim": 0.1, "z": 1.96, "groups": []}


def test_trim_zero_two_rows():
    # trim=0 keeps everything: d = [1, 3], mu = 2, sign sums +/-4, +/-2,
    # only +/-4 reach |total| = 4 -> p = 0.5; se = sqrt(2 / (2 * 1)) = 1.
    report = effect_matrix_trimmed_report(
        [_row(0, "a", 3.0), _row(5, "a", 1.0)], trim=0.0
    )
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell == {
        "key": "a",
        "n": 2,
        "trimmed": 2.0,
        "p": 0.5,
        "se": 1.0,
        "lower": 0.04,
        "upper": 3.96,
    }


def test_trimming_drops_extremes():
    # n = 5, trim = 0.2 -> k = 1, trimmed sample [2, 3, 4], mu = 3.
    # Sign sums of [2, 3, 4]: only +/-9 reach |total| = 9 -> p = 2/8 = 0.25.
    details = [_row(i, "a", d) for i, d in enumerate([1.0, 2.0, 3.0, 4.0, 100.0])]
    report = effect_matrix_trimmed_report(details, trim=0.2)
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["n"] == 5
    assert cell["trimmed"] == 3.0
    assert cell["p"] == 0.25
    # se = sqrt((1 + 0 + 1) / (3 * 2)) = sqrt(1/3).
    assert cell["se"] == pytest.approx(0.577350)
    assert cell["lower"] == pytest.approx(3.0 - 1.96 * (1 / 3) ** 0.5)
    assert cell["upper"] == pytest.approx(3.0 + 1.96 * (1 / 3) ** 0.5)


def test_untrimmed_same_rows_differ():
    # Same rows with trim=0 keep the 100 outlier: mu = 22, p = 2/32.
    details = [_row(i, "a", d) for i, d in enumerate([1.0, 2.0, 3.0, 4.0, 100.0])]
    report = effect_matrix_trimmed_report(details, trim=0.0)
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["trimmed"] == 22.0
    assert cell["p"] == 0.0625


def test_k_is_floor_of_trim_times_n():
    # n = 10, trim = 0.1 -> k = 1: the smallest and largest of the ten
    # deltas 1..10 are dropped, leaving 2..9 with mean 5.5.
    details = [_row(i, "a", float(i + 1)) for i in range(10)]
    report = effect_matrix_trimmed_report(details, trim=0.1)
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["n"] == 10
    assert cell["trimmed"] == 5.5


def test_default_trim_is_point_one():
    # n = 5, default trim = 0.1 -> k = floor(0.5) = 0, nothing trimmed.
    details = [_row(i, "a", float(i + 1)) for i in range(5)]
    report = effect_matrix_trimmed_report(details)
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["trimmed"] == 3.0
    assert '"trim":0.100000' in report


def test_deltas_sorted_before_trimming():
    # Rows arrive out of order; trimming must apply to the sorted deltas.
    details = [_row(0, "a", 100.0), _row(1, "a", 1.0), _row(2, "a", 3.0),
               _row(3, "a", 2.0), _row(4, "a", 4.0)]
    report = effect_matrix_trimmed_report(details, trim=0.2)
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["trimmed"] == 3.0


@pytest.mark.parametrize(
    "rows,trim",
    [
        ([_row(0, "a", 1.0)], 0.0),  # m = 1
        ([_row(0, "a", 1.0)], 0.1),  # m = 1
        ([_row(i, "a", 1.0) for i in range(3)], 0.4),  # k = 1, m = 1
        ([_row(i, "a", 1.0) for i in range(2)], 0.499999),  # k = 0... m = 2 ok
    ],
)
def test_too_few_rows_after_trimming_raise_value_error(rows, trim):
    if len(rows) - 2 * int(trim * len(rows)) >= 2:
        pytest.skip("keeps at least two rows")
    with pytest.raises(ValueError):
        effect_matrix_trimmed_report(rows, trim=trim)


def test_m_two_is_allowed():
    report = effect_matrix_trimmed_report(
        [_row(0, "a", 1.0), _row(1, "a", 3.0)], trim=0.499999
    )
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["n"] == 2
    assert cell["trimmed"] == 2.0


def test_trim_limit_is_per_bucket_cell():
    # One row per bucket: each bucket/cell keeps m = 1 -> ValueError only
    # for that cell; spread rows so each cell has two rows instead.
    details = [_row(i * 3600, "a", 1.0) for i in range(4)] + [
        _row(i * 3600 + 1, "a", 3.0) for i in range(4)
    ]
    report = effect_matrix_trimmed_report(details, minutes=60, trim=0.0)
    groups = json.loads(report)["groups"]
    assert len(groups) == 4
    assert all(cell["n"] == 2 for g in groups for cell in g["cells"])


def test_buckets_floor_and_cells_sorted_within_bucket():
    details = [
        _row(0, "b", 1.0),
        _row(1800, "a", 1.0),
        _row(1800, "b", 1.0),
        _row(0, "a", 1.0),
        _row(1, "b", 2.0),
        _row(1801, "a", 2.0),
        _row(1801, "b", 2.0),
        _row(1, "a", 2.0),
    ]
    report = effect_matrix_trimmed_report(details, minutes=30, trim=0.0)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [0, 1800]
    assert [cell["key"] for cell in groups[0]["cells"]] == ["a", "b"]
    assert [cell["key"] for cell in groups[1]["cells"]] == ["a", "b"]


def test_zero_z_gives_degenerate_interval():
    report = effect_matrix_trimmed_report(
        [_row(0, "a", 1.0), _row(1, "a", 3.0)], trim=0.0, z=0.0
    )
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["lower"] == 2.0
    assert cell["upper"] == 2.0


def test_negative_zero_normalized():
    report = effect_matrix_trimmed_report(
        [_row(0, "a", -0.0), _row(1, "a", 0.0)], trim=0.0
    )
    assert "-0.000000" not in report
    assert report == (
        '{"minutes":60,"trim":0.000000,"z":1.960000,"groups":['
        '{"key":0,"cells":['
        '{"key":"a","n":2,"trimmed":0.000000,"p":1.000000,"se":0.000000,'
        '"lower":0.000000,"upper":0.000000}]}]}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_matrix_trimmed_report(
        [_row(0, "céll", 1.0), _row(1, "céll", 2.0)], trim=0.0
    )
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    report = effect_matrix_trimmed_report(
        [_row(0, "a", 1.0), _row(1, "a", 2.0)], minutes=30, trim=0.25, z=2.0
    )
    assert list(json.loads(report)) == ["minutes", "trim", "z", "groups"]
    assert list(json.loads(report)["groups"][0]) == ["key", "cells"]
    assert list(json.loads(report)["groups"][0]["cells"][0]) == [
        "key", "n", "trimmed", "p", "se", "lower", "upper",
    ]
    assert '"minutes":30' in report
    assert '"trim":0.250000' in report
    assert '"z":2.000000' in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_trimmed_report("not a list")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes": 0},
        {"minutes": 7},
        {"minutes": 1441},
        {"minutes": True},
        {"trim": -0.1},
        {"trim": 0.5},
        {"trim": 1.0},
        {"trim": True},
        {"trim": float("nan")},
        {"trim": float("inf")},
        {"trim": "0.1"},
        {"z": -0.0000001},
        {"z": True},
        {"z": float("nan")},
        {"z": float("inf")},
        {"z": "1.96"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_matrix_trimmed_report([], **kwargs)


@pytest.mark.parametrize("minutes", [1, 30, 60, 1440])
def test_minutes_boundaries(minutes):
    report = effect_matrix_trimmed_report([], minutes=minutes)
    assert json.loads(report)["minutes"] == minutes


@pytest.mark.parametrize("trim", [0, 0.0, 0.25, 0.499999])
def test_trim_boundaries_allowed(trim):
    report = effect_matrix_trimmed_report([], trim=trim)
    assert json.loads(report)["trim"] == pytest.approx(trim)


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
        effect_matrix_trimmed_report([row])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 0.0)
    with pytest.raises(ValueError):
        effect_matrix_trimmed_report([row, row])
