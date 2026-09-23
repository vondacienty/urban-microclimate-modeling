"""Tests for urban_micro.effect_matrix_autocorr_report."""

import json
import math

import pytest

from urban_micro import effect_matrix_autocorr_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def test_empty_details_gives_empty_groups():
    report = effect_matrix_autocorr_report([])
    assert report == '{"minutes":60,"lag":1,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "lag": 1, "groups": []}


def test_lag_one_pairs_adjacent_buckets():
    details = [
        _row(0, "a", 1.0),
        _row(3600, "a", 3.0),
        _row(7200, "a", 6.0),
    ]
    group = json.loads(effect_matrix_autocorr_report(details))["groups"][0]
    assert group["key"] == "a"
    assert group["n"] == 2
    assert group["pairs"] == [
        {"bucket": 3600, "prev": 1.0, "curr": 3.0, "change": 2.0},
        {"bucket": 7200, "prev": 3.0, "curr": 6.0, "change": 3.0},
    ]
    assert group["mean_change"] == pytest.approx(2.5, abs=1e-6)
    assert group["se"] == pytest.approx(0.5, abs=1e-6)


def test_unpaired_buckets_and_cells_are_skipped():
    details = [
        # buckets 0 and 7200 with a gap at 3600: neither pairs
        _row(0, "a", 1.0),
        _row(7200, "a", 4.0),
        # only one occupied bucket: the whole cell is omitted
        _row(0, "b", 2.0),
        # cell c has one valid pair and must survive
        _row(0, "c", 1.0),
        _row(3600, "c", 5.0),
    ]
    groups = json.loads(effect_matrix_autocorr_report(details))["groups"]
    assert [g["key"] for g in groups] == ["c"]
    assert groups[0]["pairs"] == [
        {"bucket": 3600, "prev": 1.0, "curr": 5.0, "change": 4.0},
    ]


def test_lag_two_skips_intermediate_buckets():
    details = [_row(3600 * k, "a", float(k)) for k in range(5)]
    group = json.loads(effect_matrix_autocorr_report(details, lag=2))["groups"][0]
    assert [pair["bucket"] for pair in group["pairs"]] == [7200, 10800, 14400]
    assert [pair["prev"] for pair in group["pairs"]] == [0.0, 1.0, 2.0]
    assert [pair["curr"] for pair in group["pairs"]] == [2.0, 3.0, 4.0]
    assert all(pair["change"] == 2.0 for pair in group["pairs"])
    assert group["mean_change"] == 2.0
    assert group["se"] == 0.0


def test_within_bucket_deltas_are_averaged_before_pairing():
    details = [
        _row(0, "a", 1.0),
        _row(600, "a", 3.0),  # mean of bucket 0 = 2
        _row(3600, "a", 7.0),
        _row(5400, "a", 9.0),  # mean of bucket 3600 = 8
    ]
    pair = json.loads(effect_matrix_autocorr_report(details))["groups"][0]["pairs"][0]
    assert pair == {"bucket": 3600, "prev": 2.0, "curr": 8.0, "change": 6.0}


def test_se_formula_on_changes():
    # four buckets -> three changes 2, 4, 6: mean 4, sum sq dev 8,
    # se = sqrt(8 / (3 * 2))
    details = [_row(3600 * k, "a", float(d)) for k, d in enumerate((0.0, 2.0, 6.0, 12.0))]
    group = json.loads(effect_matrix_autocorr_report(details))["groups"][0]
    assert group["n"] == 3
    assert group["mean_change"] == pytest.approx(4.0, abs=1e-6)
    assert group["se"] == pytest.approx(math.sqrt(8 / 6), abs=1e-6)


def test_single_pair_has_zero_se():
    details = [_row(0, "a", 5.0), _row(3600, "a", 9.0)]
    group = json.loads(effect_matrix_autocorr_report(details))["groups"][0]
    assert group["n"] == 1
    assert group["mean_change"] == 4.0
    assert group["se"] == 0.0


def test_corr_perfect_positive_and_negative():
    pos = [_row(3600 * k, "a", float(2 * k)) for k in range(4)]
    group = json.loads(effect_matrix_autocorr_report(pos))["groups"][0]
    assert group["corr"] == pytest.approx(1.0, abs=1e-6)

    # alternating 0, 2, 0, 2 -> curr deviations are exactly -prev deviations
    neg = [_row(3600 * k, "c", float(d)) for k, d in enumerate((0.0, 2.0, 0.0, 2.0))]
    group = json.loads(effect_matrix_autocorr_report(neg))["groups"][0]
    assert group["corr"] == pytest.approx(-1.0, abs=1e-6)


def test_corr_zero_when_any_sum_is_zero():
    # prev values are constant (5, 5, 5) across the three pairs -> sxx = 0
    details = [
        _row(0, "a", 5.0),
        _row(3600, "a", 5.0),
        _row(7200, "a", 5.0),
        _row(10800, "a", 9.0),
    ]
    group = json.loads(effect_matrix_autocorr_report(details))["groups"][0]
    assert group["corr"] == 0.0


def test_groups_sorted_by_cell_and_pairs_by_bucket():
    details = [
        _row(7200, "b", 3.0),
        _row(3600, "b", 1.0),
        _row(0, "b", 0.0),
        _row(3600, "a", 1.0),
        _row(0, "a", 0.0),
    ]
    report = json.loads(effect_matrix_autocorr_report(details))
    assert [g["key"] for g in report["groups"]] == ["a", "b"]
    assert [p["bucket"] for p in report["groups"][1]["pairs"]] == [3600, 7200]


def test_minutes_changes_bucket_width():
    details = [_row(0, "a", 1.0), _row(1800, "a", 3.0), _row(3600, "a", 5.0)]
    # 30-minute buckets: 0 and 1800 pair; 3600 pairs with 1800.
    group = json.loads(effect_matrix_autocorr_report(details, minutes=30))["groups"][0]
    assert [p["bucket"] for p in group["pairs"]] == [1800, 3600]


def test_negative_zero_normalized():
    report = effect_matrix_autocorr_report([_row(0, "a", -0.0), _row(3600, "a", -0.0)])
    assert "-0.000000" not in report
    assert report == (
        '{"minutes":60,"lag":1,"groups":['
        '{"key":"a","n":1,"mean_change":0.000000,"se":0.000000,"corr":0.000000,'
        '"pairs":['
        '{"bucket":3600,"prev":0.000000,"curr":0.000000,"change":0.000000}]}]}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_matrix_autocorr_report(
        [_row(0, "céll", 1.0), _row(3600, "céll", 2.0)]
    )
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    report = effect_matrix_autocorr_report(
        [_row(0, "a", 1.0), _row(3600, "a", 2.0)], minutes=30, lag=2
    )
    assert list(json.loads(report)) == ["minutes", "lag", "groups"]
    assert list(json.loads(report)["groups"][0]) == [
        "key", "n", "mean_change", "se", "corr", "pairs",
    ]
    assert list(json.loads(report)["groups"][0]["pairs"][0]) == [
        "bucket", "prev", "curr", "change",
    ]
    assert '"minutes":30' in report
    assert '"lag":2' in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_autocorr_report("not a list")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes": 0},
        {"minutes": 7},
        {"minutes": 1441},
        {"minutes": True},
        {"lag": 0},
        {"lag": -1},
        {"lag": True},
        {"lag": 1.5},
        {"lag": "1"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_matrix_autocorr_report([], **kwargs)


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
        effect_matrix_autocorr_report([row])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 0.0)
    with pytest.raises(ValueError):
        effect_matrix_autocorr_report([row, row])
