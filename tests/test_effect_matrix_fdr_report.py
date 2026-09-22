"""Tests for urban_micro.effect_matrix_fdr_report."""

import json

import pytest

from urban_micro import effect_matrix_fdr_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def test_empty_details_gives_empty_groups():
    report = effect_matrix_fdr_report([])
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}'
    assert json.loads(report) == {
        "minutes": 60,
        "alpha": 0.05,
        "groups": [],
    }


def test_global_bh_q_spans_all_buckets_and_cells():
    # bucket 0:
    #   cell a: six positive deltas -> p = 2/64 = 0.03125
    #   cell c: deltas +1, -1 -> m = 2 split 1/1 -> p = 1
    # bucket 3600:
    #   cell b: five positive deltas -> p = 2/32 = 0.0625
    details = [_row(t, "a", 1.0, base=2.0, cg=1.0) for t in range(6)]
    details += [_row(0, "c", 1.0), _row(1, "c", -1.0)]
    details += [_row(3600 + t, "b", 2.0) for t in range(5)]

    report = effect_matrix_fdr_report(details)
    assert report == (
        '{"minutes":60,"alpha":0.050000,"groups":['
        '{"key":0,"cells":['
        '{"key":"a","n":6,"base":2.000000,"post":3.000000,"delta":1.000000,'
        '"cg":1.000000,"cr":0.000000,"cm":0.000000,"p":0.031250,'
        '"q":0.093750,"reject":false},'
        '{"key":"c","n":2,"base":0.000000,"post":0.000000,"delta":0.000000,'
        '"cg":0.000000,"cr":0.000000,"cm":0.000000,"p":1.000000,'
        '"q":1.000000,"reject":false}]},'
        '{"key":3600,"cells":['
        '{"key":"b","n":5,"base":0.000000,"post":2.000000,"delta":2.000000,'
        '"cg":0.000000,"cr":0.000000,"cm":0.000000,"p":0.062500,'
        '"q":0.093750,"reject":false}]}]}'
    )

    groups = {g["key"]: g["cells"] for g in json.loads(report)["groups"]}
    cell_a = groups[0][0]
    cell_b = groups[3600][0]
    # Global BH with N = 3 cells (not per-bucket N): both a and b inherit the
    # running minimum min(3*0.03125/1, 3*0.0625/2) = 0.09375.
    assert cell_a["q"] == 0.09375
    assert cell_b["q"] == 0.09375


def test_reject_compared_on_unquantized_q():
    # Same data as above: a and b share q = 0.09375, c has q = 1.
    details = [_row(t, "a", 1.0) for t in range(6)]
    details += [_row(0, "c", 1.0), _row(1, "c", -1.0)]
    details += [_row(3600 + t, "b", 1.0) for t in range(5)]
    groups = {
        (g["key"], cell["key"]): cell
        for g in json.loads(effect_matrix_fdr_report(details, alpha=0.1))["groups"]
        for cell in g["cells"]
    }
    assert groups[(0, "a")]["reject"] is True
    assert groups[(3600, "b")]["reject"] is True
    assert groups[(0, "c")]["reject"] is False


def test_sign_test_zero_deltas_give_p_one():
    report = effect_matrix_fdr_report([_row(0, "a", 0.0)])
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["n"] == 1
    assert cell["p"] == 1.0
    assert cell["q"] == 1.0
    assert cell["reject"] is False


def test_buckets_floor_and_cells_sorted_within_bucket():
    details = [
        _row(0, "b", 1.0),
        _row(1799, "a", 1.0),
        _row(1800, "a", 1.0),
        _row(1800, "b", 1.0),
    ]
    report = effect_matrix_fdr_report(details, minutes=30)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [0, 1800]
    assert [cell["key"] for cell in groups[0]["cells"]] == ["a", "b"]
    assert [cell["n"] for cell in groups[0]["cells"]] == [1, 1]
    assert [cell["key"] for cell in groups[1]["cells"]] == ["a", "b"]


def test_same_cell_rows_aggregate_within_bucket():
    # Two rows of cell a in bucket 0 (both positive) -> n = 2, p = 0.5.
    details = [_row(0, "a", 1.0, base=2.0), _row(5, "a", 3.0, base=4.0)]
    cell = json.loads(effect_matrix_fdr_report(details))["groups"][0]["cells"][0]
    assert cell["n"] == 2
    assert cell["base"] == 3.0
    assert cell["post"] == 5.0
    assert cell["delta"] == 2.0
    assert cell["p"] == 0.5
    assert cell["q"] == 0.5


def test_negative_zero_normalized():
    report = effect_matrix_fdr_report(
        [_row(0, "a", -0.0, base=-0.0, post=-0.0, cg=-0.0, cr=-0.0, cm=-0.0)]
    )
    assert "-0.000000" not in report
    assert report == (
        '{"minutes":60,"alpha":0.050000,"groups":['
        '{"key":0,"cells":['
        '{"key":"a","n":1,"base":0.000000,"post":0.000000,"delta":0.000000,'
        '"cg":0.000000,"cr":0.000000,"cm":0.000000,"p":1.000000,'
        '"q":1.000000,"reject":false}]}]}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_matrix_fdr_report([_row(0, "céll", 1.0)])
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    report = effect_matrix_fdr_report([_row(0, "a", 1.0)], minutes=30, alpha=0.1)
    assert list(json.loads(report)) == ["minutes", "alpha", "groups"]
    assert list(json.loads(report)["groups"][0]) == ["key", "cells"]
    assert list(json.loads(report)["groups"][0]["cells"][0]) == [
        "key", "n", "base", "post", "delta", "cg", "cr", "cm", "p", "q", "reject",
    ]
    assert '"minutes":30' in report
    assert '"alpha":0.100000' in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_fdr_report("not a list")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes": 0},
        {"minutes": 7},
        {"minutes": 1441},
        {"minutes": True},
        {"alpha": 0.0},
        {"alpha": 1.5},
        {"alpha": True},
        {"alpha": float("nan")},
        {"alpha": "0.05"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_matrix_fdr_report([], **kwargs)


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
        effect_matrix_fdr_report([row])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 0.0)
    with pytest.raises(ValueError):
        effect_matrix_fdr_report([row, row])
