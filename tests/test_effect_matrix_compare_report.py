"""Tests for urban_micro.effect_matrix_compare_report."""

import json

import pytest

from urban_micro import effect_matrix_compare_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def test_empty_sides_give_empty_groups():
    report = effect_matrix_compare_report([], [])
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}'
    assert json.loads(report) == {
        "minutes": 60,
        "alpha": 0.05,
        "groups": [],
    }


def test_paired_means_changes_and_global_bh_q():
    # bucket 0:
    #   cell a: six pairs with change +1 -> p = 2/64 = 0.03125
    #   cell c: changes +1 and -1 -> m = 2 split 1/1 -> p = 1
    # bucket 3600:
    #   cell b: five pairs with change +2 -> p = 2/32 = 0.0625
    before = [_row(t, "a", 0.0) for t in range(6)]
    before += [_row(0, "c", 0.0), _row(1, "c", 0.0)]
    before += [_row(3600 + t, "b", 0.0) for t in range(5)]

    after = [_row(t, "a", 1.0) for t in range(6)]
    after += [_row(0, "c", 1.0), _row(1, "c", -1.0)]
    after += [_row(3600 + t, "b", 2.0) for t in range(5)]

    report = effect_matrix_compare_report(before, after)
    assert report == (
        '{"minutes":60,"alpha":0.050000,"groups":['
        '{"key":0,"cells":['
        '{"key":"a","n":6,"before":0.000000,"after":1.000000,"change":1.000000,'
        '"p":0.031250,"q":0.093750,"reject":false},'
        '{"key":"c","n":2,"before":0.000000,"after":0.000000,"change":0.000000,'
        '"p":1.000000,"q":1.000000,"reject":false}]},'
        '{"key":3600,"cells":['
        '{"key":"b","n":5,"before":0.000000,"after":2.000000,"change":2.000000,'
        '"p":0.062500,"q":0.093750,"reject":false}]}]}'
    )

    groups = {g["key"]: g["cells"] for g in json.loads(report)["groups"]}
    cell_a = groups[0][0]
    cell_b = groups[3600][0]
    # Global BH with N = 3 cells (not per-bucket N): both a and b inherit the
    # running minimum min(3*0.03125/1, 3*0.0625/2) = 0.09375.
    assert cell_a["q"] == 0.09375
    assert cell_b["q"] == 0.09375


def test_reject_compared_on_unquantized_q():
    before = [_row(t, "a", 0.0) for t in range(6)]
    before += [_row(0, "c", 0.0), _row(1, "c", 0.0)]
    before += [_row(3600 + t, "b", 0.0) for t in range(5)]

    after = [_row(t, "a", 1.0) for t in range(6)]
    after += [_row(0, "c", 1.0), _row(1, "c", -1.0)]
    after += [_row(3600 + t, "b", 1.0) for t in range(5)]

    groups = {
        (g["key"], cell["key"]): cell
        for g in json.loads(
            effect_matrix_compare_report(before, after, alpha=0.1)
        )["groups"]
        for cell in g["cells"]
    }
    assert groups[(0, "a")]["reject"] is True
    assert groups[(3600, "b")]["reject"] is True
    assert groups[(0, "c")]["reject"] is False


def test_zero_changes_give_p_one():
    report = effect_matrix_compare_report([_row(0, "a", 0.0)], [_row(0, "a", 0.0)])
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["n"] == 1
    assert cell["change"] == 0.0
    assert cell["p"] == 1.0
    assert cell["q"] == 1.0
    assert cell["reject"] is False


def test_buckets_floor_and_cells_sorted_within_bucket():
    keys = [(0, "b"), (1799, "a"), (1800, "a"), (1800, "b")]
    before = [_row(t, c, 0.0) for t, c in keys]
    after = [_row(t, c, 1.0) for t, c in keys]
    report = effect_matrix_compare_report(before, after, minutes=30)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [0, 1800]
    assert [cell["key"] for cell in groups[0]["cells"]] == ["a", "b"]
    assert [cell["n"] for cell in groups[0]["cells"]] == [1, 1]
    assert [cell["key"] for cell in groups[1]["cells"]] == ["a", "b"]


def test_paired_rows_aggregate_within_bucket():
    # Two pairs of cell a in bucket 0 (both changes +2) -> n = 2, p = 0.5.
    before = [_row(0, "a", 2.0), _row(5, "a", 4.0)]
    after = [_row(0, "a", 4.0), _row(5, "a", 6.0)]
    cell = json.loads(effect_matrix_compare_report(before, after))["groups"][0][
        "cells"
    ][0]
    assert cell["n"] == 2
    assert cell["before"] == 3.0
    assert cell["after"] == 5.0
    assert cell["change"] == 2.0
    assert cell["p"] == 0.5
    assert cell["q"] == 0.5


def test_pairing_is_by_timestamp_cell_key_not_position():
    keys = [(0, "b"), (1799, "a"), (1800, "a"), (1800, "b")]
    before = [_row(t, c, 0.0, base=10.0) for t, c in keys]
    after = [_row(t, c, 1.0, base=99.0) for t, c in keys]
    ordered = effect_matrix_compare_report(before, after, minutes=30)
    # A shuffled ``after`` must pair by the (timestamp, cell_id) key, not index.
    shuffled = effect_matrix_compare_report(before, list(reversed(after)), minutes=30)
    assert shuffled == ordered
    cell = json.loads(ordered)["groups"][0]["cells"][0]
    # before/after are the delta means, never base/post.
    assert cell["before"] == 0.0
    assert cell["after"] == 1.0
    assert cell["change"] == 1.0


def test_negative_zero_normalized():
    report = effect_matrix_compare_report(
        [_row(0, "a", -0.0, base=-0.0, post=-0.0, cg=-0.0, cr=-0.0, cm=-0.0)],
        [_row(0, "a", -0.0, base=-0.0, post=-0.0, cg=-0.0, cr=-0.0, cm=-0.0)],
    )
    assert "-0.000000" not in report
    assert report == (
        '{"minutes":60,"alpha":0.050000,"groups":['
        '{"key":0,"cells":['
        '{"key":"a","n":1,"before":0.000000,"after":0.000000,"change":0.000000,'
        '"p":1.000000,"q":1.000000,"reject":false}]}]}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_matrix_compare_report(
        [_row(0, "céll", 0.0)], [_row(0, "céll", 1.0)]
    )
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    report = effect_matrix_compare_report(
        [_row(0, "a", 0.0)], [_row(0, "a", 1.0)], minutes=30, alpha=0.1
    )
    assert list(json.loads(report)) == ["minutes", "alpha", "groups"]
    assert list(json.loads(report)["groups"][0]) == ["key", "cells"]
    assert list(json.loads(report)["groups"][0]["cells"][0]) == [
        "key", "n", "before", "after", "change", "p", "q", "reject",
    ]
    assert '"minutes":30' in report
    assert '"alpha":0.100000' in report


@pytest.mark.parametrize("bad_side", ["before", "after"])
def test_side_not_list_raises_type_error(bad_side):
    sides = {"before": [], "after": []}
    sides[bad_side] = "not a list"
    with pytest.raises(TypeError):
        effect_matrix_compare_report(sides["before"], sides["after"])


def test_tuple_side_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_compare_report((), [])
    with pytest.raises(TypeError):
        effect_matrix_compare_report([], ())


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
        effect_matrix_compare_report([], [], **kwargs)


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
def test_invalid_rows_in_before_raise_value_error(row):
    good = _row(0, "a", 0.0)
    with pytest.raises(ValueError):
        effect_matrix_compare_report([row], [good])


def test_invalid_row_in_after_raises_value_error():
    good = _row(0, "a", 0.0)
    bad = (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        effect_matrix_compare_report([good], [bad])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 0.0)
    other = _row(0, "a", 1.0)
    with pytest.raises(ValueError):
        effect_matrix_compare_report([row, row], [row, other])
    with pytest.raises(ValueError):
        effect_matrix_compare_report([row, other], [row, row])


def test_missing_pair_in_after_raises_value_error():
    before = [_row(0, "a", 0.0), _row(1, "a", 0.0)]
    after = [_row(0, "a", 1.0)]
    with pytest.raises(ValueError):
        effect_matrix_compare_report(before, after)


def test_extra_pair_in_after_raises_value_error():
    before = [_row(0, "a", 0.0)]
    after = [_row(0, "a", 1.0), _row(1, "a", 1.0)]
    with pytest.raises(ValueError):
        effect_matrix_compare_report(before, after)


def test_one_side_empty_other_not_raises_value_error():
    with pytest.raises(ValueError):
        effect_matrix_compare_report([], [_row(0, "a", 0.0)])
    with pytest.raises(ValueError):
        effect_matrix_compare_report([_row(0, "a", 0.0)], [])
