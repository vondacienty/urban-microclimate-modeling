"""Tests for urban_micro.effect_matrix_local_moran_report."""

import json

import pytest

from urban_micro import effect_matrix_local_moran_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _group(report, index=0):
    return json.loads(report)["groups"][index]


def test_empty_details_gives_empty_groups():
    report = effect_matrix_local_moran_report([], [])
    assert report == '{"minutes":60,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "groups": []}


def test_local_values_and_permutation_p_on_three_chain():
    # d = a:0, b:1, c:3; mean = 4/3, x = (-4/3, -1/3, 5/3), S = 14/3
    # L_a = 2/7, L_b = -1/14, L_c = -5/14. |L_a| and |L_b| are the
    # smallest attainable magnitudes so p_a = p_b = 1; 4 of the 6
    # permutations reach |L_c| = 5/14, so p_c = 2/3.
    details = [_row(0, "a", 0.0), _row(0, "b", 1.0), _row(0, "c", 3.0)]
    report = effect_matrix_local_moran_report(details, [("a", "b"), ("b", "c")])
    assert report == (
        '{"minutes":60,"groups":[{"key":0,"n":3,"cells":['
        '{"key":"a","local":0.285714,"p":1.000000},'
        '{"key":"b","local":-0.071429,"p":1.000000},'
        '{"key":"c","local":-0.357143,"p":0.666667}]}]}'
    )


def test_extreme_assignment_gives_small_p():
    # Two adjacent cells, d = a:0, b:1: x = (-0.5, 0.5), L_a = L_b = -1.
    # Both permutations attain |L| = 1, so p = 1.
    details = [_row(0, "a", 0.0), _row(0, "b", 1.0)]
    cells = _group(effect_matrix_local_moran_report(details, [("a", "b")]))["cells"]
    assert cells == [
        {"key": "a", "local": -1.0, "p": 1.0},
        {"key": "b", "local": -1.0, "p": 1.0},
    ]


def test_isolated_cell_has_local_zero_and_p_one():
    # Edge a-b exists only across the two buckets, so within bucket 0 cell
    # c is isolated: local 0, p 1 even though S > 0.
    details = [
        _row(0, "a", 0.0),
        _row(0, "b", 1.0),
        _row(0, "c", 3.0),
        _row(3600, "a", 5.0),
        _row(3600, "b", 9.0),
        _row(3600, "c", 2.0),
    ]
    report = effect_matrix_local_moran_report(details, [("a", "b")])
    groups = json.loads(report)["groups"]
    bucket0 = {cell["key"]: cell for cell in groups[0]["cells"]}
    assert bucket0["c"] == {"key": "c", "local": 0.0, "p": 1.0}
    assert groups[1]["cells"]  # second bucket still emitted


def test_constant_deltas_give_zero_and_p_one():
    details = [_row(0, "a", 2.0), _row(0, "b", 2.0)]
    cells = _group(effect_matrix_local_moran_report(details, [("a", "b")]))["cells"]
    assert cells == [
        {"key": "a", "local": 0.0, "p": 1.0},
        {"key": "b", "local": 0.0, "p": 1.0},
    ]


def test_rows_averaged_within_bucket_cell_pair():
    # Bucket 0: a holds (0 + 2) / 2 = 1, b holds 3; x = (-1, 1), S = 2,
    # L_a = L_b = 2 * (-1) * 1 / 2 = -1, p = 1.
    details = [
        _row(0, "a", 0.0),
        _row(100, "a", 2.0),
        _row(0, "b", 3.0),
    ]
    cells = _group(effect_matrix_local_moran_report(details, [("a", "b")]))["cells"]
    assert cells[0]["local"] == -1.0
    assert cells[1]["local"] == -1.0


def test_buckets_and_cells_sorted_ascending():
    details = [
        _row(3600, "b", 1.0),
        _row(3600, "a", 0.0),
        _row(0, "b", 2.0),
        _row(0, "a", 3.0),
    ]
    report = effect_matrix_local_moran_report(details, [("a", "b")])
    groups = json.loads(report)["groups"]
    assert [group["key"] for group in groups] == [0, 3600]
    for group in groups:
        assert [cell["key"] for cell in group["cells"]] == ["a", "b"]


def test_duplicate_deltas_not_deduplicated_in_permutations():
    # n = 3 chain with d = (1, 1, 2): mean 4/3, x = (-1/3, -1/3, 2/3),
    # S = 2/3. L_c = 3 * (2/3) * (-1/3) / (2/3) = -1. Of the 6 ordered
    # permutations (repeated 1 kept), |L_c| = 1 in four: the two where c
    # holds 2, plus the two where c holds 1 and its neighbor b holds 2
    # (L_c = -1 there as well); the other two put the 2 at a and give
    # L_c = 1/2. Hence p = 4/6 = 2/3.
    details = [_row(0, "a", 1.0), _row(0, "b", 1.0), _row(0, "c", 2.0)]
    report = effect_matrix_local_moran_report(details, [("a", "b"), ("b", "c")])
    cells = {cell["key"]: cell for cell in _group(report)["cells"]}
    assert cells["c"]["local"] == pytest.approx(-1.0, abs=1e-6)
    assert cells["c"]["p"] == pytest.approx(2 / 3, abs=1e-6)


def test_more_than_eight_cells_raises_value_error():
    details = [_row(0, chr(97 + i), float(i)) for i in range(9)]
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report(details, [])


def test_eight_cells_are_allowed():
    details = [_row(0, chr(97 + i), float(i)) for i in range(8)]
    group = _group(effect_matrix_local_moran_report(details, [("a", "b")]))
    assert group["n"] == 8
    assert len(group["cells"]) == 8


def test_negative_zero_normalized_and_compact_format():
    details = [_row(0, "a", 0.0), _row(0, "céll", -1e-9)]
    report = effect_matrix_local_moran_report(details, [("a", "céll")])
    assert "-0.000000" not in report
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    report = effect_matrix_local_moran_report(details, [("a", "b")], minutes=30)
    assert list(json.loads(report)) == ["minutes", "groups"]
    group = json.loads(report)["groups"][0]
    assert list(group) == ["key", "n", "cells"]
    assert list(group["cells"][0]) == ["key", "local", "p"]
    assert '"minutes":30' in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_local_moran_report("not a list", [])


def test_neighbors_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_local_moran_report([_row(0, "a", 1.0)], "not a list")


@pytest.mark.parametrize(
    "neighbors",
    [
        [("a", "a")],  # self loop
        [("a", "b"), ("b", "a")],  # reversed duplicate
        [("a", "b"), ("a", "b")],  # exact duplicate
        [("a", "z")],  # unknown endpoint
        [("z", "b")],
        [("a", "")],  # empty endpoint
        [("", "b")],
        [["a", "b"]],  # not a tuple
        [("a",)],  # wrong length
        [("a", 1)],  # non-string endpoint
    ],
)
def test_invalid_neighbors_raise_value_error(neighbors):
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report(details, neighbors)


@pytest.mark.parametrize("minutes", [0, 7, 1441, True, "60", 60.0])
def test_invalid_minutes_raise_value_error(minutes):
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report([], [], minutes=minutes)


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
        effect_matrix_local_moran_report([row], [])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 1.0)
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report([row, row], [])
