"""Tests for urban_micro.effect_matrix_moran_report."""

import json

import pytest

from urban_micro import effect_matrix_moran_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def test_empty_details_gives_empty_groups():
    report = effect_matrix_moran_report([], [])
    assert report == '{"minutes":60,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "groups": []}


def test_no_edges_gives_i_zero_p_one():
    report = effect_matrix_moran_report(
        [_row(0, "a", 1.0), _row(0, "b", 3.0)], []
    )
    group = json.loads(report)["groups"][0]
    assert group == {"key": 0, "n": 2, "moran": 0.0, "p": 1.0}


def test_zero_variance_gives_i_zero_p_one():
    report = effect_matrix_moran_report(
        [_row(0, "a", 2.0), _row(0, "b", 2.0)], [("a", "b")]
    )
    group = json.loads(report)["groups"][0]
    assert group["moran"] == 0.0
    assert group["p"] == 1.0


def test_two_node_edge_negative_autocorrelation():
    # d = [1, 3], one edge: x = [-1, 1], I = 2/2 * 2*(-1)/2 = -1; the two
    # permutations give the same edges products, so p = 1.
    report = effect_matrix_moran_report(
        [_row(0, "a", 1.0), _row(0, "b", 3.0)], [("a", "b")]
    )
    group = json.loads(report)["groups"][0]
    assert group == {"key": 0, "n": 2, "moran": -1.0, "p": 1.0}


def test_three_node_path_moran_and_permutation_p():
    # d in ascending cell order = [1, 3, 2], edges a-b, b-c:
    # x = [-1, 1, 0], sum squares = 2, edge products -1 and 0,
    # I = 3/4 * 2*(-1)/2 = -0.75. Of the 6 permutations, 4 place the
    # zero on an endpoint -> |I| = 0.75; p = 4/6.
    report = effect_matrix_moran_report(
        [_row(0, "a", 1.0), _row(0, "b", 3.0), _row(0, "c", 2.0)],
        [("a", "b"), ("b", "c")],
    )
    group = json.loads(report)["groups"][0]
    assert group["n"] == 3
    assert group["moran"] == -0.75
    assert group["p"] == pytest.approx(2 / 3)
    assert '"p":0.666667' in report


def test_permutation_p_below_one():
    # d = [1, 2, 4], edges a-b, a-c: mean 7/3, x = [-4/3, -1/3, 5/3],
    # edge products 4/9 and -20/9, denominator 14/3;
    # I = 3/4 * 2*(-16/9)/(14/3) = -4/7. The star center takes value 1
    # (|I| = 4/7) in 2 permutations and value 4 (|I| = 25/28) in 2 more;
    # the 4 permutations with |I_perm| >= 4/7 give p = 4/6.
    report = effect_matrix_moran_report(
        [_row(0, "a", 1.0), _row(0, "b", 2.0), _row(0, "c", 4.0)],
        [("a", "b"), ("a", "c")],
    )
    group = json.loads(report)["groups"][0]
    assert group["moran"] == pytest.approx(-4 / 7, abs=1e-6)
    assert group["p"] == pytest.approx(2 / 3)
    assert '"p":0.666667' in report


def test_duplicate_d_values_not_deduplicated():
    # n = 3, d = [1, 4, 1], edges a-b, b-c: mean 2, x = [-1, 2, -1],
    # I = 3/4 * 2*(-4)/6 = -1. Although two values are equal, all 3! index
    # permutations are counted; the 4 at the middle (2 index permutations)
    # gives |I| = 1 while placing it at an endpoint (4 permutations) gives
    # |I| = 1/4, hence p = 2/6.
    report = effect_matrix_moran_report(
        [_row(0, "a", 1.0), _row(0, "b", 4.0), _row(0, "c", 1.0)],
        [("a", "b"), ("b", "c")],
    )
    group = json.loads(report)["groups"][0]
    assert group["moran"] == -1.0
    assert group["p"] == pytest.approx(1 / 3)


def test_delta_averaged_per_bucket_cell():
    # Cell a has two rows in the bucket: mean delta 2; cell b delta 4;
    # edge a-b. x = [-1, 1], I = -1 as in the two-node case.
    report = effect_matrix_moran_report(
        [
            _row(0, "a", 1.0),
            _row(1200, "a", 3.0),
            _row(0, "b", 4.0),
        ],
        [("a", "b")],
        minutes=60,
    )
    group = json.loads(report)["groups"][0]
    assert group == {"key": 0, "n": 2, "moran": -1.0, "p": 1.0}


def test_buckets_floor_and_cells_sorted_within_bucket():
    details = [
        _row(0, "c", 1.0),
        _row(1800, "a", 1.0),
        _row(1800, "b", 2.0),
        _row(0, "a", 3.0),
    ]
    report = effect_matrix_moran_report(details, [("a", "b"), ("a", "c")], minutes=30)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [0, 1800]
    assert groups[0]["n"] == 2
    assert groups[1]["n"] == 2


def test_edges_with_both_endpoints_in_bucket_only():
    # Edge b-c only exists in bucket 3600 (c absent from bucket 0); edge
    # a-b only in bucket 0. Each bucket therefore has one edge.
    report = effect_matrix_moran_report(
        [
            _row(0, "a", 1.0),
            _row(0, "b", 3.0),
            _row(3600, "b", 5.0),
            _row(3600, "c", 2.0),
        ],
        [("a", "b"), ("b", "c")],
    )
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [0, 3600]
    assert groups[0] == {"key": 0, "n": 2, "moran": -1.0, "p": 1.0}
    assert groups[1] == {"key": 3600, "n": 2, "moran": -1.0, "p": 1.0}


def test_more_than_8_cells_per_bucket_raises_value_error():
    with pytest.raises(ValueError):
        effect_matrix_moran_report(
            [_row(0, chr(ord("a") + i), float(i)) for i in range(9)], []
        )


def test_8_cells_per_bucket_allowed():
    cells = [chr(ord("a") + i) for i in range(8)]
    report = effect_matrix_moran_report(
        [_row(i * 100, c, float(i)) for i, c in enumerate(cells)],
        [("a", "b")],
    )
    group = json.loads(report)["groups"][0]
    assert group["key"] == 0
    assert group["n"] == 8


def test_9_cells_spread_across_buckets_allowed():
    rows = [
        _row(bucket * 3600, chr(ord("a") + cell), 1.0)
        for bucket in range(3)
        for cell in range(3)
    ]
    report = effect_matrix_moran_report(rows, [], minutes=60)
    groups = json.loads(report)["groups"]
    assert [g["n"] for g in groups] == [3, 3, 3]


def test_negative_zero_normalized():
    report = effect_matrix_moran_report(
        [_row(0, "a", -0.0), _row(0, "b", -0.0)], [("a", "b")]
    )
    assert "-0.000000" not in report
    assert report == (
        '{"minutes":60,"groups":['
        '{"key":0,"n":2,"moran":0.000000,"p":1.000000}]}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    # Cell ids (including non-ASCII ones) only drive grouping and edges;
    # group keys are the integer bucket starts.
    report = effect_matrix_moran_report(
        [_row(0, "a", 1.0), _row(0, "céll", 2.0)], [("a", "céll")]
    )
    assert " " not in report
    assert not report.endswith("\n")
    assert report == (
        '{"minutes":60,"groups":['
        '{"key":0,"n":2,"moran":-1.000000,"p":1.000000}]}'
    )


def test_top_level_and_group_key_order():
    report = effect_matrix_moran_report(
        [_row(0, "a", 1.0), _row(0, "b", 2.0)], [("a", "b")], minutes=30
    )
    assert list(json.loads(report)) == ["minutes", "groups"]
    assert list(json.loads(report)["groups"][0]) == ["key", "n", "moran", "p"]
    assert '"minutes":30' in report
    assert '"key":0' in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_moran_report("not a list", [])


def test_neighbors_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_moran_report([], (("a", "b"),))


@pytest.mark.parametrize(
    "edge",
    [
        ["a", "b"],  # not a tuple
        ("a", "b", "c"),  # wrong length
        ("a",),  # wrong length
        ("a", 1),  # non-string endpoint
        ("a", ""),  # empty endpoint
        ("", "b"),  # empty endpoint
        (True, "b"),  # non-string endpoint
        ("a", "a"),  # self loop
        ("a", "z"),  # unknown cell
        ("z", "b"),  # unknown cell
    ],
)
def test_invalid_edges_raise_value_error(edge):
    with pytest.raises(ValueError):
        effect_matrix_moran_report(
            [_row(0, "a", 1.0), _row(0, "b", 2.0)], [edge]
        )


def test_duplicate_edge_same_orientation_raises_value_error():
    with pytest.raises(ValueError):
        effect_matrix_moran_report(
            [_row(0, "a", 1.0), _row(0, "b", 2.0)],
            [("a", "b"), ("a", "b")],
        )


def test_duplicate_edge_reverse_orientation_raises_value_error():
    with pytest.raises(ValueError):
        effect_matrix_moran_report(
            [_row(0, "a", 1.0), _row(0, "b", 2.0)],
            [("a", "b"), ("b", "a")],
        )


def test_nonempty_neighbors_with_empty_details_raises_value_error():
    with pytest.raises(ValueError):
        effect_matrix_moran_report([], [("a", "b")])


@pytest.mark.parametrize("minutes", [0, 7, 1441, True, False, 1.5, "60"])
def test_invalid_minutes_raises_value_error(minutes):
    with pytest.raises(ValueError):
        effect_matrix_moran_report([], [], minutes=minutes)


@pytest.mark.parametrize("minutes", [1, 30, 60, 1440])
def test_minutes_boundaries(minutes):
    report = effect_matrix_moran_report([], [], minutes=minutes)
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
        effect_matrix_moran_report([row], [])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 0.0)
    with pytest.raises(ValueError):
        effect_matrix_moran_report([row, row], [])
