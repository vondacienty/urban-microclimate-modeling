"""Tests for urban_micro.effect_matrix_local_moran_report."""

import itertools
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


def _bruteforce(d, adjacency):
    """Reference local Moran values and permutation p-values in float."""
    n = len(d)
    mean = sum(d) / n
    x = [value - mean for value in d]
    sxx = sum(value * value for value in x)

    def locals_of(values):
        deviations = [value - mean for value in values]
        return [
            n * deviations[c] * sum(deviations[j] for j in adjacency[c]) / sxx
            for c in range(n)
        ]

    observed = locals_of(d)
    counts = [0] * n
    for perm in itertools.permutations(d):
        perm_locals = locals_of(perm)
        for c in range(n):
            if abs(perm_locals[c]) >= abs(observed[c]):
                counts[c] += 1
    return observed, [count / math_factorial(n) for count in counts]


def math_factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def test_empty_details_gives_empty_groups():
    report = effect_matrix_local_moran_report([], [])
    assert report == '{"minutes":60,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "groups": []}


def test_three_cell_path_statistics_and_permutation_p():
    # d = 1, 2, 6 with edges a-b and b-c: mean 3, x = -2, -1, 3 and
    # S = 14; L_a = 3*(-2)*(-1)/14 = 3/7, L_b = 3*(-1)*(1)/14 = -3/14
    # and L_c = 3*3*(-1)/14 = -9/14.
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0), _row(0, "c", 6.0)]
    neighbors = [("a", "b"), ("b", "c")]
    report = effect_matrix_local_moran_report(details, neighbors)
    assert report == (
        '{"minutes":60,"groups":['
        '{"key":0,"n":3,"cells":['
        '{"key":"a","local":0.428571,"p":1.000000},'
        '{"key":"b","local":-0.214286,"p":1.000000},'
        '{"key":"c","local":-0.642857,"p":0.666667}]}]}'
    )
    cells = _group(report)["cells"]
    assert [cell["key"] for cell in cells] == ["a", "b", "c"]
    assert cells[0]["local"] == pytest.approx(3 / 7, abs=1e-6)
    assert cells[1]["local"] == pytest.approx(-3 / 14, abs=1e-6)
    assert cells[2]["local"] == pytest.approx(-9 / 14, abs=1e-6)
    assert cells[2]["p"] == pytest.approx(2 / 3, abs=1e-6)


def test_matches_bruteforce_for_a_four_cell_graph():
    d_values = [0.5, -1.25, 2.0, 0.25]
    details = [_row(0, c, value) for c, value in zip("abcd", d_values)]
    neighbors = [("a", "b"), ("a", "c"), ("c", "d")]
    adjacency = [[1, 2], [0], [0, 3], [2]]
    expected_locals, expected_p = _bruteforce(d_values, adjacency)
    cells = _group(effect_matrix_local_moran_report(details, neighbors))["cells"]
    for cell, local, p_value in zip(cells, expected_locals, expected_p):
        assert cell["local"] == pytest.approx(local, abs=1e-6)
        assert cell["p"] == pytest.approx(p_value, abs=1e-6)


def test_rows_average_within_bucket_before_computing():
    # Bucket 0: cell a holds 0 and 2 (mean 1), b holds 3, c holds 2; the
    # bucket vector is therefore [1, 3, 2] for cells a, b, c.
    details = [
        _row(0, "a", 0.0),
        _row(100, "a", 2.0),
        _row(0, "b", 3.0),
        _row(0, "c", 2.0),
    ]
    neighbors = [("a", "b"), ("b", "c")]
    expected_locals, expected_p = _bruteforce([1.0, 3.0, 2.0], [[1], [0, 2], [1]])
    cells = _group(effect_matrix_local_moran_report(details, neighbors))["cells"]
    for cell, local, p_value in zip(cells, expected_locals, expected_p):
        assert cell["local"] == pytest.approx(local, abs=1e-6)
        assert cell["p"] == pytest.approx(p_value, abs=1e-6)


def test_groups_in_ascending_bucket_order_with_minutes_interval():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 3.0),
        _row(1800, "a", 2.0),
        _row(1800, "b", 4.0),
        _row(3600, "a", 5.0),
        _row(3600, "b", -1.0),
    ]
    neighbors = [("a", "b")]
    report = effect_matrix_local_moran_report(details, neighbors, minutes=30)
    groups = json.loads(report)["groups"]
    assert [group["key"] for group in groups] == [0, 1800, 3600]
    assert all(group["n"] == 2 for group in groups)
    assert '"minutes":30' in report


def test_only_edges_inside_bucket_count():
    # a and b share bucket 0; c only appears in bucket 3600. The a-c edge
    # is inactive in bucket 0, so a has no in-bucket neighbor there:
    # L_a = 0 and p = 1.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "c", 3.0),
    ]
    report = effect_matrix_local_moran_report(details, [("a", "c")])
    first, second = json.loads(report)["groups"]
    assert first["key"] == 0
    a_cell = first["cells"][0]
    assert a_cell == {"key": "a", "local": 0.0, "p": 1.0}
    assert second["cells"][0] == {"key": "c", "local": 0.0, "p": 1.0}


def test_zero_sum_of_squares_gives_local_zero_and_p_one():
    details = [_row(0, "a", 2.0), _row(0, "b", 2.0), _row(0, "c", 2.0)]
    report = effect_matrix_local_moran_report(
        details, [("a", "b"), ("b", "c")]
    )
    cells = _group(report)["cells"]
    assert cells == [
        {"key": "a", "local": 0.0, "p": 1.0},
        {"key": "b", "local": 0.0, "p": 1.0},
        {"key": "c", "local": 0.0, "p": 1.0},
    ]


def test_duplicate_values_are_not_deduplicated_in_permutations():
    # d = 1, 1, 2 on a path a-b-c: permutations() yields 6 arrangements
    # (the two equal 1s counted separately). mean 4/3, x = -1/3, -1/3,
    # 2/3, S = 2/3; L_a = 1/2, L_b = -1/2 and L_c = -1, the latter
    # attained by 4 of the 6 permutations -> p = 2/3.
    details = [_row(0, "a", 1.0), _row(0, "b", 1.0), _row(0, "c", 2.0)]
    neighbors = [("a", "b"), ("b", "c")]
    cells = _group(effect_matrix_local_moran_report(details, neighbors))["cells"]
    assert cells[0]["local"] == pytest.approx(1 / 2, abs=1e-6)
    assert cells[1]["local"] == pytest.approx(-1 / 2, abs=1e-6)
    assert cells[2]["local"] == pytest.approx(-1.0, abs=1e-6)
    assert cells[0]["p"] == 1.0
    assert cells[1]["p"] == 1.0
    assert cells[2]["p"] == pytest.approx(2 / 3, abs=1e-6)


def test_eight_cells_are_allowed():
    # Star centered at "a" with leaf values 0..7. The extreme leaf "h"
    # has L_h = 8 * 3.5 * (-3.5) / 42 = -7/3; only permutations putting
    # values 0 and 7 at (center, h) reach that magnitude: 2*6! of 8!
    # arrangements, so p = 1/28.
    details = [_row(0, c, float(index)) for index, c in enumerate("abcdefgh")]
    neighbors = [("a", b) for b in "bcdefgh"]
    report = effect_matrix_local_moran_report(details, neighbors)
    group = _group(report)
    assert group["n"] == 8
    h_cell = group["cells"][-1]
    assert h_cell["key"] == "h"
    assert h_cell["local"] == pytest.approx(-7 / 3, abs=1e-6)
    assert h_cell["p"] == pytest.approx(1 / 28, abs=1e-6)


def test_more_than_eight_cells_raises_value_error():
    details = [_row(0, chr(97 + i), float(i)) for i in range(9)]
    neighbors = [("a", "b")]
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report(details, neighbors)


def test_negative_zero_normalized():
    # Cell b is isolated in bucket 0 (its neighbor c lives in another
    # bucket), so L_b = 0 must render as 0.000000 rather than -0.000000.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "c", 3.0),
    ]
    report = effect_matrix_local_moran_report(details, [("b", "c")])
    assert "-0.000000" not in report
    assert '"local":0.000000' in report


def test_compact_utf8_no_spaces_no_trailing_newline():
    details = [
        _row(0, "céll", 1.0),
        _row(0, "λ", 2.0),
    ]
    report = effect_matrix_local_moran_report(details, [("céll", "λ")])
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report
    assert '"key":"λ"' in report


def test_key_orders():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    report = effect_matrix_local_moran_report(
        details, [("a", "b")], minutes=30
    )
    assert list(json.loads(report)) == ["minutes", "groups"]
    group = json.loads(report)["groups"][0]
    assert list(group) == ["key", "n", "cells"]
    assert list(group["cells"][0]) == ["key", "local", "p"]


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_local_moran_report("not a list", [])


def test_neighbors_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_local_moran_report([], (("a", "b"),))


@pytest.mark.parametrize("minutes", [0, 7, 1441, True, "60", 60.0])
def test_invalid_minutes_raise_value_error(minutes):
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report([], [], minutes=minutes)


@pytest.mark.parametrize(
    "neighbor",
    [
        ("a",),  # not a two-tuple
        ["a", "b"],  # not a tuple
        ("a", "b", "c"),  # wrong length
        ("", "b"),  # empty endpoint
        ("a", ""),  # empty endpoint
        ("a", 1),  # non-string endpoint
        ("a", "a"),  # self-loop
    ],
)
def test_invalid_neighbors_raise_value_error(neighbor):
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report(details, [neighbor])


def test_unknown_endpoint_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report(details, [("a", "c")])


def test_duplicate_edge_in_either_orientation_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report(details, [("a", "b"), ("a", "b")])
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report(details, [("a", "b"), ("b", "a")])


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
        (0, "a", "x", 0.0, 0.0, 0.0, 0.0, 0.0),  # non-number
    ],
)
def test_invalid_rows_raise_value_error(row):
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report([row], [])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 1.0)
    with pytest.raises(ValueError):
        effect_matrix_local_moran_report([row, row], [])
