"""Tests for urban_micro.effect_matrix_spatial_lag_report."""

import itertools
import json

import pytest

from urban_micro import effect_matrix_spatial_lag_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _group(report, index=0):
    return json.loads(report)["groups"][index]


def _factorial(n):
    result = 1
    for k in range(2, n + 1):
        result *= k
    return result


def _bruteforce(d, adjacency):
    """Reference lag/local values and permutation p-values in float."""
    n = len(d)
    mean = sum(d) / n
    x = [value - mean for value in d]
    constant = all(value == d[0] for value in d)

    lag = [0.0] * n
    local = [0.0] * n
    for c in range(n):
        if adjacency[c] and not constant:
            lag[c] = sum(x[j] for j in adjacency[c]) / len(adjacency[c])
            local[c] = x[c] * lag[c]

    counts = [0] * n
    for perm in itertools.permutations(x):
        for c in range(n):
            if not adjacency[c] or constant:
                continue
            perm_lag = sum(perm[j] for j in adjacency[c]) / len(adjacency[c])
            perm_local = perm[c] * perm_lag
            if abs(perm_local) >= abs(local[c]):
                counts[c] += 1
    p = [
        count / _factorial(n) if (adjacency[c] and not constant) else 1.0
        for c, count in enumerate(counts)
    ]
    return lag, local, p


def test_empty_details_gives_empty_groups():
    report = effect_matrix_spatial_lag_report([], [])
    assert report == '{"minutes":60,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "groups": []}


def test_three_cell_path_statistics_and_permutation_p():
    # d = 1, 2, 6 with edges a-b and b-c: mean 3, x = -2, -1, 3.
    # lag_a = -1, local_a = 2; lag_b = (-2 + 3)/2 = 0.5, local_b = -0.5;
    # lag_c = -1, local_c = -3 (p = 4/6 = 2/3).
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0), _row(0, "c", 6.0)]
    neighbors = [("a", "b"), ("b", "c")]
    report = effect_matrix_spatial_lag_report(details, neighbors)
    assert report == (
        '{"minutes":60,"groups":['
        '{"key":0,"cells":['
        '{"key":"a","lag":-1.000000,"local":2.000000,"p":1.000000},'
        '{"key":"b","lag":0.500000,"local":-0.500000,"p":1.000000},'
        '{"key":"c","lag":-1.000000,"local":-3.000000,"p":0.666667}]}]}'
    )


def test_matches_bruteforce_for_a_four_cell_graph():
    d_values = [0.5, -1.25, 2.0, 0.25]
    details = [_row(0, c, value) for c, value in zip("abcd", d_values)]
    neighbors = [("a", "b"), ("a", "c"), ("c", "d")]
    adjacency = [[1, 2], [0], [0, 3], [2]]
    expected_lag, expected_local, expected_p = _bruteforce(d_values, adjacency)
    cells = _group(effect_matrix_spatial_lag_report(details, neighbors))["cells"]
    for cell, lag, local, p_value in zip(
        cells, expected_lag, expected_local, expected_p
    ):
        assert cell["lag"] == pytest.approx(lag, abs=1e-6)
        assert cell["local"] == pytest.approx(local, abs=1e-6)
        assert cell["p"] == pytest.approx(p_value, abs=1e-6)


def test_rows_average_within_bucket_before_computing():
    details = [
        _row(0, "a", 0.0),
        _row(100, "a", 2.0),
        _row(0, "b", 3.0),
        _row(0, "c", 2.0),
    ]
    neighbors = [("a", "b"), ("b", "c")]
    expected_lag, expected_local, expected_p = _bruteforce(
        [1.0, 3.0, 2.0], [[1], [0, 2], [1]]
    )
    cells = _group(effect_matrix_spatial_lag_report(details, neighbors))["cells"]
    for cell, lag, local, p_value in zip(
        cells, expected_lag, expected_local, expected_p
    ):
        assert cell["lag"] == pytest.approx(lag, abs=1e-6)
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
    report = effect_matrix_spatial_lag_report(details, neighbors, minutes=30)
    groups = json.loads(report)["groups"]
    assert [group["key"] for group in groups] == [0, 1800, 3600]
    assert all(len(group["cells"]) == 2 for group in groups)
    assert '"minutes":30' in report


def test_only_edges_inside_bucket_count():
    # a and b share bucket 0; c only appears in bucket 3600. The a-c edge
    # is inactive in bucket 0, so a has no in-bucket neighbor there.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "c", 3.0),
    ]
    report = effect_matrix_spatial_lag_report(details, [("a", "c")])
    first, second = json.loads(report)["groups"]
    assert first["cells"][0] == {"key": "a", "lag": 0.0, "local": 0.0, "p": 1.0}
    assert second["cells"][0] == {"key": "c", "lag": 0.0, "local": 0.0, "p": 1.0}


def test_constant_d_gives_zero_lag_local_and_p_one_even_with_edges():
    details = [_row(0, "a", 2.0), _row(0, "b", 2.0), _row(0, "c", 2.0)]
    report = effect_matrix_spatial_lag_report(
        details, [("a", "b"), ("b", "c")]
    )
    cells = _group(report)["cells"]
    assert cells == [
        {"key": "a", "lag": 0.0, "local": 0.0, "p": 1.0},
        {"key": "b", "lag": 0.0, "local": 0.0, "p": 1.0},
        {"key": "c", "lag": 0.0, "local": 0.0, "p": 1.0},
    ]


def test_duplicate_values_are_not_deduplicated_in_permutations():
    # d = 1, 1, 2 on a path a-b-c: x = -1/3, -1/3, 2/3.
    # lag_a = -1/3, local_a = 1/9; lag_b = ((-1/3)+(2/3))/2 = 1/6,
    # local_b = -1/18; lag_c = -1/3, local_c = -2/9.
    details = [_row(0, "a", 1.0), _row(0, "b", 1.0), _row(0, "c", 2.0)]
    neighbors = [("a", "b"), ("b", "c")]
    cells = _group(effect_matrix_spatial_lag_report(details, neighbors))["cells"]
    assert cells[0]["lag"] == pytest.approx(-1 / 3, abs=1e-6)
    assert cells[0]["local"] == pytest.approx(1 / 9, abs=1e-6)
    assert cells[1]["lag"] == pytest.approx(1 / 6, abs=1e-6)
    assert cells[1]["local"] == pytest.approx(-1 / 18, abs=1e-6)
    assert cells[2]["lag"] == pytest.approx(-1 / 3, abs=1e-6)
    assert cells[2]["local"] == pytest.approx(-2 / 9, abs=1e-6)
    # Enumerating all 6 permutations (duplicates counted) for cell c:
    # |x_c * x_b| >= 2/9 holds for 4 of 6.
    assert cells[2]["p"] == pytest.approx(2 / 3, abs=1e-6)


def test_more_than_eight_cells_raises_value_error():
    details = [_row(0, chr(97 + i), float(i)) for i in range(9)]
    neighbors = [("a", "b")]
    with pytest.raises(ValueError):
        effect_matrix_spatial_lag_report(details, neighbors)


def test_negative_zero_normalized():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "c", 3.0),
    ]
    report = effect_matrix_spatial_lag_report(details, [("b", "c")])
    assert "-0.000000" not in report
    assert '"lag":0.000000' in report


def test_compact_utf8_no_spaces_no_trailing_newline():
    details = [
        _row(0, "céll", 1.0),
        _row(0, "λ", 2.0),
    ]
    report = effect_matrix_spatial_lag_report(details, [("céll", "λ")])
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report
    assert '"key":"λ"' in report


def test_key_orders():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    report = effect_matrix_spatial_lag_report(
        details, [("a", "b")], minutes=30
    )
    assert list(json.loads(report)) == ["minutes", "groups"]
    group = json.loads(report)["groups"][0]
    assert list(group) == ["key", "cells"]
    assert list(group["cells"][0]) == ["key", "lag", "local", "p"]


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_spatial_lag_report("not a list", [])


def test_neighbors_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_spatial_lag_report([], (("a", "b"),))


@pytest.mark.parametrize("minutes", [0, 7, 1441, True, "60", 60.0])
def test_invalid_minutes_raise_value_error(minutes):
    with pytest.raises(ValueError):
        effect_matrix_spatial_lag_report([], [], minutes=minutes)


@pytest.mark.parametrize(
    "neighbor",
    [
        ("a",),
        ["a", "b"],
        ("a", "b", "c"),
        ("", "b"),
        ("a", ""),
        ("a", 1),
        ("a", "a"),
    ],
)
def test_invalid_neighbors_raise_value_error(neighbor):
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_spatial_lag_report(details, [neighbor])


def test_unknown_endpoint_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_spatial_lag_report(details, [("a", "c")])


def test_duplicate_edge_in_either_orientation_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_spatial_lag_report(details, [("a", "b"), ("a", "b")])
    with pytest.raises(ValueError):
        effect_matrix_spatial_lag_report(details, [("a", "b"), ("b", "a")])


@pytest.mark.parametrize(
    "row",
    [
        [0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0),
        (True, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (-1, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0, "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0, "a", 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0),
        (0, "a", 0.0, 0.0, True, 0.0, 0.0, 0.0),
        (0, "a", "x", 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_invalid_rows_raise_value_error(row):
    with pytest.raises(ValueError):
        effect_matrix_spatial_lag_report([row], [])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 1.0)
    with pytest.raises(ValueError):
        effect_matrix_spatial_lag_report([row, row], [])
