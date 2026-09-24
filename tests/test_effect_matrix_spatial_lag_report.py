"""Tests for urban_micro.effect_matrix_spatial_lag_report."""

import itertools
import json
from fractions import Fraction

import pytest

from urban_micro import effect_matrix_spatial_lag_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _group(report, index=0):
    return json.loads(report)["groups"][index]


def _bruteforce(d, adjacency):
    """Reference lag/local values and permutation p-values in Fraction."""
    n = len(d)
    values = [Fraction(value) for value in d]
    total = sum(values, Fraction(0))
    mean = total / n
    x = [value - mean for value in values]

    def stats_of(ordered):
        deviations = [value - mean for value in ordered]
        lags = []
        locals_ = []
        for c in range(n):
            k = len(adjacency[c])
            if k == 0:
                lag = Fraction(0)
            else:
                lag = sum((deviations[j] for j in adjacency[c]), Fraction(0)) / k
            lags.append(lag)
            locals_.append(deviations[c] * lag)
        return lags, locals_

    observed_lag, observed_local = stats_of(values)
    counts = [0] * n
    factorial = 1
    for k in range(2, n + 1):
        factorial *= k
    for perm in itertools.permutations(values):
        _, perm_locals = stats_of(perm)
        for c in range(n):
            if abs(perm_locals[c]) >= abs(observed_local[c]):
                counts[c] += 1
    return observed_lag, observed_local, [
        Fraction(count, factorial) for count in counts
    ]


def test_empty_details_gives_empty_groups():
    report = effect_matrix_spatial_lag_report([], [])
    assert report == '{"minutes":60,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "groups": []}


def test_three_cell_path_statistics_and_permutation_p():
    # d = 1, 2, 6 with edges a-b and b-c: mean 3, x = -2, -1, 3.
    # lag_a = -1, local_a = 2; lag_b = ((-2) + 3) / 2 = 1/2,
    # local_b = -1/2; lag_c = -1, local_c = -3.
    # Permutation p: 6 arrangements each.
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0), _row(0, "c", 6.0)]
    neighbors = [("a", "b"), ("b", "c")]
    report = effect_matrix_spatial_lag_report(details, neighbors)
    cells = _group(report)["cells"]
    assert cells[0]["key"] == "a"
    assert cells[0]["lag"] == pytest.approx(-1.0, abs=1e-6)
    assert cells[0]["local"] == pytest.approx(2.0, abs=1e-6)
    assert cells[1]["lag"] == pytest.approx(0.5, abs=1e-6)
    assert cells[1]["local"] == pytest.approx(-0.5, abs=1e-6)
    assert cells[2]["lag"] == pytest.approx(-1.0, abs=1e-6)
    assert cells[2]["local"] == pytest.approx(-3.0, abs=1e-6)


def test_matches_bruteforce_for_a_four_cell_graph():
    d_values = [0.5, -1.25, 2.0, 0.25]
    details = [_row(0, c, value) for c, value in zip("abcd", d_values)]
    neighbors = [("a", "b"), ("a", "c"), ("c", "d")]
    adjacency = [[1, 2], [0], [0, 3], [2]]
    expected_lag, expected_local, expected_p = _bruteforce(d_values, adjacency)
    cells = _group(effect_matrix_spatial_lag_report(details, neighbors))["cells"]
    for cell, lag, local, p_value in zip(cells, expected_lag, expected_local, expected_p):
        assert cell["lag"] == pytest.approx(float(lag), abs=1e-6)
        assert cell["local"] == pytest.approx(float(local), abs=1e-6)
        assert cell["p"] == pytest.approx(float(p_value), abs=1e-6)


def test_rows_average_within_bucket_before_computing():
    # Bucket 0: cell a holds 0 and 2 (mean 1), b holds 3, c holds 2.
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
    for cell, lag, local, p_value in zip(cells, expected_lag, expected_local, expected_p):
        assert cell["lag"] == pytest.approx(float(lag), abs=1e-6)
        assert cell["local"] == pytest.approx(float(local), abs=1e-6)
        assert cell["p"] == pytest.approx(float(p_value), abs=1e-6)


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
    assert '"minutes":30' in report


def test_isolated_cell_gets_zero_lag_local_and_p_one():
    # c is isolated in bucket 0 (its edge leaves the bucket).
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(0, "c", 7.0),
        _row(3600, "d", 3.0),
    ]
    report = effect_matrix_spatial_lag_report(details, [("a", "b"), ("c", "d")])
    first = _group(report, 0)
    cells = {cell["key"]: cell for cell in first["cells"]}
    assert cells["c"] == {"key": "c", "lag": 0.0, "local": 0.0, "p": 1.0}


def test_no_active_edges_gives_zero_and_p_one_everywhere():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0), _row(3600, "c", 3.0)]
    report = effect_matrix_spatial_lag_report(details, [("a", "c")])
    for group in json.loads(report)["groups"]:
        for cell in group["cells"]:
            assert cell == {"key": cell["key"], "lag": 0.0, "local": 0.0, "p": 1.0}


def test_constant_d_gives_zero_lag_local_and_p_one():
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
    # d = 1, 1, 2 on a path a-b-c: permutations() yields 6 arrangements
    # (the two equal 1s counted separately), and the 6-decimal p values must
    # match a brute force that counts those arrangements separately.
    details = [_row(0, "a", 1.0), _row(0, "b", 1.0), _row(0, "c", 2.0)]
    neighbors = [("a", "b"), ("b", "c")]
    expected_lag, expected_local, expected_p = _bruteforce(
        [1.0, 1.0, 2.0], [[1], [0, 2], [1]]
    )
    cells = _group(effect_matrix_spatial_lag_report(details, neighbors))["cells"]
    for cell, lag, local, p_value in zip(cells, expected_lag, expected_local, expected_p):
        assert cell["lag"] == pytest.approx(float(lag), abs=1e-6)
        assert cell["local"] == pytest.approx(float(local), abs=1e-6)
        assert cell["p"] == pytest.approx(float(p_value), abs=1e-6)
    # mean 4/3, x = -1/3, -1/3, 2/3.
    assert cells[0]["lag"] == pytest.approx(-1 / 3, abs=1e-6)
    assert cells[0]["local"] == pytest.approx(1 / 9, abs=1e-6)
    assert cells[1]["lag"] == pytest.approx(1 / 6, abs=1e-6)
    assert cells[1]["local"] == pytest.approx(-1 / 18, abs=1e-6)
    assert cells[2]["lag"] == pytest.approx(-1 / 3, abs=1e-6)
    assert cells[2]["local"] == pytest.approx(-2 / 9, abs=1e-6)


def test_eight_cells_are_allowed():
    details = [_row(0, c, float(index)) for index, c in enumerate("abcdefgh")]
    neighbors = [("a", b) for b in "bcdefgh"]
    report = effect_matrix_spatial_lag_report(details, neighbors)
    group = _group(report)
    assert group["key"] == 0
    assert len(group["cells"]) == 8


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
    assert '"local":0.000000' in report


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
        effect_matrix_spatial_lag_report([row], [])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 1.0)
    with pytest.raises(ValueError):
        effect_matrix_spatial_lag_report([row, row], [])


def test_decimal_inputs_are_used_exactly():
    # 0.1-style floats must enter via Decimal(str(x)); check against the
    # Fraction-based reference using the same decimal values.
    d_values = [0.1, 0.2, 0.3, -0.4]
    details = [_row(0, c, value) for c, value in zip("abcd", d_values)]
    neighbors = [("a", "b"), ("b", "c"), ("c", "d")]
    adjacency = [[1], [0, 2], [1, 3], [2]]
    expected_lag, expected_local, expected_p = _bruteforce(d_values, adjacency)
    cells = _group(effect_matrix_spatial_lag_report(details, neighbors))["cells"]
    for cell, lag, local, p_value in zip(cells, expected_lag, expected_local, expected_p):
        assert cell["lag"] == pytest.approx(float(lag), abs=1e-6)
        assert cell["local"] == pytest.approx(float(local), abs=1e-6)
        assert cell["p"] == pytest.approx(float(p_value), abs=1e-6)
