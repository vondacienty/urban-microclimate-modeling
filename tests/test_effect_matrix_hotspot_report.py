"""Tests for urban_micro.effect_matrix_hotspot_report."""

import itertools
import json
import math

import pytest

from urban_micro import effect_matrix_hotspot_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _cells(report):
    return [(group["key"], cell) for group in json.loads(report)["groups"]
            for cell in group["cells"]]


def _bruteforce(details, neighbors, *, minutes=60, alpha=0.05):
    """Reference bucketed local-Moran p-values, BH q-values and kinds."""
    buckets = {}
    for t, c, delta in ((r[0], r[1], r[4]) for r in details):
        bucket = (t // (minutes * 60)) * (minutes * 60)
        buckets.setdefault(bucket, {}).setdefault(c, [0.0, 0])
        buckets[bucket][c][0] += delta
        buckets[bucket][c][1] += 1

    records = []
    for bucket in sorted(buckets):
        cell_ids = sorted(buckets[bucket])
        n = len(cell_ids)
        d = [buckets[bucket][c][0] / buckets[bucket][c][1] for c in cell_ids]
        position = {c: i for i, c in enumerate(cell_ids)}
        adjacency = [set() for _ in cell_ids]
        for a, b in neighbors:
            if a in position and b in position:
                adjacency[position[a]].add(position[b])
                adjacency[position[b]].add(position[a])
        mean = sum(d) / n
        x = [value - mean for value in d]
        sxx = sum(value * value for value in x)
        neighbor_sums = [sum(x[j] for j in adj) for adj in adjacency]

        def locals_of(values):
            deviations = [value - sum(values) / n for value in values]
            denom = sum(value * value for value in deviations)
            return [
                n * deviations[i] * sum(deviations[j] for j in adjacency[i]) / denom
                for i in range(n)
            ]

        observed = locals_of(d)
        p_values = []
        for i in range(n):
            if sxx == 0 or not adjacency[i]:
                p_values.append(1.0)
                continue
            hits = 0
            for perm in itertools.permutations(d):
                perm_locals = locals_of(perm)
                if abs(perm_locals[i]) >= abs(observed[i]) - 1e-12:
                    hits += 1
            p_values.append(hits / math.factorial(n))
        for i, c in enumerate(cell_ids):
            records.append((bucket, c, n, observed[i], p_values[i],
                            x[i], neighbor_sums[i]))

    total = len(records)
    ranked = sorted(range(total),
                    key=lambda i: (records[i][4], records[i][0], records[i][1]))
    q_values = [1.0] * total
    running = 1.0
    for rank in range(total, 0, -1):
        i = ranked[rank - 1]
        running = min(running, total * records[i][4] / rank)
        q_values[i] = min(1.0, running)

    expected = []
    for i, (bucket, c, n, local, p_value, x_c, v_c) in enumerate(records):
        reject = q_values[i] <= alpha
        kind = "NS"
        if reject and x_c != 0 and v_c != 0:
            kind = {
                (1, 1): "HH", (-1, -1): "LL",
                (1, -1): "HL", (-1, 1): "LH",
            }[(1 if x_c > 0 else -1, 1 if v_c > 0 else -1)]
        expected.append((bucket, c, n, local, p_value, q_values[i],
                         reject, kind))
    return expected


def test_empty_details_gives_empty_groups():
    report = effect_matrix_hotspot_report([], [])
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "alpha": 0.05, "groups": []}


def test_empty_with_explicit_minutes_and_alpha():
    report = effect_matrix_hotspot_report([], [], minutes=30, alpha=0.1)
    assert report == '{"minutes":30,"alpha":0.100000,"groups":[]}'


def test_three_cell_path_matches_bruteforce():
    # d = 1, 2, 6 on a path a-b-c: L_a = 3/7, L_b = -3/14, L_c = -9/14
    # and the p-values are 1, 1 and 2/3; all BH q-values are 1, so nothing
    # is rejected at alpha 0.05 and every cell is NS.
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0), _row(0, "c", 6.0)]
    neighbors = [("a", "b"), ("b", "c")]
    report = effect_matrix_hotspot_report(details, neighbors)
    assert report == (
        '{"minutes":60,"alpha":0.050000,"groups":['
        '{"key":0,"n":3,"cells":['
        '{"key":"a","local":0.428571,"p":1.000000,"q":1.000000,'
        '"reject":false,"kind":"NS"},'
        '{"key":"b","local":-0.214286,"p":1.000000,"q":1.000000,'
        '"reject":false,"kind":"NS"},'
        '{"key":"c","local":-0.642857,"p":0.666667,"q":1.000000,'
        '"reject":false,"kind":"NS"}]}]}'
    )
    expected = _bruteforce(details, neighbors)
    for (bucket, cell), (ebucket, ecell, _, local, p_value, q_value,
                         reject, kind) in zip(_cells(report), expected):
        assert bucket == ebucket and cell["key"] == ecell
        assert cell["local"] == pytest.approx(local, abs=1e-6)
        assert cell["p"] == pytest.approx(p_value, abs=1e-6)
        assert cell["q"] == pytest.approx(q_value, abs=1e-6)
        assert cell["reject"] is reject
        assert cell["kind"] == kind


def test_alpha_one_classifies_hotspot_kinds():
    # Every q-value is 1, so alpha = 1 rejects every cell; x = (-2, -1, 3)
    # and neighbor sums v = (-1, 1, -1) give LL, LH and HL.
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0), _row(0, "c", 6.0)]
    neighbors = [("a", "b"), ("b", "c")]
    cells = _cells(effect_matrix_hotspot_report(
        details, neighbors, alpha=1.0))
    assert [cell["kind"] for _, cell in cells] == ["LL", "LH", "HL"]
    assert all(cell["reject"] for _, cell in cells)


def test_zero_neighbor_sum_is_ns_even_when_rejected():
    # d = (1, -1, -1, 1) on path a-b-c-d has x = d: the inner cells b and c
    # each have neighbor sums x_a + x_c = 0 and x_b + x_d = 0, so they are
    # NS despite rejection at alpha 1; a and d are HL.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", -1.0),
        _row(0, "c", -1.0),
        _row(0, "d", 1.0),
    ]
    neighbors = [("a", "b"), ("b", "c"), ("c", "d")]
    cells = _cells(effect_matrix_hotspot_report(
        details, neighbors, alpha=1.0))
    assert [cell["kind"] for _, cell in cells] == ["HL", "NS", "NS", "HL"]
    inner = [cell for _, cell in cells[1:3]]
    assert all(cell["local"] == 0.0 for cell in inner)


def test_zero_sum_of_squares_is_ns_even_at_alpha_one():
    details = [_row(0, "a", 2.0), _row(0, "b", 2.0), _row(0, "c", 2.0)]
    cells = _cells(effect_matrix_hotspot_report(
        details, [("a", "b"), ("b", "c")], alpha=1.0))
    for _, cell in cells:
        assert cell["local"] == 0.0
        assert cell["p"] == 1.0
        assert cell["q"] == 1.0
        assert cell["reject"] is True
        assert cell["kind"] == "NS"


def test_bh_runs_across_all_bucket_cell_cells():
    # Two buckets, three and two cells; q-values must be ranked over all
    # five bucket/cell cells jointly rather than per bucket.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(0, "c", 6.0),
        _row(3600, "a", 5.0),
        _row(3600, "b", 0.0),
    ]
    neighbors = [("a", "b"), ("b", "c")]
    report = effect_matrix_hotspot_report(details, neighbors)
    groups = json.loads(report)["groups"]
    assert [group["key"] for group in groups] == [0, 3600]
    assert [group["n"] for group in groups] == [3, 2]
    expected = _bruteforce(details, neighbors)
    for (bucket, cell), (ebucket, ecell, _, local, p_value, q_value,
                         reject, kind) in zip(_cells(report), expected):
        assert bucket == ebucket and cell["key"] == ecell
        assert cell["local"] == pytest.approx(local, abs=1e-6)
        assert cell["p"] == pytest.approx(p_value, abs=1e-6)
        assert cell["q"] == pytest.approx(q_value, abs=1e-6)
        assert cell["reject"] is reject
        assert cell["kind"] == kind


def test_rows_average_within_bucket_before_computing():
    # Bucket 0 holds means a=1 (0 and 2), b=3, c=2 -> vector [1, 3, 2].
    details = [
        _row(0, "a", 0.0),
        _row(100, "a", 2.0),
        _row(0, "b", 3.0),
        _row(0, "c", 2.0),
    ]
    neighbors = [("a", "b"), ("b", "c")]
    report = effect_matrix_hotspot_report(details, neighbors, alpha=1.0)
    expected = _bruteforce(details, neighbors, alpha=1.0)
    for (_, cell), (_, _, _, local, p_value, q_value, reject, kind) in zip(
        _cells(report), expected
    ):
        assert cell["local"] == pytest.approx(local, abs=1e-6)
        assert cell["p"] == pytest.approx(p_value, abs=1e-6)
        assert cell["q"] == pytest.approx(q_value, abs=1e-6)
        assert cell["reject"] is reject
        assert cell["kind"] == kind


def test_more_than_eight_cells_raises_value_error():
    details = [_row(0, chr(97 + i), float(i)) for i in range(9)]
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report(details, [("a", "b")])


def test_neighbor_outside_bucket_leaves_cell_inactive():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "c", 3.0),
    ]
    report = effect_matrix_hotspot_report(
        details, [("a", "c")], alpha=1.0)
    first, second = json.loads(report)["groups"]
    a_cell = first["cells"][0]
    assert a_cell["local"] == 0.0
    assert a_cell["p"] == 1.0
    assert a_cell["q"] == 1.0
    assert a_cell["reject"] is True
    assert a_cell["kind"] == "NS"  # no in-bucket neighbor
    assert second["cells"][0]["kind"] == "NS"


def test_compact_utf8_no_spaces_no_trailing_newline():
    details = [_row(0, "céll", 1.0), _row(0, "λ", 2.0)]
    report = effect_matrix_hotspot_report(details, [("céll", "λ")])
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report
    assert '"key":"λ"' in report


def test_key_orders():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    report = effect_matrix_hotspot_report(details, [("a", "b")])
    assert list(json.loads(report)) == ["minutes", "alpha", "groups"]
    group = json.loads(report)["groups"][0]
    assert list(group) == ["key", "n", "cells"]
    assert list(group["cells"][0]) == [
        "key", "local", "p", "q", "reject", "kind"
    ]


def test_negative_zero_normalized():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "c", 3.0),
    ]
    report = effect_matrix_hotspot_report(details, [("b", "c")])
    assert "-0.000000" not in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_hotspot_report("not a list", [])


def test_neighbors_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_hotspot_report([], (("a", "b"),))


@pytest.mark.parametrize("minutes", [0, 7, 1441, True, "60", 60.0])
def test_invalid_minutes_raise_value_error(minutes):
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report([], [], minutes=minutes)


@pytest.mark.parametrize("alpha", [0, -0.01, 1.1, True, False, "0.05",
                                   float("nan"), float("inf")])
def test_invalid_alpha_raises_value_error(alpha):
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report([], [], alpha=alpha)


@pytest.mark.parametrize(
    "neighbor",
    [
        ("a",),
        ["a", "b"],
        ("a", "b", "c"),
        ("", "b"),
        ("a", 1),
        ("a", "a"),
    ],
)
def test_invalid_neighbors_raise_value_error(neighbor):
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report(details, [neighbor])


def test_unknown_and_duplicate_neighbor_endpoints_raise_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report(details, [("a", "z")])
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report(details, [("a", "b"), ("b", "a")])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "a", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report(details, [])
