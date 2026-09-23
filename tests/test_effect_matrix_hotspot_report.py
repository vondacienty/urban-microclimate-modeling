"""Tests for urban_micro.effect_matrix_hotspot_report."""

import itertools
import json
import random
from fractions import Fraction

import pytest

from urban_micro import effect_matrix_hotspot_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _group(report, index=0):
    return json.loads(report)["groups"][index]


def _reference(d, adjacency, alpha):
    """Exact local Moran values, permutation p, BH q, reject and kind.

    Everything is kept as exact ``Fraction`` values so permutation ties that
    compare equal in exact arithmetic are never split by float rounding.
    """
    n = len(d)
    values = [Fraction(value) for value in d]
    mean = sum(values, Fraction(0)) / n
    x = [value - mean for value in values]
    sxx = sum((value * value for value in x), Fraction(0))

    def locals_of(deviations):
        return [
            Fraction(n)
            * deviations[c]
            * sum((deviations[j] for j in adjacency[c]), Fraction(0))
            / sxx
            for c in range(n)
        ]

    factorial = 1
    for k in range(2, n + 1):
        factorial *= k

    if sxx == 0:
        observed = [Fraction(0)] * n
        p_values = [Fraction(1)] * n
    else:
        observed = locals_of(x)
        p_values = []
        for c in range(n):
            if not adjacency[c]:
                p_values.append(Fraction(1))
            else:
                hits = sum(
                    1
                    for perm in itertools.permutations(x)
                    if abs(locals_of(perm)[c]) >= abs(observed[c])
                )
                p_values.append(Fraction(hits, factorial))

    order = sorted(range(n), key=lambda c: (p_values[c], c))
    q_values = [Fraction(1)] * n
    running = Fraction(1)
    for rank in range(n, 0, -1):
        c = order[rank - 1]
        running = min(running, Fraction(n) * p_values[c] / rank)
        q_values[c] = min(Fraction(1), running)

    alpha_value = Fraction(str(alpha))
    expected = []
    for c in range(n):
        reject = q_values[c] <= alpha_value
        xc, vc = x[c], sum((x[j] for j in adjacency[c]), Fraction(0))
        if reject and xc != 0 and vc != 0:
            if xc > 0:
                kind = "HH" if vc > 0 else "HL"
            else:
                kind = "LH" if vc > 0 else "LL"
        else:
            kind = "NS"
        expected.append(
            (float(observed[c]), float(p_values[c]), float(q_values[c]), reject, kind)
        )
    return expected


def test_empty_details_gives_empty_groups():
    report = effect_matrix_hotspot_report([], [])
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "alpha": 0.05, "groups": []}


def test_three_cell_path_statistics_p_and_bh_q():
    # d = 1, 2, 6 on a path a-b-c: locals 3/7, -3/14, -9/14; only c has a
    # non-trivial p (2/3). BH over N = 3 lifts every q to 1.
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


def test_matches_reference_on_random_graphs():
    rng = random.Random(1234)
    for _ in range(60):
        n = rng.randint(2, 6)
        cells = [chr(97 + i) for i in range(n)]
        values = [rng.choice([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]) for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    edges.append((cells[i], cells[j]))
        adjacency = [[] for _ in range(n)]
        for a, b in edges:
            i, j = ord(a) - 97, ord(b) - 97
            adjacency[i].append(j)
            adjacency[j].append(i)
        alpha = rng.choice([0.01, 0.05, 0.2, 0.5])

        details = [_row(0, c, value) for c, value in zip(cells, values)]
        cells_out = _group(
            effect_matrix_hotspot_report(details, edges, alpha=alpha)
        )["cells"]
        expected = _reference(values, adjacency, alpha)
        for cell, (local, p_value, q_value, reject, kind) in zip(cells_out, expected):
            assert cell["local"] == pytest.approx(local, abs=1e-6)
            assert cell["p"] == pytest.approx(p_value, abs=1e-6)
            assert cell["q"] == pytest.approx(q_value, abs=1e-6)
            assert cell["reject"] is reject
            assert cell["kind"] == kind


def test_classification_quadrants_when_all_rejected():
    # d = 10, -10, 10, -10 (mean 0) on a path a-b-c-d:
    # a (+, lag -10) HL; b (-, lag +20) LH; c (+, lag -20) HL; d (-, lag +10) LH.
    details = [_row(0, c, value) for c, value in zip("abcd", [10, -10, 10, -10])]
    neighbors = [("a", "b"), ("b", "c"), ("c", "d")]
    cells = _group(effect_matrix_hotspot_report(details, neighbors, alpha=1.0))["cells"]
    assert [(c["reject"], c["kind"]) for c in cells] == [
        (True, "HL"),
        (True, "LH"),
        (True, "HL"),
        (True, "LH"),
    ]


def test_hh_and_ll_clusters():
    # Two disjoint pairs: a,b both high (HH), c,d both low (LL), mean 0.
    details = [_row(0, c, value) for c, value in zip("abcd", [10, 10, -10, -10])]
    neighbors = [("a", "b"), ("c", "d")]
    cells = _group(effect_matrix_hotspot_report(details, neighbors, alpha=1.0))["cells"]
    assert [c["kind"] for c in cells] == ["HH", "HH", "LL", "LL"]


def test_bh_rejection_threshold_on_eight_cell_star():
    # Star centered on a (value 0) with leaves 1..7. Only leaf h reaches an
    # extreme local statistic: its permutation p is 1/28 and BH maps it to
    # q = 8 * (1/28) / 1 = 2/7. The center a has p = 1/4, q >= 1/2. Hence at
    # alpha = 0.2 nobody rejects, while at alpha = 0.3 only h rejects.
    details = [_row(0, c, float(index)) for index, c in enumerate("abcdefgh")]
    neighbors = [("a", b) for b in "bcdefgh"]
    cells_02 = _group(
        effect_matrix_hotspot_report(details, neighbors, alpha=0.2)
    )["cells"]
    by_key_02 = {c["key"]: c for c in cells_02}
    assert by_key_02["h"]["q"] == pytest.approx(2 / 7, abs=1e-6)
    assert all(not c["reject"] for c in cells_02)

    cells_03 = _group(
        effect_matrix_hotspot_report(details, neighbors, alpha=0.3)
    )["cells"]
    by_key_03 = {c["key"]: c for c in cells_03}
    rejected = {c["key"] for c in cells_03 if c["reject"]}
    assert rejected == {"h"}
    # h is below its neighbors (lag positive) -> HL.
    assert by_key_03["h"]["kind"] == "HL"
    assert by_key_03["a"]["reject"] is False
    assert by_key_03["a"]["kind"] == "NS"


def test_bh_q_is_computed_independently_per_bucket():
    # Two buckets each hold their own N; c in bucket 0 with p = 2/3 gets
    # q = 1 over N = 3 regardless of the other bucket.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(0, "c", 6.0),
        _row(3600, "a", 5.0),
        _row(3600, "b", -1.0),
    ]
    report = effect_matrix_hotspot_report(details, [("a", "b"), ("b", "c")])
    groups = json.loads(report)["groups"]
    assert [group["key"] for group in groups] == [0, 3600]
    assert [group["n"] for group in groups] == [3, 2]
    assert all(not cell["reject"] for group in groups for cell in group["cells"])


def test_rows_average_within_bucket_before_computing():
    details = [
        _row(0, "a", 0.0),
        _row(100, "a", 2.0),
        _row(0, "b", 3.0),
        _row(0, "c", 2.0),
    ]
    neighbors = [("a", "b"), ("b", "c")]
    cells = _group(effect_matrix_hotspot_report(details, neighbors, alpha=1.0))[
        "cells"
    ]
    expected = _reference([1.0, 3.0, 2.0], [[1], [0, 2], [1]], 1.0)
    for cell, (local, p_value, q_value, reject, kind) in zip(cells, expected):
        assert cell["local"] == pytest.approx(local, abs=1e-6)
        assert cell["p"] == pytest.approx(p_value, abs=1e-6)
        assert cell["q"] == pytest.approx(q_value, abs=1e-6)
        assert cell["reject"] is reject
        assert cell["kind"] == kind


def test_isolated_and_zero_sxx_cells_are_ns():
    # In bucket 0, a's only neighbor c is absent (different bucket), so a is
    # isolated there: local 0, p 1, q 1 and NS. c is likewise isolated in its
    # own bucket.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "c", 3.0),
    ]
    report = effect_matrix_hotspot_report(details, [("a", "c")])
    groups = json.loads(report)["groups"]
    assert groups[0]["cells"][0] == {
        "key": "a",
        "local": 0.0,
        "p": 1.0,
        "q": 1.0,
        "reject": False,
        "kind": "NS",
    }
    assert groups[1]["cells"][0] == {
        "key": "c",
        "local": 0.0,
        "p": 1.0,
        "q": 1.0,
        "reject": False,
        "kind": "NS",
    }

    # Constant vector -> S = 0: every cell is local 0, p/q 1 and NS.
    constant = [_row(0, c, 2.0) for c in "abc"]
    cells = _group(
        effect_matrix_hotspot_report(constant, [("a", "b"), ("b", "c")])
    )["cells"]
    assert [
        (cell["local"], cell["p"], cell["q"], cell["reject"], cell["kind"])
        for cell in cells
    ] == [(0.0, 1.0, 1.0, False, "NS")] * 3


def test_zero_lag_stays_ns_even_when_rejected():
    # A rejected cell whose neighbor lag is exactly 0 must still read NS.
    # The bridging cells b and c have lag 0; with alpha = 1 every q rejects.
    details = [_row(0, c, value) for c, value in zip("abcd", [10, 10, -10, -10])]
    neighbors = [("a", "b"), ("b", "c"), ("c", "d")]
    cells = _group(effect_matrix_hotspot_report(details, neighbors, alpha=1.0))["cells"]
    assert {c["key"]: (c["reject"], c["kind"]) for c in cells} == {
        "a": (True, "HH"),
        "b": (True, "NS"),
        "c": (True, "NS"),
        "d": (True, "LL"),
    }


def test_eight_cells_allowed_nine_raises():
    details = [_row(0, c, float(index)) for index, c in enumerate("abcdefgh")]
    assert _group(effect_matrix_hotspot_report(details, [("a", "b")]))["n"] == 8

    nine = [_row(0, chr(97 + i), float(i)) for i in range(9)]
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report(nine, [("a", "b")])


def test_compact_utf8_no_spaces_no_trailing_newline():
    details = [_row(0, "céll", 1.0), _row(0, "λ", 2.0)]
    report = effect_matrix_hotspot_report(details, [("céll", "λ")])
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report
    assert '"key":"λ"' in report
    assert "-0.000000" not in report


def test_key_orders():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    report = effect_matrix_hotspot_report(details, [("a", "b")], minutes=30)
    assert list(json.loads(report)) == ["minutes", "alpha", "groups"]
    group = json.loads(report)["groups"][0]
    assert list(group) == ["key", "n", "cells"]
    assert list(group["cells"][0]) == ["key", "local", "p", "q", "reject", "kind"]


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


@pytest.mark.parametrize("alpha", [0.0, -0.05, 1.01, True, "0.05", float("nan"), float("inf")])
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


def test_unknown_and_duplicate_endpoints_raise_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report(details, [("a", "c")])
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report(details, [("a", "b"), ("b", "a")])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 1.0)
    with pytest.raises(ValueError):
        effect_matrix_hotspot_report([row, row], [])
