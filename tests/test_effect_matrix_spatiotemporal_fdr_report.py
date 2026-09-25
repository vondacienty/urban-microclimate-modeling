"""Tests for urban_micro.effect_matrix_spatiotemporal_fdr_report."""

import itertools
import json
from fractions import Fraction

import pytest

from urban_micro import effect_matrix_spatiotemporal_fdr_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _bruteforce(x_values):
    """Reference mean/p in Fraction for the paired differences ``x``."""
    values = [Fraction(value) for value in x_values]
    n = len(values)
    mean = sum(values, Fraction(0)) / n
    hits = 0
    for signs in itertools.product([1, -1], repeat=n):
        signed = sum(sign * value for sign, value in zip(signs, values))
        if abs(signed / n) >= abs(mean):
            hits += 1
    return mean, Fraction(hits, 1 << n)


def _bh(records):
    """Reference BH q-values: records is {bucket: (p,)} -> {bucket: q}."""
    count = len(records)
    ranked = sorted(records, key=lambda bucket: (records[bucket], bucket))
    q_values = {}
    running = Fraction(1)
    for rank in range(count, 0, -1):
        bucket = ranked[rank - 1]
        running = min(running, Fraction(count) * records[bucket] / rank)
        q_values[bucket] = running
    return q_values


def _groups(report):
    return json.loads(report)["groups"]


def test_empty_details_gives_empty_groups():
    report = effect_matrix_spatiotemporal_fdr_report([], [])
    assert report == '{"minutes":60,"lag":1,"alpha":0.050000,"groups":[]}'
    assert json.loads(report) == {
        "minutes": 60,
        "lag": 1,
        "alpha": 0.05,
        "groups": [],
    }


def test_single_pair_statistics():
    # Da=2, Db=6, x=-4: n=1 -> every sign vector reaches |mean| so p=1.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 4.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 10.0),
    ]
    group = _groups(
        effect_matrix_spatiotemporal_fdr_report(details, [("a", "b")])
    )[0]
    assert group == {
        "key": 3600,
        "n": 1,
        "mean": -4.0,
        "p": 1.0,
        "q": 1.0,
        "reject": False,
    }


def test_multiple_buckets_match_bruteforce_and_bh():
    data = {
        0: {"a": 1, "b": 4, "c": 0},
        3600: {"a": 3, "b": 10, "c": 2},
        7200: {"a": 6, "b": -1, "c": 5},
        10800: {"a": 2, "b": 2, "c": 2},
    }
    neighbors = [("a", "b"), ("a", "c"), ("b", "c")]
    details = [
        _row(bucket, cell, value)
        for bucket, cells in data.items()
        for cell, value in cells.items()
    ]
    expected = {}
    for bucket in sorted(data):
        previous = data.get(bucket - 3600)
        if previous is None:
            continue
        xs = [
            (data[bucket][a] - previous[a]) - (data[bucket][b] - previous[b])
            for a, b in neighbors
        ]
        expected[bucket] = _bruteforce(xs)
    q_values = _bh({bucket: p for bucket, (_, p) in expected.items()})

    groups = _groups(
        effect_matrix_spatiotemporal_fdr_report(details, neighbors)
    )
    assert [group["key"] for group in groups] == sorted(expected)
    for group in groups:
        bucket = group["key"]
        mean, p_value = expected[bucket]
        assert group["n"] == 3
        assert group["mean"] == pytest.approx(float(mean), abs=5e-7)
        assert group["p"] == pytest.approx(float(p_value), abs=5e-7)
        assert group["q"] == pytest.approx(float(q_values[bucket]), abs=5e-7)


def test_bh_ranks_by_p_then_bucket_and_maps_back():
    # Two buckets, single edge:
    # bucket 3600: x = 0 -> p = 1
    # bucket 7200: x = 4 -> n=1 -> p = 1 as well; ties broken by bucket.
    details = [
        _row(0, "a", 1.0), _row(0, "b", 1.0),
        _row(3600, "a", 3.0), _row(3600, "b", 3.0),
        _row(7200, "a", 7.0), _row(7200, "b", 3.0),
    ]
    groups = _groups(
        effect_matrix_spatiotemporal_fdr_report(details, [("a", "b")])
    )
    assert [group["key"] for group in groups] == [3600, 7200]
    assert [group["p"] for group in groups] == [1.0, 1.0]
    assert [group["q"] for group in groups] == [1.0, 1.0]


def test_reject_compared_on_unquantized_q():
    # Deltas (a, b, c) = (10, 2, 1) give edge pairings x = (8, 9, 1); with
    # total 18 only the two all-same-sign vectors reach |sum| >= 18, so
    # p = 2/8 = 0.25; N = 1 hence q = 0.25. alpha = 0.25 exactly must
    # reject (comparison on the unquantized values).
    details = [
        _row(0, "a", 0.0), _row(0, "b", 0.0), _row(0, "c", 0.0),
        _row(3600, "a", 10.0), _row(3600, "b", 2.0), _row(3600, "c", 1.0),
    ]
    neighbors = [("a", "b"), ("a", "c"), ("b", "c")]
    report = effect_matrix_spatiotemporal_fdr_report(
        details, neighbors, alpha=0.25
    )
    group = _groups(report)[0]
    assert group["mean"] == 6.0
    assert group["p"] == 0.25
    assert group["q"] == 0.25
    assert group["reject"] is True
    assert '"reject":true' in report
    below = _groups(
        effect_matrix_spatiotemporal_fdr_report(details, neighbors, alpha=0.2)
    )[0]
    assert below["reject"] is False


def test_rows_average_within_bucket_cell_before_diff():
    details = [
        _row(0, "a", 1.0),
        _row(100, "a", 3.0),
        _row(0, "b", 4.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 10.0),
    ]
    group = _groups(
        effect_matrix_spatiotemporal_fdr_report(details, [("a", "b")])
    )[0]
    assert group["mean"] == -5.0


def test_lag_two_uses_bucket_two_steps_back():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 2.0),
        _row(3600, "b", 5.0),
        _row(7200, "a", 10.0),
        _row(7200, "b", 10.0),
    ]
    report = effect_matrix_spatiotemporal_fdr_report(
        details, [("a", "b")], lag=2
    )
    groups = _groups(report)
    assert [group["key"] for group in groups] == [7200]
    assert groups[0]["mean"] == 1.0
    assert '"lag":2' in report


def test_minutes_changes_bucket_size_and_offset():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 4.0),
        _row(1800, "a", 3.0),
        _row(1800, "b", 10.0),
    ]
    report = effect_matrix_spatiotemporal_fdr_report(
        details, [("a", "b")], minutes=30
    )
    assert [group["key"] for group in _groups(report)] == [1800]
    assert '"minutes":30' in report


def test_buckets_without_pairings_are_omitted():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 3.0),
    ]
    report = effect_matrix_spatiotemporal_fdr_report(details, [("a", "b")])
    assert _groups(report) == []


def test_groups_in_ascending_bucket_order():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 4.0),
        _row(7200, "a", 6.0),
        _row(7200, "b", -1.0),
    ]
    groups = _groups(
        effect_matrix_spatiotemporal_fdr_report(details, [("a", "b")])
    )
    assert [group["key"] for group in groups] == [3600, 7200]


def test_sixteen_pairings_are_allowed():
    cells = [f"c{i}" for i in range(17)]
    details = [_row(0, cell, 0.0) for cell in cells]
    details += [_row(3600, cell, 1.0) for cell in cells]
    neighbors = [("c0", cell) for cell in cells[1:]]
    assert _groups(
        effect_matrix_spatiotemporal_fdr_report(details, neighbors)
    )[0]["n"] == 16


def test_more_than_sixteen_pairings_raises_value_error():
    cells = [f"c{i}" for i in range(18)]
    details = [_row(0, cell, 0.0) for cell in cells]
    details += [_row(3600, cell, 1.0) for cell in cells]
    neighbors = [("c0", cell) for cell in cells[1:]]
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_fdr_report(details, neighbors)


def test_negative_zero_normalized():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 1.0),
        _row(3600, "a", 1.0),
        _row(3600, "b", 1.0),
    ]
    report = effect_matrix_spatiotemporal_fdr_report(details, [("a", "b")])
    assert "-0.000000" not in report
    assert '"mean":0.000000' in report


def test_decimal_inputs_are_used_exactly():
    details = [
        _row(0, "a", 0.1),
        _row(0, "b", 0.2),
        _row(3600, "a", 0.3),
        _row(3600, "b", 0.5),
    ]
    group = _groups(
        effect_matrix_spatiotemporal_fdr_report(details, [("a", "b")])
    )[0]
    # Da=0.2, Db=0.3, x=-0.1 exactly via Decimal(str(...)).
    assert group["mean"] == pytest.approx(-0.1, abs=1e-12)
    assert group["p"] == 1.0
    assert group["q"] == 1.0


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_matrix_spatiotemporal_fdr_report(
        [
            _row(0, "a", 1.0),
            _row(0, "b", 2.0),
            _row(3600, "a", 3.0),
            _row(3600, "b", 4.0),
        ],
        [("a", "b")],
    )
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":3600' in report


def test_key_orders():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 4.0),
    ]
    report = effect_matrix_spatiotemporal_fdr_report(
        details, [("a", "b")], alpha=0.1
    )
    assert list(json.loads(report)) == ["minutes", "lag", "alpha", "groups"]
    assert list(_groups(report)[0]) == ["key", "n", "mean", "p", "q", "reject"]
    assert '"alpha":0.100000' in report


def test_alpha_one_rejects():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 4.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 10.0),
    ]
    report = effect_matrix_spatiotemporal_fdr_report(
        details, [("a", "b")], alpha=1
    )
    assert '"alpha":1.000000' in report
    assert _groups(report)[0]["reject"] is True


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_spatiotemporal_fdr_report("not a list", [])


def test_neighbors_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_spatiotemporal_fdr_report([], (("a", "b"),))


@pytest.mark.parametrize("minutes", [0, 7, 1441, True, "60", 60.0])
def test_invalid_minutes_raise_value_error(minutes):
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_fdr_report([], [], minutes=minutes)


@pytest.mark.parametrize("lag", [0, -1, True, 1.0, "1"])
def test_invalid_lag_raises_value_error(lag):
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_fdr_report([], [], lag=lag)


@pytest.mark.parametrize(
    "alpha",
    [0.0, -0.01, 1.5, True, float("inf"), float("nan"), "0.05"],
)
def test_invalid_alpha_raises_value_error(alpha):
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_fdr_report([], [], alpha=alpha)


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
        effect_matrix_spatiotemporal_fdr_report(details, [neighbor])


def test_unknown_endpoint_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_fdr_report(details, [("a", "c")])


def test_duplicate_edge_in_either_orientation_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_fdr_report(
            details, [("a", "b"), ("a", "b")]
        )
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_fdr_report(
            details, [("a", "b"), ("b", "a")]
        )


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
        effect_matrix_spatiotemporal_fdr_report([row], [])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 1.0)
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_fdr_report([row, row], [])
