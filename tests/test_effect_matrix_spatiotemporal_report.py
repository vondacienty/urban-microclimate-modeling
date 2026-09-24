"""Tests for urban_micro.effect_matrix_spatiotemporal_report."""

import json
import math
from fractions import Fraction

import pytest

from urban_micro import effect_matrix_spatiotemporal_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _reference(details, neighbors, *, minutes=60, lag=1, z=1.96):
    """Exact Fraction reference: bucket/cell means then edge differencing.

    Returns ``{bucket: {field: value}}`` for every bucket with at least one
    contributing edge.
    """
    bucket_seconds = minutes * 60
    acc = {}
    for t, c, _base, _post, delta, *_ in details:
        bucket = (t // bucket_seconds) * bucket_seconds
        acc.setdefault((bucket, c), [Fraction(0), 0])
        acc[(bucket, c)][0] += Fraction(str(delta))
        acc[(bucket, c)][1] += 1
    means = {key: total / count for key, (total, count) in acc.items()}

    edges = sorted({tuple(sorted(edge)) for edge in neighbors})
    step = lag * bucket_seconds
    groups = {}
    for (bucket, _c) in sorted(acc):
        prev = bucket - step
        values = []
        for a, b in edges:
            if all(
                key in means
                for key in ((bucket, a), (bucket, b), (prev, a), (prev, b))
            ):
                delta_a = means[(bucket, a)] - means[(prev, a)]
                delta_b = means[(bucket, b)] - means[(prev, b)]
                values.append(delta_a - delta_b)
        if not values:
            continue
        n = len(values)
        mean = sum(values, Fraction(0)) / n
        if n > 1:
            se = math.sqrt(
                float(sum((x - mean) ** 2 for x in values) / (n * (n - 1)))
            )
        else:
            se = 0.0
        hits = 0
        for mask in range(1 << n):
            signed = sum(
                (-x if (mask >> i) & 1 else x) for i, x in enumerate(values)
            )
            if abs(signed / n) >= abs(mean):
                hits += 1
        p = Fraction(hits, 1 << n)
        zf = Fraction(str(z))
        groups[bucket] = {
            "n": n,
            "mean": float(mean),
            "p": float(p),
            "se": se,
            "lower": float(mean) - float(zf) * se,
            "upper": float(mean) + float(zf) * se,
        }
    return groups


def _groups_map(report):
    return {group["key"]: group for group in json.loads(report)["groups"]}


def test_empty_details_gives_empty_groups():
    report = effect_matrix_spatiotemporal_report([], [])
    assert report == '{"minutes":60,"lag":1,"z":1.960000,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "lag": 1, "z": 1.96, "groups": []}


def test_single_edge_two_buckets_statistics():
    # B=0: d_a=1, d_b=2 ; B=3600: d_a=4, d_b=3.
    # Da=3, Db=1, x=2; n=1 -> se=0, p=1, interval collapses onto the mean.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 4.0),
        _row(3600, "b", 3.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")])
    groups = json.loads(report)["groups"]
    assert groups == [
        {
            "key": 3600,
            "n": 1,
            "mean": 2.0,
            "p": 1.0,
            "se": 0.0,
            "lower": 2.0,
            "upper": 2.0,
        }
    ]


def test_first_bucket_without_predecessor_is_omitted():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")])
    assert json.loads(report)["groups"] == []


def test_matches_fraction_reference_on_rich_dataset():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(0, "c", 3.0),
        _row(0, "d", 4.0),
        _row(3600, "a", 2.0),
        _row(3600, "b", 1.0),
        _row(3600, "c", 5.0),
        _row(7200, "a", 0.0),
        _row(7200, "b", 3.0),
        _row(7200, "c", 2.0),
        _row(7200, "d", -1.0),
    ]
    neighbors = [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]
    report = effect_matrix_spatiotemporal_report(details, neighbors)
    groups = _groups_map(report)
    expected = _reference(details, neighbors)
    assert sorted(groups) == [3600, 7200]
    assert sorted(groups) == sorted(expected)
    for bucket, want in expected.items():
        got = groups[bucket]
        assert got["n"] == want["n"]
        assert got["mean"] == pytest.approx(want["mean"], abs=1e-6)
        assert got["p"] == pytest.approx(want["p"], abs=1e-6)
        assert got["se"] == pytest.approx(want["se"], abs=1e-6)
        assert got["lower"] == pytest.approx(want["lower"], abs=1e-6)
        assert got["upper"] == pytest.approx(want["upper"], abs=1e-6)
    # B=3600 samples are 2 (a-b) and -3 (b-c): mean -1/2.
    assert groups[3600]["n"] == 2
    assert groups[3600]["mean"] == pytest.approx(-0.5, abs=1e-12)


def test_edge_requires_both_endpoints_in_both_buckets():
    # At B=3600 cell b is missing, so the a-b edge contributes nowhere.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 4.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")])
    assert json.loads(report)["groups"] == []


def test_inactive_edge_does_not_block_other_edges_in_bucket():
    # a-c cannot pair (c absent at B=0), but a-b can.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 4.0),
        _row(3600, "b", 3.0),
        _row(3600, "c", 9.0),
    ]
    report = effect_matrix_spatiotemporal_report(
        details, [("a", "c"), ("a", "b")]
    )
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [3600]
    assert groups[0]["n"] == 1
    assert groups[0]["mean"] == pytest.approx(2.0, abs=1e-12)


def test_rows_average_within_bucket_cell_before_differencing():
    # B=0: a rows 0 and 2 -> mean 1, b=3 ; B=3600: a=4, b=1.
    details = [
        _row(0, "a", 0.0),
        _row(100, "a", 2.0),
        _row(0, "b", 3.0),
        _row(3600, "a", 4.0),
        _row(3600, "b", 1.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")])
    group = json.loads(report)["groups"][0]
    # Da = 4-1 = 3, Db = 1-3 = -2, x = 5.
    assert group["mean"] == pytest.approx(5.0, abs=1e-12)


def test_lag_two_skips_intermediate_bucket():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 5.0),
        _row(3600, "b", 5.0),
        _row(7200, "a", 3.0),
        _row(7200, "b", 0.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")], lag=2)
    groups = json.loads(report)["groups"]
    # Only B=7200 pairs with B-7200=0: Da=2, Db=-2, x=4.
    assert [g["key"] for g in groups] == [7200]
    assert groups[0]["mean"] == pytest.approx(4.0, abs=1e-12)
    assert '"lag":2' in report


def test_minutes_changes_bucket_keys_and_step():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(1800, "a", 4.0),
        _row(1800, "b", 3.0),
    ]
    report = effect_matrix_spatiotemporal_report(
        details, [("a", "b")], minutes=30
    )
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [1800]
    assert '"minutes":30' in report


def test_sign_flip_p_value_exact():
    # Three independent samples all equal to 1: mean=1; only the all-plus and
    # all-minus vectors reach |sum|=3, so p = 2/8 = 1/4.
    cells_a = ["a", "c", "e"]
    cells_b = ["b", "d", "f"]
    details = [
        _row(0, c, 0.0) for c in cells_a + cells_b
    ] + [
        _row(3600, c, 1.0) for c in cells_a
    ] + [
        _row(3600, c, 0.0) for c in cells_b
    ]
    neighbors = [("a", "b"), ("c", "d"), ("e", "f")]
    report = effect_matrix_spatiotemporal_report(details, neighbors)
    group = json.loads(report)["groups"][0]
    assert group["n"] == 3
    assert group["p"] == pytest.approx(0.25, abs=1e-12)
    assert group["mean"] == pytest.approx(1.0, abs=1e-12)


def test_sixteen_contributing_edges_allowed():
    # 17 cells form a path -> 16 edges, all present in both buckets.
    cells = [chr(97 + i) for i in range(17)]
    details = [_row(0, c, float(i)) for i, c in enumerate(cells)]
    details += [_row(3600, c, float(i + 1)) for i, c in enumerate(cells)]
    neighbors = [(cells[i], cells[i + 1]) for i in range(16)]
    report = effect_matrix_spatiotemporal_report(details, neighbors)
    group = json.loads(report)["groups"][0]
    assert group["n"] == 16


def test_more_than_sixteen_contributing_edges_raises_value_error():
    cells = [chr(97 + i) for i in range(18)]
    details = [_row(0, c, float(i)) for i, c in enumerate(cells)]
    details += [_row(3600, c, float(i + 1)) for i, c in enumerate(cells)]
    neighbors = [(cells[i], cells[i + 1]) for i in range(17)]
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report(details, neighbors)


def test_interval_uses_z():
    # n=2 samples 1 and -1: mean 0, se = sqrt((1 + 1) / (2 * 1)) = 1.
    details = [
        _row(0, "a", 0.0),
        _row(0, "b", 0.0),
        _row(0, "c", 0.0),
        _row(0, "d", 0.0),
        _row(3600, "a", 1.0),
        _row(3600, "b", 0.0),
        _row(3600, "c", -1.0),
        _row(3600, "d", 0.0),
    ]
    neighbors = [("a", "b"), ("c", "d")]
    report = effect_matrix_spatiotemporal_report(details, neighbors, z=2.0)
    group = json.loads(report)["groups"][0]
    assert group["se"] == pytest.approx(1.0, abs=1e-6)
    assert group["lower"] == pytest.approx(-2.0, abs=1e-6)
    assert group["upper"] == pytest.approx(2.0, abs=1e-6)
    assert '"z":2.000000' in report


def test_zero_z_is_allowed():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 4.0),
        _row(3600, "b", 3.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")], z=0)
    group = json.loads(report)["groups"][0]
    assert group["lower"] == group["mean"] == group["upper"]


def test_decimal_inputs_are_used_exactly():
    values_b0 = [0.1, 0.2, 0.3, 0.4]
    values_b1 = [-0.3, 0.5, 0.2, 0.1]
    details = [
        _row(0, c, value) for c, value in zip("abcd", values_b0)
    ] + [
        _row(3600, c, value) for c, value in zip("abcd", values_b1)
    ]
    neighbors = [("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")]
    report = effect_matrix_spatiotemporal_report(details, neighbors)
    groups = _groups_map(report)
    expected = _reference(details, neighbors)
    assert sorted(groups) == sorted(expected)
    for bucket, want in expected.items():
        got = groups[bucket]
        assert got["mean"] == pytest.approx(want["mean"], abs=1e-6)
        assert got["p"] == pytest.approx(want["p"], abs=1e-6)
        assert got["se"] == pytest.approx(want["se"], abs=1e-6)
        assert got["lower"] == pytest.approx(want["lower"], abs=1e-6)
        assert got["upper"] == pytest.approx(want["upper"], abs=1e-6)


def test_negative_zero_normalized():
    # Two samples cancel to mean 0; nothing may render as -0.000000.
    details = [
        _row(0, "a", 0.0),
        _row(0, "b", 0.0),
        _row(0, "c", 0.0),
        _row(0, "d", 0.0),
        _row(3600, "a", 1.0),
        _row(3600, "b", 0.0),
        _row(3600, "c", -1.0),
        _row(3600, "d", 0.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b"), ("c", "d")])
    assert "-0.000000" not in report
    assert '"mean":0.000000' in report


def test_compact_utf8_no_spaces_no_trailing_newline():
    details = [
        _row(0, "céll", 1.0),
        _row(0, "λ", 2.0),
        _row(3600, "céll", 3.0),
        _row(3600, "λ", 0.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("céll", "λ")])
    assert " " not in report
    assert not report.endswith("\n")


def test_key_orders():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 4.0),
        _row(3600, "b", 3.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")])
    assert list(json.loads(report)) == ["minutes", "lag", "z", "groups"]
    assert list(json.loads(report)["groups"][0]) == [
        "key",
        "n",
        "mean",
        "p",
        "se",
        "lower",
        "upper",
    ]


def test_groups_in_ascending_bucket_order():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 1.0),
        _row(3600, "a", 2.0),
        _row(3600, "b", 0.0),
        _row(7200, "a", 3.0),
        _row(7200, "b", 8.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")])
    assert [g["key"] for g in json.loads(report)["groups"]] == [3600, 7200]


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_spatiotemporal_report("not a list", [])


def test_neighbors_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_spatiotemporal_report([], (("a", "b"),))


@pytest.mark.parametrize("minutes", [0, 7, 1441, True, "60", 60.0])
def test_invalid_minutes_raise_value_error(minutes):
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report([], [], minutes=minutes)


@pytest.mark.parametrize("lag", [0, -1, True, 1.0, "1"])
def test_invalid_lag_raises_value_error(lag):
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report([], [], lag=lag)


@pytest.mark.parametrize("z", [-0.1, True, "1.96", float("inf"), float("nan")])
def test_invalid_z_raises_value_error(z):
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report([], [], z=z)


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
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 1.0),
        _row(3600, "b", 2.0),
    ]
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report(details, [neighbor])


def test_unknown_endpoint_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report(details, [("a", "c")])


def test_duplicate_edge_in_either_orientation_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report(details, [("a", "b"), ("a", "b")])
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report(details, [("a", "b"), ("b", "a")])


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
        effect_matrix_spatiotemporal_report([row], [])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 1.0)
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report([row, row], [])
