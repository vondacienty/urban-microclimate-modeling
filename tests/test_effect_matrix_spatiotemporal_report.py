"""Tests for urban_micro.effect_matrix_spatiotemporal_report."""

import json
from fractions import Fraction

import pytest

from urban_micro import effect_matrix_spatiotemporal_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _bruteforce(x_values):
    """Reference mean/p/se in Fraction for the paired differences ``x``."""
    values = [Fraction(value) for value in x_values]
    n = len(values)
    mean = sum(values, Fraction(0)) / n
    if n > 1:
        squared = sum((value - mean) ** 2 for value in values)
        se = (squared / (n * (n - 1))) ** Fraction(1, 2)
    else:
        se = Fraction(0)
    hits = 0
    for mask in range(1 << n):
        signed = sum(
            -value if (mask >> index) & 1 else value
            for index, value in enumerate(values)
        )
        if abs(signed / n) >= abs(mean):
            hits += 1
    p_value = Fraction(hits, 1 << n)
    return mean, p_value, se


def _group(report, index=0):
    return json.loads(report)["groups"][index]


def test_empty_details_gives_empty_groups():
    report = effect_matrix_spatiotemporal_report([], [])
    assert report == '{"minutes":60,"lag":1,"z":1.960000,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "lag": 1, "z": 1.96, "groups": []}


def test_single_pair_statistics():
    # Bucket 0: a=1, b=4; bucket 3600: a=3, b=10.
    # Da=2, Db=6, x=-4: n=1 -> se 0 and p=1 (every flip reaches |mean|).
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 4.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 10.0),
    ]
    group = _group(effect_matrix_spatiotemporal_report(details, [("a", "b")]))
    assert group == {
        "key": 3600,
        "n": 1,
        "mean": -4.0,
        "p": 1.0,
        "se": 0.0,
        "lower": -4.0,
        "upper": -4.0,
    }


def test_three_pairs_match_bruteforce():
    # Da=1, Db=3, Dc=-3 -> edges give x_ab=-2, x_ac=4, x_bc=6.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(0, "c", 3.0),
        _row(3600, "a", 2.0),
        _row(3600, "b", 5.0),
        _row(3600, "c", 0.0),
    ]
    neighbors = [("a", "b"), ("a", "c"), ("b", "c")]
    expected_mean, expected_p, expected_se = _bruteforce([-2, 4, 6])
    group = _group(effect_matrix_spatiotemporal_report(details, neighbors))
    assert group["n"] == 3
    assert group["mean"] == pytest.approx(float(expected_mean), abs=5e-7)
    assert group["p"] == pytest.approx(float(expected_p), abs=5e-7)
    assert group["se"] == pytest.approx(float(expected_se), abs=5e-7)
    assert group["lower"] == pytest.approx(
        float(expected_mean) - 1.96 * float(expected_se), abs=5e-7
    )
    assert group["upper"] == pytest.approx(
        float(expected_mean) + 1.96 * float(expected_se), abs=5e-7
    )


def test_rows_average_within_bucket_cell_before_diff():
    # Bucket 0: cell a holds 1 and 3 (mean 2), b=4; bucket 3600: a=3, b=10.
    # Da=1, Db=6, x=-5.
    details = [
        _row(0, "a", 1.0),
        _row(100, "a", 3.0),
        _row(0, "b", 4.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 10.0),
    ]
    group = _group(effect_matrix_spatiotemporal_report(details, [("a", "b")]))
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
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")], lag=2)
    groups = json.loads(report)["groups"]
    # Bucket 3600 has no bucket -3600, so only 7200 pairs against 0:
    # x = (10 - 1) - (10 - 2) = 1.
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
    report = effect_matrix_spatiotemporal_report(
        details, [("a", "b")], minutes=30
    )
    groups = json.loads(report)["groups"]
    assert [group["key"] for group in groups] == [1800]
    assert '"minutes":30' in report


def test_buckets_without_previous_bucket_are_omitted():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 3.0),
    ]
    # Edge a-b cannot pair at 3600 (b missing there) and bucket 0 has no
    # previous bucket, so no group is emitted.
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")])
    assert json.loads(report)["groups"] == []


def test_edge_missing_an_endpoint_at_either_bucket_is_skipped():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(0, "c", 4.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 5.0),
        # c is absent at 3600: edges touching c do not pair.
    ]
    report = effect_matrix_spatiotemporal_report(
        details, [("a", "b"), ("a", "c"), ("b", "c")]
    )
    group = _group(report)
    assert group["n"] == 1  # only a-b: x = (3-1) - (5-2) = -1
    assert group["mean"] == -1.0


def test_groups_in_ascending_bucket_order():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 4.0),
        _row(7200, "a", 6.0),
        _row(7200, "b", -1.0),
    ]
    groups = json.loads(
        effect_matrix_spatiotemporal_report(details, [("a", "b")])
    )["groups"]
    assert [group["key"] for group in groups] == [3600, 7200]


def test_edges_visit_in_lexicographic_order_regardless_of_input_order():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(0, "c", 3.0),
        _row(3600, "a", 2.0),
        _row(3600, "b", 5.0),
        _row(3600, "c", 0.0),
    ]
    canonical = [("a", "b"), ("a", "c"), ("b", "c")]
    shuffled = [("c", "b"), ("b", "a"), ("c", "a")]
    assert effect_matrix_spatiotemporal_report(
        details, shuffled
    ) == effect_matrix_spatiotemporal_report(details, canonical)


def test_sixteen_pairings_are_allowed():
    cells = [f"c{i}" for i in range(17)]
    details = [_row(0, cell, 0.0) for cell in cells]
    details += [_row(3600, cell, 1.0) for cell in cells]
    neighbors = [("c0", cell) for cell in cells[1:]]
    assert len(neighbors) == 16
    group = _group(effect_matrix_spatiotemporal_report(details, neighbors))
    assert group["n"] == 16


def test_more_than_sixteen_pairings_raises_value_error():
    cells = [f"c{i}" for i in range(18)]
    details = [_row(0, cell, 0.0) for cell in cells]
    details += [_row(3600, cell, 1.0) for cell in cells]
    neighbors = [("c0", cell) for cell in cells[1:]]
    assert len(neighbors) == 17
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report(details, neighbors)


def test_z_zero_collapses_interval():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 4.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 10.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")], z=0)
    group = _group(report)
    assert group["lower"] == group["mean"] == group["upper"]
    assert '"z":0.000000' in report


def test_negative_zero_normalized():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 1.0),
        _row(3600, "a", 1.0),
        _row(3600, "b", 1.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")])
    assert "-0.000000" not in report
    assert '"mean":0.000000' in report
    assert '"se":0.000000' in report


def test_decimal_inputs_are_used_exactly():
    details = [
        _row(0, "a", 0.1),
        _row(0, "b", 0.2),
        _row(3600, "a", 0.3),
        _row(3600, "b", 0.5),
    ]
    group = _group(effect_matrix_spatiotemporal_report(details, [("a", "b")]))
    # Da=0.2, Db=0.3, x=-0.1 exactly via Decimal(str(...)).
    assert group["mean"] == pytest.approx(-0.1, abs=1e-12)
    assert group["se"] == 0.0
    assert group["p"] == 1.0


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
    assert '"key":3600' in report


def test_key_orders():
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 2.0),
        _row(3600, "a", 3.0),
        _row(3600, "b", 4.0),
    ]
    report = effect_matrix_spatiotemporal_report(details, [("a", "b")])
    assert list(json.loads(report)) == ["minutes", "lag", "z", "groups"]
    assert list(_group(report)) == ["key", "n", "mean", "p", "se", "lower", "upper"]


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


@pytest.mark.parametrize("z_value", [-0.01, float("inf"), float("nan"), True, "1.96"])
def test_invalid_z_raises_value_error(z_value):
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report([], [], z=z_value)


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
        effect_matrix_spatiotemporal_report(details, [neighbor])


def test_unknown_endpoint_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report(details, [("a", "c")])


def test_duplicate_edge_in_either_orientation_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report(
            details, [("a", "b"), ("a", "b")]
        )
    with pytest.raises(ValueError):
        effect_matrix_spatiotemporal_report(
            details, [("a", "b"), ("b", "a")]
        )


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
