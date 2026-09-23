"""Tests for urban_micro.effect_matrix_theilsen_report."""

import json

import pytest

from urban_micro import effect_matrix_theilsen_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _cell(report, index=0):
    return json.loads(report)["groups"][index]


def test_empty_details_gives_empty_groups():
    report = effect_matrix_theilsen_report([])
    assert report == '{"minutes":60,"groups":[]}'
    assert json.loads(report) == {"minutes": 60, "groups": []}


def test_monotonic_three_points():
    # y = 1, 2, 3 at hourly buckets: every pair slope is 1/3600, S = 3,
    # tau = 1; of the 6 permutations only the two strictly ordered ones
    # reach |S| = 3, so p = 2/6 = 1/3.
    details = [
        _row(0, "a", 1.0),
        _row(3600, "a", 2.0),
        _row(7200, "a", 3.0),
    ]
    report = effect_matrix_theilsen_report(details)
    assert report == (
        '{"minutes":60,"groups":['
        '{"key":"a","n":3,"slope":0.000278,"tau":1.000000,"p":0.333333}]}'
    )
    cell = _cell(report)
    assert cell["slope"] == pytest.approx(1 / 3600, abs=1e-6)
    assert cell["tau"] == 1.0
    assert cell["p"] == pytest.approx(1 / 3, abs=1e-6)


def test_ties_in_y():
    # y = 1, 1, 2: pair signs 0, +1, +1 -> S = 2, tau = 2/3; pair slopes
    # are 0, 1/7200, 1/3600 whose median is 1/7200; perms with |S'| >= 2
    # are 4 of 6, p = 2/3.
    details = [_row(0, "a", 1.0), _row(3600, "a", 1.0), _row(7200, "a", 2.0)]
    cell = _cell(effect_matrix_theilsen_report(details))
    assert cell == {
        "key": "a",
        "n": 3,
        "slope": pytest.approx(1 / 7200, abs=1e-6),
        "tau": pytest.approx(2 / 3, abs=1e-6),
        "p": pytest.approx(2 / 3, abs=1e-6),
    }


def test_even_pair_count_averages_middle_slopes():
    # y = 0, 1, 1, 2 at hourly buckets; sorted pair slopes:
    # 0, 1/7200, 1/7200, 1/5400, 1/3600, 1/3600 -> median
    # (1/7200 + 1/5400) / 2 = 7/43200.
    details = [
        _row(0, "a", 0.0),
        _row(3600, "a", 1.0),
        _row(7200, "a", 1.0),
        _row(10800, "a", 2.0),
    ]
    cell = _cell(effect_matrix_theilsen_report(details))
    assert cell["n"] == 4
    assert cell["slope"] == pytest.approx(7 / 43200, abs=1e-6)
    assert cell["tau"] == pytest.approx(5 / 6, abs=1e-6)
    # Permutation S-scores of a multiset with one duplicated value: 4 of
    # the 24 permutations reach |S| = 5, so p = 1/6.
    assert cell["p"] == pytest.approx(1 / 6, abs=1e-6)


def test_strictly_decreasing_trend():
    details = [_row(0, "a", 3.0), _row(3600, "a", 2.0), _row(7200, "a", 1.0)]
    cell = _cell(effect_matrix_theilsen_report(details))
    assert cell["slope"] == pytest.approx(-1 / 3600, abs=1e-6)
    assert cell["tau"] == -1.0
    assert cell["p"] == pytest.approx(1 / 3, abs=1e-6)


def test_all_equal_y_gives_zero_slope_tau_and_p_one():
    details = [_row(3600 * i, "a", 4.0) for i in range(3)]
    cell = _cell(effect_matrix_theilsen_report(details))
    assert cell == {"key": "a", "n": 3, "slope": 0.0, "tau": 0.0, "p": 1.0}


def test_rows_average_within_bucket_before_fitting():
    # Bucket 0 holds deltas 0 and 2 -> mean 1; buckets at 3600 and 7200
    # hold y = 2 and y = 3, so the fitted points are (0, 1), (3600, 2),
    # (7200, 3): slope 1/3600, tau 1, p 1/3.
    details = [
        _row(0, "a", 0.0),
        _row(100, "a", 2.0),
        _row(3600, "a", 2.0),
        _row(7200, "a", 3.0),
    ]
    cell = _cell(effect_matrix_theilsen_report(details))
    assert cell["n"] == 3
    assert cell["slope"] == pytest.approx(1 / 3600, abs=1e-6)
    assert cell["tau"] == 1.0
    assert cell["p"] == pytest.approx(1 / 3, abs=1e-6)


def test_buckets_floor_to_minutes_interval():
    details = [
        _row(0, "a", 1.0),
        _row(1799, "a", 3.0),
        _row(1800, "a", 2.0),
    ]
    # minutes=30: bucket 0 averages (1 + 3) / 2 = 2, bucket 1800 holds 2;
    # only two occupied buckets -> omitted under the default min_points=3.
    assert effect_matrix_theilsen_report(details, minutes=30) == (
        '{"minutes":30,"groups":[]}'
    )
    # minutes=60: all three rows share bucket 0 -> a single bucket, omitted.
    assert effect_matrix_theilsen_report(details, minutes=60) == (
        '{"minutes":60,"groups":[]}'
    )


def test_cells_sorted_ascending_and_sparse_cells_omitted():
    details = [
        _row(0, "b", 0.0),
        _row(3600, "b", 1.0),
        _row(7200, "b", 2.0),
        _row(0, "a", 5.0),  # cell "a" occupies only one bucket -> omitted
        _row(0, "c", 3.0),
        _row(3600, "c", 2.0),
        _row(7200, "c", 1.0),
    ]
    report = effect_matrix_theilsen_report(details)
    groups = json.loads(report)["groups"]
    assert [group["key"] for group in groups] == ["b", "c"]


def test_min_points_threshold_includes_exact_match():
    details = [_row(3600 * i, "a", float(i)) for i in range(4)]
    assert json.loads(effect_matrix_theilsen_report(details, min_points=4))["groups"]
    assert not json.loads(
        effect_matrix_theilsen_report(details, min_points=5)
    )["groups"]


def test_more_than_eight_buckets_raises_value_error():
    details = [_row(3600 * i, "a", float(i)) for i in range(9)]
    with pytest.raises(ValueError):
        effect_matrix_theilsen_report(details)


def test_eight_buckets_are_allowed_and_permutation_p_exact():
    # Strictly monotonic y over 8 points: S = 28, tau 1 and only 2 of 8!
    # permutations reach |S'| = 28, so p = 2/40320 = 1/20160.
    details = [_row(3600 * i, "a", float(i)) for i in range(8)]
    cell = _cell(effect_matrix_theilsen_report(details))
    assert cell["n"] == 8
    assert cell["tau"] == 1.0
    assert cell["p"] == pytest.approx(1 / 20160, abs=1e-6)


def test_negative_zero_normalized():
    details = [
        _row(0, "a", 0.0),
        _row(3600, "a", 0.0),
        _row(7200, "a", -1e-9),
    ]
    report = effect_matrix_theilsen_report(details)
    assert "-0.000000" not in report
    assert '"slope":0.000000' in report


def test_compact_utf8_no_spaces_no_trailing_newline():
    details = [
        _row(0, "céll", 1.0),
        _row(3600, "céll", 2.0),
        _row(7200, "céll", 3.0),
    ]
    report = effect_matrix_theilsen_report(details)
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    details = [
        _row(0, "a", 1.0),
        _row(3600, "a", 2.0),
        _row(7200, "a", 3.0),
    ]
    report = effect_matrix_theilsen_report(details, minutes=30)
    assert list(json.loads(report)) == ["minutes", "groups"]
    assert list(json.loads(report)["groups"][0]) == ["key", "n", "slope", "tau", "p"]
    assert '"minutes":30' in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_theilsen_report("not a list")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes": 0},
        {"minutes": 7},
        {"minutes": 1441},
        {"minutes": True},
        {"minutes": "60"},
        {"min_points": 2},
        {"min_points": 0},
        {"min_points": True},
        {"min_points": 3.0},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_matrix_theilsen_report([], **kwargs)


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
        effect_matrix_theilsen_report([row])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 1.0)
    with pytest.raises(ValueError):
        effect_matrix_theilsen_report([row, row])
