"""Tests for urban_micro.attribute_effects."""

import math

import pytest

from urban_micro import attribute_effects, scenario


def _row(t, c, base=0.0, post=0.0, delta=0.0, cg=0.0, cr=0.0, cm=0.0):
    return (t, c, base, post, delta, cg, cr, cm)


def test_empty_details_returns_empty_list():
    assert attribute_effects([], by="cell") == []
    assert attribute_effects([], by="time") == []


def test_cell_grouping_means_and_counts():
    details = [
        _row(0, "a", base=1.0, post=3.0, delta=2.0, cg=1.0),
        _row(60, "b", base=10.0, post=20.0, delta=10.0),
        _row(120, "a", base=3.0, post=5.0, delta=2.0, cg=3.0),
    ]
    result = attribute_effects(details, by="cell")
    assert result == [
        ("a", 2, 2.0, 4.0, 2.0, 2.0, 0.0, 0.0, 0.5),
        ("b", 1, 10.0, 20.0, 10.0, 0.0, 0.0, 0.0, 1.0),
    ]


def test_time_grouping_floors_to_minute_buckets():
    details = [
        _row(0, "a", base=1.0),
        _row(3599, "b", base=3.0),
        _row(3600, "c", base=9.0),
    ]
    result = attribute_effects(details, by="time", minutes=60)
    assert [(row[0], row[1], row[2]) for row in result] == [
        (0, 2, 2.0),
        (3600, 1, 9.0),
    ]


def test_time_grouping_custom_minutes():
    # 15-minute buckets: 0..899 -> 0; 1440 // 900 == 1 -> 900; 2339 -> 1800
    details = [_row(0, "a"), _row(1440, "b"), _row(2339, "c")]
    result = attribute_effects(details, by="time", minutes=15)
    assert [row[0] for row in result] == [0, 900, 1800]


def test_results_sorted_by_key():
    details = [_row(0, "c"), _row(0, "a"), _row(0, "b")]
    result = attribute_effects(details, by="cell")
    assert [row[0] for row in result] == ["a", "b", "c"]


def test_p_value_all_zero_deltas_is_one():
    details = [_row(0, "a", delta=0.0), _row(1, "a", delta=-0.0)]
    result = attribute_effects(details, by="cell")
    assert result[0][8] == 1.0
    # zeros still counted in n
    assert result[0][1] == 2


@pytest.mark.parametrize(
    "r,s,expected",
    [
        (1, 0, 1.0),        # 2*1/2 = 1
        (2, 0, 0.5),        # 2*1/4
        (3, 0, 0.25),       # 2*1/8
        (4, 0, 0.125),      # 2*1/16
        (3, 1, 0.625),      # m=4: 2*(1+4)/16
        (4, 1, 0.375),      # m=5: 2*(1+5)/32
        (5, 1, 0.21875),    # m=6: 2*(1+6)/64
        (9, 1, 0.021484),   # m=10: 2*11/1024
        (2, 2, 1.0),        # capped at 1: 2*(1+4+6)/16 > 1
    ],
)
def test_sign_test_p_values(r, s, expected):
    details = []
    for i in range(r):
        details.append(_row(2 * i, "a", delta=1.0))
    for i in range(s):
        details.append(_row(2 * i + 1, "a", delta=-1.0))
    result = attribute_effects(details, by="cell")
    assert result[0][8] == expected


def test_zeros_ignored_in_sign_count_but_kept_in_n():
    # r=2, s=1 plus two zeros -> m=3, q=1 -> p = 2*(1+3)/8 = 1
    details = [
        _row(0, "a", delta=1.0),
        _row(1, "a", delta=1.0),
        _row(2, "a", delta=-1.0),
        _row(3, "a", delta=0.0),
        _row(4, "a", delta=0.0),
    ]
    result = attribute_effects(details, by="cell")
    assert result[0][1] == 5
    assert result[0][8] == 1.0


def test_zero_deltas_group_only_p_one():
    details = [_row(0, "a"), _row(1, "b")]
    result = attribute_effects(details, by="time", minutes=60)
    assert len(result) == 1
    assert result[0][8] == 1.0


def test_six_means_quantized_six_decimals_half_even():
    # 1/3 mean repeated-quantized
    details = [_row(0, "a", base=1.0), _row(1, "a", base=0.0)]
    result = attribute_effects(details, by="cell")
    assert result[0][2] == 0.5
    details2 = [_row(0, "a", cg=1.0), _row(1, "a", cg=0.0), _row(2, "a", cg=0.0)]
    result2 = attribute_effects(details2, by="cell")
    assert result2[0][5] == round(1.0 / 3.0, 6)


def test_negative_zero_normalized():
    # mean of cr values -0.0000002 and 0 quantizes to -0.0 -> normalized
    details = [_row(0, "a", cr=-0.0000002), _row(1, "a", cr=0.0)]
    result = attribute_effects(details, by="cell")
    row = result[0]
    # six means are exactly 0.0 with positive sign; p is 1.0 (zero deltas)
    for value in row[2:8]:
        assert value == 0.0
        assert math.copysign(1.0, value) == 1.0
    assert row[8] == 1.0


def test_all_returned_numbers_are_float():
    details = [_row(0, "a", base=1, post=2, delta=1, cg=1, cr=0, cm=0)]
    result = attribute_effects(details, by="cell")
    row = result[0]
    assert row[0] == "a"
    assert row[1] == 1
    assert all(isinstance(v, float) for v in row[2:])


def test_int_inputs_accepted():
    details = [_row(0, "a", base=1, post=2, delta=1, cg=1, cr=0, cm=0)]
    result = attribute_effects(details, by="time", minutes=1440)
    assert result == [(0, 1, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0, 1.0)]


def test_accepts_scenario_details_output():
    model = (1, 1.0, 0.0, 0.0, -1.0, 2.0, 0.5)
    rows = [
        (0, "a", 1.0, 2.0, 0.0, 0.0, 0.5, 0.6, 0.0),
        (1, "b", 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    details, _summary = scenario(
        model, rows, [("a", 0.5, 0.2, 0.1)], roof_c=0.3, material_c=0.4
    )
    result = attribute_effects(details, by="cell")
    assert [row[0] for row in result] == ["a", "b"]
    assert result[0][1] == 1


def test_details_must_be_list():
    with pytest.raises(TypeError):
        attribute_effects((_row(0, "a"),), by="cell")
    with pytest.raises(TypeError):
        attribute_effects({}, by="cell")


def test_detail_must_be_eight_tuple():
    for bad in (
        [1, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        _row(0, "a")[:7],
        _row(0, "a") + (0.0,),
        "nope",
    ):
        with pytest.raises(ValueError):
            attribute_effects([bad], by="cell")


def test_t_must_be_non_negative_non_bool_int():
    for bad in (True, False, 1.0, -1, "0", None):
        with pytest.raises(ValueError):
            attribute_effects([_row(bad, "a")], by="cell")


def test_c_must_be_non_empty_string():
    for bad in ("", 1, None, b"a"):
        with pytest.raises(ValueError):
            attribute_effects([_row(0, bad)], by="cell")


def test_numeric_fields_must_be_finite_non_bool_numbers():
    base = [0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for idx in range(2, 8):
        for bad in (True, False, "0.0", float("nan"), float("inf"),
                    float("-inf"), 1 + 0j):
            row = list(base)
            row[idx] = bad
            with pytest.raises(ValueError):
                attribute_effects([tuple(row)], by="cell")


def test_duplicate_t_c_pair_rejected():
    details = [_row(0, "a"), _row(60, "a")]
    # distinct pairs are fine even in one time bucket
    attribute_effects(details, by="time", minutes=60)
    with pytest.raises(ValueError):
        attribute_effects([_row(0, "a"), _row(0, "a")], by="cell")
    with pytest.raises(ValueError):
        attribute_effects([_row(0, "a"), _row(0, "a")], by="time")


def test_by_must_be_time_or_cell():
    details = [_row(0, "a")]
    for bad in ("TIME", "Cell", "", None, 1, ("time",)):
        with pytest.raises(ValueError):
            attribute_effects(details, by=bad)


def test_by_is_keyword_only():
    with pytest.raises(TypeError):
        attribute_effects([], "cell")  # type: ignore[misc]


def test_minutes_validation():
    details = [_row(0, "a")]
    for bad in (0, -1, 1441, 61, 100, True, False, 1.5, "60", None):
        with pytest.raises(ValueError):
            attribute_effects(details, by="time", minutes=bad)
    # divisors of 1440 are accepted
    for valid in (1, 2, 3, 5, 15, 30, 60, 120, 144, 360, 720, 1440):
        attribute_effects(details, by="time", minutes=valid)


def test_independent_of_default_context_precision():
    from decimal import getcontext

    previous = getcontext().prec
    getcontext().prec = 2
    try:
        details = [
            _row(0, "a", base=1e300, delta=1.0),
            _row(1, "a", base=3e300, delta=-1.0),
        ]
        result = attribute_effects(details, by="cell")
        assert result[0][2] == 2e300
        assert result[0][8] == 1.0
    finally:
        getcontext().prec = previous
