"""Tests for urban_micro.grid_features."""

from decimal import Decimal, localcontext

import pytest

from urban_micro import align_temp, grid_features

ALIGNED = [
    (0, "c1", 30.0, 29.0, 1.0),
    (0, "c2", 20.0, 19.0, 1.0),
    (3600, "c1", 31.0, 30.0, 1.0),
]


def test_basic_features():
    # c1: buildings 2@h10 area 40 + 2@h20 area 10 -> mean h = 600/50 = 12
    # building frac 0.5, impervious 0.3, green 0.2 (coverage sums to g)
    items = [
        ("c1", "building", 10, 40, 100),
        ("c1", "building", 20, 10, 100),
        ("c1", "impervious", 0, 30, 100),
        ("c1", "green", 0, 20, 100),
        ("c1", "other", 0, 50, 100),
    ]
    result = grid_features(ALIGNED, items)
    assert result == [
        (0, "c1", 30.0, 29.0, 1.0, 12.0, 0.5, 0.3, 0.2),
        (3600, "c1", 31.0, 30.0, 1.0, 12.0, 0.5, 0.3, 0.2),
    ]


def test_zero_area_coverage_item_still_counts():
    # c2 has a zero-area impervious item plus a full-area green one
    items = [
        ("c1", "building", 5, 10, 100),
        ("c1", "green", 0, 100, 100),
        ("c2", "building", 5, 60, 100),
        ("c2", "impervious", 0, 0, 100),
        ("c2", "green", 0, 100, 100),
    ]
    result = grid_features(ALIGNED, items)
    cells = {(r[0], r[1]) for r in result}
    assert (0, "c2") in cells


def test_cell_without_building_dropped():
    items = [
        ("c1", "impervious", 0, 60, 100),
        ("c1", "green", 0, 40, 100),
    ]
    assert grid_features(ALIGNED, items) == []


def test_cell_without_coverage_dropped():
    items = [("c1", "building", 5, 100, 100)]
    assert grid_features(ALIGNED, items) == []


def test_coverage_sum_not_equal_grid_dropped():
    items = [
        ("c1", "building", 5, 60, 100),
        ("c1", "impervious", 0, 30, 100),
    ]
    assert grid_features(ALIGNED, items) == []


def test_unknown_cell_in_aligned_skipped():
    items = [
        ("c1", "building", 5, 50, 100),
        ("c1", "green", 0, 100, 100),
    ]
    result = grid_features(ALIGNED, items)
    assert [r[1] for r in result] == ["c1", "c1"]


def test_zero_building_area_mean_height_zero():
    items = [
        ("c1", "building", 0, 0, 100),
        ("c1", "green", 0, 100, 100),
    ]
    row = grid_features(ALIGNED, items)[0]
    assert row[5] == 0.0
    assert row[6] == 0.0
    assert row[8] == 1.0


def test_sort_order_timestamp_then_cell():
    aligned = [
        (3600, "b", 1.0, 1.0, 0.0),
        (0, "b", 1.0, 1.0, 0.0),
        (0, "a", 1.0, 1.0, 0.0),
    ]
    items = []
    for cell in ("a", "b"):
        items += [
            (cell, "building", 1, 50, 100),
            (cell, "green", 0, 100, 100),
        ]
    result = grid_features(aligned, items)
    assert [(r[0], r[1]) for r in result] == [(0, "a"), (0, "b"), (3600, "b")]


def test_nine_tuple_length_and_original_fields_preserved():
    items = [
        ("c1", "building", 3, 25, 100),
        ("c1", "other", 0, 100, 100),
    ]
    row = grid_features(ALIGNED, items)[0]
    assert len(row) == 9
    assert row[:5] == (0, "c1", 30.0, 29.0, 1.0)


def test_decimal_str_semantics_no_float_noise():
    # 0.1 + 0.2 style sums must compare exactly against grid area
    items = [
        ("c1", "building", 1, 0.1, 1.0),
        ("c1", "building", 1, 0.2, 1.0),
        ("c1", "impervious", 0, 0.3, 1.0),
        ("c1", "green", 0, 0.4, 1.0),
        ("c1", "other", 0, 0.3, 1.0),
    ]
    row = grid_features(ALIGNED, items)[0]
    assert row[6] == 0.3
    assert row[7] == 0.3
    assert row[8] == 0.4


def test_fractions_round_half_even_to_four_decimals():
    # 2/12 = 0.16666... -> 0.1667; 4/12 = 0.33333... -> 0.3333; 6/12 = 0.5
    items = [
        ("c1", "building", 10, 2, 12),
        ("c1", "impervious", 0, 4, 12),
        ("c1", "green", 0, 6, 12),
        ("c1", "other", 0, 2, 12),
    ]
    row = grid_features(ALIGNED, items)[0]
    assert row[5] == 10.0
    assert row[6] == 0.1667
    assert row[7] == 0.3333
    assert row[8] == 0.5


def test_mean_height_round_half_even():
    # weighted mean exactly 12.00005 -> half-even on 4 decimals -> 12.0
    items = [
        ("c1", "building", 24.0001, 1, 100),
        ("c1", "building", 0, 1, 100),
        ("c1", "green", 0, 100, 100),
    ]
    row = grid_features(ALIGNED, items)[0]
    assert row[5] == 12.0


def test_negative_zero_normalized():
    items = [
        ("c1", "building", 0, 0, 100),
        ("c1", "green", 0, 100, 100),
    ]
    row = grid_features(ALIGNED, items)[0]
    for value in row[5:]:
        assert value == 0.0 or value == 1.0
        assert str(value) != "-0.0"


def test_independent_of_default_context_precision():
    items = [
        ("c1", "building", 1, 1, 3),
        ("c1", "impervious", 0, 1, 3),
        ("c1", "green", 0, 1, 3),
        ("c1", "other", 0, 1, 3),
    ]
    with localcontext() as ctx:
        ctx.prec = 1
        result = grid_features(ALIGNED, items)
    assert result[0][6:] == (0.3333, 0.3333, 0.3333)


def test_inconsistent_grid_area_rejected():
    items = [
        ("c1", "building", 1, 10, 100),
        ("c1", "green", 0, 100, 100.0),
    ]
    # 100 == Decimal("100.0") -> fine
    assert grid_features(ALIGNED, items)
    items[-1] = ("c1", "green", 0, 100, 99)
    with pytest.raises(ValueError):
        grid_features(ALIGNED, items)


def test_building_area_exceeds_grid_rejected():
    items = [
        ("c1", "building", 1, 60, 100),
        ("c1", "building", 1, 50, 100),
        ("c1", "green", 0, 0, 100),
    ]
    with pytest.raises(ValueError):
        grid_features(ALIGNED, items)


def test_coverage_area_exceeds_grid_rejected():
    items = [
        ("c1", "building", 1, 10, 100),
        ("c1", "impervious", 0, 60, 100),
        ("c1", "green", 0, 50, 100),
    ]
    with pytest.raises(ValueError):
        grid_features(ALIGNED, items)


def test_inputs_must_be_lists():
    with pytest.raises(TypeError):
        grid_features(tuple(ALIGNED), [])
    with pytest.raises(TypeError):
        grid_features(ALIGNED, tuple())


def test_item_must_be_five_tuple():
    for bad in [
        ("c1", "building", 1, 10),
        ("c1", "building", 1, 10, 100, 1),
        ["c1", "building", 1, 10, 100],
    ]:
        with pytest.raises(ValueError):
            grid_features(ALIGNED, [bad])


def test_cell_id_validation():
    base = ["building", 1, 10, 100]
    for bad in ("", 1, None, b"c1"):
        with pytest.raises(ValueError):
            grid_features(ALIGNED, [(bad, *base)])


def test_kind_validation():
    for bad in ("Building", "water", "", 0):
        with pytest.raises(ValueError):
            grid_features(ALIGNED, [("c1", bad, 1, 10, 100)])


def test_dimensions_must_be_finite_numbers():
    for bad in (True, False, "1", None, 1 + 0j, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            grid_features(ALIGNED, [("c1", "building", bad, 10, 100)])
        with pytest.raises(ValueError):
            grid_features(ALIGNED, [("c1", "building", 1, bad, 100)])
        with pytest.raises(ValueError):
            grid_features(ALIGNED, [("c1", "green", 0, 10, bad)])


def test_height_and_area_signs():
    with pytest.raises(ValueError):
        grid_features(ALIGNED, [("c1", "building", -1, 10, 100)])
    with pytest.raises(ValueError):
        grid_features(ALIGNED, [("c1", "building", 1, -0.1, 100)])
    with pytest.raises(ValueError):
        grid_features(ALIGNED, [("c1", "green", 0, 10, 0)])
    with pytest.raises(ValueError):
        grid_features(ALIGNED, [("c1", "green", 0, 10, -5)])


def test_non_building_height_must_be_zero():
    for kind in ("impervious", "green", "other"):
        with pytest.raises(ValueError):
            grid_features(ALIGNED, [(("c1"), kind, 0.1, 10, 100)])


def test_int_and_float_mix_decimal_equal():
    items = [
        ("c1", "building", 10, 25, 100.0),
        ("c1", "impervious", 0, 25.0, 100),
        ("c1", "green", 0, 50, 100),
        ("c1", "other", 0, 25, 100),
    ]
    result = grid_features(ALIGNED, items)
    assert result[0][6:] == (0.25, 0.25, 0.5)


def test_empty_inputs():
    assert grid_features([], []) == []


def test_works_with_align_temp_output():
    stations = [
        {"station_id": "s1", "timestamp": 0, "temp_c": 30.0},
    ]
    satellite = [
        {"cell_id": "c1", "timestamp": 0, "lst_c": 29.0},
    ]
    aligned = align_temp(stations, satellite, {"s1": "c1"}, min_count=1)
    items = [
        ("c1", "building", 10, 40, 100),
        ("c1", "impervious", 0, 100, 100),
    ]
    result = grid_features(aligned, items)
    assert result == [(0, "c1", 30.0, 29.0, 1.0, 10.0, 0.4, 1.0, 0.0)]
