"""Tests for urban_micro.grid_features."""

from decimal import getcontext

import pytest

from urban_micro import grid_features


def _row(timestamp, cell_id, station=1.0, satellite=2.0, diff=-1.0):
    return (timestamp, cell_id, station, satellite, diff)


def test_basic_features():
    aligned = [_row(0, "x")]
    items = [
        ("x", "building", 10, 40, 100),
        ("x", "building", 30, 10, 100),
        ("x", "impervious", 0, 30, 100),
        ("x", "green", 0, 20, 100),
        ("x", "other", 0, 50, 100),
    ]
    # weighted height = (10*40 + 30*10) / 50 = 14
    assert grid_features(aligned, items) == [
        (0, "x", 1.0, 2.0, -1.0, 14.0, 0.5, 0.3, 0.2)
    ]


def test_nine_tuple_length_and_passthrough_fields():
    aligned = [_row(3600, "c", 31.5, 29.0, 2.5)]
    items = [
        ("c", "building", 9, 25, 100),
        ("c", "impervious", 0, 100, 100),
    ]
    result = grid_features(aligned, items)
    assert len(result) == 1
    assert len(result[0]) == 9
    assert result[0][:5] == (3600, "c", 31.5, 29.0, 2.5)
    assert result[0][5:] == (9.0, 0.25, 1.0, 0.0)


def test_sorted_by_timestamp_then_cell_id():
    aligned = [
        _row(7200, "b"),
        _row(0, "b"),
        _row(0, "a"),
        _row(3600, "c"),
    ]
    items = []
    for cell in ("a", "b", "c"):
        items += [
            (cell, "building", 1, 50, 100),
            (cell, "green", 0, 50, 100),
            (cell, "impervious", 0, 50, 100),
        ]
    result = grid_features(aligned, items)
    assert [(r[0], r[1]) for r in result] == [
        (0, "a"),
        (0, "b"),
        (3600, "c"),
        (7200, "b"),
    ]


def test_grid_without_building_dropped():
    aligned = [_row(0, "x")]
    items = [
        ("x", "impervious", 0, 50, 100),
        ("x", "green", 0, 50, 100),
    ]
    assert grid_features(aligned, items) == []


def test_grid_without_coverage_dropped():
    aligned = [_row(0, "x")]
    items = [
        ("x", "building", 10, 60, 100),
        ("x", "building", 20, 40, 100),
    ]
    assert grid_features(aligned, items) == []


def test_grid_coverage_sum_not_equal_g_dropped():
    aligned = [_row(0, "x")]
    items = [
        ("x", "building", 10, 50, 100),
        ("x", "impervious", 0, 40, 100),
        ("x", "green", 0, 9, 100),
    ]
    assert grid_features(aligned, items) == []


def test_zero_area_coverage_still_counts_as_present():
    aligned = [_row(0, "y")]
    items = [
        ("y", "building", 1, 50, 100),
        ("y", "impervious", 0, 0, 100),
        ("y", "green", 0, 100, 100),
    ]
    result = grid_features(aligned, items)
    assert result == [(0, "y", 1.0, 2.0, -1.0, 1.0, 0.5, 0.0, 1.0)]


def test_zero_total_building_area_mean_height_zero():
    aligned = [_row(0, "z")]
    items = [
        ("z", "building", 5, 0, 100),
        ("z", "green", 0, 100, 100),
    ]
    result = grid_features(aligned, items)
    assert result[0][5:] == (0.0, 0.0, 0.0, 1.0)


def test_aligned_cell_without_items_dropped():
    aligned = [_row(0, "known"), _row(0, "unknown")]
    items = [
        ("known", "building", 3, 10, 100),
        ("known", "impervious", 0, 100, 100),
    ]
    result = grid_features(aligned, items)
    assert [r[1] for r in result] == ["known"]


def test_items_without_aligned_rows_have_no_output():
    assert grid_features([], [("x", "building", 1, 10, 100),
                              ("x", "green", 0, 100, 100)]) == []


def test_decimal_str_semantics_for_coverage_equality():
    # 0.1 + 0.2 != 0.3 in binary floats, but Decimal(str(x)) makes them equal
    aligned = [_row(0, "x")]
    items = [
        ("x", "building", 1, 0, 0.3),
        ("x", "impervious", 0, 0.1, 0.3),
        ("x", "green", 0, 0.2, 0.3),
    ]
    result = grid_features(aligned, items)
    assert len(result) == 1
    assert result[0][5:] == (0.0, 0.0, 0.3333, 0.6667)


def test_repeating_division_quantized_half_even_four_places():
    aligned = [_row(0, "x")]
    items = [
        ("x", "building", 10, 30, 90),  # weighted height 10*30/30 = 10
        ("x", "green", 0, 30, 90),
        ("x", "impervious", 0, 60, 90),
    ]
    result = grid_features(aligned, items)
    # 30/90 = 0.3333..., 60/90 = 0.6666...
    assert result[0][5:] == (10.0, 0.3333, 0.6667, 0.3333)


def test_half_even_quantization():
    aligned = [_row(0, "c")]
    items = [
        ("c", "building", 0, 0, 100000),
        ("c", "impervious", 0, 12345, 100000),  # 0.12345 -> 0.1234 (even)
        ("c", "green", 0, 87655, 100000),       # 0.87655 -> 0.8766 (even)
    ]
    result = grid_features(aligned, items)
    assert result[0][5:] == (0.0, 0.0, 0.1234, 0.8766)


def test_negative_zero_normalized():
    aligned = [_row(0, "x")]
    items = [
        ("x", "building", -0.0, 50, 100),
        ("x", "green", 0, 100, 100),
    ]
    result = grid_features(aligned, items)
    for value in result[0][5:]:
        assert value == 0.0 or value > 0.0
        assert str(result[0][5]) != "-0.0"


def test_independent_of_default_context_precision():
    previous = getcontext().prec
    getcontext().prec = 2
    try:
        aligned = [_row(0, "x")]
        items = [
            ("x", "building", 10, 30, 90),
            ("x", "green", 0, 30, 90),
            ("x", "impervious", 0, 60, 90),
        ]
        result = grid_features(aligned, items)
        assert result[0][5:] == (10.0, 0.3333, 0.6667, 0.3333)

        # large magnitudes: under prec=2, raw accumulation would round sums
        # to 2 significant digits, corrupting both the coverage equality and
        # the building-area cap
        aligned2 = [_row(0, "y")]
        items2 = [
            ("y", "building", 5, 60, 101),
            ("y", "impervious", 0, 100.99, 101),
            ("y", "green", 0, 0.01, 101),
        ]
        result2 = grid_features(aligned2, items2)
        # prec=2 accumulation would turn 100.99+0.01=101 into 100 -> dropped
        assert result2[0][5] == 5.0
        assert result2[0][6] == 0.5941       # 60/101
        assert result2[0][7] == 0.9999       # 100.99/101
        assert result2[0][8] == 0.0001       # 0.01/101

        # prec=2 accumulation would turn 60+42=102 into 100, hiding the
        # non-building area overflow
        with pytest.raises(ValueError):
            grid_features(
                aligned2,
                [
                    ("y", "building", 5, 0, 101),
                    ("y", "impervious", 0, 60, 101),
                    ("y", "green", 0, 42, 101),
                ],
            )
    finally:
        getcontext().prec = previous


def test_int_inputs_accepted():
    aligned = [_row(0, "x")]
    items = [
        ("x", "building", 12, 1, 4),
        ("x", "impervious", 0, 4, 4),
    ]
    result = grid_features(aligned, items)
    assert result[0][5:] == (12.0, 0.25, 1.0, 0.0)


def test_aligned_must_be_list():
    with pytest.raises(TypeError):
        grid_features((_row(0, "x"),), [])
    with pytest.raises(TypeError):
        grid_features({}, [])


def test_items_must_be_list():
    aligned = [_row(0, "x")]
    with pytest.raises(TypeError):
        grid_features(aligned, (("x", "building", 1, 10, 100),))
    with pytest.raises(TypeError):
        grid_features(aligned, {})


def test_aligned_row_must_be_five_tuple():
    items = [
        ("x", "building", 1, 10, 100),
        ("x", "green", 0, 100, 100),
    ]
    for bad in ([0, "x", 1.0, 2.0, -1.0], (0, "x", 1.0, 2.0), "nope"):
        with pytest.raises(ValueError):
            grid_features([bad], items)


def test_item_must_be_five_tuple():
    aligned = [_row(0, "x")]
    for bad in (
        ["x", "building", 1, 10, 100],
        ("x", "building", 1, 10),
        ("x", "building", 1, 10, 100, 1),
    ):
        with pytest.raises(ValueError):
            grid_features(aligned, [bad])


def test_cell_id_must_be_nonempty_string():
    aligned = [_row(0, "x")]
    for bad in ("", 7, None, b"x"):
        with pytest.raises(ValueError):
            grid_features(aligned, [
                (bad, "building", 1, 0, 100),
                (bad, "green", 0, 100, 100),
            ])


def test_kind_validation():
    aligned = [_row(0, "x")]
    for bad in ("Building", "BUILDING", "water", 7, None):
        with pytest.raises(ValueError):
            grid_features(aligned, [
                ("x", "building", 1, 0, 100),
                ("x", bad, 0, 100, 100),
            ])


def test_height_area_grid_area_number_validation():
    aligned = [_row(0, "x")]
    valid = ("x", "building", 1, 10, 100)
    for bad in (True, False, "1", None, 1 + 0j):
        with pytest.raises(ValueError):
            grid_features(aligned, [("x", "building", bad, 10, 100)])
        with pytest.raises(ValueError):
            grid_features(aligned, [("x", "building", 1, bad, 100)])
        with pytest.raises(ValueError):
            grid_features(aligned, [valid, ("x", "green", 0, 0, bad)])
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            grid_features(aligned, [("x", "building", bad, 10, 100)])
        with pytest.raises(ValueError):
            grid_features(aligned, [valid, ("x", "green", 0, bad, 100)])
        with pytest.raises(ValueError):
            grid_features(aligned, [valid, ("x", "green", 0, 0, bad)])


def test_height_and_area_nonnegative_grid_area_positive():
    aligned = [_row(0, "x")]
    with pytest.raises(ValueError):
        grid_features(aligned, [("x", "building", -1, 10, 100)])
    with pytest.raises(ValueError):
        grid_features(aligned, [("x", "building", 1, -0.1, 100)])
    for bad in (0, -100):
        with pytest.raises(ValueError):
            grid_features(aligned, [("x", "building", 1, 10, bad)])


def test_nonbuilding_height_must_be_zero():
    aligned = [_row(0, "x")]
    for kind in ("impervious", "green", "other"):
        with pytest.raises(ValueError):
            grid_features(aligned, [
                ("x", "building", 1, 0, 100),
                ("x", kind, 0.5, 100, 100),
            ])


def test_grid_area_consistent_within_cell():
    aligned = [_row(0, "x")]
    with pytest.raises(ValueError):
        grid_features(aligned, [
            ("x", "building", 1, 10, 100),
            ("x", "green", 0, 90, 100.5),
        ])


def test_grid_area_equivalence_int_float():
    aligned = [_row(0, "x")]
    items = [
        ("x", "building", 1, 10, 100),
        ("x", "green", 0, 100, 100.0),
    ]
    # Decimal("100") == Decimal("100.0")
    result = grid_features(aligned, items)
    assert result[0][5:] == (1.0, 0.1, 0.0, 1.0)


def test_building_area_sum_exceeds_grid():
    aligned = [_row(0, "x")]
    with pytest.raises(ValueError):
        grid_features(aligned, [
            ("x", "building", 1, 60, 100),
            ("x", "building", 2, 41, 100),
            ("x", "green", 0, 0, 100),
        ])


def test_nonbuilding_area_sum_exceeds_grid():
    aligned = [_row(0, "x")]
    with pytest.raises(ValueError):
        grid_features(aligned, [
            ("x", "building", 1, 0, 100),
            ("x", "impervious", 0, 60, 100),
            ("x", "green", 0, 41, 100),
        ])


def test_independent_cells_validated_separately():
    aligned = [_row(0, "a"), _row(0, "b")]
    items = [
        ("a", "building", 3, 20, 80),
        ("a", "green", 0, 80, 80),
        ("b", "building", 6, 40, 160),
        ("b", "impervious", 0, 160, 160),
    ]
    result = grid_features(aligned, items)
    assert result[0][5:] == (3.0, 0.25, 0.0, 1.0)
    assert result[1][5:] == (6.0, 0.25, 1.0, 0.0)


def test_other_area_excluded_from_three_fractions():
    aligned = [_row(0, "x")]
    items = [
        ("x", "building", 2, 25, 100),
        ("x", "impervious", 0, 50, 100),
        ("x", "other", 0, 50, 100),
    ]
    result = grid_features(aligned, items)
    assert result[0][5:] == (2.0, 0.25, 0.5, 0.0)


def test_empty_inputs():
    assert grid_features([], []) == []
