"""Tests for urban_micro.fit_uhi_model."""

import math

import pytest

from urban_micro import fit_uhi_model


def _row(t, c, d=0.0, h=0.0, b=0.0, i=0.0, g=0.0):
    return (t, c, 1.0, 2.0, d, h, b, i, g)


def test_perfect_linear_fit_recovers_coefficients():
    rows = []
    k = 0
    for h in (0.0, 1.0, 2.5, 7.0):
        for b in (0.0, 0.25, 0.5, 1.0):
            for i in (0.0, 0.5, 1.0):
                for g in (0.0, 0.3, 1.0):
                    d = 2 + 3 * h - b + 0.5 * i - 2 * g
                    rows.append(_row(k, f"c{k % 7}", d, h, b, i, g))
                    k += 1
    result = fit_uhi_model(rows)
    assert result == (len(rows), 2.0, 3.0, -1.0, 0.5, -2.0, 1.0)


def test_returns_seven_tuple_of_right_types():
    result = fit_uhi_model(
        [_row(0, "a", 1.0, 1.0, 0.0, 0.0, 0.0),
         _row(1, "b", 2.0, 2.0, 1.0, 0.0, 0.0)]
    )
    assert len(result) == 7
    assert result[0] == 2 and isinstance(result[0], int)
    assert all(isinstance(v, float) for v in result[1:])


def test_constant_target_gives_r2_zero():
    rows = [_row(i, f"c{i}", 5.0, float(i), 0.1, 0.2, 0.3) for i in range(6)]
    result = fit_uhi_model(rows)
    assert result[1] == 5.0
    assert result[2:6] == (0.0, 0.0, 0.0, 0.0)
    assert result[6] == 0.0


def test_sst_zero_with_negative_deviations():
    rows = [_row(i, "same", 3.0, float(i), 0.5, 0.5, 0.5) for i in range(4)]
    assert fit_uhi_model(rows)[6] == 0.0


def test_rows_sorted_by_timestamp_cell_id_before_fitting():
    # three points on d = h, supplied out of order
    rows = [
        _row(5, "z", 1.0, 1.0),
        _row(0, "b", 1.0, 1.0),
        _row(0, "a", 0.0, 0.0),
    ]
    n, b0, bh, bb, bi, bg, r2 = fit_uhi_model(rows)
    assert n == 3
    assert b0 == pytest.approx(0.0, abs=1e-5)
    assert bh == pytest.approx(1.0, abs=1e-5)
    assert (bb, bi, bg) == (0.0, 0.0, 0.0)
    assert r2 == 1.0


def test_duplicate_timestamp_cell_pair_rejected():
    with pytest.raises(ValueError):
        fit_uhi_model([_row(0, "x"), _row(0, "x", 1.0)])


def test_same_timestamp_different_cells_allowed():
    assert fit_uhi_model([_row(0, "a"), _row(0, "b")])[0] == 2


def test_same_cell_different_timestamps_allowed():
    assert fit_uhi_model([_row(0, "x"), _row(1, "x")])[0] == 2


def test_rows_must_be_list():
    with pytest.raises(TypeError):
        fit_uhi_model((_row(0, "x"),))
    with pytest.raises(TypeError):
        fit_uhi_model({0: _row(0, "x")})


def test_empty_rows_rejected():
    with pytest.raises(ValueError):
        fit_uhi_model([])


def test_row_must_be_nine_tuple():
    for bad in (
        [0, "x", 1.0, 2.0, 0.5, 1.0, 0.2, 0.3, 0.4],
        (0, "x", 1.0, 2.0, 0.5, 1.0, 0.2, 0.3),
        (0, "x", 1.0, 2.0, 0.5, 1.0, 0.2, 0.3, 0.4, 9),
        "nope",
    ):
        with pytest.raises(ValueError):
            fit_uhi_model([bad])


def test_timestamp_validation():
    for bad in (-1, 1.5, True, "0"):
        with pytest.raises(ValueError):
            fit_uhi_model([_row(bad, "x")])
    fit_uhi_model([_row(0, "x")])
    fit_uhi_model([_row(10**12, "x", 1.0, 1.0)])


def test_cell_id_validation():
    for bad in ("", 7, None, b"x"):
        with pytest.raises(ValueError):
            fit_uhi_model([(0, bad, 1.0, 2.0, 0.5, 1.0, 0.2, 0.3, 0.4)])


def test_target_field_validation():
    for idx, bad in (
        (2, True),
        (2, float("nan")),
        (3, float("inf")),
        (4, "0.5"),
        (4, True),
    ):
        row = [0, "x", 1.0, 2.0, 0.5, 1.0, 0.2, 0.3, 0.4]
        row[idx] = bad
        with pytest.raises(ValueError):
            fit_uhi_model([tuple(row)])


def test_height_validation():
    for bad in (True, False, -0.000001, float("inf"), float("-inf"),
                float("nan"), "1", 1 + 0j):
        with pytest.raises(ValueError):
            fit_uhi_model([_row(0, "x", h=bad)])


def test_fraction_fields_validation():
    base = _row(0, "x")
    for idx in (6, 7, 8):
        for bad in (-0.000001, 1.000001, True, False, float("nan"),
                    float("inf"), float("-inf"), "0.5"):
            row = list(base)
            row[idx] = bad
            with pytest.raises(ValueError):
                fit_uhi_model([tuple(row)])


def test_boundary_values_accepted():
    result = fit_uhi_model(
        [
            _row(0, "x", 0.5, 0.0, 0.0, 0.0, 0.0),
            _row(1, "y", 0.7, 100.0, 1.0, 1.0, 1.0),
            _row(2, "z", 0.9, 50.0, 0.5, 0.5, 0.5),
        ]
    )
    assert result[0] == 3


def test_negative_zero_normalized():
    result = fit_uhi_model([_row(i, "c", 0.0, 0.0, 0.0, 0.0, 0.0)
                            for i in range(3)])
    for value in result[1:]:
        assert value == 0.0
        assert math.copysign(1.0, value) == 1.0


def test_ridge_penalty_on_feature_diagonal_only():
    # Two points (h=0, d=0) and (h=1, d=1), other features zero.
    # Penalized normal equations [[2, 1], [1, 1+1e-6]] with rhs [1, 1]:
    # b0 = 1e-6 / (1 + 2e-6), bh = 1 / (1 + 2e-6) = 0.9999980000...
    result = fit_uhi_model(
        [_row(0, "a", 0.0, 0.0), _row(1, "b", 1.0, 1.0)]
    )
    n, b0, bh, bb, bi, bg, r2 = result
    assert n == 2
    assert b0 == 0.000001
    assert bh == 0.999998
    assert (bb, bi, bg) == (0.0, 0.0, 0.0)
    assert r2 == 1.0


def test_independent_of_default_context_precision():
    from decimal import getcontext

    previous = getcontext().prec
    getcontext().prec = 2
    try:
        rows = [
            _row(0, "a", 1e300, 1e300),
            _row(1, "b", 2e300, 2e300),
        ]
        result = fit_uhi_model(rows)
        assert result[0] == 2
        assert all(isinstance(v, float) for v in result[1:])
    finally:
        getcontext().prec = previous
