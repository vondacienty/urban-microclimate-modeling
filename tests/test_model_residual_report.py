"""Tests for urban_micro.model_residual_report."""

import json

import pytest

from urban_micro import fit_uhi_model, model_residual_report


def _row(t, c, d=0.0, h=0.0, b=0.0, i=0.0, g=0.0):
    return (t, c, 1.0, 2.0, d, h, b, i, g)


def _model(n=5, b0=1.0, bh=2.0, bb=0.0, bi=0.0, bg=0.0, r2=0.9):
    return (n, b0, bh, bb, bi, bg, r2)


def test_empty_rows_yield_empty_groups():
    assert model_residual_report(_model(), []) == '{"by":"time","minutes":60,"groups":[]}'
    assert (
        model_residual_report(_model(), [], by="cell", minutes=30)
        == '{"by":"cell","minutes":30,"groups":[]}'
    )


def test_residuals_bias_mae_rmse():
    # p = 1 + 2*h: residuals 0 (d=1,h=0), 1 (d=4,h=1), 0 (d=3,h=1)
    rows = [
        _row(0, "a", 1.0, 0.0),
        _row(60, "b", 4.0, 1.0),
        _row(120, "a", 3.0, 1.0),
    ]
    result = model_residual_report(_model(), rows, minutes=60)
    assert result == (
        '{"by":"time","minutes":60,"groups":[{"key":0,"n":3,'
        '"bias":"0.333333","mae":"0.333333","rmse":"0.577350"}]}'
    )
    group = json.loads(result)["groups"][0]
    assert list(group) == ["key", "n", "bias", "mae", "rmse"]


def test_all_four_features_enter_prediction():
    # p = 1 + 2h + 3b + 4i + 5g = 15 for the first row, 1 for the second
    model = (1, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0)
    rows = [
        _row(0, "a", 16.0, 1.0, 1.0, 1.0, 1.0),  # e = 1
        _row(10, "b", 0.0, 0.0),                # e = -1
    ]
    result = model_residual_report(model, rows, minutes=60)
    assert result == (
        '{"by":"time","minutes":60,"groups":[{"key":0,"n":2,'
        '"bias":"0.000000","mae":"1.000000","rmse":"1.000000"}]}'
    )


def test_time_bucketing_keys_and_order():
    rows = [
        _row(7200, "c", 1.0, 0.0),
        _row(0, "a", 1.0, 0.0),
        _row(3600, "b", 1.0, 0.0),
        _row(3599, "a", 2.0, 0.0),
    ]
    result = model_residual_report(_model(), rows, minutes=60)
    groups = json.loads(result)["groups"]
    assert [g["key"] for g in groups] == [0, 3600, 7200]
    assert [g["n"] for g in groups] == [2, 1, 1]
    assert groups[0]["bias"] == "0.500000"


def test_cell_grouping_keys_and_order():
    rows = [
        _row(0, "b", 4.0, 1.0),
        _row(10, "a", 1.0, 0.0),
        _row(20, "a", 4.0, 1.0),
    ]
    result = model_residual_report(_model(), rows, by="cell")
    groups = json.loads(result)["groups"]
    assert [g["key"] for g in groups] == ["a", "b"]
    assert groups[0]["n"] == 2
    assert groups[1]["bias"] == "1.000000"
    assert result.startswith('{"by":"cell"')


def test_compact_json_has_no_spaces_or_trailing_newline():
    result = model_residual_report(_model(), [_row(0, "a", 1.0, 0.0)])
    assert " " not in result
    assert not result.endswith("\n")
    json.loads(result)


def test_numeric_fields_are_six_decimal_strings():
    result = model_residual_report(_model(), [_row(0, "a", 1.0, 0.0)])
    group = json.loads(result)["groups"][0]
    for field in ("bias", "mae", "rmse"):
        assert isinstance(group[field], str)
        assert len(group[field].split(".")[1]) == 6
    assert isinstance(group["key"], int)
    assert isinstance(group["n"], int)


def test_negative_zero_normalized():
    model = (1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a", 2.0), _row(10, "b", 0.0)]  # residuals +1 and -1
    result = model_residual_report(model, rows)
    assert '"bias":"0.000000"' in result
    assert '"bias":"-0.000000"' not in result


def test_unicode_cell_key_preserved():
    result = model_residual_report(_model(), [_row(0, "格子", 1.0, 0.0)], by="cell")
    assert '"key":"格子"' in result
    json.loads(result)


def test_round_trips_with_fit_uhi_model_output():
    rows = [
        _row(0, "a", 2.0, 0.0),
        _row(60, "b", 5.0, 1.0),
        _row(3600, "a", 5.0, 1.0),
        _row(7200, "c", 8.0, 2.0),
    ]
    model = fit_uhi_model(rows)
    result = model_residual_report(model, rows, minutes=60)
    assert list(json.loads(result)) == ["by", "minutes", "groups"]
    assert [g["key"] for g in json.loads(result)["groups"]] == [0, 3600, 7200]


def test_model_must_be_tuple():
    with pytest.raises(TypeError):
        model_residual_report([], [])
    with pytest.raises(TypeError):
        model_residual_report("x", [])
    with pytest.raises(TypeError):
        model_residual_report(None, [])


def test_rows_must_be_list():
    with pytest.raises(TypeError):
        model_residual_report(_model(), (_row(0, "a", 1.0),))
    with pytest.raises(TypeError):
        model_residual_report(_model(), {0: _row(0, "a", 1.0)})


def test_model_structure_validation():
    with pytest.raises(ValueError):
        model_residual_report((1, 1.0, 2.0, 0.0, 0.0, 0.0), [])
    for n in (0, -1, True, 1.0, "1"):
        with pytest.raises(ValueError):
            model_residual_report(_model(n=n), [])


def test_model_coefficients_validation():
    for idx, bad in (
        (1, float("nan")),
        (2, float("inf")),
        (3, True),
        (4, "0.0"),
        (5, None),
        (6, float("-inf")),
    ):
        model = list(_model())
        model[idx] = bad
        with pytest.raises(ValueError):
            model_residual_report(tuple(model), [])


def test_rows_contract_validation():
    with pytest.raises(ValueError):
        model_residual_report(_model(), ["nope"])
    with pytest.raises(ValueError):
        model_residual_report(
            _model(),
            [_row(0, "a", 1.0), _row(0, "a", 1.0)],
        )
    with pytest.raises(ValueError):
        model_residual_report(_model(), [_row(-1, "a", 1.0)])
    with pytest.raises(ValueError):
        model_residual_report(_model(), [_row(0, "", 1.0)])


def test_by_validation():
    with pytest.raises(ValueError):
        model_residual_report(_model(), [], by="hour")
    with pytest.raises(ValueError):
        model_residual_report(_model(), [], by="TIME")


def test_minutes_validation():
    for bad in (0, 1441, 7, True, False, 60.0, "60"):
        with pytest.raises(ValueError):
            model_residual_report(_model(), [], minutes=bad)
    assert '"minutes":1' in model_residual_report(_model(), [], minutes=1)
    assert '"minutes":1440' in model_residual_report(_model(), [], minutes=1440)
