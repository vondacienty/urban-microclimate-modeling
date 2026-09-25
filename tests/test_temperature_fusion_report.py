"""Tests for urban_micro.temperature_fusion_report."""

import json

import pytest

from urban_micro import temperature_fusion_report


CELLS = {"s1": "A", "s2": "A", "s3": "B"}


def _station(station_id, timestamp, temp_c):
    return {"station_id": station_id, "timestamp": timestamp, "temp_c": temp_c}


def _satellite(cell_id, timestamp, lst_c):
    return {"cell_id": cell_id, "timestamp": timestamp, "lst_c": lst_c}


def test_empty_inputs_yield_empty_groups():
    assert temperature_fusion_report([], [], CELLS) == '{"minutes":60,"groups":[]}'
    assert (
        temperature_fusion_report([], [], CELLS, minutes=30)
        == '{"minutes":30,"groups":[]}'
    )


def test_basic_single_bucket_single_cell():
    station = [
        _station("s1", 0, 30.0),
        _station("s1", 10, 32.0),  # s1 mean 31
        _station("s2", 5, 28.0),  # s2 mean 28
    ]
    satellite = [
        _satellite("A", 0, 25.0),
        _satellite("A", 3, 27.0),  # lst mean 26
    ]
    # d = 31 - 26 = 5 and 28 - 26 = 2 -> bias 3.5, rmse sqrt(14.5)
    result = temperature_fusion_report(station, satellite, CELLS)
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"cells":[{"key":"A",'
        '"n_station":2,"n_satellite":2,"bias":"3.500000",'
        '"rmse":"3.807887"}]}]}'
    )
    json.loads(result)  # valid JSON


def test_min_count_filters_pairs():
    station = [_station("s1", 0, 30.0)]
    satellite = [_satellite("A", 0, 25.0)]
    assert (
        temperature_fusion_report(station, satellite, CELLS, min_count=2)
        == '{"minutes":60,"groups":[]}'
    )
    result = temperature_fusion_report(station, satellite, CELLS, min_count=1)
    assert '"n_station":1' in result


def test_missing_satellite_source_dropped():
    station = [_station("s1", 0, 30.0), _station("s1", 7200, 20.0)]
    satellite = [_satellite("A", 0, 25.0)]
    result = temperature_fusion_report(station, satellite, CELLS)
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"cells":[{"key":"A",'
        '"n_station":1,"n_satellite":1,"bias":"5.000000",'
        '"rmse":"5.000000"}]}]}'
    )


def test_bucketing_and_sort_order():
    station = [
        _station("s3", 7200, 20.0),
        _station("s1", 0, 30.0),
        _station("s1", 7200, 10.0),
    ]
    satellite = [
        _satellite("B", 7200, 18.0),
        _satellite("A", 0, 25.0),
        _satellite("A", 7200, 8.0),
    ]
    result = temperature_fusion_report(station, satellite, CELLS)
    assert result == (
        '{"minutes":60,"groups":['
        '{"key":0,"cells":[{"key":"A","n_station":1,"n_satellite":1,'
        '"bias":"5.000000","rmse":"5.000000"}]},'
        '{"key":7200,"cells":[{"key":"A","n_station":1,"n_satellite":1,'
        '"bias":"2.000000","rmse":"2.000000"},'
        '{"key":"B","n_station":1,"n_satellite":1,'
        '"bias":"2.000000","rmse":"2.000000"}]}]}'
    )


def test_minutes_30_buckets():
    station = [_station("s1", 1799, 30.0), _station("s1", 1800, 40.0)]
    satellite = [_satellite("A", 1799, 20.0), _satellite("A", 1800, 30.0)]
    result = temperature_fusion_report(station, satellite, CELLS, minutes=30)
    assert result == (
        '{"minutes":30,"groups":['
        '{"key":0,"cells":[{"key":"A","n_station":1,"n_satellite":1,'
        '"bias":"10.000000","rmse":"10.000000"}]},'
        '{"key":1800,"cells":[{"key":"A","n_station":1,"n_satellite":1,'
        '"bias":"10.000000","rmse":"10.000000"}]}]}'
    )


def test_decimal_str_semantics_and_half_even():
    # 0.1 + 0.2 float noise must not leak in: Decimal(str(x))
    station = [_station("s1", 0, 0.1), _station("s1", 1, 0.2)]
    satellite = [_satellite("A", 0, 0.0)]
    result = temperature_fusion_report(station, satellite, CELLS)
    assert '"bias":"0.150000"' in result
    assert '"rmse":"0.150000"' in result


def test_negative_zero_normalized():
    station = [_station("s1", 0, -0.0)]
    satellite = [_satellite("A", 0, 0.0)]
    result = temperature_fusion_report(station, satellite, CELLS)
    assert '"bias":"0.000000"' in result
    assert '"rmse":"0.000000"' in result
    assert "-0.000000" not in result


def test_no_trailing_newline_and_no_spaces():
    station = [_station("s1", 0, 30.0)]
    satellite = [_satellite("A", 0, 25.0)]
    result = temperature_fusion_report(station, satellite, CELLS)
    assert not result.endswith("\n")
    assert " " not in result


def test_type_errors():
    with pytest.raises(TypeError):
        temperature_fusion_report("x", [], CELLS)
    with pytest.raises(TypeError):
        temperature_fusion_report([], "x", CELLS)
    with pytest.raises(TypeError):
        temperature_fusion_report([], [], "x")


def test_value_errors():
    station = [_station("s1", 0, 30.0)]
    satellite = [_satellite("A", 0, 25.0)]
    with pytest.raises(ValueError):
        temperature_fusion_report(station, satellite, {"s1": ""})
    with pytest.raises(ValueError):
        temperature_fusion_report(station, satellite, {1: "A"})
    with pytest.raises(ValueError):
        temperature_fusion_report(station, satellite, CELLS, minutes=7)
    with pytest.raises(ValueError):
        temperature_fusion_report(station, satellite, CELLS, minutes=True)
    with pytest.raises(ValueError):
        temperature_fusion_report(station, satellite, CELLS, minutes=0)
    with pytest.raises(ValueError):
        temperature_fusion_report(station, satellite, CELLS, min_count=0)
    with pytest.raises(ValueError):
        temperature_fusion_report(station, satellite, CELLS, min_count=True)
    with pytest.raises(ValueError):
        temperature_fusion_report([_station("s9", 0, 1.0)], satellite, CELLS)
    with pytest.raises(ValueError):
        temperature_fusion_report([_station("", 0, 1.0)], satellite, CELLS)
    with pytest.raises(ValueError):
        temperature_fusion_report([_station("s1", -1, 1.0)], satellite, CELLS)
    with pytest.raises(ValueError):
        temperature_fusion_report([_station("s1", 0, float("nan"))], satellite, CELLS)
    with pytest.raises(ValueError):
        temperature_fusion_report([_station("s1", 0, True)], satellite, CELLS)
    with pytest.raises(ValueError):
        temperature_fusion_report(
            [{"station_id": "s1", "timestamp": 0}], satellite, CELLS
        )
    with pytest.raises(ValueError):
        temperature_fusion_report(station, [_satellite("", 0, 25.0)], CELLS)
    with pytest.raises(ValueError):
        temperature_fusion_report(station, [_satellite("A", 0, True)], CELLS)
