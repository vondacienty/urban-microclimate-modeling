"""Tests for urban_micro.temperature_fusion_uhi_report."""

import json

import pytest

from urban_micro import temperature_fusion_uhi_report


CELLS = {"s1": "A", "s2": "A", "s3": "B"}
ZONES = {"A": "urban", "B": "rural"}


def _station(station_id, timestamp, temp_c):
    return {"station_id": station_id, "timestamp": timestamp, "temp_c": temp_c}


def _satellite(cell_id, timestamp, lst_c):
    return {"cell_id": cell_id, "timestamp": timestamp, "lst_c": lst_c}


def test_empty_inputs_yield_empty_groups():
    assert (
        temperature_fusion_uhi_report([], [], CELLS, ZONES)
        == '{"minutes":60,"groups":[]}'
    )
    assert (
        temperature_fusion_uhi_report([], [], CELLS, ZONES, minutes=30)
        == '{"minutes":30,"groups":[]}'
    )


def test_basic_single_bucket():
    station = [
        _station("s1", 0, 30.0),
        _station("s1", 10, 32.0),  # s1 mean 31
        _station("s2", 5, 28.0),  # s2 mean 28 -> cell A station mean 29.5
        _station("s3", 0, 20.0),  # cell B station mean 20
    ]
    satellite = [
        _satellite("A", 0, 25.0),
        _satellite("A", 3, 27.0),  # lst mean 26 -> urban bias 3.5
        _satellite("B", 0, 18.0),  # rural bias 2
    ]
    result = temperature_fusion_uhi_report(station, satellite, CELLS, ZONES)
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"n_urban":1,"n_rural":1,'
        '"urban_bias":"3.500000","rural_bias":"2.000000",'
        '"uhi":"1.500000"}]}'
    )
    json.loads(result)  # valid JSON


def test_multiple_cells_per_zone():
    cells = {"s1": "A", "s2": "B", "s3": "C"}
    zones = {"A": "urban", "B": "urban", "C": "rural"}
    station = [
        _station("s1", 0, 30.0),  # A bias 30 - 25 = 5
        _station("s2", 0, 26.0),  # B bias 26 - 25 = 1 -> urban mean 3
        _station("s3", 0, 22.0),  # C bias 22 - 20 = 2
    ]
    satellite = [
        _satellite("A", 0, 25.0),
        _satellite("B", 0, 25.0),
        _satellite("C", 0, 20.0),
    ]
    result = temperature_fusion_uhi_report(station, satellite, cells, zones)
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"n_urban":2,"n_rural":1,'
        '"urban_bias":"3.000000","rural_bias":"2.000000",'
        '"uhi":"1.000000"}]}'
    )


def test_min_count_filters_pairs():
    station = [_station("s1", 0, 30.0), _station("s3", 0, 20.0)]
    satellite = [_satellite("A", 0, 25.0), _satellite("B", 0, 18.0)]
    assert (
        temperature_fusion_uhi_report(station, satellite, CELLS, ZONES, min_count=2)
        == '{"minutes":60,"groups":[]}'
    )
    result = temperature_fusion_uhi_report(
        station, satellite, CELLS, ZONES, min_count=1
    )
    assert '"n_urban":1' in result


def test_missing_satellite_source_dropped():
    station = [_station("s1", 0, 30.0), _station("s3", 0, 20.0)]
    satellite = [_satellite("B", 0, 18.0)]  # no satellite for urban cell A
    assert (
        temperature_fusion_uhi_report(station, satellite, CELLS, ZONES)
        == '{"minutes":60,"groups":[]}'
    )


def test_bucket_missing_zone_dropped():
    # bucket 0 has both zones, bucket 7200 only urban -> 7200 dropped
    station = [
        _station("s1", 0, 30.0),
        _station("s3", 0, 20.0),
        _station("s1", 7200, 30.0),
    ]
    satellite = [
        _satellite("A", 0, 25.0),
        _satellite("B", 0, 18.0),
        _satellite("A", 7200, 25.0),
    ]
    result = temperature_fusion_uhi_report(station, satellite, CELLS, ZONES)
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"n_urban":1,"n_rural":1,'
        '"urban_bias":"5.000000","rural_bias":"2.000000",'
        '"uhi":"3.000000"}]}'
    )


def test_bucketing_and_minutes_30():
    station = [
        _station("s1", 1799, 30.0),
        _station("s1", 1800, 40.0),
        _station("s3", 1799, 20.0),
        _station("s3", 1800, 24.0),
    ]
    satellite = [
        _satellite("A", 1799, 20.0),
        _satellite("A", 1800, 30.0),
        _satellite("B", 1799, 18.0),
        _satellite("B", 1800, 20.0),
    ]
    result = temperature_fusion_uhi_report(
        station, satellite, CELLS, ZONES, minutes=30
    )
    assert result == (
        '{"minutes":30,"groups":['
        '{"key":0,"n_urban":1,"n_rural":1,"urban_bias":"10.000000",'
        '"rural_bias":"2.000000","uhi":"8.000000"},'
        '{"key":1800,"n_urban":1,"n_rural":1,"urban_bias":"10.000000",'
        '"rural_bias":"4.000000","uhi":"6.000000"}]}'
    )


def test_decimal_str_semantics_and_negative_zero():
    # 0.1 + 0.2 float noise must not leak in: Decimal(str(x))
    station = [
        _station("s1", 0, 0.1),
        _station("s1", 1, 0.2),
        _station("s3", 0, -0.0),
    ]
    satellite = [_satellite("A", 0, 0.0), _satellite("B", 0, 0.0)]
    result = temperature_fusion_uhi_report(station, satellite, CELLS, ZONES)
    assert '"urban_bias":"0.150000"' in result
    assert '"rural_bias":"0.000000"' in result
    assert '"uhi":"0.150000"' in result
    assert "-0.000000" not in result


def test_no_trailing_newline_and_no_spaces():
    station = [_station("s1", 0, 30.0), _station("s3", 0, 20.0)]
    satellite = [_satellite("A", 0, 25.0), _satellite("B", 0, 18.0)]
    result = temperature_fusion_uhi_report(station, satellite, CELLS, ZONES)
    assert not result.endswith("\n")
    assert " " not in result


def test_type_errors():
    with pytest.raises(TypeError):
        temperature_fusion_uhi_report("x", [], CELLS, ZONES)
    with pytest.raises(TypeError):
        temperature_fusion_uhi_report([], "x", CELLS, ZONES)
    with pytest.raises(TypeError):
        temperature_fusion_uhi_report([], [], "x", ZONES)
    with pytest.raises(TypeError):
        temperature_fusion_uhi_report([], [], CELLS, "x")


def test_value_errors():
    station = [_station("s1", 0, 30.0), _station("s3", 0, 20.0)]
    satellite = [_satellite("A", 0, 25.0), _satellite("B", 0, 18.0)]
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, satellite, {"s1": ""}, ZONES)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, satellite, {1: "A"}, ZONES)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, satellite, CELLS, {"A": "urban"})
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, satellite, CELLS, {"A": "city", "B": "rural"})
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, satellite, CELLS, {"": "urban", "B": "rural"})
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, satellite, CELLS, ZONES, minutes=7)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, satellite, CELLS, ZONES, minutes=True)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, satellite, CELLS, ZONES, minutes=0)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, satellite, CELLS, ZONES, min_count=0)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, satellite, CELLS, ZONES, min_count=True)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report([_station("s9", 0, 1.0)], satellite, CELLS, ZONES)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report([_station("", 0, 1.0)], satellite, CELLS, ZONES)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report([_station("s1", -1, 1.0)], satellite, CELLS, ZONES)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(
            [_station("s1", 0, float("nan"))], satellite, CELLS, ZONES
        )
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report([_station("s1", 0, True)], satellite, CELLS, ZONES)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(
            [{"station_id": "s1", "timestamp": 0}], satellite, CELLS, ZONES
        )
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, [_satellite("", 0, 25.0)], CELLS, ZONES)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, [_satellite("A", 0, True)], CELLS, ZONES)
    with pytest.raises(ValueError):
        temperature_fusion_uhi_report(station, [_satellite("Z", 0, 25.0)], CELLS, ZONES)
