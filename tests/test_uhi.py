"""Tests for urban_micro.compute_uhi."""

from decimal import Decimal

import pytest

from urban_micro import compute_uhi


STATIONS = {"u1": "urban", "u2": "urban", "r1": "rural", "r2": "rural"}


def test_empty_records_returns_empty_list():
    assert compute_uhi([], STATIONS) == []


def test_basic_single_bucket():
    records = [
        {"station_id": "u1", "timestamp": 0, "temp_c": 30.0},
        {"station_id": "u2", "timestamp": 100, "temp_c": 32.0},
        {"station_id": "r1", "timestamp": 200, "temp_c": 25.0},
        {"station_id": "r2", "timestamp": 300, "temp_c": 27.0},
    ]
    assert compute_uhi(records, STATIONS) == [(0, 31.0, 26.0, 5.0)]


def test_same_station_same_bucket_averaged_first():
    # u1 has two readings averaging 30; station mean (not reading mean) feeds zone mean
    records = [
        {"station_id": "u1", "timestamp": 0, "temp_c": 20.0},
        {"station_id": "u1", "timestamp": 10, "temp_c": 40.0},
        {"station_id": "u2", "timestamp": 0, "temp_c": 40.0},
        {"station_id": "r1", "timestamp": 0, "temp_c": 10.0},
        {"station_id": "r2", "timestamp": 0, "temp_c": 20.0},
    ]
    assert compute_uhi(records, STATIONS) == [(0, 35.0, 15.0, 20.0)]


def test_min_count_filters_buckets():
    records = [
        {"station_id": "u1", "timestamp": 0, "temp_c": 30.0},
        {"station_id": "r1", "timestamp": 0, "temp_c": 20.0},
        {"station_id": "r2", "timestamp": 0, "temp_c": 22.0},
    ]
    # only one urban station -> bucket dropped with default min_count=2
    assert compute_uhi(records, STATIONS) == []
    # min_count=1 keeps it
    assert compute_uhi(records, STATIONS, min_count=1) == [(0, 30.0, 21.0, 9.0)]


def test_bucketing_and_sort_order():
    records = [
        {"station_id": "u1", "timestamp": 7200, "temp_c": 30.0},
        {"station_id": "u2", "timestamp": 7200, "temp_c": 30.0},
        {"station_id": "r1", "timestamp": 7200, "temp_c": 20.0},
        {"station_id": "r2", "timestamp": 7200, "temp_c": 20.0},
        {"station_id": "u1", "timestamp": 3599, "temp_c": 10.0},
        {"station_id": "u2", "timestamp": 3599, "temp_c": 10.0},
        {"station_id": "r1", "timestamp": 3599, "temp_c": 5.0},
        {"station_id": "r2", "timestamp": 3599, "temp_c": 5.0},
    ]
    result = compute_uhi(records, STATIONS)
    assert result == [(0, 10.0, 5.0, 5.0), (7200, 30.0, 20.0, 10.0)]


def test_minutes_30_buckets():
    records = [
        {"station_id": "u1", "timestamp": 1799, "temp_c": 30.0},
        {"station_id": "u2", "timestamp": 1799, "temp_c": 30.0},
        {"station_id": "r1", "timestamp": 1799, "temp_c": 20.0},
        {"station_id": "r2", "timestamp": 1799, "temp_c": 20.0},
        {"station_id": "u1", "timestamp": 1800, "temp_c": 40.0},
        {"station_id": "u2", "timestamp": 1800, "temp_c": 40.0},
        {"station_id": "r1", "timestamp": 1800, "temp_c": 10.0},
        {"station_id": "r2", "timestamp": 1800, "temp_c": 10.0},
    ]
    assert compute_uhi(records, STATIONS, minutes=30) == [
        (0, 30.0, 20.0, 10.0),
        (1800, 40.0, 10.0, 30.0),
    ]


def test_decimal_str_semantics_and_half_even():
    # 0.1 + 0.2 style float noise must not leak in: Decimal(str(x))
    records = [
        {"station_id": "u1", "timestamp": 0, "temp_c": 0.1},
        {"station_id": "u1", "timestamp": 1, "temp_c": 0.2},
        {"station_id": "u2", "timestamp": 0, "temp_c": 0.0005},  # half-even -> 0.000
        {"station_id": "r1", "timestamp": 0, "temp_c": 0.0},
        {"station_id": "r2", "timestamp": 0, "temp_c": 0.0},
    ]
    result = compute_uhi(records, STATIONS, min_count=1)
    # u1 mean = 0.15, u2 = 0.0005; urban mean = 0.07525 -> 0.075
    # uhi = 0.07525 - 0 -> 0.075
    assert result == [(0, 0.075, 0.0, 0.075)]


def test_half_even_rounding():
    # urban mean exactly x.xxx5 rounds to even
    records = [
        {"station_id": "u1", "timestamp": 0, "temp_c": 1.0005},
        {"station_id": "u2", "timestamp": 0, "temp_c": 1.0005},
        {"station_id": "r1", "timestamp": 0, "temp_c": 0.0025},
        {"station_id": "r2", "timestamp": 0, "temp_c": 0.0025},
    ]
    assert compute_uhi(records, STATIONS) == [(0, 1.0, 0.002, 0.998)]


def test_negative_zero_becomes_positive_zero():
    records = [
        {"station_id": "u1", "timestamp": 0, "temp_c": -0.0001},
        {"station_id": "u2", "timestamp": 0, "temp_c": -0.0001},
        {"station_id": "r1", "timestamp": 0, "temp_c": 0.0},
        {"station_id": "r2", "timestamp": 0, "temp_c": 0.0},
    ]
    result = compute_uhi(records, STATIONS)
    assert result == [(0, 0.0, 0.0, 0.0)]
    for value in result[0][1:]:
        assert value == 0.0 and str(value) != "-0.0"


def test_uhi_uses_unrounded_means():
    # urban mean 0.0004 -> rounds to 0.000; rural 0.0; uhi 0.0004 -> 0.000
    records = [
        {"station_id": "u1", "timestamp": 0, "temp_c": 0.0004},
        {"station_id": "u2", "timestamp": 0, "temp_c": 0.0004},
        {"station_id": "r1", "timestamp": 0, "temp_c": 0.0},
        {"station_id": "r2", "timestamp": 0, "temp_c": 0.0},
    ]
    assert compute_uhi(records, STATIONS) == [(0, 0.0, 0.0, 0.0)]


def test_records_must_be_list():
    with pytest.raises(TypeError):
        compute_uhi((r for r in []), STATIONS)
    with pytest.raises(TypeError):
        compute_uhi({}, STATIONS)


def test_stations_must_be_dict():
    with pytest.raises(TypeError):
        compute_uhi([], [("u1", "urban")])


def test_stations_zone_validation():
    with pytest.raises(ValueError):
        compute_uhi([], {"u1": "city"})


def test_minutes_validation():
    for bad in (0, 1441, 7, 90.0, True, "60", -60):
        with pytest.raises(ValueError):
            compute_uhi([], STATIONS, minutes=bad)
    for good in (1, 2, 3, 30, 60, 720, 1440):
        assert compute_uhi([], STATIONS, minutes=good) == []


def test_min_count_validation():
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError):
            compute_uhi([], STATIONS, min_count=bad)


def test_record_key_validation():
    with pytest.raises(ValueError):
        compute_uhi([{"station_id": "u1", "timestamp": 0}], STATIONS)
    with pytest.raises(ValueError):
        compute_uhi(
            [{"station_id": "u1", "timestamp": 0, "temp_c": 1.0, "extra": 1}], STATIONS
        )
    with pytest.raises(ValueError):
        compute_uhi([("u1", 0, 1.0)], STATIONS)


def test_unknown_station():
    with pytest.raises(ValueError):
        compute_uhi(
            [{"station_id": "nope", "timestamp": 0, "temp_c": 1.0}], STATIONS
        )


def test_timestamp_validation():
    for bad in (-1, 1.5, True, "0"):
        with pytest.raises(ValueError):
            compute_uhi(
                [{"station_id": "u1", "timestamp": bad, "temp_c": 1.0}], STATIONS
            )


def test_temp_c_validation():
    for bad in (float("nan"), float("inf"), float("-inf"), True, "20"):
        with pytest.raises(ValueError):
            compute_uhi(
                [{"station_id": "u1", "timestamp": 0, "temp_c": bad}], STATIONS
            )


def test_int_timestamps_and_temps_accepted():
    records = [
        {"station_id": "u1", "timestamp": 0, "temp_c": 30},
        {"station_id": "u2", "timestamp": 0, "temp_c": 30},
        {"station_id": "r1", "timestamp": 0, "temp_c": 20},
        {"station_id": "r2", "timestamp": 0, "temp_c": 20},
    ]
    assert compute_uhi(records, STATIONS) == [(0, 30.0, 20.0, 10.0)]


def test_thirds_do_not_accumulate_float_error():
    # 1/3-style repeating decimals via Decimal keep full precision
    records = [
        {"station_id": "u1", "timestamp": 0, "temp_c": 1},
        {"station_id": "u2", "timestamp": 0, "temp_c": 2},
        {"station_id": "r1", "timestamp": 0, "temp_c": 0},
        {"station_id": "r2", "timestamp": 0, "temp_c": 0},
    ]
    assert compute_uhi(records, STATIONS) == [(0, 1.5, 0.0, 1.5)]
