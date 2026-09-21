"""Tests for urban_micro.compute_uhi."""

from __future__ import annotations

import math

import pytest

from urban_micro import compute_uhi

STATIONS = {
    "u1": "urban",
    "u2": "urban",
    "r1": "rural",
    "r2": "rural",
}


def test_empty_records():
    assert compute_uhi([], STATIONS) == []


def test_basic_bucketing_and_averages():
    records = [
        ("u1", 0, 30.0),
        ("u1", 1800, 32.0),  # averaged with 30 -> u1 mean 31 in bucket 0
        ("u2", 60, 34.0),
        ("r1", 30, 20.0),
        ("r1", 3700, 22.0),  # later bucket lacks enough stations -> dropped
        ("r2", 100, 21.0),
    ]
    assert compute_uhi(records, STATIONS) == [(0, 32.5, 20.5, 12.0)]


def test_bucket_start_is_floor_of_epoch():
    records = [
        ("u1", 86399, 10.0),
        ("u2", 0, 10.0),
        ("r1", 86399, 5.0),
        ("r2", 0, 5.0),
    ]
    assert compute_uhi(records, STATIONS, minutes=1440, min_count=1) == [
        (0, 10.0, 5.0, 5.0)
    ]


def test_minute_buckets():
    records = [
        ("u1", 0, 1.0), ("u2", 59, 1.2),
        ("r1", 0, 0.0), ("r2", 59, 0.2),
        ("u1", 60, 3.0), ("u2", 119, 3.2),
        ("r1", 60, 2.0), ("r2", 119, 2.2),
    ]
    assert compute_uhi(records, STATIONS, minutes=1) == [
        (0, 1.1, 0.1, 1.0),
        (60, 3.1, 2.1, 1.0),
    ]


def test_results_sorted_by_bucket():
    records = [
        ("u1", 7200, 1.0), ("u2", 7200, 1.0),
        ("r1", 7200, 0.0), ("r2", 7200, 0.0),
        ("u1", 0, 3.0), ("u2", 0, 3.0),
        ("r1", 0, 2.0), ("r2", 0, 2.0),
    ]
    assert [row[0] for row in compute_uhi(records, STATIONS)] == [0, 7200]


def test_min_count_counts_distinct_stations_not_records():
    records = [
        ("u1", 0, 1.0), ("u1", 10, 3.0),  # only one urban station
        ("r1", 0, 2.0), ("r2", 0, 2.0),
    ]
    assert compute_uhi(records, STATIONS) == []


def test_mapping_records():
    records = [
        {"station_id": "u1", "timestamp": 0, "temp_c": 1.0},
        {"station_id": "u2", "timestamp": 0, "temp_c": 1.0},
        {"station_id": "r1", "timestamp": 0, "temp_c": 0.0},
        {"station_id": "r2", "timestamp": 0, "temp_c": 0.0},
    ]
    assert compute_uhi(records, STATIONS) == [(0, 1.0, 0.0, 1.0)]


def test_result_rows_are_tuples_of_floats():
    records = [
        ("u1", 0, 2), ("u2", 0, 4),
        ("r1", 0, 1), ("r2", 0, 3),
    ]
    rows = compute_uhi(records, STATIONS)
    assert rows == [(0, 3.0, 2.0, 1.0)]
    (row,) = rows
    assert type(row) is tuple
    assert isinstance(row[0], int)
    assert all(isinstance(v, float) for v in row[1:])


def test_uhi_uses_unrounded_means():
    # urban mean 1.0006 -> 1.001, rural mean 1.0004 -> 1.000,
    # but unrounded difference 0.0002 -> 0.0
    records = [
        ("u1", 0, 1.0006), ("u2", 0, 1.0006),
        ("r1", 0, 1.0004), ("r2", 0, 1.0004),
    ]
    assert compute_uhi(records, STATIONS) == [(0, 1.001, 1.0, 0.0)]


def test_round_half_even():
    stations = {"u": "urban", "r": "rural"}
    rows = compute_uhi(
        [("u", 0, 2.6785), ("r", 0, 0.0)], stations, min_count=1
    )
    assert rows == [(0, 2.678, 0.0, 2.678)]


def test_negative_zero_becomes_positive_zero():
    records = [
        ("u1", 0, -0.0005), ("u2", 0, 0.0005),
        ("r1", 0, 0.0), ("r2", 0, 0.0),
    ]
    rows = compute_uhi(records, STATIONS)
    assert rows == [(0, 0.0, 0.0, 0.0)]
    assert all(math.copysign(1.0, v) == 1.0 for v in rows[0][1:])


def test_keyword_only_optional_arguments():
    with pytest.raises(TypeError):
        compute_uhi([], STATIONS, 60, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes": 0},
        {"minutes": 7},
        {"minutes": 1441},
        {"minutes": True},
        {"minutes": 60.0},
        {"minutes": "60"},
        {"min_count": 0},
        {"min_count": False},
        {"min_count": 1.0},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        compute_uhi([], STATIONS, **kwargs)


def test_top_level_container_types():
    with pytest.raises(TypeError):
        compute_uhi((), STATIONS)
    with pytest.raises(TypeError):
        compute_uhi([], [])


@pytest.mark.parametrize(
    "record",
    [
        ("x", 0, 1.0),          # unknown station
        ("u1", -1, 1.0),        # negative timestamp
        ("u1", 1.5, 1.0),       # non-integral timestamp
        ("u1", True, 1.0),      # bool timestamp
        ("u1", 0, float("nan")),
        ("u1", 0, float("inf")),
        ("u1", 0, True),        # bool temperature
        ("u1", 0, 1 + 2j),
        ("u1", 0, "1.0"),
        ("u1", 0),              # wrong arity
        123,
        None,
        {},
        {"station_id": "u1", "timestamp": 0, "temp_c": 1.0, "extra": 1},
        {"station_id": "u1", "timestamp": 0},
    ],
)
def test_invalid_records(record):
    with pytest.raises(ValueError):
        compute_uhi([record], STATIONS)


def test_invalid_station_zone():
    with pytest.raises(ValueError):
        compute_uhi([], {"u1": "suburb"})
