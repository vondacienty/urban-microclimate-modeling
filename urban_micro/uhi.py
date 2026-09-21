"""Urban heat island (UHI) intensity computation."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal
import math

__all__ = ["compute_uhi"]

_RECORD_KEYS = frozenset({"station_id", "timestamp", "temp_c"})
_ZONES = frozenset({"urban", "rural"})
_QUANT = Decimal("0.001")


def _validate_minutes(minutes: object) -> int:
    if isinstance(minutes, bool) or not isinstance(minutes, int):
        raise ValueError("minutes must be an integer")
    if not 1 <= minutes <= 1440 or 1440 % minutes != 0:
        raise ValueError("minutes must be an integer in 1..1440 that divides 1440")
    return minutes


def _validate_min_count(min_count: object) -> int:
    if isinstance(min_count, bool) or not isinstance(min_count, int):
        raise ValueError("min_count must be an integer")
    if min_count < 1:
        raise ValueError("min_count must be a positive integer")
    return min_count


def _validate_stations(stations: object) -> dict:
    if not isinstance(stations, dict):
        raise TypeError("stations must be a dict mapping station IDs to 'urban' or 'rural'")
    for station_id, zone in stations.items():
        if zone not in _ZONES:
            raise ValueError(
                f"station {station_id!r} must map to 'urban' or 'rural', got {zone!r}"
            )
    return stations


def _validate_record(record: object, stations: dict) -> tuple[object, int, Decimal]:
    if not isinstance(record, Mapping):
        raise ValueError("each record must be a mapping")
    if set(record.keys()) != _RECORD_KEYS:
        raise ValueError(
            "each record must contain exactly the keys "
            "'station_id', 'timestamp' and 'temp_c'"
        )

    station_id = record["station_id"]
    if station_id not in stations:
        raise ValueError(f"unknown station_id: {station_id!r}")

    timestamp = record["timestamp"]
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise ValueError("timestamp must be an integer")
    if timestamp < 0:
        raise ValueError("timestamp must be non-negative")

    temp_c = record["temp_c"]
    if isinstance(temp_c, bool) or not isinstance(temp_c, (int, float)):
        raise ValueError("temp_c must be a number")
    if not math.isfinite(temp_c):
        raise ValueError("temp_c must be finite")

    return station_id, timestamp, Decimal(str(temp_c))


def _quantize3(value: Decimal) -> float:
    result = float(value.quantize(_QUANT, rounding=ROUND_HALF_EVEN))
    return 0.0 if result == 0.0 else result


def compute_uhi(
    records: list,
    stations: dict,
    *,
    minutes: int = 60,
    min_count: int = 2,
) -> list[tuple[int, float, float, float]]:
    """Compute urban heat island intensity per time bucket.

    Records are bucketed by Unix epoch (floor to ``minutes``-sized buckets).
    Within each bucket, temperatures are averaged per station first, then
    per zone (urban/rural). A bucket is emitted only when both zones have at
    least ``min_count`` distinct stations. Returns a list of
    ``(timestamp, urban_mean_c, rural_mean_c, uhi_c)`` tuples sorted by
    bucket start, with temperatures quantized to 3 decimals (ROUND_HALF_EVEN).
    """
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    stations = _validate_stations(stations)
    minutes = _validate_minutes(minutes)
    min_count = _validate_min_count(min_count)

    if not records:
        return []

    bucket_seconds = minutes * 60
    # bucket -> station_id -> [sum, count]
    buckets: dict[int, dict[object, list]] = {}
    for record in records:
        station_id, timestamp, temp = _validate_record(record, stations)
        bucket = (timestamp // bucket_seconds) * bucket_seconds
        station_acc = buckets.setdefault(bucket, {}).setdefault(station_id, [Decimal(0), 0])
        station_acc[0] += temp
        station_acc[1] += 1

    results = []
    for bucket in sorted(buckets):
        zone_sums: dict[str, Decimal] = {"urban": Decimal(0), "rural": Decimal(0)}
        zone_counts: dict[str, int] = {"urban": 0, "rural": 0}
        for station_id, (total, count) in buckets[bucket].items():
            zone = stations[station_id]
            zone_sums[zone] += total / count
            zone_counts[zone] += 1
        if zone_counts["urban"] < min_count or zone_counts["rural"] < min_count:
            continue
        urban_mean = zone_sums["urban"] / zone_counts["urban"]
        rural_mean = zone_sums["rural"] / zone_counts["rural"]
        uhi = urban_mean - rural_mean
        results.append(
            (bucket, _quantize3(urban_mean), _quantize3(rural_mean), _quantize3(uhi))
        )
    return results
