"""Urban heat island (UHI) computation from station temperature records."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from decimal import Decimal, ROUND_HALF_EVEN
from math import isfinite
from typing import Any

__all__ = ["compute_uhi"]

_FIELDS = ("station_id", "timestamp", "temp_c")
_QUANT = Decimal("0.001")


def _is_plain_int(x: Any) -> bool:
    """True for ints but not bool (bool is a subclass of int)."""
    return isinstance(x, int) and not isinstance(x, bool)


def _q(value: Decimal) -> float:
    """Quantize a Decimal to 3 fractional digits (half-even) and return a plain float."""
    f = float(value.quantize(_QUANT, rounding=ROUND_HALF_EVEN))
    return 0.0 if f == 0 else f


def compute_uhi(records, stations, *, minutes=60, min_count=2):
    """Compute per-time-bucket urban/rural mean temperatures and the UHI.

    Records are bucketed by Unix epoch time (floor division), temperatures are
    averaged per station within a bucket, then per-zone station means are taken.
    A bucket is emitted only when both zones have at least ``min_count`` distinct
    stations reporting.
    """
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    if not isinstance(stations, dict):
        raise TypeError("stations must be a dict")

    if not _is_plain_int(minutes) or not (1 <= minutes <= 1440) or 1440 % minutes != 0:
        raise ValueError("minutes must be an integer in 1..1440 that divides 1440")
    if not _is_plain_int(min_count) or min_count < 1:
        raise ValueError("min_count must be a positive integer")

    for zone in stations.values():
        if zone not in ("urban", "rural"):
            raise ValueError("stations values must be 'urban' or 'rural'")

    bucket_seconds = minutes * 60

    # (station_id, bucket_start) -> list of Decimal temperatures
    station_temps: dict[tuple[Any, int], list[Decimal]] = defaultdict(list)

    for record in records:
        if isinstance(record, Mapping):
            if set(record) != set(_FIELDS):
                raise ValueError(
                    "mapping records must contain exactly the keys "
                    "station_id, timestamp, temp_c"
                )
            station_id = record["station_id"]
            timestamp = record["timestamp"]
            temp_c = record["temp_c"]
        elif (
            isinstance(record, Sequence)
            and not isinstance(record, (str, bytes, bytearray))
            and len(record) == 3
        ):
            station_id, timestamp, temp_c = record
        else:
            raise ValueError(
                "each record must be a mapping with station_id, timestamp, "
                "temp_c or a 3-item sequence"
            )

        try:
            known = station_id in stations
        except TypeError as exc:
            raise ValueError("station_id must be a valid station key") from exc
        if not known:
            raise ValueError(f"unknown station_id: {station_id!r}")

        if not _is_plain_int(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be a non-negative integer of Unix seconds")

        if (
            isinstance(temp_c, bool)
            or not isinstance(temp_c, (int, float))
            or not isfinite(temp_c)
        ):
            raise ValueError("temp_c must be a finite number")

        bucket = (timestamp // bucket_seconds) * bucket_seconds
        station_temps[(station_id, bucket)].append(Decimal(str(temp_c)))

    # Per-station, per-bucket averages.
    station_means = {
        key: sum(values, Decimal(0)) / Decimal(len(values))
        for key, values in station_temps.items()
    }

    # Per-bucket, per-zone lists of station means.
    buckets: dict[int, dict[str, list[Decimal]]] = defaultdict(
        lambda: {"urban": [], "rural": []}
    )
    for (station_id, bucket), mean in station_means.items():
        buckets[bucket][stations[station_id]].append(mean)

    result = []
    for bucket in sorted(buckets):
        urban_vals = buckets[bucket]["urban"]
        rural_vals = buckets[bucket]["rural"]
        if len(urban_vals) < min_count or len(rural_vals) < min_count:
            continue

        n_urban = Decimal(len(urban_vals))
        n_rural = Decimal(len(rural_vals))
        urban_mean = sum(urban_vals, Decimal(0)) / n_urban
        rural_mean = sum(rural_vals, Decimal(0)) / n_rural
        uhi = urban_mean - rural_mean

        result.append(
            (bucket, _q(urban_mean), _q(rural_mean), _q(uhi))
        )

    return result
