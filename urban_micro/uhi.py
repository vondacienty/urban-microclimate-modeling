"""Urban heat island (UHI) intensity computation."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
import math

__all__ = ["compute_uhi", "align_temp", "grid_features"]

_RECORD_KEYS = frozenset({"station_id", "timestamp", "temp_c"})
_SATELLITE_KEYS = frozenset({"cell_id", "timestamp", "lst_c"})
_ZONES = frozenset({"urban", "rural"})
_GRID_CATEGORIES = frozenset({"building", "impervious", "green", "other"})
_QUANT = Decimal("0.001")
_QUANT4 = Decimal("0.0001")


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


def _validate_cells(cells: object) -> dict:
    if not isinstance(cells, dict):
        raise TypeError("cells must be a dict mapping station IDs to grid cell IDs")
    for station_id, cell_id in cells.items():
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(
                f"station {station_id!r} must map to a non-empty grid cell ID "
                f"string, got {cell_id!r}"
            )
    return cells


def _check_hashable(value: object, name: str) -> None:
    try:
        hash(value)
    except TypeError:
        raise ValueError(f"{name} must be hashable, got {value!r}") from None


def _validate_timestamp(timestamp: object) -> int:
    if isinstance(timestamp, bool) or not isinstance(timestamp, int):
        raise ValueError("timestamp must be an integer")
    if timestamp < 0:
        raise ValueError("timestamp must be non-negative")
    return timestamp


def _validate_temperature(temp: object, name: str) -> Decimal:
    if isinstance(temp, bool) or not isinstance(temp, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(temp):
        raise ValueError(f"{name} must be finite")
    return Decimal(str(temp))


def _validate_record(record: object, stations: dict) -> tuple[object, int, Decimal]:
    if not isinstance(record, Mapping):
        raise ValueError("each record must be a mapping")
    if set(record.keys()) != _RECORD_KEYS:
        raise ValueError(
            "each record must contain exactly the keys "
            "'station_id', 'timestamp' and 'temp_c'"
        )

    station_id = record["station_id"]
    _check_hashable(station_id, "station_id")
    if station_id not in stations:
        raise ValueError(f"unknown station_id: {station_id!r}")

    timestamp = _validate_timestamp(record["timestamp"])
    temp = _validate_temperature(record["temp_c"], "temp_c")
    return station_id, timestamp, temp


def _validate_satellite_record(record: object) -> tuple[object, int, Decimal]:
    if not isinstance(record, Mapping):
        raise ValueError("each satellite record must be a mapping")
    if set(record.keys()) != _SATELLITE_KEYS:
        raise ValueError(
            "each satellite record must contain exactly the keys "
            "'cell_id', 'timestamp' and 'lst_c'"
        )

    cell_id = record["cell_id"]
    _check_hashable(cell_id, "cell_id")

    timestamp = _validate_timestamp(record["timestamp"])
    temp = _validate_temperature(record["lst_c"], "lst_c")
    return cell_id, timestamp, temp


def _precision_for(temps: list[Decimal]) -> int:
    """Precision needed so means and 3-decimal quantization are unaffected by
    the default decimal context.

    Covers the exact digit span of any sum of ``temps`` (most significant to
    least significant digit, plus carry digits from adding ``n`` values), the
    3 quantization decimals and guard digits for division.
    """
    max_adj = 0
    min_exp = 0
    n = 0
    for temp in temps:
        n += 1
        if not temp.is_zero():
            max_adj = max(max_adj, temp.adjusted())
            min_exp = min(min_exp, temp.as_tuple().exponent)
    span = max_adj - min_exp + 1
    return span + 5 * len(str(n)) + 10


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

    Station IDs must be hashable. Means and quantization are computed under a
    local decimal context whose precision is raised to fit the magnitude of
    the input temperatures, independent of the default context.
    """
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    stations = _validate_stations(stations)
    minutes = _validate_minutes(minutes)
    min_count = _validate_min_count(min_count)

    if not records:
        return []

    validated = [_validate_record(record, stations) for record in records]

    with localcontext() as ctx:
        ctx.prec = _precision_for([temp for _, _, temp in validated])
        bucket_seconds = minutes * 60
        # bucket -> station_id -> [sum, count]
        buckets: dict[int, dict[object, list]] = {}
        for station_id, timestamp, temp in validated:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            station_acc = buckets.setdefault(bucket, {}).setdefault(
                station_id, [Decimal(0), 0]
            )
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


def align_temp(
    station: list,
    satellite: list,
    cells: dict,
    *,
    minutes: int = 60,
    min_count: int = 1,
) -> list[tuple[int, str, float, float, float]]:
    """Align station and satellite temperatures per time bucket and grid cell.

    ``station`` records follow the ``compute_uhi`` three-key contract
    (``station_id``/``timestamp``/``temp_c``); ``satellite`` records must
    contain exactly ``cell_id``/``timestamp``/``lst_c``. ``cells`` maps each
    station ID to a non-empty grid cell ID string.

    Both sources are bucketed by Unix epoch (floor to ``minutes``-sized
    buckets). Station temperatures are averaged per station first, then per
    grid cell; satellite temperatures are averaged per grid cell. A
    bucket/cell pair is emitted only when both sources are present and at
    least ``min_count`` distinct stations contribute. Returns a list of
    ``(timestamp, cell_id, station_c, satellite_c, diff_c)`` tuples sorted by
    bucket start then cell ID, where ``diff_c`` is the station mean minus the
    satellite mean computed from the unrounded means. Temperatures are
    computed as ``Decimal(str(x))`` and quantized to 3 decimals
    (ROUND_HALF_EVEN); negative zero is normalized to ``0.0``.
    """
    if not isinstance(station, list):
        raise TypeError("station must be a list")
    if not isinstance(satellite, list):
        raise TypeError("satellite must be a list")
    cells = _validate_cells(cells)
    minutes = _validate_minutes(minutes)
    min_count = _validate_min_count(min_count)

    station_rows = [_validate_record(record, cells) for record in station]
    satellite_rows = [_validate_satellite_record(record) for record in satellite]

    with localcontext() as ctx:
        ctx.prec = _precision_for(
            [temp for _, _, temp in station_rows]
            + [temp for _, _, temp in satellite_rows]
        )
        bucket_seconds = minutes * 60
        # bucket -> cell_id -> station_id -> [sum, count]
        station_buckets: dict[int, dict[object, dict[object, list]]] = {}
        for station_id, timestamp, temp in station_rows:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            cell_id = cells[station_id]
            station_acc = (
                station_buckets.setdefault(bucket, {})
                .setdefault(cell_id, {})
                .setdefault(station_id, [Decimal(0), 0])
            )
            station_acc[0] += temp
            station_acc[1] += 1

        # bucket -> cell_id -> [sum, count]
        satellite_buckets: dict[int, dict[object, list]] = {}
        for cell_id, timestamp, temp in satellite_rows:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            cell_acc = satellite_buckets.setdefault(bucket, {}).setdefault(
                cell_id, [Decimal(0), 0]
            )
            cell_acc[0] += temp
            cell_acc[1] += 1

        results = []
        for bucket in sorted(station_buckets):
            sat_cells = satellite_buckets.get(bucket)
            if not sat_cells:
                continue
            for cell_id in sorted(station_buckets[bucket]):
                if cell_id not in sat_cells:
                    continue
                per_station = station_buckets[bucket][cell_id]
                if len(per_station) < min_count:
                    continue
                station_sum = Decimal(0)
                for total, count in per_station.values():
                    station_sum += total / count
                station_mean = station_sum / len(per_station)
                sat_total, sat_count = sat_cells[cell_id]
                satellite_mean = sat_total / sat_count
                diff = station_mean - satellite_mean
                results.append(
                    (
                        bucket,
                        cell_id,
                        _quantize3(station_mean),
                        _quantize3(satellite_mean),
                        _quantize3(diff),
                    )
                )
    return results


def _validate_dimension(value: object, name: str, *, allow_zero: bool = True) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a non-bool finite int or float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    decimal_value = Decimal(str(value))
    if decimal_value < 0:
        raise ValueError(f"{name} must be non-negative")
    if not allow_zero and decimal_value <= 0:
        raise ValueError(f"{name} must be positive")
    return decimal_value


def _grid_precision_for(
    heights: list[Decimal], areas: list[Decimal], grids: list[Decimal]
) -> int:
    """Precision covering the exact digit span of every sum, product and
    ratio in :func:`grid_features`, independent of the default context.
    """
    add_digits = len(str(max(len(areas), 1)))

    def extrema(values: list[Decimal]) -> tuple[int, int]:
        max_adj = 0
        min_exp = 0
        for value in values:
            if not value.is_zero():
                max_adj = max(max_adj, value.adjusted())
                min_exp = min(min_exp, value.as_tuple().exponent)
        return max_adj, min_exp

    h_adj, h_exp = extrema(heights)
    a_adj, a_exp = extrema(areas)
    g_adj, g_exp = extrema(grids)

    # Most-significant position reachable by area sums and h*a products.
    top = max(a_adj + add_digits + 1, h_adj + a_adj + add_digits + 2, g_adj)
    # Least-significant position touched by sums and products.
    bottom = min(a_exp, h_exp + a_exp, g_exp)
    # Ratios are quantized to 4 decimals; keep guard digits beyond that.
    return max(top - bottom + 1, top + 16)


def _quantize4(value: Decimal) -> float:
    result = float(value.quantize(_QUANT4, rounding=ROUND_HALF_EVEN))
    return 0.0 if result == 0.0 else result


def grid_features(aligned: list, items: list) -> list:
    """Attach building/coverage grid features to ``align_temp`` rows.

    ``aligned`` is a list of ``align_temp`` five-tuples
    ``(timestamp, cell_id, station_c, satellite_c, diff_c)``; ``items`` is a
    list of strict ``(cell_id, kind, height, item_area, grid_area)`` tuples.
    ``cell_id`` must be a non-empty string, ``kind`` one of
    ``building``/``impervious``/``green``/``other``; ``height`` and the two
    areas must be non-bool finite ints/floats, with height/item area
    non-negative, grid area positive, and non-building items having zero
    height. Items sharing a cell must agree on grid area (compared via
    ``Decimal(str(x))``); total building area and total non-building area
    must each not exceed the grid area.

    Non-building items are coverage items (zero-area ones still count). Only
    cells with at least one building item and at least one coverage item, and
    whose coverage area sums exactly to the grid area, are kept. Each
    surviving aligned row gets four features appended: the area-weighted mean
    building height ``sum(h*a)/sum(a)`` (0 when total building area is 0) and
    the building/impervious/green area fractions. All arithmetic is exact
    ``Decimal`` arithmetic; the four features are quantized to 4 decimals
    (ROUND_HALF_EVEN) independent of the default context precision, converted
    to float, and negative zero is normalized to ``0.0``. Sorted by timestamp
    then cell ID.
    """
    if not isinstance(aligned, list):
        raise TypeError("aligned must be a list")
    if not isinstance(items, list):
        raise TypeError("items must be a list")

    # Validate every item first; only Decimal conversions happen here, all
    # sums/products below run under the raised-precision local context.
    parsed: list[tuple[str, str, Decimal, Decimal, Decimal]] = []
    heights: list[Decimal] = []
    areas: list[Decimal] = []
    grids: list[Decimal] = []
    grid_by_cell: dict[str, Decimal] = {}
    for item in items:
        if not isinstance(item, tuple) or len(item) != 5:
            raise ValueError("each item must be a strict (c, k, h, a, g) tuple")
        cell_id, kind, height, item_area, grid_area = item
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell_id must be a non-empty string")
        if kind not in _GRID_CATEGORIES:
            raise ValueError(
                "kind must be one of 'building', 'impervious', 'green' or 'other'"
            )
        h_dec = _validate_dimension(height, "height")
        a_dec = _validate_dimension(item_area, "item_area")
        g_dec = _validate_dimension(grid_area, "grid_area", allow_zero=False)
        if kind != "building" and h_dec != 0:
            raise ValueError("height must be 0 for non-building items")
        previous_g = grid_by_cell.setdefault(cell_id, g_dec)
        if previous_g != g_dec:
            raise ValueError(
                f"grid area for cell {cell_id!r} must be consistent across items"
            )
        parsed.append((cell_id, kind, h_dec, a_dec, g_dec))
        heights.append(h_dec)
        areas.append(a_dec)
        grids.append(g_dec)

    results = []
    with localcontext() as ctx:
        ctx.prec = _grid_precision_for(heights, areas, grids)

        # cell_id -> per-kind area sums, item counts and weighted height sum.
        cells: dict[str, dict] = {}
        for cell_id, kind, h_dec, a_dec, _g_dec in parsed:
            cell = cells.get(cell_id)
            if cell is None:
                cell = {
                    "g": grid_by_cell[cell_id],
                    "sums": {
                        "building": Decimal(0),
                        "impervious": Decimal(0),
                        "green": Decimal(0),
                        "other": Decimal(0),
                    },
                    "counts": {
                        "building": 0,
                        "impervious": 0,
                        "green": 0,
                        "other": 0,
                    },
                    "b_h": Decimal(0),
                }
                cells[cell_id] = cell
            if kind == "building":
                cell["b_h"] += h_dec * a_dec
            cell["sums"][kind] += a_dec
            cell["counts"][kind] += 1

        for cell_id, cell in cells.items():
            g_dec = cell["g"]
            building_sum = cell["sums"]["building"]
            coverage_sum = (
                cell["sums"]["impervious"]
                + cell["sums"]["green"]
                + cell["sums"]["other"]
            )
            if building_sum > g_dec:
                raise ValueError(
                    f"total building area for cell {cell_id!r} exceeds grid area"
                )
            if coverage_sum > g_dec:
                raise ValueError(
                    f"total non-building area for cell {cell_id!r} exceeds grid area"
                )
            cell["cover"] = coverage_sum

        for row in sorted(aligned, key=lambda r: (r[0], r[1])):
            _timestamp, cell_id = row[0], row[1]
            cell = cells.get(cell_id)
            if cell is None:
                continue
            if cell["counts"]["building"] < 1:
                continue
            coverage_count = (
                cell["counts"]["impervious"]
                + cell["counts"]["green"]
                + cell["counts"]["other"]
            )
            if coverage_count < 1:
                continue
            g_dec = cell["g"]
            if cell["cover"] != g_dec:
                continue
            building_sum = cell["sums"]["building"]
            mean_height = (
                cell["b_h"] / building_sum if building_sum != 0 else Decimal(0)
            )
            results.append(
                (
                    row[0],
                    cell_id,
                    row[2],
                    row[3],
                    row[4],
                    _quantize4(mean_height),
                    _quantize4(building_sum / g_dec),
                    _quantize4(cell["sums"]["impervious"] / g_dec),
                    _quantize4(cell["sums"]["green"] / g_dec),
                )
            )
    return results
