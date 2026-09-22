"""Urban heat island (UHI) intensity computation."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
import json
import math

__all__ = [
    "compute_uhi",
    "align_temp",
    "grid_features",
    "fit_uhi_model",
    "scenario",
    "attribute_effects",
    "effect_report",
    "effect_report_csv",
    "effect_jackknife_report",
    "effect_permutation_report",
    "effect_trend_report",
]

_RECORD_KEYS = frozenset({"station_id", "timestamp", "temp_c"})
_SATELLITE_KEYS = frozenset({"cell_id", "timestamp", "lst_c"})
_ZONES = frozenset({"urban", "rural"})
_ITEM_KINDS = frozenset({"building", "impervious", "green", "other"})
_COVERAGE_KINDS = frozenset({"impervious", "green", "other"})
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


def _validate_min_points(min_points: object) -> int:
    if isinstance(min_points, bool) or not isinstance(min_points, int):
        raise ValueError("min_points must be an integer")
    if min_points < 2:
        raise ValueError("min_points must be an integer greater than or equal to 2")
    return min_points


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


def _validate_aligned_row(row: object) -> tuple[int, str, float, float, float]:
    """Validate one ``align_temp`` ``(t, c, s, l, d)`` five-tuple."""
    if not isinstance(row, tuple) or len(row) != 5:
        raise ValueError(
            "each aligned row must be an align_temp "
            "(timestamp, cell_id, station_c, satellite_c, diff_c) five-tuple"
        )
    timestamp, cell_id, station, satellite, diff = row
    _validate_timestamp(timestamp)
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")
    _validate_temperature(station, "station_c")
    _validate_temperature(satellite, "satellite_c")
    _validate_temperature(diff, "diff_c")
    return timestamp, cell_id, station, satellite, diff


def _validate_grid_number(value: object, name: str, *, positive: bool) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite int or float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    decimal_value = Decimal(str(value))
    if positive:
        if decimal_value <= 0:
            raise ValueError(f"{name} must be greater than 0")
    elif decimal_value < 0:
        raise ValueError(f"{name} must be non-negative")
    return decimal_value


def _validate_grid_item(item: object) -> tuple[str, str, Decimal, Decimal, Decimal]:
    if not isinstance(item, tuple) or len(item) != 5:
        raise ValueError(
            "each item must be a (cell_id, kind, height, area, grid_area) tuple"
        )
    cell_id, kind, height, area, grid_area = item
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")
    _check_hashable(kind, "kind")
    if kind not in _ITEM_KINDS:
        raise ValueError(
            "kind must be one of 'building', 'impervious', 'green' or 'other'"
        )
    height_d = _validate_grid_number(height, "height", positive=False)
    area_d = _validate_grid_number(area, "area", positive=False)
    grid_area_d = _validate_grid_number(grid_area, "grid_area", positive=True)
    if kind != "building" and height_d != 0:
        raise ValueError("height of non-building items must be 0")
    return cell_id, kind, height_d, area_d, grid_area_d


def _grid_precision(decimals: list[Decimal]) -> int:
    """Precision wide enough for exact sums/products and guarded division,
    independent of the default decimal context."""
    max_adj = 0
    min_exp = 0
    n = 0
    for value in decimals:
        n += 1
        if not value.is_zero():
            max_adj = max(max_adj, value.adjusted())
            min_exp = min(min_exp, value.as_tuple().exponent)
    span = max_adj - min_exp + 1
    # products join two spans; summing n values adds log10(n) digits; keep a
    # guard margin for division and the 4-decimal quantization.
    return 2 * span + 5 * len(str(max(n, 1))) + 20


def _quantize4(value: Decimal) -> float:
    result = float(value.quantize(_QUANT4, rounding=ROUND_HALF_EVEN))
    return 0.0 if result == 0.0 else result


def grid_features(
    aligned: list, items: list
) -> list[tuple[int, object, float, float, float, float, float, float, float]]:
    """Attach grid-geometry features to aligned temperature rows.

    ``aligned`` is a list of ``align_temp`` five-tuples
    ``(timestamp, cell_id, station_c, satellite_c, diff_c)``. ``items`` is a
    list of ``(cell_id, kind, height, area, grid_area)`` tuples, where kind is
    ``building``, ``impervious``, ``green`` or ``other``. Heights and areas are
    non-negative finite numbers (booleans rejected); ``grid_area`` must be
    positive; non-building items must have height 0. All items sharing a
    cell_id must have the same grid area (compared via ``Decimal(str(x))``);
    per cell, both the total building area and the total non-building area
    must not exceed the grid area.

    Non-building kinds are coverage items (zero-area items still count as
    present). A grid cell is kept only when it has at least one building
    item, at least one coverage item and the coverage area equals the grid
    area. For each kept cell referenced by an aligned row, four features are
    appended: the area-weighted mean building height
    ``sum(h*a)/sum(a)`` (0 when the total building area is 0), and the
    building, impervious and green area fractions (each sum divided by the
    grid area). All sums, products, comparisons and division use
    ``Decimal(str(x))`` under a raised-precision local context; features are
    quantized to 4 decimals (ROUND_HALF_EVEN) and returned as floats, with
    negative zero normalized to ``0.0``. Rows are sorted by timestamp then
    cell_id.
    """
    if not isinstance(aligned, list):
        raise TypeError("aligned must be a list")
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    aligned_rows = [_validate_aligned_row(row) for row in aligned]

    parsed = [_validate_grid_item(item) for item in items]

    precision_inputs: list[Decimal] = []
    for _, _, height, area, grid_area in parsed:
        precision_inputs.extend((height, area, grid_area))

    zero = Decimal(0)
    with localcontext() as ctx:
        ctx.prec = _grid_precision(precision_inputs)
        # cell_id -> {"g", "buildings": [(h, a)], "b_area", "cov_area",
        #              "imp_area", "green_area", "n_build", "n_cov"}
        cells: dict[str, dict] = {}
        for cell_id, kind, height, area, grid_area in parsed:
            entry = cells.get(cell_id)
            if entry is None:
                entry = {
                    "g": grid_area,
                    "buildings": [],
                    "b_area": zero,
                    "cov_area": zero,
                    "imp_area": zero,
                    "green_area": zero,
                    "n_build": 0,
                    "n_cov": 0,
                }
                cells[cell_id] = entry
            elif entry["g"] != grid_area:
                raise ValueError(
                    f"grid_area for cell {cell_id!r} must be consistent across items"
                )
            if kind == "building":
                entry["n_build"] += 1
                entry["b_area"] += area
                entry["buildings"].append((height, area))
            else:
                entry["n_cov"] += 1
                entry["cov_area"] += area
                if kind == "impervious":
                    entry["imp_area"] += area
                elif kind == "green":
                    entry["green_area"] += area

        for cell_id, entry in cells.items():
            if entry["b_area"] > entry["g"]:
                raise ValueError(
                    f"total building area for cell {cell_id!r} exceeds grid_area"
                )
            if entry["cov_area"] > entry["g"]:
                raise ValueError(
                    f"total non-building area for cell {cell_id!r} exceeds grid_area"
                )

        features: dict[str, tuple[float, float, float, float]] = {}
        for cell_id, entry in cells.items():
            if entry["n_build"] < 1 or entry["n_cov"] < 1:
                continue
            if entry["cov_area"] != entry["g"]:
                continue
            grid_area = entry["g"]
            building_area = entry["b_area"]
            if building_area == 0:
                mean_height = zero
            else:
                weighted = zero
                for height, area in entry["buildings"]:
                    weighted += height * area
                mean_height = weighted / building_area
            features[cell_id] = (
                _quantize4(mean_height),
                _quantize4(building_area / grid_area),
                _quantize4(entry["imp_area"] / grid_area),
                _quantize4(entry["green_area"] / grid_area),
            )

    results = []
    for row in aligned_rows:
        cell_features = features.get(row[1])
        if cell_features is None:
            continue
        results.append(
            (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                cell_features[0],
                cell_features[1],
                cell_features[2],
                cell_features[3],
            )
        )
    results.sort(key=lambda row: (row[0], row[1]))
    return results


_QUANT6 = Decimal("0.000001")
_RIDGE_PENALTY = Decimal("0.000001")
_MODEL_PRECISION = 1000
_MODEL_FEATURES = 4


def _validate_model_number(
    value: object, name: str, *, low: Decimal, high: Decimal
) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite int or float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    decimal_value = Decimal(str(value))
    if decimal_value < low or decimal_value > high:
        raise ValueError(f"{name} must be between {low} and {high} inclusive")
    return decimal_value


def _quantize6(value: Decimal) -> float:
    result = float(value.quantize(_QUANT6, rounding=ROUND_HALF_EVEN))
    return 0.0 if result == 0.0 else result


def _validate_feature_row(row: object) -> tuple[int, str, Decimal, Decimal, Decimal, Decimal, Decimal]:
    """Validate one ``(t, c, s, l, d, h, b, i, g)`` nine-tuple.

    Returns ``(timestamp, cell_id, h, b, i, g, d)`` as Decimals, sharing the
    non-empty rows contract of ``fit_uhi_model`` with ``scenario``.
    """
    if not isinstance(row, tuple) or len(row) != 9:
        raise ValueError(
            "each row must be a "
            "(timestamp, cell_id, station_c, satellite_c, d, h, b, i, g) "
            "nine-tuple"
        )
    timestamp, cell_id, station, satellite, diff, height, build, imp, green = row
    _validate_timestamp(timestamp)
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")
    _validate_temperature(station, "station_c")
    _validate_temperature(satellite, "satellite_c")
    diff_d = _validate_temperature(diff, "d")
    height_d = _validate_grid_number(height, "h", positive=False)
    build_d = _validate_model_number(build, "b", low=Decimal(0), high=Decimal(1))
    imp_d = _validate_model_number(imp, "i", low=Decimal(0), high=Decimal(1))
    green_d = _validate_model_number(green, "g", low=Decimal(0), high=Decimal(1))
    return timestamp, cell_id, height_d, build_d, imp_d, green_d, diff_d


def _parse_feature_rows(
    rows: object,
) -> list[tuple[int, str, Decimal, Decimal, Decimal, Decimal, Decimal]]:
    """Validate the non-empty rows contract shared by model fitting and scenarios."""
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    if not rows:
        raise ValueError("rows must not be empty")
    parsed: list[tuple[int, str, Decimal, Decimal, Decimal, Decimal, Decimal]] = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        timestamp, cell_id, height_d, build_d, imp_d, green_d, diff_d = (
            _validate_feature_row(row)
        )
        key = (timestamp, cell_id)
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append((timestamp, cell_id, height_d, build_d, imp_d, green_d, diff_d))
    parsed.sort(key=lambda row: (row[0], row[1]))
    return parsed


def fit_uhi_model(
    rows: list,
) -> tuple[int, float, float, float, float, float, float]:
    """Fit a ridge-regression model of the temperature difference on grid features.

    ``rows`` is a non-empty list of nine-tuples
    ``(timestamp, cell_id, station_c, satellite_c, d, h, b, i, g)`` where the
    first five fields follow the ``align_temp`` row contract, ``h`` (mean
    building height) is a non-negative finite number and ``b``/``i``/``g``
    (building, impervious and green area fractions) are finite numbers in
    ``[0, 1]``; booleans are rejected everywhere. ``(timestamp, cell_id)``
    pairs must be unique.

    The target is ``d`` and the features are ``h``, ``b``, ``i`` and ``g``,
    with an intercept. Rows are processed in ascending ``(timestamp,
    cell_id)`` order. Every number enters the computation as
    ``Decimal(str(x))``; the ridge normal equations are assembled and solved
    under a precision-1000, ROUND_HALF_EVEN local context. The penalty
    ``0.000001`` is added to the four feature diagonal entries (not the
    intercept's) and the system is solved by Gauss-Jordan elimination, each
    pivot chosen as the largest absolute value in the remaining rows, ties
    keeping the earliest row; a zero pivot raises ``ValueError``.

    Returns ``(n, b0, bh, bb, bi, bg, r2)``: the sample count, intercept,
    feature coefficients and ``1 - SSE/SST`` with
    ``SST = sum((d - mean(d)) ** 2)`` (``r2`` is 0 when ``SST`` is 0). The
    final six values are quantized to 6 decimals (ROUND_HALF_EVEN) and
    returned as floats, with negative zero normalized to ``0.0``.
    """
    parsed = _parse_feature_rows(rows)
    n = len(parsed)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zero = Decimal(0)
        one = Decimal(1)
        # Columns: 0 = intercept, 1..4 = h, b, i, g.
        size = _MODEL_FEATURES + 1
        xtx = [[zero] * size for _ in range(size)]
        xty = [zero] * size
        for _, _, height, build, imp, green, diff in parsed:
            features = (one, height, build, imp, green)
            for j in range(size):
                xty[j] += features[j] * diff
                for k in range(j, size):
                    xtx[j][k] += features[j] * features[k]
        for j in range(size):
            for k in range(j):
                xtx[j][k] = xtx[k][j]
        for j in range(1, size):
            xtx[j][j] += _RIDGE_PENALTY

        # Augmented matrix for Gauss-Jordan elimination.
        matrix = [xtx[row_idx] + [xty[row_idx]] for row_idx in range(size)]
        for col in range(size):
            pivot = col
            pivot_magnitude = abs(matrix[col][col])
            for row_idx in range(col + 1, size):
                magnitude = abs(matrix[row_idx][col])
                if magnitude > pivot_magnitude:
                    pivot_magnitude = magnitude
                    pivot = row_idx
            if pivot_magnitude == 0:
                raise ValueError("singular design matrix: zero pivot in normal equations")
            if pivot != col:
                matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            pivot_value = matrix[col][col]
            for k in range(col, size + 1):
                matrix[col][k] /= pivot_value
            for row_idx in range(size):
                if row_idx == col:
                    continue
                factor = matrix[row_idx][col]
                if factor == 0:
                    continue
                for k in range(col, size + 1):
                    matrix[row_idx][k] -= factor * matrix[col][k]

        beta = [matrix[row_idx][size] for row_idx in range(size)]

        diff_sum = zero
        for _, _, _, _, _, _, diff in parsed:
            diff_sum += diff
        diff_mean = diff_sum / n

        sst = zero
        sse = zero
        for _, _, height, build, imp, green, diff in parsed:
            residual = diff - (
                beta[0]
                + beta[1] * height
                + beta[2] * build
                + beta[3] * imp
                + beta[4] * green
            )
            sse += residual * residual
            deviation = diff - diff_mean
            sst += deviation * deviation
        r2 = zero if sst == 0 else one - sse / sst

        return (
            n,
            _quantize6(beta[0]),
            _quantize6(beta[1]),
            _quantize6(beta[2]),
            _quantize6(beta[3]),
            _quantize6(beta[4]),
            _quantize6(r2),
        )


def _validate_finite_number(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite int or float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return Decimal(str(value))


def scenario(
    model: tuple,
    rows: list,
    actions: list,
    *,
    roof_c: float = 0.0,
    material_c: float = 0.0,
) -> tuple[list[tuple], tuple[int, float, float, float]]:
    """Apply green-space, cool-roof and cool-material actions to fitted rows.

    ``model`` is a ``fit_uhi_model`` return value
    ``(n, b0, bh, bb, bi, bg, r2)`` and ``rows`` follows the same non-empty
    ``(timestamp, cell_id, station_c, satellite_c, d, h, b, i, g)`` contract
    as ``fit_uhi_model`` (unique ``(timestamp, cell_id)`` pairs). ``actions``
    is a list of ``(c, G, R, M)`` tuples: ``c`` is a cell id that must be
    unique within ``actions`` and occur in ``rows``; ``G`` (green), ``R``
    (roof) and ``M`` (material) are finite non-boolean numbers in ``[0, 1]``.
    For every row sharing a listed cell, ``G + M <= i`` and ``R <= b`` must
    hold; unlisted cells get ``G = R = M = 0``. ``roof_c`` and
    ``material_c`` are non-negative finite non-boolean cost coefficients
    (keyword-only, default ``0.0``).

    For each row, ``base = b0 + bh*h + bb*b + bi*i + bg*g``,
    ``cg = (bg - bi) * G``, ``cr = -roof_c * R``, ``cm = -material_c * M``,
    ``post = base + cg + cr + cm`` and ``delta = post - base``. All numbers
    enter as ``Decimal(str(x))`` under a precision-1000, ROUND_HALF_EVEN
    local context.

    Returns ``(details, summary)``: ``details`` lists
    ``(timestamp, cell_id, base, post, delta, cg, cr, cm)`` tuples in
    ascending ``(timestamp, cell_id)`` order, with the six numeric fields
    quantized to 6 decimals (ROUND_HALF_EVEN) as floats, negative zero
    normalized to ``0.0``; ``summary`` is
    ``(count, base_mean, post_mean, delta_mean)`` with the means computed
    from the unquantized per-row values before 6-decimal quantization.
    ``rows`` or ``actions`` not being a list raises ``TypeError``; every
    other contract violation raises ``ValueError``.
    """
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    if not isinstance(actions, list):
        raise TypeError("actions must be a list")

    parsed = _parse_feature_rows(rows)

    if not isinstance(model, tuple) or len(model) != 7:
        raise ValueError(
            "model must be a fit_uhi_model (n, b0, bh, bb, bi, bg, r2) seven-tuple"
        )
    model_n, b0_v, bh_v, bb_v, bi_v, bg_v, r2_v = model
    if isinstance(model_n, bool) or not isinstance(model_n, int) or model_n < 1:
        raise ValueError("model n must be a positive integer")
    b0 = _validate_finite_number(b0_v, "b0")
    bh = _validate_finite_number(bh_v, "bh")
    bb = _validate_finite_number(bb_v, "bb")
    bi = _validate_finite_number(bi_v, "bi")
    bg = _validate_finite_number(bg_v, "bg")
    _validate_finite_number(r2_v, "r2")

    roof_cost = _validate_grid_number(roof_c, "roof_c", positive=False)
    material_cost = _validate_grid_number(material_c, "material_c", positive=False)

    zero = Decimal(0)
    one = Decimal(1)
    cell_ids = {cell_id for _, cell_id, *_ in parsed}
    action_map: dict[str, tuple[Decimal, Decimal, Decimal]] = {}
    for item in actions:
        if not isinstance(item, tuple) or len(item) != 4:
            raise ValueError("each action must be a (c, G, R, M) four-tuple")
        cell_id, green_v, roof_v, material_v = item
        _check_hashable(cell_id, "c")
        if cell_id not in cell_ids:
            raise ValueError(f"unknown cell id in action: {cell_id!r}")
        if cell_id in action_map:
            raise ValueError(f"duplicate action cell id: {cell_id!r}")
        green = _validate_model_number(green_v, "G", low=zero, high=one)
        roof = _validate_model_number(roof_v, "R", low=zero, high=one)
        material = _validate_model_number(material_v, "M", low=zero, high=one)
        action_map[cell_id] = (green, roof, material)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        details = []
        base_sum = zero
        post_sum = zero
        delta_sum = zero
        for timestamp, cell_id, height, build, imp, green_frac, _d in parsed:
            green, roof, material = action_map.get(cell_id, (zero, zero, zero))
            if cell_id in action_map and (green + material > imp or roof > build):
                raise ValueError(
                    f"action for cell {cell_id!r} exceeds available area: "
                    "G + M must be <= i and R must be <= b"
                )
            base = b0 + bh * height + bb * build + bi * imp + bg * green_frac
            cg = (bg - bi) * green
            cr = -roof_cost * roof
            cm = -material_cost * material
            post = base + cg + cr + cm
            delta = post - base

            base_sum += base
            post_sum += post
            delta_sum += delta
            details.append(
                (
                    timestamp,
                    cell_id,
                    _quantize6(base),
                    _quantize6(post),
                    _quantize6(delta),
                    _quantize6(cg),
                    _quantize6(cr),
                    _quantize6(cm),
                )
            )

        count = len(details)
        summary = (
            count,
            _quantize6(base_sum / count),
            _quantize6(post_sum / count),
            _quantize6(delta_sum / count),
        )
    return details, summary


_DETAIL_FIELDS = ("base", "post", "delta", "cg", "cr", "cm")


def _validate_detail_row(
    row: object,
) -> tuple[int, str, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]:
    """Validate one ``scenario`` ``(t, c, base, post, delta, cg, cr, cm)`` eight-tuple."""
    if not isinstance(row, tuple) or len(row) != 8:
        raise ValueError(
            "each detail must be a scenario "
            "(timestamp, cell_id, base, post, delta, cg, cr, cm) eight-tuple"
        )
    timestamp, cell_id = row[0], row[1]
    _validate_timestamp(timestamp)
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")
    values = tuple(
        _validate_finite_number(value, name)
        for value, name in zip(row[2:], _DETAIL_FIELDS)
    )
    return (timestamp, cell_id) + values


def attribute_effects(
    details: list,
    *,
    by: str,
    minutes: int = 60,
) -> list[tuple]:
    """Aggregate scenario details and sign-test the deltas per group.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``; the six numeric fields must be finite
    non-boolean int/float values and ``(timestamp, cell_id)`` pairs must be
    unique. An empty list returns ``[]``. ``by`` is ``"time"`` (rows are
    bucketed by Unix epoch, floored to ``minutes``-sized buckets) or
    ``"cell"`` (rows are grouped by cell id); ``minutes`` must be a
    non-boolean integer in ``1..1440`` that divides 1440.

    Returns a list of ``(key, n, base, post, delta, cg, cr, cm, p)``
    nine-tuples sorted by ascending group key, where ``n`` is the group size
    and ``base``..``cm`` are the within-group arithmetic means. ``p`` is the
    two-sided sign-test p-value of the deltas: with ``r``/``s`` the counts of
    positive/negative deltas (zeros ignored), ``m = r + s`` and
    ``q = min(r, s)``, ``p`` is 1 when ``m`` is 0 and
    ``min(1, 2 * sum(C(m, k) for k in 0..q) / 2**m)`` otherwise.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context; the six means and ``p``
    are quantized to 6 decimals (ROUND_HALF_EVEN) and returned as floats,
    with negative zero normalized to ``0.0``. ``details`` not being a list
    raises ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if by not in ("time", "cell"):
        raise ValueError("by must be 'time' or 'cell'")
    minutes = _validate_minutes(minutes)

    if not details:
        return []

    parsed = []
    seen: set[tuple[int, str]] = set()
    for row in details:
        validated = _validate_detail_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append(validated)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        groups: dict[object, list[tuple[Decimal, ...]]] = {}
        for validated in parsed:
            timestamp, cell_id, values = validated[0], validated[1], validated[2:]
            if by == "time":
                group_key = (timestamp // bucket_seconds) * bucket_seconds
            else:
                group_key = cell_id
            groups.setdefault(group_key, []).append(values)

        results = []
        for group_key in sorted(groups):
            rows = groups[group_key]
            n = len(rows)
            means = []
            for index in range(6):
                total = Decimal(0)
                for values in rows:
                    total += values[index]
                means.append(_quantize6(total / n))

            r = sum(1 for values in rows if values[2] > 0)
            s = sum(1 for values in rows if values[2] < 0)
            m = r + s
            if m == 0:
                p = Decimal(1)
            else:
                q = min(r, s)
                tail = sum(math.comb(m, k) for k in range(q + 1))
                p = min(Decimal(1), 2 * Decimal(tail) / Decimal(2**m))
            results.append((group_key, n, *means, _quantize6(p)))
    return results


def _format6(value: Decimal) -> str:
    """Quantize to 6 decimals (ROUND_HALF_EVEN) and render with exactly six
    fractional digits, normalizing negative zero to ``0.000000``."""
    quantized = value.quantize(_QUANT6, rounding=ROUND_HALF_EVEN)
    if quantized == 0:
        quantized = abs(quantized)
    return f"{quantized:.6f}"


def _effect_groups(
    details: list, by: str, minutes: int, z: float
) -> tuple[int, Decimal, list[tuple]]:
    """Validate inputs and compute the per-group statistics shared by
    ``effect_report`` and ``effect_report_csv``.

    Returns ``(minutes, z_value, rows)`` where ``rows`` is a list of
    ``(group_key, n, means, p, se, lower, upper)`` tuples in ascending key
    order; ``means`` holds the six unquantized ``Decimal`` means and ``p``,
    ``se``, ``lower`` and ``upper`` are unquantized ``Decimal`` values.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if by not in ("time", "cell"):
        raise ValueError("by must be 'time' or 'cell'")
    minutes = _validate_minutes(minutes)
    z_value = _validate_finite_number(z, "z")
    if z_value < 0:
        raise ValueError("z must be non-negative")

    parsed = []
    seen: set[tuple[int, str]] = set()
    for row in details:
        validated = _validate_detail_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append(validated)

    rows = []
    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        if parsed:
            bucket_seconds = minutes * 60
            groups: dict[object, list[tuple[Decimal, ...]]] = {}
            for validated in parsed:
                timestamp, cell_id, values = validated[0], validated[1], validated[2:]
                if by == "time":
                    group_key = (timestamp // bucket_seconds) * bucket_seconds
                else:
                    group_key = cell_id
                groups.setdefault(group_key, []).append(values)

            for group_key in sorted(groups):
                group_rows = groups[group_key]
                n = len(group_rows)
                means = []
                for index in range(6):
                    total = Decimal(0)
                    for values in group_rows:
                        total += values[index]
                    means.append(total / n)

                mu = means[2]
                if n > 1:
                    squared = Decimal(0)
                    for values in group_rows:
                        deviation = values[2] - mu
                        squared += deviation * deviation
                    se = (squared / (n * (n - 1))).sqrt()
                else:
                    se = Decimal(0)
                lower = mu - z_value * se
                upper = mu + z_value * se

                r = sum(1 for values in group_rows if values[2] > 0)
                s = sum(1 for values in group_rows if values[2] < 0)
                m = r + s
                if m == 0:
                    p = Decimal(1)
                else:
                    q = min(r, s)
                    tail = sum(math.comb(m, k) for k in range(q + 1))
                    p = min(Decimal(1), 2 * Decimal(tail) / Decimal(2**m))
                rows.append((group_key, n, means, p, se, lower, upper))
    return minutes, z_value, rows


def effect_report(
    details: list,
    *,
    by: str,
    minutes: int = 60,
    z: float = 1.96,
) -> str:
    """Aggregate scenario details per group and emit a compact JSON report.

    ``details`` follows the ``attribute_effects`` contract: a list of
    ``scenario`` eight-tuples ``(timestamp, cell_id, base, post, delta, cg,
    cr, cm)`` with finite non-boolean numeric fields and unique
    ``(timestamp, cell_id)`` pairs; an empty list yields empty ``groups``.
    ``by`` is ``"time"`` (Unix-epoch buckets floored to ``minutes``-sized
    buckets) or ``"cell"`` (grouped by cell id); ``minutes`` must be a
    non-boolean integer in ``1..1440`` that divides 1440. ``z`` is a
    non-boolean finite number greater than or equal to 0.

    Groups are emitted in ascending key order (bucket-start seconds for
    ``time``, cell id strings for ``cell``). The six means and the sign-test
    ``p`` are computed exactly as in ``attribute_effects``. With ``n`` the
    group size and ``mu`` the mean of the deltas, ``se`` is
    ``sqrt(sum((delta - mu) ** 2) / (n * (n - 1)))`` when ``n > 1`` and 0
    otherwise, and ``lower``/``upper`` are ``mu - z * se`` / ``mu + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``by, minutes, z, groups`` and each group object uses the key
    order ``key, n, base, post, delta, cg, cr, cm, p, se, lower, upper``.
    ``z`` and every numeric result are rendered with exactly six decimals,
    negative zero normalized to ``0.000000``. ``details`` not being a list
    raises ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    minutes, z_value, rows = _effect_groups(details, by, minutes, z)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        items = []
        for group_key, n, means, p, se, lower, upper in rows:
            if by == "time":
                key_json = str(group_key)
            else:
                key_json = json.dumps(group_key, ensure_ascii=False)
            items.append(
                '{"key":' + key_json
                + ',"n":' + str(n)
                + ',"base":' + _format6(means[0])
                + ',"post":' + _format6(means[1])
                + ',"delta":' + _format6(means[2])
                + ',"cg":' + _format6(means[3])
                + ',"cr":' + _format6(means[4])
                + ',"cm":' + _format6(means[5])
                + ',"p":' + _format6(p)
                + ',"se":' + _format6(se)
                + ',"lower":' + _format6(lower)
                + ',"upper":' + _format6(upper)
                + '}'
            )

        return (
            '{"by":' + json.dumps(by)
            + ',"minutes":' + str(minutes)
            + ',"z":' + _format6(z_value)
            + ',"groups":[' + ",".join(items) + ']}'
        )


_CSV_HEADER = "key,n,base,post,delta,cg,cr,cm,p,se,lower,upper"


def _csv_field(text: str) -> str:
    """Render one RFC4180 field: double-quote when it contains a comma, a
    double quote or a line break, doubling any inner double quotes."""
    if any(char in text for char in (",", '"', "\r", "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


def effect_report_csv(
    details: list,
    *,
    by: str,
    minutes: int = 60,
    z: float = 1.96,
) -> str:
    """Aggregate scenario details per group and emit an RFC4180 CSV report.

    ``details``, ``by``, ``minutes`` and ``z`` follow the ``effect_report``
    contract exactly, and the grouping, ordering, means, ``p``, ``se``,
    ``lower`` and ``upper`` values are computed identically: every number
    enters as ``Decimal(str(x))`` under a precision-1000, ROUND_HALF_EVEN
    local context.

    Returns a UTF-8 RFC4180 CSV string whose first line is the fixed header
    ``key,n,base,post,delta,cg,cr,cm,p,se,lower,upper``, followed by one
    line per group in ascending key order (an empty ``details`` still
    yields the header). Lines end with CRLF, including the final line.
    ``time`` keys are rendered as decimal bucket-start seconds, ``cell``
    keys as the original cell id strings; any field containing a comma, a
    double quote or a line break is wrapped in double quotes with inner
    double quotes doubled. ``n`` is a decimal integer and every other
    numeric field is rendered with exactly six decimals, negative zero
    normalized to ``0.000000``. ``details`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    _, _, rows = _effect_groups(details, by, minutes, z)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        lines = [_CSV_HEADER]
        for group_key, n, means, p, se, lower, upper in rows:
            if by == "time":
                key_field = str(group_key)
            else:
                key_field = _csv_field(group_key)
            lines.append(
                key_field
                + "," + str(n)
                + "," + _format6(means[0])
                + "," + _format6(means[1])
                + "," + _format6(means[2])
                + "," + _format6(means[3])
                + "," + _format6(means[4])
                + "," + _format6(means[5])
                + "," + _format6(p)
                + "," + _format6(se)
                + "," + _format6(lower)
                + "," + _format6(upper)
            )
        return "\r\n".join(lines) + "\r\n"


def effect_jackknife_report(
    details: list,
    *,
    by: str,
    minutes: int = 60,
    z: float = 1.96,
) -> str:
    """Aggregate scenario deltas per group and emit a jackknife JSON report.

    ``details`` follows the ``effect_report`` contract: a list of
    ``scenario`` eight-tuples ``(timestamp, cell_id, base, post, delta, cg,
    cr, cm)`` with finite non-boolean numeric fields and unique
    ``(timestamp, cell_id)`` pairs; an empty list yields empty ``groups``.
    ``by`` is ``"time"`` (Unix-epoch buckets floored to ``minutes``-sized
    buckets, with key ``floor(t / (minutes * 60)) * (minutes * 60)``) or
    ``"cell"`` (grouped by cell id, key ``c``); ``minutes`` must be a
    non-boolean integer in ``1..1440`` that divides 1440. ``z`` is a
    non-boolean finite number greater than or equal to 0.

    Groups are emitted in ascending key order (bucket-start seconds for
    ``time``, cell id strings for ``cell``). With ``d`` the per-row deltas
    and ``n`` the group size, ``delta`` is ``mu = sum(d) / n``; the
    jackknife standard error is 0 when ``n <= 1`` and otherwise
    ``sqrt((n - 1) / n * sum((mu_i - mu_bar) ** 2))`` where
    ``mu_i = (sum(d) - d_i) / (n - 1)`` and ``mu_bar = sum(mu_i) / n``;
    ``lower``/``upper`` are ``mu - z * se`` / ``mu + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``by, minutes, z, groups`` and each group object uses the key
    order ``key, n, delta, se, lower, upper``. ``z`` and every numeric
    result are rendered with exactly six decimals, negative zero
    normalized to ``0.000000``. ``details`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if by not in ("time", "cell"):
        raise ValueError("by must be 'time' or 'cell'")
    minutes = _validate_minutes(minutes)
    z_value = _validate_finite_number(z, "z")
    if z_value < 0:
        raise ValueError("z must be non-negative")

    parsed = []
    seen: set[tuple[int, str]] = set()
    for row in details:
        validated = _validate_detail_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append(validated)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        groups: dict[object, list[Decimal]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id = validated[0], validated[1]
                delta = validated[4]
                if by == "time":
                    group_key = (timestamp // bucket_seconds) * bucket_seconds
                else:
                    group_key = cell_id
                groups.setdefault(group_key, []).append(delta)

        items = []
        for group_key in sorted(groups):
            deltas = groups[group_key]
            n = len(deltas)
            total = Decimal(0)
            for delta in deltas:
                total += delta
            mu = total / n
            if n > 1:
                leave_one_means = [(total - delta) / (n - 1) for delta in deltas]
                mean_sum = Decimal(0)
                for leave_one in leave_one_means:
                    mean_sum += leave_one
                mean_of_means = mean_sum / n
                squared = Decimal(0)
                for leave_one in leave_one_means:
                    deviation = leave_one - mean_of_means
                    squared += deviation * deviation
                se = (Decimal(n - 1) / n * squared).sqrt()
            else:
                se = Decimal(0)
            lower = mu - z_value * se
            upper = mu + z_value * se

            if by == "time":
                key_json = str(group_key)
            else:
                key_json = json.dumps(group_key, ensure_ascii=False)
            items.append(
                '{"key":' + key_json
                + ',"n":' + str(n)
                + ',"delta":' + _format6(mu)
                + ',"se":' + _format6(se)
                + ',"lower":' + _format6(lower)
                + ',"upper":' + _format6(upper)
                + '}'
            )

        return (
            '{"by":' + json.dumps(by)
            + ',"minutes":' + str(minutes)
            + ',"z":' + _format6(z_value)
            + ',"groups":[' + ",".join(items) + ']}'
        )


_PERMUTATION_MAX_N = 16


def effect_permutation_report(
    details: list,
    *,
    by: str,
    minutes: int = 60,
    z: float = 1.96,
) -> str:
    """Aggregate scenario deltas per group and emit a permutation JSON report.

    ``details`` follows the ``effect_report`` contract: a list of
    ``scenario`` eight-tuples ``(timestamp, cell_id, base, post, delta, cg,
    cr, cm)`` with finite non-boolean numeric fields and unique
    ``(timestamp, cell_id)`` pairs; an empty list yields empty ``groups``.
    ``by`` is ``"time"`` (Unix-epoch buckets floored to ``minutes``-sized
    buckets, with key ``floor(t / (minutes * 60)) * (minutes * 60)``) or
    ``"cell"`` (grouped by cell id, key ``c``); ``minutes`` must be a
    non-boolean integer in ``1..1440`` that divides 1440. ``z`` is a
    non-boolean finite number greater than or equal to 0. Every group must
    contain at most 16 rows; a larger group raises ``ValueError``.

    Groups are emitted in ascending key order (bucket-start seconds for
    ``time``, cell id strings for ``cell``). With ``d`` the per-row deltas
    and ``n`` the group size, ``delta`` is ``sum(d) / n``; ``p`` is the
    exact two-sided sign-flip permutation p-value: all ``2 ** n`` sign
    vectors ``s_i`` in ``{-1, 1}`` are enumerated and
    ``p = 2 ** -n * #{|sum(s_i * d_i) / n| >= |delta|}``. The jackknife
    standard error is 0 when ``n <= 1`` and otherwise
    ``sqrt((n - 1) / n * sum((mu_i - mu_bar) ** 2))`` where
    ``mu_i = (sum(d) - d_i) / (n - 1)`` and ``mu_bar = sum(mu_i) / n``;
    ``lower``/``upper`` are ``delta - z * se`` / ``delta + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``by, minutes, z, groups`` and each group object uses the key
    order ``key, n, delta, p, se, lower, upper``. ``z`` and every numeric
    result are rendered with exactly six decimals, negative zero
    normalized to ``0.000000``; ``cell`` keys render as the original cell
    id strings. ``details`` not being a list raises ``TypeError``; every
    other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if by not in ("time", "cell"):
        raise ValueError("by must be 'time' or 'cell'")
    minutes = _validate_minutes(minutes)
    z_value = _validate_finite_number(z, "z")
    if z_value < 0:
        raise ValueError("z must be non-negative")

    parsed = []
    seen: set[tuple[int, str]] = set()
    for row in details:
        validated = _validate_detail_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append(validated)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        groups: dict[object, list[Decimal]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id = validated[0], validated[1]
                delta = validated[4]
                if by == "time":
                    group_key = (timestamp // bucket_seconds) * bucket_seconds
                else:
                    group_key = cell_id
                groups.setdefault(group_key, []).append(delta)

        items = []
        for group_key in sorted(groups):
            deltas = groups[group_key]
            n = len(deltas)
            if n > _PERMUTATION_MAX_N:
                raise ValueError(
                    f"group {group_key!r} has {n} rows; permutation report "
                    f"requires at most {_PERMUTATION_MAX_N} rows per group"
                )
            total = Decimal(0)
            for delta in deltas:
                total += delta
            mu = total / n

            # |sum(s_i * d_i) / n| >= |delta| is equivalent (n > 0) to
            # |sum(s_i * d_i)| >= |sum(d_i)|; compare the raw sums so exact
            # ties are decided without any division rounding.
            hits = 0
            for mask in range(1 << n):
                signed_sum = Decimal(0)
                for index, delta in enumerate(deltas):
                    if (mask >> index) & 1:
                        signed_sum -= delta
                    else:
                        signed_sum += delta
                if abs(signed_sum) >= abs(total):
                    hits += 1
            p = Decimal(hits) / Decimal(1 << n)

            if n > 1:
                leave_one_means = [(total - delta) / (n - 1) for delta in deltas]
                mean_sum = Decimal(0)
                for leave_one in leave_one_means:
                    mean_sum += leave_one
                mean_of_means = mean_sum / n
                squared = Decimal(0)
                for leave_one in leave_one_means:
                    deviation = leave_one - mean_of_means
                    squared += deviation * deviation
                se = (Decimal(n - 1) / n * squared).sqrt()
            else:
                se = Decimal(0)
            lower = mu - z_value * se
            upper = mu + z_value * se

            if by == "time":
                key_json = str(group_key)
            else:
                key_json = json.dumps(group_key, ensure_ascii=False)
            items.append(
                '{"key":' + key_json
                + ',"n":' + str(n)
                + ',"delta":' + _format6(mu)
                + ',"p":' + _format6(p)
                + ',"se":' + _format6(se)
                + ',"lower":' + _format6(lower)
                + ',"upper":' + _format6(upper)
                + '}'
            )

        return (
            '{"by":' + json.dumps(by)
            + ',"minutes":' + str(minutes)
            + ',"z":' + _format6(z_value)
            + ',"groups":[' + ",".join(items) + ']}'
        )


def effect_trend_report(
    details: list,
    *,
    minutes: int = 60,
    min_points: int = 2,
    z: float = 1.96,
) -> str:
    """Aggregate scenario deltas into per-cell time trends and emit JSON.

    ``details`` follows the ``effect_report`` contract: a list of
    ``scenario`` eight-tuples ``(timestamp, cell_id, base, post, delta, cg,
    cr, cm)`` with finite non-boolean numeric fields and unique
    ``(timestamp, cell_id)`` pairs; an empty list yields empty ``groups``.
    ``minutes`` must be a non-boolean integer in ``1..1440`` that divides
    1440; ``min_points`` must be a non-boolean integer greater than or equal
    to 2; ``z`` is a non-boolean finite number greater than or equal to 0.

    For each cell id ``c``, rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within a
    bucket are averaged. Cells with fewer than ``min_points`` occupied
    buckets are omitted; the remaining cells are emitted in ascending cell
    id order. For each kept cell an ordinary least-squares line is fitted to
    the points ``(x, y)`` with ``x`` the bucket start and ``y`` the bucket
    mean delta:

    ``Sxx = sum((x - x_bar) ** 2)``,
    ``Sxy = sum((x - x_bar) * (y - y_bar))``,
    ``slope = Sxy / Sxx``, ``a = y_bar - slope * x_bar`` and
    ``SSE = sum((y - a - slope * x) ** 2)``. The slope standard error is
    ``sqrt(SSE / ((n - 2) * Sxx))`` when ``n > 2`` and 0 otherwise, and
    ``lower``/``upper`` are ``slope - z * se`` / ``slope + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, min_points, z, groups`` and each group object uses
    the key order ``key, n, slope, se, lower, upper`` with ``n`` the number
    of fitted buckets. ``z`` and every numeric result are rendered with
    exactly six decimals, negative zero normalized to ``0.000000``.
    ``details`` not being a list raises ``TypeError``; every other contract
    violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)
    min_points = _validate_min_points(min_points)
    z_value = _validate_finite_number(z, "z")
    if z_value < 0:
        raise ValueError("z must be non-negative")

    parsed = []
    seen: set[tuple[int, str]] = set()
    for row in details:
        validated = _validate_detail_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append(validated)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # cell_id -> bucket start -> [delta sum, row count]
        cells: dict[str, dict[int, list]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id, delta = validated[0], validated[1], validated[4]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                acc = cells.setdefault(cell_id, {}).setdefault(
                    bucket, [Decimal(0), 0]
                )
                acc[0] += delta
                acc[1] += 1

        items = []
        for cell_id in sorted(cells):
            buckets = cells[cell_id]
            if len(buckets) < min_points:
                continue
            points = [
                (Decimal(bucket), total / count)
                for bucket, (total, count) in sorted(buckets.items())
            ]
            n = len(points)
            x_total = Decimal(0)
            y_total = Decimal(0)
            for x, y in points:
                x_total += x
                y_total += y
            x_bar = x_total / n
            y_bar = y_total / n
            sxx = Decimal(0)
            sxy = Decimal(0)
            for x, y in points:
                dx = x - x_bar
                sxx += dx * dx
                sxy += dx * (y - y_bar)
            slope = sxy / sxx
            intercept = y_bar - slope * x_bar
            sse = Decimal(0)
            for x, y in points:
                residual = y - intercept - slope * x
                sse += residual * residual
            if n > 2:
                se = (sse / (Decimal(n - 2) * sxx)).sqrt()
            else:
                se = Decimal(0)
            lower = slope - z_value * se
            upper = slope + z_value * se

            items.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"slope":' + _format6(slope)
                + ',"se":' + _format6(se)
                + ',"lower":' + _format6(lower)
                + ',"upper":' + _format6(upper)
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"min_points":' + str(min_points)
            + ',"z":' + _format6(z_value)
            + ',"groups":[' + ",".join(items) + ']}'
        )
