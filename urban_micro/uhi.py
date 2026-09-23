"""Urban heat island (UHI) intensity computation."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from itertools import permutations
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
    "effect_bootstrap_report",
    "effect_fdr_report",
    "effect_significance_report",
    "effect_matrix_report",
    "effect_matrix_csv",
    "effect_matrix_significance_report",
    "effect_matrix_fdr_report",
    "effect_matrix_compare_report",
    "effect_matrix_bootstrap_report",
    "effect_matrix_jackknife_report",
    "effect_matrix_permutation_report",
    "effect_matrix_robust_report",
    "effect_matrix_contribution_report",
    "effect_matrix_autocorr_report",
    "effect_matrix_moran_report",
    "effect_matrix_theilsen_report",
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


def effect_fdr_report(
    details: list,
    *,
    by: str,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Aggregate scenario deltas per group and emit an FDR JSON report.

    ``details`` follows the ``effect_report`` contract: a list of
    ``scenario`` eight-tuples ``(timestamp, cell_id, base, post, delta, cg,
    cr, cm)`` with finite non-boolean numeric fields and unique
    ``(timestamp, cell_id)`` pairs; an empty list yields empty ``groups``.
    ``by`` is ``"time"`` (Unix-epoch buckets floored to ``minutes``-sized
    buckets, with key ``floor(t / (minutes * 60)) * (minutes * 60)``) or
    ``"cell"`` (grouped by cell id, key ``c``); ``minutes`` must be a
    non-boolean integer in ``1..1440`` that divides 1440. ``alpha`` is a
    non-boolean finite number with ``0 < alpha <= 1``.

    Groups are emitted in ascending key order (bucket-start seconds for
    ``time``, cell id strings for ``cell``). With ``d`` the per-row deltas
    and ``n`` the group size, ``delta`` is ``sum(d) / n`` and ``p`` is the
    two-sided sign-test p-value of the deltas: with ``r``/``s`` the counts
    of positive/negative deltas (zeros ignored), ``m = r + s`` and
    ``q0 = min(r, s)``, ``p`` is 1 when ``m`` is 0 and
    ``min(1, 2 * sum(C(m, k) for k in 0..q0) / 2**m)`` otherwise.

    With ``N`` the number of groups, the p-values are ranked ascending by
    ``(p, key)`` and each rank ``j`` (1-based) gets the Benjamini-Hochberg
    q-value ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to
    its group key; ``reject`` is ``q <= alpha`` (compared on the unquantized
    values).

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``by, minutes, alpha, groups`` and each group object uses the
    key order ``key, n, delta, p, q, reject``. ``alpha`` and every numeric
    result are rendered with exactly six decimals, negative zero normalized
    to ``0.000000``. ``details`` not being a list raises ``TypeError``;
    every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if by not in ("time", "cell"):
        raise ValueError("by must be 'time' or 'cell'")
    minutes = _validate_minutes(minutes)
    alpha_value = _validate_finite_number(alpha, "alpha")
    if alpha_value <= 0 or alpha_value > 1:
        raise ValueError("alpha must be greater than 0 and at most 1")

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

        # group_key -> (n, mean delta, sign-test p)
        stats: dict[object, tuple[int, Decimal, Decimal]] = {}
        for group_key, deltas in groups.items():
            n = len(deltas)
            total = Decimal(0)
            for delta in deltas:
                total += delta
            mu = total / n
            r = sum(1 for delta in deltas if delta > 0)
            s = sum(1 for delta in deltas if delta < 0)
            m = r + s
            if m == 0:
                p = Decimal(1)
            else:
                q0 = min(r, s)
                tail = sum(math.comb(m, k) for k in range(q0 + 1))
                p = min(Decimal(1), 2 * Decimal(tail) / Decimal(2**m))
            stats[group_key] = (n, mu, p)

        # Benjamini-Hochberg q-values: rank ascending by (p, key), then
        # accumulate the running minimum of N * p_l / l from the top rank down.
        count = len(stats)
        ranked = sorted(stats, key=lambda group_key: (stats[group_key][2], group_key))
        q_values: dict[object, Decimal] = {}
        running = Decimal(1)
        for rank in range(count, 0, -1):
            group_key = ranked[rank - 1]
            candidate = Decimal(count) * stats[group_key][2] / rank
            if candidate < running:
                running = candidate
            q_values[group_key] = running

        items = []
        for group_key in sorted(groups):
            n, mu, p = stats[group_key]
            q_value = q_values[group_key]
            reject = q_value <= alpha_value
            if by == "time":
                key_json = str(group_key)
            else:
                key_json = json.dumps(group_key, ensure_ascii=False)
            items.append(
                '{"key":' + key_json
                + ',"n":' + str(n)
                + ',"delta":' + _format6(mu)
                + ',"p":' + _format6(p)
                + ',"q":' + _format6(q_value)
                + ',"reject":' + ("true" if reject else "false")
                + '}'
            )

        return (
            '{"by":' + json.dumps(by)
            + ',"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(items) + ']}'
        )


def effect_significance_report(
    details: list,
    *,
    by: str,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Aggregate scenario deltas per group and z-test the means, as JSON.

    ``details`` follows the ``effect_report`` contract: a list of
    ``scenario`` eight-tuples ``(timestamp, cell_id, base, post, delta, cg,
    cr, cm)`` with finite non-boolean numeric fields and unique
    ``(timestamp, cell_id)`` pairs; an empty list yields empty ``groups``.
    ``by`` is ``"time"`` (Unix-epoch buckets floored to ``minutes``-sized
    buckets, with key ``floor(t / (minutes * 60)) * (minutes * 60)``) or
    ``"cell"`` (grouped by cell id, key ``c``); ``minutes`` must be a
    non-boolean integer in ``1..1440`` that divides 1440. ``alpha`` is a
    non-boolean finite number with ``0 < alpha <= 1``.

    Groups are emitted in ascending key order (bucket-start seconds for
    ``time``, cell id strings for ``cell``). With ``d`` the per-row deltas
    and ``n`` the group size, ``delta`` is ``mu = sum(d) / n`` and ``se`` is
    ``sqrt(sum((d - mu) ** 2) / (n * (n - 1)))`` when ``n > 1`` and 0
    otherwise. When ``se`` is 0, ``z`` is 0 and ``p`` is 1; otherwise
    ``z = mu / se`` and ``p = erfc(abs(float(z)) / sqrt(2))`` (the two-sided
    normal p-value). ``significant`` is ``p <= alpha``, compared on the
    unquantized values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``by, minutes, alpha, groups`` and each group object uses the
    key order ``key, n, delta, se, z, p, significant``. ``alpha`` and every
    numeric result are rendered with exactly six decimals, negative zero
    normalized to ``0.000000``. ``details`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if by not in ("time", "cell"):
        raise ValueError("by must be 'time' or 'cell'")
    minutes = _validate_minutes(minutes)
    alpha_value = _validate_finite_number(alpha, "alpha")
    if alpha_value <= 0 or alpha_value > 1:
        raise ValueError("alpha must be greater than 0 and at most 1")

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
                squared = Decimal(0)
                for delta in deltas:
                    deviation = delta - mu
                    squared += deviation * deviation
                se = (squared / (n * (n - 1))).sqrt()
            else:
                se = Decimal(0)
            if se == 0:
                z_value = Decimal(0)
                p_value = Decimal(1)
            else:
                z_value = mu / se
                p_value = Decimal(
                    str(math.erfc(abs(float(z_value)) / math.sqrt(2)))
                )
            significant = p_value <= alpha_value

            if by == "time":
                key_json = str(group_key)
            else:
                key_json = json.dumps(group_key, ensure_ascii=False)
            items.append(
                '{"key":' + key_json
                + ',"n":' + str(n)
                + ',"delta":' + _format6(mu)
                + ',"se":' + _format6(se)
                + ',"z":' + _format6(z_value)
                + ',"p":' + _format6(p_value)
                + ',"significant":' + ("true" if significant else "false")
                + '}'
            )

        return (
            '{"by":' + json.dumps(by)
            + ',"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(items) + ']}'
        )


_BOOTSTRAP_MAX_N = 8


def _bootstrap_interval(
    deltas: list[Decimal], n: int, confidence_value: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    """Return ``(delta, lower, upper)`` for one bootstrap group.

    ``delta`` is the sample mean ``sum(deltas) / n``; ``lower``/``upper`` are
    the interpolated quantiles of the sorted multiset of the ``n ** n``
    resample means at ``q = (1 - confidence) / 2`` and ``1 - q``. The sorted
    multiset is built from the draw-count compositions (each composition
    ``c`` with sum ``n`` has multiplicity ``n! / prod(c_j!)``) instead of
    materializing all ``n ** n`` means, giving an identical result.
    """
    total = Decimal(0)
    for delta in deltas:
        total += delta
    mu = total / n

    # The sorted multiset of the n**n resample means only depends on how
    # many times each delta is drawn: a composition (c_0, ..., c_{n-1}) with
    # sum c_j = n contributes the mean sum(c_j * d_j) / n with multiplicity
    # n! / prod(c_j!).
    weighted_sums: list[tuple[Decimal, int]] = []
    counts = [0] * n
    factorial_n = math.factorial(n)

    def enumerate_compositions(index: int, remaining: int) -> None:
        if index == n - 1:
            counts[index] = remaining
            weighted_sum = Decimal(0)
            multiplicity = factorial_n
            for j, count_j in enumerate(counts):
                if count_j:
                    weighted_sum += count_j * deltas[j]
                    multiplicity //= math.factorial(count_j)
            weighted_sums.append((weighted_sum, multiplicity))
            return
        for count_j in range(remaining + 1):
            counts[index] = count_j
            enumerate_compositions(index + 1, remaining - count_j)

    enumerate_compositions(0, n)
    weighted_sums.sort(key=lambda item: item[0])

    # Merge equal means into (sorted mean numerator, cumulative count);
    # block_sums[k] repeats block_cumulative[k] - block_cumulative[k-1]
    # times in the sorted n**n-length sequence of resample means.
    block_sums: list[Decimal] = []
    block_cumulative: list[int] = []
    running = 0
    for weighted_sum, multiplicity in weighted_sums:
        running += multiplicity
        if block_sums and block_sums[-1] == weighted_sum:
            block_cumulative[-1] = running
        else:
            block_sums.append(weighted_sum)
            block_cumulative.append(running)

    def value_at(position: int) -> Decimal:
        block = bisect_left(block_cumulative, position + 1)
        return block_sums[block] / n

    resample_count = n**n
    q_value = (Decimal(1) - confidence_value) / 2
    r_value = (Decimal(resample_count) - 1) * q_value
    floor_index = int(r_value.to_integral_value(rounding=ROUND_FLOOR))
    lower = value_at(floor_index)
    upper = value_at(resample_count - 1 - floor_index)
    weight = r_value - Decimal(floor_index)
    if weight != 0:
        # r is non-integral: interpolate against ceil(r) using the unrounded
        # fractional part of r; the symmetric upper bound interpolates with
        # the same weight at N-1-floor(r).
        lower = lower + weight * (value_at(floor_index + 1) - lower)
        upper = upper + weight * (
            value_at(resample_count - 2 - floor_index) - upper
        )
    return mu, lower, upper


def effect_bootstrap_report(
    details: list,
    *,
    confidence: float = 0.95,
) -> str:
    """Group scenario deltas by cell id and emit a bootstrap CI JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. Timestamps do not participate in grouping: rows are
    grouped by ``cell_id`` only, and groups are emitted in ascending cell id
    order. An empty list yields empty ``groups``. Every group must contain at
    most 8 rows; a larger group raises ``ValueError``. ``confidence`` is a
    non-boolean finite number with ``0 < confidence < 1``.

    With ``d`` the per-row deltas and ``n`` the group size, ``delta`` is the
    original sample mean ``sum(d) / n``. The multiset of resample means is
    defined by the ``n ** n`` index tuples with replacement (the base-``n``
    digits of ``k`` for ``k`` in ``0 .. n**n - 1``, in index-dictionary
    order), each giving the mean ``sum(d_i for i in digits) / n``; the means
    are then sorted. With ``q = (1 - confidence) / 2`` and
    ``r = (n ** n - 1) * q``, ``lower`` and ``upper`` linearly interpolate
    the sorted resample means between the positions ``floor(r)`` and
    ``ceil(r)``, taking the value at that position when ``r`` is integral.
    The sorted multiset is built directly from the draw-count compositions
    (each composition ``c`` with sum ``n`` has multiplicity
    ``n! / prod(c_j!)``), so the result is identical to the full
    enumeration without materializing all ``n ** n`` means.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``confidence, groups`` and each group object uses the key order
    ``key, n, delta, lower, upper``. ``confidence`` and every numeric result
    are rendered with exactly six decimals, negative zero normalized to
    ``0.000000``. ``details`` not being a list raises ``TypeError``; every
    other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    confidence_value = _validate_finite_number(confidence, "confidence")
    if confidence_value <= 0 or confidence_value >= 1:
        raise ValueError("confidence must be greater than 0 and less than 1")

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

        groups: dict[str, list[Decimal]] = {}
        for validated in parsed:
            groups.setdefault(validated[1], []).append(validated[4])

        items = []
        for cell_id in sorted(groups):
            deltas = groups[cell_id]
            n = len(deltas)
            if n > _BOOTSTRAP_MAX_N:
                raise ValueError(
                    f"group {cell_id!r} has {n} rows; bootstrap report "
                    f"requires at most {_BOOTSTRAP_MAX_N} rows per group"
                )
            mu, lower, upper = _bootstrap_interval(deltas, n, confidence_value)

            items.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"delta":' + _format6(mu)
                + ',"lower":' + _format6(lower)
                + ',"upper":' + _format6(upper)
                + '}'
            )

        return (
            '{"confidence":' + _format6(confidence_value)
            + ',"groups":[' + ",".join(items) + ']}'
        )


def _effect_matrix_groups(
    details: list, minutes: int, z: float
) -> tuple[int, Decimal, list[tuple[int, list[tuple]]]]:
    """Validate inputs and compute the per-bucket, per-cell statistics shared
    by ``effect_matrix_report`` and ``effect_matrix_csv``.

    Returns ``(minutes, z_value, buckets)`` where ``buckets`` is a list of
    ``(bucket_start, cells)`` tuples in ascending bucket order, each
    ``cells`` listing ``(cell_id, n, means, se, lower, upper)`` tuples in
    ascending cell id order; ``means`` holds the six unquantized ``Decimal``
    means and ``se``, ``lower`` and ``upper`` are unquantized ``Decimal``
    values.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
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

    buckets_out: list[tuple[int, list[tuple]]] = []
    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        if parsed:
            bucket_seconds = minutes * 60
            # bucket start -> cell_id -> list of six-tuples of Decimal values
            buckets: dict[int, dict[str, list[tuple[Decimal, ...]]]] = {}
            for validated in parsed:
                timestamp, cell_id, values = validated[0], validated[1], validated[2:]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(values)

            for bucket in sorted(buckets):
                cells = []
                for cell_id in sorted(buckets[bucket]):
                    rows = buckets[bucket][cell_id]
                    n = len(rows)
                    means = []
                    for index in range(6):
                        total = Decimal(0)
                        for values in rows:
                            total += values[index]
                        means.append(total / n)
                    mu = means[2]
                    if n > 1:
                        squared = Decimal(0)
                        for values in rows:
                            deviation = values[2] - mu
                            squared += deviation * deviation
                        se = (squared / (n * (n - 1))).sqrt()
                    else:
                        se = Decimal(0)
                    lower = mu - z_value * se
                    upper = mu + z_value * se
                    cells.append((cell_id, n, means, se, lower, upper))
                buckets_out.append((bucket, cells))
    return minutes, z_value, buckets_out


def effect_matrix_report(
    details: list,
    *,
    minutes: int = 60,
    z: float = 1.96,
) -> str:
    """Aggregate scenario details into a time-bucket x cell matrix JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``z`` is a non-boolean finite number
    greater than or equal to 0.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string order.
    Each cell object carries the group size ``n``, the within-group arithmetic
    means of the six numeric fields and the delta confidence interval: with
    ``mu`` the mean of the deltas, ``se`` is
    ``sqrt(sum((delta - mu) ** 2) / (n * (n - 1)))`` when ``n > 1`` and 0
    otherwise, and ``lower``/``upper`` are ``mu - z * se`` / ``mu + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, z, groups``, each group object uses the key order
    ``key, cells`` and each cell object the key order
    ``key, n, base, post, delta, cg, cr, cm, se, lower, upper`` with
    ``key`` the cell id. An empty ``details`` yields
    ``{"minutes":60,"z":1.960000,"groups":[]}``. ``z`` and every numeric
    result are rendered with exactly six decimals, negative zero normalized
    to ``0.000000``. ``details`` not being a list raises ``TypeError``; every
    other contract violation raises ``ValueError``.
    """
    minutes, z_value, matrix_buckets = _effect_matrix_groups(details, minutes, z)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        groups = []
        for bucket, cells in matrix_buckets:
            cell_items = []
            for cell_id, n, means, se, lower, upper in cells:
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"base":' + _format6(means[0])
                    + ',"post":' + _format6(means[1])
                    + ',"delta":' + _format6(means[2])
                    + ',"cg":' + _format6(means[3])
                    + ',"cr":' + _format6(means[4])
                    + ',"cm":' + _format6(means[5])
                    + ',"se":' + _format6(se)
                    + ',"lower":' + _format6(lower)
                    + ',"upper":' + _format6(upper)
                    + '}'
                )
            groups.append(
                '{"key":' + str(bucket)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"z":' + _format6(z_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


_MATRIX_CSV_HEADER = "timestamp,cell_id,n,base,post,delta,cg,cr,cm,se,lower,upper"


def effect_matrix_csv(
    details: list,
    *,
    minutes: int = 60,
    z: float = 1.96,
) -> str:
    """Aggregate scenario details into a time-bucket x cell RFC4180 CSV report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``z`` is a non-boolean finite number
    greater than or equal to 0.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string order,
    one CSV line per bucket/cell pair. With ``n`` the group size, the six
    numeric fields ``base``..``cm`` are the within-group arithmetic means;
    with ``mu`` the mean of the deltas, ``se`` is
    ``sqrt(sum((delta - mu) ** 2) / (n * (n - 1)))`` when ``n > 1`` and 0
    otherwise, and ``lower``/``upper`` are ``mu - z * se`` / ``mu + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a UTF-8 RFC4180
    CSV string whose first line is the fixed header
    ``timestamp,cell_id,n,base,post,delta,cg,cr,cm,se,lower,upper``, followed
    by one line per bucket/cell pair (an empty ``details`` still yields the
    header). Lines end with CRLF, including the final line; ``timestamp``
    and ``n`` are decimal integers and every other numeric field is rendered
    with exactly six decimals, negative zero normalized to ``0.000000``. Any
    field containing a comma, a double quote or a line break is wrapped in
    double quotes with inner double quotes doubled. ``details`` not being a
    list raises ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    _, _, matrix_buckets = _effect_matrix_groups(details, minutes, z)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        lines = [_MATRIX_CSV_HEADER]
        for bucket, cells in matrix_buckets:
            for cell_id, n, means, se, lower, upper in cells:
                lines.append(
                    str(bucket)
                    + "," + _csv_field(cell_id)
                    + "," + str(n)
                    + "," + _format6(means[0])
                    + "," + _format6(means[1])
                    + "," + _format6(means[2])
                    + "," + _format6(means[3])
                    + "," + _format6(means[4])
                    + "," + _format6(means[5])
                    + "," + _format6(se)
                    + "," + _format6(lower)
                    + "," + _format6(upper)
                )
        return "\r\n".join(lines) + "\r\n"


def effect_matrix_significance_report(
    details: list,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Aggregate scenario details into a time-bucket x cell matrix z-test report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``alpha`` is a non-boolean finite number
    with ``0 < alpha <= 1``.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string order.
    Each cell object carries the group size ``n``, the within-group arithmetic
    means ``base``..``cm`` (``delta`` being ``mu``, the mean of the deltas)
    and the two-sided normal z-test of the mean: ``se`` is
    ``sqrt(sum((delta - mu) ** 2) / (n * (n - 1)))`` when ``n > 1`` and 0
    otherwise; when ``se`` is 0, ``z`` is 0 and ``p`` is 1, otherwise
    ``z = mu / se`` and ``p = erfc(abs(float(z)) / sqrt(2))``;
    ``significant`` is ``p <= alpha``, compared on the unquantized values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, alpha, groups``, each group object uses the key order
    ``key, cells`` and each cell object the key order
    ``key, n, base, post, delta, cg, cr, cm, se, z, p, significant`` with
    ``key`` the cell id. An empty ``details`` yields
    ``{"minutes":60,"alpha":0.050000,"groups":[]}``. ``alpha`` and every
    numeric result are rendered with exactly six decimals, negative zero
    normalized to ``0.000000``. ``details`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)
    alpha_value = _validate_finite_number(alpha, "alpha")
    if alpha_value <= 0 or alpha_value > 1:
        raise ValueError("alpha must be greater than 0 and at most 1")

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

        # bucket start -> cell_id -> list of six-tuples of Decimal values
        buckets: dict[int, dict[str, list[tuple[Decimal, ...]]]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id, values = validated[0], validated[1], validated[2:]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(values)

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                rows = buckets[bucket][cell_id]
                n = len(rows)
                means = []
                for index in range(6):
                    total = Decimal(0)
                    for values in rows:
                        total += values[index]
                    means.append(total / n)
                mu = means[2]
                if n > 1:
                    squared = Decimal(0)
                    for values in rows:
                        deviation = values[2] - mu
                        squared += deviation * deviation
                    se = (squared / (n * (n - 1))).sqrt()
                else:
                    se = Decimal(0)
                if se == 0:
                    z_value = Decimal(0)
                    p_value = Decimal(1)
                else:
                    z_value = mu / se
                    p_value = Decimal(
                        str(math.erfc(abs(float(z_value)) / math.sqrt(2)))
                    )
                significant = p_value <= alpha_value
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"base":' + _format6(means[0])
                    + ',"post":' + _format6(means[1])
                    + ',"delta":' + _format6(means[2])
                    + ',"cg":' + _format6(means[3])
                    + ',"cr":' + _format6(means[4])
                    + ',"cm":' + _format6(means[5])
                    + ',"se":' + _format6(se)
                    + ',"z":' + _format6(z_value)
                    + ',"p":' + _format6(p_value)
                    + ',"significant":' + ("true" if significant else "false")
                    + '}'
                )
            groups.append(
                '{"key":' + str(bucket)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def effect_matrix_fdr_report(
    details: list,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Aggregate scenario details into a time-bucket x cell matrix FDR report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``alpha`` is a non-boolean finite number
    with ``0 < alpha <= 1``.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string order.
    Each bucket/cell cell carries the group size ``n`` and the within-group
    arithmetic means of the six numeric fields. Its ``p`` is the two-sided
    sign-test p-value of the deltas: with ``r``/``s`` the counts of
    positive/negative deltas (zeros ignored), ``m = r + s``, ``p`` is 1 when
    ``m`` is 0 and ``min(1, 2 * sum(C(m, k) for k in 0..min(r, s)) / 2**m)``
    otherwise.

    With ``N`` the number of bucket/cell cells, all cells are ranked
    ascending by ``(p, bucket, cell)`` and each rank ``j`` (1-based) gets the
    Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    bucket/cell; ``reject`` is ``q <= alpha``, compared on the unquantized
    values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, alpha, groups``, each group object uses the key order
    ``key, cells`` and each cell object the key order
    ``key, n, base, post, delta, cg, cr, cm, p, q, reject`` with ``key`` the
    cell id. An empty ``details`` yields
    ``{"minutes":60,"alpha":0.050000,"groups":[]}``. ``alpha`` and every
    numeric result are rendered with exactly six decimals, negative zero
    normalized to ``0.000000``. ``details`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)
    alpha_value = _validate_finite_number(alpha, "alpha")
    if alpha_value <= 0 or alpha_value > 1:
        raise ValueError("alpha must be greater than 0 and at most 1")

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

        # bucket start -> cell_id -> list of six-tuples of Decimal values
        buckets: dict[int, dict[str, list[tuple[Decimal, ...]]]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id, values = validated[0], validated[1], validated[2:]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(values)

        # One record per bucket/cell in ascending (bucket, cell) output order:
        # ``[bucket, cell_id, n, means, p, q]`` with q filled in below.
        records: list[list] = []
        for bucket in sorted(buckets):
            for cell_id in sorted(buckets[bucket]):
                rows = buckets[bucket][cell_id]
                n = len(rows)
                means = []
                for index in range(6):
                    total = Decimal(0)
                    for values in rows:
                        total += values[index]
                    means.append(total / n)
                r = sum(1 for values in rows if values[2] > 0)
                s = sum(1 for values in rows if values[2] < 0)
                m = r + s
                if m == 0:
                    p_value = Decimal(1)
                else:
                    q0 = min(r, s)
                    tail = sum(math.comb(m, k) for k in range(q0 + 1))
                    p_value = min(Decimal(1), 2 * Decimal(tail) / Decimal(2**m))
                records.append([bucket, cell_id, n, means, p_value, None])

        # Benjamini-Hochberg q-values across ALL bucket/cell cells: rank
        # ascending by (p, bucket, cell), then accumulate the running minimum
        # of N * p_l / l from the top rank down, mapping q back to each cell.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (records[idx][4], records[idx][0], records[idx][1]),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Decimal(count) * records[idx][4] / rank
            if candidate < running:
                running = candidate
            records[idx][5] = running

        groups = []
        cell_items = []
        current_bucket = None
        for bucket, cell_id, n, means, p_value, q_value in records:
            if current_bucket is not None and bucket != current_bucket:
                groups.append(
                    '{"key":' + str(current_bucket)
                    + ',"cells":[' + ",".join(cell_items) + ']}'
                )
                cell_items = []
            current_bucket = bucket
            reject = q_value <= alpha_value
            cell_items.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"base":' + _format6(means[0])
                + ',"post":' + _format6(means[1])
                + ',"delta":' + _format6(means[2])
                + ',"cg":' + _format6(means[3])
                + ',"cr":' + _format6(means[4])
                + ',"cm":' + _format6(means[5])
                + ',"p":' + _format6(p_value)
                + ',"q":' + _format6(q_value)
                + ',"reject":' + ("true" if reject else "false")
                + '}'
            )
        if current_bucket is not None:
            groups.append(
                '{"key":' + str(current_bucket)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def effect_matrix_bootstrap_report(
    details: list,
    *,
    minutes: int = 60,
    confidence: float = 0.95,
) -> str:
    """Aggregate scenario details into a time-bucket x cell bootstrap CI report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``confidence`` is a non-boolean finite
    number with ``0 < confidence < 1``.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string order.
    Each bucket/cell cell must aggregate at most 8 rows; a larger cell raises
    ``ValueError``. With ``d`` the per-row deltas and ``n`` the cell size,
    ``delta`` is the sample mean ``sum(d) / n`` and ``lower``/``upper`` are
    the bootstrap quantiles of the sorted ``n ** n`` with-replacement
    resample means: with ``q = (1 - confidence) / 2`` and
    ``r = (n ** n - 1) * q``, the bounds are taken directly at integer ``r``
    and otherwise linearly interpolated between ``floor(r)`` and ``ceil(r)``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, confidence, groups``, each group object uses the key
    order ``key, cells`` and each cell object the key order
    ``key, n, delta, lower, upper`` with ``key`` the cell id. An empty
    ``details`` yields ``{"minutes":60,"confidence":0.950000,"groups":[]}``.
    Every numeric result is rendered with exactly six decimals, negative
    zero normalized to ``0.000000``. ``details`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)
    confidence_value = _validate_finite_number(confidence, "confidence")
    if confidence_value <= 0 or confidence_value >= 1:
        raise ValueError("confidence must be greater than 0 and less than 1")

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

        # bucket start -> cell_id -> list of delta Decimal values
        buckets: dict[int, dict[str, list[Decimal]]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id, delta = validated[0], validated[1], validated[4]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(delta)

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                deltas = buckets[bucket][cell_id]
                n = len(deltas)
                if n > _BOOTSTRAP_MAX_N:
                    raise ValueError(
                        f"bucket {bucket} cell {cell_id!r} has {n} rows; "
                        f"matrix bootstrap report requires at most "
                        f"{_BOOTSTRAP_MAX_N} rows per bucket/cell"
                    )
                mu, lower, upper = _bootstrap_interval(deltas, n, confidence_value)
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"delta":' + _format6(mu)
                    + ',"lower":' + _format6(lower)
                    + ',"upper":' + _format6(upper)
                    + '}'
                )
            groups.append(
                '{"key":' + str(bucket)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"confidence":' + _format6(confidence_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def effect_matrix_jackknife_report(
    details: list,
    *,
    minutes: int = 60,
    z: float = 1.96,
) -> str:
    """Aggregate scenario deltas into a time-bucket x cell jackknife JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``z`` is a non-boolean finite number
    greater than or equal to 0.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string order.
    Each cell object carries the group size and the delta jackknife confidence
    interval: with ``d`` the per-row deltas and ``n`` the cell size,
    ``delta`` is ``sum(d) / n``; the jackknife standard error is 0 when
    ``n <= 1`` and otherwise
    ``sqrt((n - 1) / n * sum((mu_i - mu_bar) ** 2))`` where
    ``mu_i = (sum(d) - d_i) / (n - 1)`` and ``mu_bar = sum(mu_i) / n``;
    ``lower``/``upper`` are ``delta - z * se`` / ``delta + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, z, groups``, each group object uses the key order
    ``key, cells`` and each cell object the key order
    ``key, n, delta, se, lower, upper`` with ``key`` the cell id. An empty
    ``details`` yields ``{"minutes":60,"z":1.960000,"groups":[]}``. ``z`` and
    every numeric result are rendered with exactly six decimals, negative
    zero normalized to ``0.000000``. ``details`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
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

        # bucket start -> cell_id -> list of delta Decimal values
        buckets: dict[int, dict[str, list[Decimal]]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id, delta = validated[0], validated[1], validated[4]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(delta)

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                deltas = buckets[bucket][cell_id]
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
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"delta":' + _format6(mu)
                    + ',"se":' + _format6(se)
                    + ',"lower":' + _format6(lower)
                    + ',"upper":' + _format6(upper)
                    + '}'
                )
            groups.append(
                '{"key":' + str(bucket)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"z":' + _format6(z_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def effect_matrix_permutation_report(
    details: list,
    *,
    minutes: int = 60,
    z: float = 1.96,
) -> str:
    """Aggregate scenario deltas into a time-bucket x cell sign-flip permutation report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``z`` is a non-boolean finite number
    greater than or equal to 0.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string order.
    Each bucket/cell cell must aggregate at most 16 rows; a larger cell
    raises ``ValueError``. With ``d`` the per-row deltas and ``n`` the cell
    size, ``delta`` is ``sum(d) / n``; ``p`` is the exact two-sided
    sign-flip permutation p-value: all ``2 ** n`` sign vectors ``s_i`` in
    ``{-1, 1}`` are enumerated and
    ``p = 2 ** -n * #{|sum(s_i * d_i) / n| >= |delta|}``. The jackknife
    standard error is 0 when ``n <= 1`` and otherwise
    ``sqrt((n - 1) / n * sum((mu_i - mu_bar) ** 2))`` where
    ``mu_i = (sum(d) - d_i) / (n - 1)`` and ``mu_bar = sum(mu_i) / n``;
    ``lower``/``upper`` are ``delta - z * se`` / ``delta + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, z, groups``, each group object uses the key order
    ``key, cells`` and each cell object the key order
    ``key, n, delta, p, se, lower, upper`` with ``key`` the cell id. An
    empty ``details`` yields ``{"minutes":60,"z":1.960000,"groups":[]}``.
    ``z`` and every numeric result are rendered with exactly six decimals,
    negative zero normalized to ``0.000000``; cell ids are JSON-escaped with
    Unicode preserved. ``details`` not being a list raises ``TypeError``;
    every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
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

        # bucket start -> cell_id -> list of delta Decimal values
        buckets: dict[int, dict[str, list[Decimal]]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id, delta = validated[0], validated[1], validated[4]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(delta)

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                deltas = buckets[bucket][cell_id]
                n = len(deltas)
                if n > _PERMUTATION_MAX_N:
                    raise ValueError(
                        f"bucket {bucket} cell {cell_id!r} has {n} rows; "
                        f"matrix permutation report requires at most "
                        f"{_PERMUTATION_MAX_N} rows per bucket/cell"
                    )
                total = Decimal(0)
                for delta in deltas:
                    total += delta
                mu = total / n

                # |sum(s_i * d_i) / n| >= |delta| is equivalent (n > 0) to
                # |sum(s_i * d_i)| >= |sum(d_i)|; compare the raw sums so
                # exact ties are decided without any division rounding.
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
                p_value = Decimal(hits) / Decimal(1 << n)

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
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"delta":' + _format6(mu)
                    + ',"p":' + _format6(p_value)
                    + ',"se":' + _format6(se)
                    + ',"lower":' + _format6(lower)
                    + ',"upper":' + _format6(upper)
                    + '}'
                )
            groups.append(
                '{"key":' + str(bucket)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"z":' + _format6(z_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def effect_matrix_quantile_report(
    details: list,
    *,
    minutes: int = 60,
    low: float = 0.25,
    high: float = 0.75,
) -> str:
    """Aggregate scenario deltas into a time-bucket x cell quantile JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``low`` and ``high`` are non-boolean
    finite numbers with ``0 <= low < high <= 1``.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string
    order. Within each cell the deltas are sorted ascending. With ``d`` the
    sorted per-row deltas and ``n`` the cell size, ``mean`` is
    ``sum(d) / n`` and the order-statistic quantile is
    ``Q(q) = (1 - f) * d[i] + f * d[min(i + 1, n - 1)]`` with
    ``r = (n - 1) * q``, ``i = floor(r)`` and ``f = r - i``; the cell
    reports ``median = Q(0.5)``, ``lower = Q(low)``, ``upper = Q(high)``
    and ``iqr = upper - lower``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, low, high, groups``, each group object uses the key
    order ``key, cells`` and each cell object the key order
    ``key, n, mean, median, lower, upper, iqr`` with ``key`` the cell id.
    An empty ``details`` yields
    ``{"minutes":60,"low":0.250000,"high":0.750000,"groups":[]}``. ``low``
    and ``high`` and every numeric result are rendered with exactly six
    decimals, negative zero normalized to ``0.000000``; bucket keys and
    cell sizes are integers and cell ids are JSON-escaped with Unicode
    preserved. ``details`` not being a list raises ``TypeError``; every
    other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)
    low_value = _validate_finite_number(low, "low")
    high_value = _validate_finite_number(high, "high")
    if low_value < 0 or high_value > 1 or low_value >= high_value:
        raise ValueError("low and high must satisfy 0 <= low < high <= 1")

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

        # bucket start -> cell_id -> list of delta Decimal values
        buckets: dict[int, dict[str, list[Decimal]]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id, delta = validated[0], validated[1], validated[4]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(delta)

        half = Decimal("0.5")

        def quantile(sorted_deltas: list[Decimal], q: Decimal) -> Decimal:
            n = len(sorted_deltas)
            r = Decimal(n - 1) * q
            i = int(r.to_integral_value(rounding=ROUND_FLOOR))
            f = r - Decimal(i)
            if i + 1 < n:
                return (Decimal(1) - f) * sorted_deltas[i] + f * sorted_deltas[i + 1]
            return sorted_deltas[i]

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                deltas = sorted(buckets[bucket][cell_id])
                n = len(deltas)
                total = Decimal(0)
                for delta in deltas:
                    total += delta
                mean = total / n
                median = quantile(deltas, half)
                lower = quantile(deltas, low_value)
                upper = quantile(deltas, high_value)
                iqr = upper - lower
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"mean":' + _format6(mean)
                    + ',"median":' + _format6(median)
                    + ',"lower":' + _format6(lower)
                    + ',"upper":' + _format6(upper)
                    + ',"iqr":' + _format6(iqr)
                    + '}'
                )
            groups.append(
                '{"key":' + str(bucket)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"low":' + _format6(low_value)
            + ',"high":' + _format6(high_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def effect_matrix_contribution_report(
    details: list,
    *,
    minutes: int = 60,
) -> str:
    """Aggregate scenario contributions into a time-bucket x cell JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string
    order. Each cell object carries the group size ``n`` and the
    within-group arithmetic means of ``delta``, ``cg``, ``cr`` and ``cm``.
    With the mean contributions, ``ag = abs(cg)``, ``ar = abs(cr)``,
    ``am = abs(cm)`` and ``A = ag + ar + am``; when ``A > 0`` the three
    shares are ``ag / A``, ``ar / A`` and ``am / A`` (all 0 otherwise) and
    ``dominant`` is ``"g"``, ``"r"`` or ``"m"`` for the largest share, with
    ties resolved in g, r, m priority; when ``A == 0`` it is ``"none"``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups``, each group object uses the key order
    ``key, cells`` and each cell object the key order
    ``key, n, delta, cg, cr, cm, share_g, share_r, share_m, dominant`` with
    ``key`` the cell id. An empty ``details`` yields
    ``{"minutes":60,"groups":[]}``. Every numeric result is rendered with
    exactly six decimals, negative zero normalized to ``0.000000``; bucket
    keys and cell sizes are integers and cell ids are JSON-escaped with
    Unicode preserved. ``details`` not being a list raises ``TypeError``;
    every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)

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

        # bucket start -> cell_id -> list of (delta, cg, cr, cm) Decimals
        buckets: dict[int, dict[str, list[tuple[Decimal, ...]]]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id = validated[0], validated[1]
                values = validated[4:]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(values)

        zero = Decimal(0)
        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                rows = buckets[bucket][cell_id]
                n = len(rows)
                means = []
                for index in range(4):
                    total = Decimal(0)
                    for values in rows:
                        total += values[index]
                    means.append(total / n)
                delta, cg, cr, cm = means
                ag, ar, am = abs(cg), abs(cr), abs(cm)
                total_abs = ag + ar + am
                if total_abs > 0:
                    share_g = ag / total_abs
                    share_r = ar / total_abs
                    share_m = am / total_abs
                    if ag >= ar and ag >= am:
                        dominant = "g"
                    elif ar >= am:
                        dominant = "r"
                    else:
                        dominant = "m"
                else:
                    share_g = share_r = share_m = zero
                    dominant = "none"
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"delta":' + _format6(delta)
                    + ',"cg":' + _format6(cg)
                    + ',"cr":' + _format6(cr)
                    + ',"cm":' + _format6(cm)
                    + ',"share_g":' + _format6(share_g)
                    + ',"share_r":' + _format6(share_r)
                    + ',"share_m":' + _format6(share_m)
                    + ',"dominant":' + json.dumps(dominant)
                    + '}'
                )
            groups.append(
                '{"key":' + str(bucket)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def _validate_detail_rows(rows: list) -> list:
    """Validate one scenario-details table: rows follow the eight-tuple
    contract and ``(timestamp, cell_id)`` pairs must be unique."""
    parsed = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        validated = _validate_detail_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append(validated)
    return parsed


def effect_matrix_compare_report(
    before: list,
    after: list,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Pair two scenario-detail tables into a time-bucket x cell FDR comparison.

    ``before`` and ``after`` are lists of ``scenario`` eight-tuples
    ``(timestamp, cell_id, base, post, delta, cg, cr, cm)``: ``timestamp``
    must be a non-boolean non-negative integer, ``cell_id`` a non-empty
    string and the other six fields finite non-boolean int/float values; the
    ``(timestamp, cell_id)`` pairs within each table must be unique, and the
    two tables must share exactly the same set of pairs. Each row's delta is
    its fifth field. An empty pair yields
    ``{"minutes":60,"alpha":0.050000,"groups":[]}``.

    ``minutes`` must be a non-boolean integer in ``1..1440`` that divides
    1440; ``alpha`` is a non-boolean finite number with ``0 < alpha <= 1``.

    Rows are paired on ``(timestamp, cell_id)`` and bucketed by Unix epoch
    with key ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are
    emitted in ascending order and, within each bucket, cells in ascending
    string order. With ``n`` the number of pairs in a bucket/cell cell,
    ``before``/``after`` are the mean before/after deltas and
    ``change = after - before`` (means of the unquantized paired values).
    The sign test is applied to ``change``: with ``r``/``s`` the counts of
    positive/negative changes (zeros ignored), ``m = r + s``, ``p`` is 1
    when ``m`` is 0 and
    ``min(1, 2 * sum(C(m, k) for k in 0..min(r, s)) / 2**m)`` otherwise.

    With ``N`` the number of bucket/cell cells, all cells are ranked
    ascending by ``(p, bucket, cell)`` and each rank ``j`` (1-based) gets the
    Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    bucket/cell; ``reject`` is ``q <= alpha``, compared on the unquantized
    values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, alpha, groups``, each group object uses the key order
    ``key, cells`` and each cell object the key order
    ``key, n, before, after, change, p, q, reject`` with ``key`` the cell
    id. ``alpha`` and every numeric result are rendered with exactly six
    decimals, negative zero normalized to ``0.000000``; bucket keys and cell
    sizes are integers and cell ids are JSON-escaped with Unicode preserved.
    ``before`` or ``after`` not being a list raises ``TypeError``; every
    other contract violation raises ``ValueError``.
    """
    if not isinstance(before, list):
        raise TypeError("before must be a list")
    if not isinstance(after, list):
        raise TypeError("after must be a list")
    minutes = _validate_minutes(minutes)
    alpha_value = _validate_finite_number(alpha, "alpha")
    if alpha_value <= 0 or alpha_value > 1:
        raise ValueError("alpha must be greater than 0 and at most 1")

    before_rows = _validate_detail_rows(before)
    after_rows = _validate_detail_rows(after)

    before_map = {(row[0], row[1]): row[4] for row in before_rows}
    after_map = {(row[0], row[1]): row[4] for row in after_rows}
    before_keys = set(before_map)
    after_keys = set(after_map)
    if before_keys != after_keys:
        missing = sorted(before_keys - after_keys, key=lambda key: (key[0], key[1]))
        extra = sorted(after_keys - before_keys, key=lambda key: (key[0], key[1]))
        if missing:
            raise ValueError(f"key missing from after: {missing[0]!r}")
        raise ValueError(f"key missing from before: {extra[0]!r}")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # (bucket start, cell_id) -> [before sum, after sum, n, r, s]
        cells: dict[tuple[int, str], list] = {}
        if before_map:
            bucket_seconds = minutes * 60
            for (timestamp, cell_id), before_delta in before_map.items():
                after_delta = after_map[(timestamp, cell_id)]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                acc = cells.setdefault((bucket, cell_id), [Decimal(0), Decimal(0), 0, 0, 0])
                acc[0] += before_delta
                acc[1] += after_delta
                acc[2] += 1
                change = after_delta - before_delta
                if change > 0:
                    acc[3] += 1
                elif change < 0:
                    acc[4] += 1

        # One record per bucket/cell in ascending (bucket, cell) output order:
        # ``[bucket, cell_id, n, before mean, after mean, change mean, p, q]``
        # with q filled in below.
        records: list[list] = []
        for bucket, cell_id in sorted(cells):
            before_sum, after_sum, n, r, s = cells[(bucket, cell_id)]
            before_mean = before_sum / n
            after_mean = after_sum / n
            change_mean = after_mean - before_mean
            m = r + s
            if m == 0:
                p_value = Decimal(1)
            else:
                q0 = min(r, s)
                tail = sum(math.comb(m, k) for k in range(q0 + 1))
                p_value = min(Decimal(1), 2 * Decimal(tail) / Decimal(2**m))
            records.append(
                [bucket, cell_id, n, before_mean, after_mean, change_mean, p_value, None]
            )

        # Benjamini-Hochberg q-values across ALL bucket/cell cells: rank
        # ascending by (p, bucket, cell), then accumulate the running minimum
        # of N * p_l / l from the top rank down, mapping q back to each cell.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (records[idx][6], records[idx][0], records[idx][1]),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Decimal(count) * records[idx][6] / rank
            if candidate < running:
                running = candidate
            records[idx][7] = running

        groups = []
        cell_items = []
        current_bucket = None
        for bucket, cell_id, n, before_mean, after_mean, change_mean, p_value, q_value in records:
            if current_bucket is not None and bucket != current_bucket:
                groups.append(
                    '{"key":' + str(current_bucket)
                    + ',"cells":[' + ",".join(cell_items) + ']}'
                )
                cell_items = []
            current_bucket = bucket
            reject = q_value <= alpha_value
            cell_items.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"before":' + _format6(before_mean)
                + ',"after":' + _format6(after_mean)
                + ',"change":' + _format6(change_mean)
                + ',"p":' + _format6(p_value)
                + ',"q":' + _format6(q_value)
                + ',"reject":' + ("true" if reject else "false")
                + '}'
            )
        if current_bucket is not None:
            groups.append(
                '{"key":' + str(current_bucket)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def effect_matrix_robust_report(
    details: list,
    *,
    minutes: int = 60,
    z: float = 1.96,
) -> str:
    """Aggregate scenario deltas into a robust (median/MAD) matrix JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``z`` is a non-boolean finite number
    greater than or equal to 0.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string
    order. Within each cell the deltas are sorted ascending as ``d`` of size
    ``n`` and the order-statistic quantile is
    ``Q(q) = (1 - f) * d[i] + f * d[min(i + 1, n - 1)]`` with
    ``r = (n - 1) * q``, ``i = floor(r)`` and ``f = r - i``; the cell
    reports ``median = Q(0.5)``, ``mad`` the same ``Q(0.5)`` of the ascending
    absolute deviations ``|d - median|``, ``se = 1.4826 * mad / sqrt(n)`` and
    ``lower``/``upper`` as ``median - z * se`` / ``median + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, z, groups``, each group object uses the key order
    ``key, cells`` and each cell object the key order
    ``key, n, median, mad, se, lower, upper`` with ``key`` the cell id. An
    empty ``details`` yields
    ``{"minutes":60,"z":1.960000,"groups":[]}``. ``z`` and every numeric
    result are rendered with exactly six decimals, negative zero normalized
    to ``0.000000``; bucket keys and cell sizes are integers and cell ids are
    JSON-escaped with Unicode preserved. ``details`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)
    z_value = _validate_finite_number(z, "z")
    if z_value < 0:
        raise ValueError("z must be non-negative")

    parsed = _validate_detail_rows(details)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # bucket start -> cell_id -> list of delta Decimal values
        buckets: dict[int, dict[str, list[Decimal]]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id, delta = validated[0], validated[1], validated[4]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(delta)

        half = Decimal("0.5")
        mad_scale = Decimal("1.4826")

        def quantile(sorted_values: list[Decimal], q: Decimal) -> Decimal:
            n = len(sorted_values)
            r = Decimal(n - 1) * q
            i = int(r.to_integral_value(rounding=ROUND_FLOOR))
            f = r - Decimal(i)
            if i + 1 < n:
                return (Decimal(1) - f) * sorted_values[i] + f * sorted_values[i + 1]
            return sorted_values[i]

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                deltas = sorted(buckets[bucket][cell_id])
                n = len(deltas)
                median = quantile(deltas, half)
                deviations = sorted(abs(delta - median) for delta in deltas)
                mad = quantile(deviations, half)
                se = mad_scale * mad / Decimal(n).sqrt()
                lower = median - z_value * se
                upper = median + z_value * se
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"median":' + _format6(median)
                    + ',"mad":' + _format6(mad)
                    + ',"se":' + _format6(se)
                    + ',"lower":' + _format6(lower)
                    + ',"upper":' + _format6(upper)
                    + '}'
                )
            groups.append(
                '{"key":' + str(bucket)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"z":' + _format6(z_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def effect_matrix_autocorr_report(
    details: list,
    *,
    minutes: int = 60,
    lag: int = 1,
) -> str:
    """Aggregate scenario deltas into a per-cell lag-autocorrelation JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``lag`` must be a positive non-boolean
    integer.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within each
    bucket/cell pair are averaged. For each cell id ``c``, every occupied
    bucket ``B`` whose predecessor bucket ``B - lag * minutes * 60`` is also
    occupied forms a pair with ``prev``/``curr`` the predecessor/current
    bucket mean deltas and ``change = curr - prev``; cells without any pair
    are omitted. The remaining cells are emitted in ascending cell id order,
    each with its pairs in ascending bucket order, the pair count ``n``, the
    mean ``mean_change`` of the changes, the standard error
    ``se = sqrt(sum((change - mean_change) ** 2) / (n * (n - 1)))`` (0 when
    ``n <= 1``) and the Pearson correlation
    ``corr = sum((prev - p_bar) * (curr - c_bar)) /
    sqrt(sum((prev - p_bar) ** 2) * sum((curr - c_bar) ** 2))`` (0 when
    either sum of squares is 0).

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, lag, groups``, each group object uses the key order
    ``key, n, mean_change, se, corr, pairs`` and each pair object the key
    order ``bucket, prev, curr, change``. An empty ``details`` yields
    ``{"minutes":60,"lag":1,"groups":[]}``. Every numeric result is rendered
    with exactly six decimals, negative zero normalized to ``0.000000``;
    bucket keys and pair counts are integers and cell ids are JSON-escaped
    with Unicode preserved. ``details`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)
    if isinstance(lag, bool) or not isinstance(lag, int):
        raise ValueError("lag must be an integer")
    if lag < 1:
        raise ValueError("lag must be a positive integer")

    parsed = _validate_detail_rows(details)

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

        step = lag * minutes * 60
        groups = []
        for cell_id in sorted(cells):
            means = {
                bucket: total / count
                for bucket, (total, count) in cells[cell_id].items()
            }
            pairs = []
            for bucket in sorted(means):
                prev_bucket = bucket - step
                if prev_bucket not in means:
                    continue
                prev = means[prev_bucket]
                curr = means[bucket]
                pairs.append((bucket, prev, curr, curr - prev))
            if not pairs:
                continue
            n = len(pairs)

            change_total = Decimal(0)
            for _, _, _, change in pairs:
                change_total += change
            mean_change = change_total / n
            if n > 1:
                squared = Decimal(0)
                for _, _, _, change in pairs:
                    deviation = change - mean_change
                    squared += deviation * deviation
                se = (squared / (n * (n - 1))).sqrt()
            else:
                se = Decimal(0)

            prev_total = Decimal(0)
            curr_total = Decimal(0)
            for _, prev, curr, _ in pairs:
                prev_total += prev
                curr_total += curr
            prev_mean = prev_total / n
            curr_mean = curr_total / n
            sxy = Decimal(0)
            sxx = Decimal(0)
            syy = Decimal(0)
            for _, prev, curr, _ in pairs:
                prev_deviation = prev - prev_mean
                curr_deviation = curr - curr_mean
                sxy += prev_deviation * curr_deviation
                sxx += prev_deviation * prev_deviation
                syy += curr_deviation * curr_deviation
            denominator = sxx * syy
            corr = Decimal(0) if denominator == 0 else sxy / denominator.sqrt()

            pair_items = []
            for bucket, prev, curr, change in pairs:
                pair_items.append(
                    '{"bucket":' + str(bucket)
                    + ',"prev":' + _format6(prev)
                    + ',"curr":' + _format6(curr)
                    + ',"change":' + _format6(change)
                    + '}'
                )
            groups.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"mean_change":' + _format6(mean_change)
                + ',"se":' + _format6(se)
                + ',"corr":' + _format6(corr)
                + ',"pairs":[' + ",".join(pair_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"lag":' + str(lag)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


_MORAN_MAX_N = 8


def _moran_statistic(
    values: list[Decimal],
    mean: Decimal,
    edges: list[tuple[int, int]],
    factor: Decimal,
    sxx: Decimal,
) -> Decimal:
    """Moran's I of ``values`` over ``edges``: ``factor * (2 * sum_edges(x_i *
    x_j)) / sxx`` with ``x_i = values[i] - mean``."""
    deviations = [value - mean for value in values]
    edge_sum = Decimal(0)
    for i, j in edges:
        edge_sum += deviations[i] * deviations[j]
    return factor * (2 * edge_sum) / sxx


def effect_matrix_moran_report(
    details: list,
    neighbors: list,
    *,
    minutes: int = 60,
) -> str:
    """Aggregate scenario deltas per bucket and emit a Moran's I JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``neighbors`` is a list of ``(a, b)`` two-tuples
    describing an undirected adjacency: ``a`` and ``b`` must be distinct
    non-empty cell id strings occurring in ``details``; self-loops and
    repeated edges (in either orientation) are illegal. ``minutes`` must be
    a non-boolean integer in ``1..1440`` that divides 1440.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within each
    bucket/cell pair are averaged. Each bucket holds the vector ``d`` of its
    per-cell mean deltas in ascending cell id order; a bucket with more than
    8 cells raises ``ValueError``. With ``n`` the size of ``d``,
    ``x_i = d_i - mean(d)`` and ``e`` the number of neighbor edges whose
    endpoints both occur in the bucket, Moran's I is 0 and ``p`` is 1 when
    ``e`` is 0 or ``sum(x_i ** 2)`` is 0; otherwise
    ``I = (n / (2 * e)) * (2 * sum_edges(x_i * x_j)) / sum(x_i ** 2)`` and
    ``p`` is the exact permutation p-value: all ``n!`` permutations of ``d``
    are enumerated in lexicographic order (duplicate values not
    deduplicated), I is recomputed for each and ``p`` is the proportion with
    ``|I_perm| >= |I|``, compared on the unquantized values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups`` and each group object uses the key order
    ``key, n, moran, p``, with groups in ascending bucket order. ``moran``
    and ``p`` are rendered with exactly six decimals, negative zero
    normalized to ``0.000000``; bucket keys and ``n`` are integers. An empty
    ``details`` yields ``{"minutes":60,"groups":[]}``. ``details`` or
    ``neighbors`` not being a list raises ``TypeError``; every other
    contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if not isinstance(neighbors, list):
        raise TypeError("neighbors must be a list")
    minutes = _validate_minutes(minutes)

    parsed = _validate_detail_rows(details)

    cell_ids = {cell_id for _, cell_id, *_ in parsed}
    edges: set[tuple[str, str]] = set()
    for item in neighbors:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("each neighbor must be an (a, b) two-tuple")
        endpoint_a, endpoint_b = item
        for endpoint in (endpoint_a, endpoint_b):
            if not isinstance(endpoint, str) or not endpoint:
                raise ValueError(
                    "neighbor endpoints must be non-empty cell id strings"
                )
            if endpoint not in cell_ids:
                raise ValueError(f"unknown cell id in neighbor: {endpoint!r}")
        if endpoint_a == endpoint_b:
            raise ValueError("neighbor self-loops are not allowed")
        edge = (
            (endpoint_a, endpoint_b)
            if endpoint_a < endpoint_b
            else (endpoint_b, endpoint_a)
        )
        if edge in edges:
            raise ValueError(f"duplicate neighbor edge: {edge!r}")
        edges.add(edge)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # bucket start -> cell_id -> [delta sum, row count]
        buckets: dict[int, dict[str, list]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for validated in parsed:
                timestamp, cell_id, delta = validated[0], validated[1], validated[4]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                acc = buckets.setdefault(bucket, {}).setdefault(
                    cell_id, [Decimal(0), 0]
                )
                acc[0] += delta
                acc[1] += 1

        groups = []
        for bucket in sorted(buckets):
            cells = sorted(buckets[bucket])
            n = len(cells)
            if n > _MORAN_MAX_N:
                raise ValueError(
                    f"bucket {bucket} has {n} cells; Moran report requires "
                    f"at most {_MORAN_MAX_N} cells per bucket"
                )
            d = []
            for cell_id in cells:
                total, count = buckets[bucket][cell_id]
                d.append(total / count)
            position = {cell_id: index for index, cell_id in enumerate(cells)}
            bucket_edges = [
                (position[a], position[b])
                for a, b in sorted(edges)
                if a in position and b in position
            ]
            e = len(bucket_edges)

            total_d = Decimal(0)
            for value in d:
                total_d += value
            mean_d = total_d / n
            sxx = Decimal(0)
            for value in d:
                deviation = value - mean_d
                sxx += deviation * deviation

            if e == 0 or sxx == 0:
                moran = Decimal(0)
                p = Decimal(1)
            else:
                factor = Decimal(n) / (2 * e)
                moran = _moran_statistic(d, mean_d, bucket_edges, factor, sxx)
                abs_moran = abs(moran)
                hits = 0
                for perm in permutations(d):
                    perm_moran = _moran_statistic(
                        list(perm), mean_d, bucket_edges, factor, sxx
                    )
                    if abs(perm_moran) >= abs_moran:
                        hits += 1
                p = Decimal(hits) / Decimal(math.factorial(n))

            groups.append(
                '{"key":' + str(bucket)
                + ',"n":' + str(n)
                + ',"moran":' + _format6(moran)
                + ',"p":' + _format6(p)
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


_THEILSEN_MAX_N = 8


def effect_matrix_theilsen_report(
    details: list,
    *,
    minutes: int = 60,
    min_points: int = 3,
) -> str:
    """Aggregate scenario deltas into per-cell Theil-Sen/Kendall JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``min_points`` must be a non-boolean
    integer greater than or equal to 3.

    Rows are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within
    each ``(B, cell_id)`` pair are averaged. For each cell id, the occupied
    buckets form points ``(x, y)`` with ``x = B`` and ``y`` the bucket mean
    delta, ordered by ascending ``B``. Cells with fewer than ``min_points``
    occupied buckets are omitted; every remaining cell must have at most 8
    buckets, otherwise ``ValueError`` is raised. Kept cells are emitted in
    ascending cell id order.

    For each kept cell with ``n`` points, ``slope`` is the Theil-Sen
    estimator: the median of the ``C(n, 2)`` pairwise slopes
    ``(y_j - y_i) / (x_j - x_i)`` for ``i < j`` (the mean of the two middle
    slopes when their count is even). ``S`` is
    ``sum(sign(y_j - y_i))`` for ``i < j`` with ``sign`` in ``{-1, 0, 1}``
    and ``tau`` is Kendall's ``S / C(n, 2)``. ``p`` is the exact two-sided
    permutation p-value: all ``n!`` permutations of ``y`` are enumerated
    without deduplicating equal values, ``S'`` is recomputed for each and
    ``p`` is the proportion with ``|S'| >= |S|``, compared on the exact
    integer values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups`` and each group object uses the key order
    ``key, n, slope, tau, p`` with ``key`` the cell id and ``n`` the number
    of fitted buckets. Every numeric result is rendered with exactly six
    decimals, negative zero normalized to ``0.000000``; cell ids are
    JSON-escaped with Unicode preserved. An empty ``details`` yields
    ``{"minutes":60,"groups":[]}``. ``details`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)
    if isinstance(min_points, bool) or not isinstance(min_points, int):
        raise ValueError("min_points must be an integer")
    if min_points < 3:
        raise ValueError("min_points must be an integer greater than or equal to 3")

    parsed = _validate_detail_rows(details)

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

        def sign(value: Decimal) -> int:
            if value > 0:
                return 1
            if value < 0:
                return -1
            return 0

        items = []
        for cell_id in sorted(cells):
            buckets = cells[cell_id]
            if len(buckets) < min_points:
                continue
            n = len(buckets)
            if n > _THEILSEN_MAX_N:
                raise ValueError(
                    f"cell {cell_id!r} has {n} buckets; Theil-Sen report "
                    f"requires at most {_THEILSEN_MAX_N} buckets per cell"
                )
            points = [
                (Decimal(bucket), total / count)
                for bucket, (total, count) in sorted(buckets.items())
            ]
            ys = [y for _, y in points]

            slopes = []
            for i in range(n):
                for j in range(i + 1, n):
                    slopes.append((ys[j] - ys[i]) / (points[j][0] - points[i][0]))
            slopes.sort()
            slope_count = len(slopes)
            if slope_count % 2:
                slope = slopes[slope_count // 2]
            else:
                slope = (
                    slopes[slope_count // 2 - 1] + slopes[slope_count // 2]
                ) / 2

            s_value = 0
            for i in range(n):
                for j in range(i + 1, n):
                    s_value += sign(ys[j] - ys[i])
            tau = Decimal(s_value) / Decimal(math.comb(n, 2))

            hits = 0
            abs_s = abs(s_value)
            for perm in permutations(ys):
                perm_s = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        perm_s += sign(perm[j] - perm[i])
                if abs(perm_s) >= abs_s:
                    hits += 1
            p_value = Decimal(hits) / Decimal(math.factorial(n))

            items.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"slope":' + _format6(slope)
                + ',"tau":' + _format6(tau)
                + ',"p":' + _format6(p_value)
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(items) + ']}'
        )
