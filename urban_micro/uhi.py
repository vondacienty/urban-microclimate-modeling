"""Urban heat island (UHI) intensity computation."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
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
    "effect_matrix_block_bootstrap_report",
    "effect_matrix_jackknife_report",
    "effect_matrix_permutation_report",
    "effect_matrix_trimmed_report",
    "effect_matrix_robust_report",
    "effect_matrix_theilsen_report",
    "effect_matrix_contribution_report",
    "effect_matrix_autocorr_report",
    "effect_matrix_moran_report",
    "effect_matrix_local_moran_report",
    "effect_matrix_geary_report",
    "effect_matrix_hotspot_report",
    "effect_matrix_wilcoxon_report",
    "effect_matrix_spatial_lag_report",
    "effect_matrix_spatiotemporal_report",
    "effect_matrix_spatiotemporal_fdr_report",
    "effect_matrix_temporal_lag_report",
    "ventilation_report",
    "microclimate_coupling_report",
    "microclimate_coupling_uhi_report",
    "microclimate_coupling_trend_report",
    "microclimate_coupling_trend_significance_report",
    "vent_effect_report",
    "effect_matrix_ventilation_effect_report",
    "effect_matrix_cluster_report",
    "energy_balance_report",
    "energy_balance_scenario_report",
    "energy_temperature_report",
    "energy_temperature_scenario_report",
    "energy_temperature_scenario_uhi_report",
    "energy_temperature_uhi_report",
    "temperature_fusion_report",
    "temperature_fusion_uhi_report",
    "surface_morphology_report",
    "surface_thermal_zone_scenario_report",
]

_RECORD_KEYS = frozenset({"station_id", "timestamp", "temp_c"})
_SATELLITE_KEYS = frozenset({"cell_id", "timestamp", "lst_c"})
_VENTILATION_KEYS = frozenset(
    {"timestamp", "cell_id", "wind_u", "wind_v", "height", "density"}
)
_COUPLING_KEYS = frozenset(
    {
        "timestamp", "cell_id", "residual", "temp",
        "wind_u", "wind_v", "height", "density",
    }
)
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


_TREND_SIGNIFICANCE_MAX_N = 8


def _validate_trend_significance_min_points(min_points: object) -> int:
    if isinstance(min_points, bool) or not isinstance(min_points, int):
        raise ValueError("min_points must be an integer")
    if not 3 <= min_points <= _TREND_SIGNIFICANCE_MAX_N:
        raise ValueError(
            "min_points must be an integer in 3.."
            f"{_TREND_SIGNIFICANCE_MAX_N}"
        )
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
    # Every int is finite; math.isfinite() on an int converts it through a
    # C double and raises OverflowError once it exceeds float range, so guard
    # the check to floats.
    if isinstance(value, float) and not math.isfinite(value):
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
    fractional digits, normalizing negative zero to ``0.000000``.

    Quantization runs under a context wide enough to hold the value's integer
    digits (plus one for a rounding carry and six fractional ones), so finite
    arbitrarily large integers never trigger ``InvalidOperation``."""
    integer_digits = max(0, value.adjusted() + 1)
    with localcontext() as ctx:
        ctx.prec = max(_MODEL_PRECISION, integer_digits + 7)
        ctx.rounding = ROUND_HALF_EVEN
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


_BLOCK_BOOTSTRAP_MAX_N = 8


def _block_bootstrap_interval(
    values: list[Decimal], n: int, block: int, confidence_value: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    """Return ``(delta, lower, upper)`` for one circular block-bootstrap cell.

    ``delta`` is the sample mean ``sum(values) / n``. A resample draws
    ``k = ceil(n / block)`` block-start indices; each start ``i`` contributes
    the ``block`` cyclically consecutive items ``y[i], y[(i + 1) % n], ...``,
    the concatenated draws are truncated to the first ``n`` items and
    averaged. ``lower``/``upper`` are the interpolated quantiles of the
    sorted multiset of the ``n ** k`` resample means at
    ``q = (1 - confidence) / 2`` and ``1 - q``. The sorted multiset is built
    from the draw-count compositions of the ``k - 1`` full blocks times the
    ``n`` choices for the truncated last block instead of materializing all
    ``n ** k`` means, giving an identical result.
    """
    total = Decimal(0)
    for value in values:
        total += value
    mu = total / n

    k = -(-n // block)
    last_length = n - (k - 1) * block

    # full_sums[i] sums the block cyclically consecutive items starting at
    # index i; last_sums[i] sums only the first last_length of them (the
    # last drawn block is truncated so the resample holds exactly n items).
    full_sums = []
    last_sums = []
    for i in range(n):
        full = Decimal(0)
        last = Decimal(0)
        for j in range(block):
            value = values[(i + j) % n]
            full += value
            if j < last_length:
                last += value
        full_sums.append(full)
        last_sums.append(last)

    # A resample (idx_0, ..., idx_{k-1}) sums full_sums[idx_j] over the
    # first k-1 draws plus last_sums[idx_{k-1}]. The sorted multiset of the
    # n**k resample sums only depends on how many times each start appears
    # among the first k-1 draws and on the last draw: a composition
    # (c_0, ..., c_{n-1}) with sum k-1 together with last index L
    # contributes sum(c_j * full_sums[j]) + last_sums[L] with multiplicity
    # (k-1)! / prod(c_j!).
    weighted_sums: list[tuple[Decimal, int]] = []
    counts = [0] * n
    factorial = math.factorial(k - 1)

    def enumerate_compositions(index: int, remaining: int) -> None:
        if index == n - 1:
            counts[index] = remaining
            weighted_sum = Decimal(0)
            multiplicity = factorial
            for j, count_j in enumerate(counts):
                if count_j:
                    weighted_sum += count_j * full_sums[j]
                    multiplicity //= math.factorial(count_j)
            for last_index in range(n):
                weighted_sums.append(
                    (weighted_sum + last_sums[last_index], multiplicity)
                )
            return
        for count_j in range(remaining + 1):
            counts[index] = count_j
            enumerate_compositions(index + 1, remaining - count_j)

    enumerate_compositions(0, k - 1)
    weighted_sums.sort(key=lambda item: item[0])

    # Merge equal sums into (sorted sum numerator, cumulative count);
    # merged_sums[k] repeats merged_cumulative[k] - merged_cumulative[k-1]
    # times in the sorted n**k-length sequence of resample means.
    merged_sums: list[Decimal] = []
    merged_cumulative: list[int] = []
    running = 0
    for weighted_sum, multiplicity in weighted_sums:
        running += multiplicity
        if merged_sums and merged_sums[-1] == weighted_sum:
            merged_cumulative[-1] = running
        else:
            merged_sums.append(weighted_sum)
            merged_cumulative.append(running)

    def value_at(position: int) -> Decimal:
        merged_index = bisect_left(merged_cumulative, position + 1)
        return merged_sums[merged_index] / n

    resample_count = n**k
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


def effect_matrix_block_bootstrap_report(
    details: list,
    *,
    minutes: int = 60,
    block: int = 2,
    confidence: float = 0.95,
) -> str:
    """Aggregate scenario deltas into a per-cell block-bootstrap JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``block`` must be a non-boolean integer
    in ``1..8``; ``confidence`` is a non-boolean finite number with
    ``0 < confidence < 1``.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within
    each bucket/cell pair are averaged, giving each cell a series ``y`` of
    ``n`` bucket means in ascending bucket order. Cells with ``n < 2`` are
    omitted; a cell with ``n > 8`` raises ``ValueError``. With
    ``k = ceil(n / block)``, every index tuple in ``{0, ..., n-1} ** k``
    (lexicographic order) draws ``k`` blocks: from start ``i`` the ``block``
    cyclically consecutive items ``y[i], y[(i + 1) % n], ...`` are taken,
    the concatenated draws are truncated to the first ``n`` items and
    averaged. ``delta`` is ``sum(y) / n`` and ``lower``/``upper`` are the
    quantiles of the sorted ``N = n ** k`` resample means: with
    ``q = (1 - confidence) / 2`` and ``r = (N - 1) * q``, the bounds are
    taken directly at integer ``r`` (positions ``r`` and ``N - 1 - r``) and
    otherwise linearly interpolated between ``floor(r)`` and ``ceil(r)``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, block, confidence, groups`` and each group object
    uses the key order ``key, n, delta, lower, upper`` with ``key`` the
    cell id and ``n`` the number of occupied buckets, groups in ascending
    cell id order. An empty ``details`` yields
    ``{"minutes":60,"block":2,"confidence":0.950000,"groups":[]}``.
    ``confidence`` and every numeric result are rendered with exactly six
    decimals, negative zero normalized to ``0.000000``. ``details`` not
    being a list raises ``TypeError``; every other contract violation
    raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)
    if isinstance(block, bool) or not isinstance(block, int):
        raise ValueError("block must be an integer")
    if not 1 <= block <= 8:
        raise ValueError("block must be an integer in 1..8")
    confidence_value = _validate_finite_number(confidence, "confidence")
    if confidence_value <= 0 or confidence_value >= 1:
        raise ValueError("confidence must be greater than 0 and less than 1")

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

        groups = []
        for cell_id in sorted(cells):
            buckets = cells[cell_id]
            n = len(buckets)
            if n < 2:
                continue
            if n > _BLOCK_BOOTSTRAP_MAX_N:
                raise ValueError(
                    f"cell {cell_id!r} has {n} buckets; block bootstrap report "
                    f"requires at most {_BLOCK_BOOTSTRAP_MAX_N} buckets per cell"
                )
            values = [
                bucket_total / bucket_count
                for _, (bucket_total, bucket_count) in sorted(buckets.items())
            ]
            mu, lower, upper = _block_bootstrap_interval(
                values, n, block, confidence_value
            )
            groups.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"delta":' + _format6(mu)
                + ',"lower":' + _format6(lower)
                + ',"upper":' + _format6(upper)
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"block":' + str(block)
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


def effect_matrix_trimmed_report(
    details: list,
    *,
    minutes: int = 60,
    trim: float = 0.1,
    z: float = 1.96,
) -> str:
    """Aggregate scenario deltas into a time-bucket x cell trimmed-mean report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``trim`` is a non-boolean finite number
    with ``0 <= trim < 0.5``; ``z`` is a non-boolean finite number greater
    than or equal to 0.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string order.
    Within each bucket/cell the ``n`` deltas are sorted ascending and, with
    ``k = floor(trim * n)``, the trimmed sample is ``d[k : n - k]`` of size
    ``m = n - 2 * k``; ``m < 2`` raises ``ValueError``. ``trimmed`` is the
    trimmed mean ``mu = sum(d[k : n - k]) / m``; ``p`` is the exact two-sided
    sign-flip permutation p-value of the trimmed sample: all ``2 ** m`` sign
    vectors ``s_i`` in ``{-1, 1}`` are enumerated and
    ``p = 2 ** -m * #{|sum(s_i * d_i) / m| >= |mu|}``. The standard error is
    ``se = sqrt(sum((d_i - mu) ** 2) / (m * (m - 1)))`` over the trimmed
    sample and ``lower``/``upper`` are ``mu - z * se`` / ``mu + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, trim, z, groups``, each group object uses the key
    order ``key, cells`` and each cell object the key order
    ``key, n, trimmed, p, se, lower, upper`` with ``key`` the cell id and
    ``n`` the untrimmed cell size. An empty ``details`` yields
    ``{"minutes":60,"trim":0.100000,"z":1.960000,"groups":[]}``. ``trim``,
    ``z`` and every numeric result are rendered with exactly six decimals,
    negative zero normalized to ``0.000000``; cell ids are JSON-escaped with
    Unicode preserved. ``details`` not being a list raises ``TypeError``;
    every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    minutes = _validate_minutes(minutes)
    trim_value = _validate_finite_number(trim, "trim")
    if trim_value < 0 or trim_value >= Decimal("0.5"):
        raise ValueError("trim must satisfy 0 <= trim < 0.5")
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
                deltas = sorted(buckets[bucket][cell_id])
                n = len(deltas)
                k = int(trim_value * n)
                trimmed = deltas[k : n - k]
                m = len(trimmed)
                if m < 2:
                    raise ValueError(
                        f"bucket {bucket} cell {cell_id!r} keeps {m} of {n} rows "
                        f"after trimming; trimmed report requires at least 2"
                    )
                total = Decimal(0)
                for delta in trimmed:
                    total += delta
                mu = total / m

                # |sum(s_i * d_i) / m| >= |mu| is equivalent (m > 0) to
                # |sum(s_i * d_i)| >= |sum(d_i)|; compare the raw sums so
                # exact ties are decided without any division rounding.
                hits = 0
                for mask in range(1 << m):
                    signed_sum = Decimal(0)
                    for index, delta in enumerate(trimmed):
                        if (mask >> index) & 1:
                            signed_sum -= delta
                        else:
                            signed_sum += delta
                    if abs(signed_sum) >= abs(total):
                        hits += 1
                p_value = Decimal(hits) / Decimal(1 << m)

                squared = Decimal(0)
                for delta in trimmed:
                    deviation = delta - mu
                    squared += deviation * deviation
                se = (squared / (m * (m - 1))).sqrt()
                lower = mu - z_value * se
                upper = mu + z_value * se
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"trimmed":' + _format6(mu)
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
            + ',"trim":' + _format6(trim_value)
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


_GEARY_MAX_N = 8


def effect_matrix_geary_report(
    details: list,
    neighbors: list,
    *,
    minutes: int = 60,
) -> str:
    """Aggregate scenario deltas per bucket and emit a Geary's C JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``neighbors`` is a list of ``(a, b)`` two-tuples
    describing an undirected adjacency: ``a`` and ``b`` must be distinct
    non-empty cell id strings occurring in ``details``; self-loops and
    repeated edges (in either orientation) are illegal. ``minutes`` must be a
    non-boolean integer in ``1..1440`` that divides 1440.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within each
    bucket/cell pair are averaged. Each bucket holds the vector ``d`` of its
    per-cell mean deltas in ascending cell id order; a bucket with more than
    8 cells raises ``ValueError``. With ``n`` the size of ``d``,
    ``x_i = d_i - mean(d)``, ``S = sum(x_i ** 2)`` and ``e`` the number of
    neighbor edges whose endpoints both occur in the bucket, Geary's C is 1
    and ``p`` is 1 when ``e`` is 0 or ``S`` is 0; otherwise
    ``C = (n - 1) * sum_edges((x_a - x_b) ** 2) / (2 * e * S)`` and ``p`` is
    the exact permutation p-value: all ``n!`` permutations of ``d`` are
    enumerated in lexicographic order (duplicate values not deduplicated), C
    is recomputed for each and ``p`` is the proportion with
    ``|C_perm - 1| >= |C - 1|``, compared on the unquantized values via exact
    rational arithmetic so theoretically-equal distances are never split by
    rounding.

    Numbers enter as ``Decimal(str(x))`` and bucket accumulation happens
    under a precision-1000, ROUND_HALF_EVEN local context; the per-bucket
    means are then lifted to exact fractions for the statistic and its
    permutation test. Returns a compact UTF-8 JSON string with no spaces and
    no trailing newline; the top-level key order is ``minutes, groups`` and
    each group object uses the key order ``key, n, geary, p``, with groups in
    ascending bucket order. ``geary`` and ``p`` are rendered with exactly six
    decimals, negative zero normalized to ``0.000000``; bucket keys and ``n``
    are integers. An empty ``details`` yields
    ``{"minutes":60,"groups":[]}``. ``details`` or ``neighbors`` not being a
    list raises ``TypeError``; every other contract violation raises
    ``ValueError``.
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
            if n > _GEARY_MAX_N:
                raise ValueError(
                    f"bucket {bucket} has {n} cells; Geary report requires "
                    f"at most {_GEARY_MAX_N} cells per bucket"
                )
            # Lift the precision-1000 means to exact fractions so that
            # equidistant permutation statistics are never split by rounding.
            d = []
            for cell_id in cells:
                total, count = buckets[bucket][cell_id]
                d.append(Fraction(total) / count)
            position = {cell_id: index for index, cell_id in enumerate(cells)}
            bucket_edges = [
                (position[a], position[b])
                for a, b in sorted(edges)
                if a in position and b in position
            ]
            e = len(bucket_edges)

            mean_d = sum(d, Fraction(0)) / n
            deviations = [value - mean_d for value in d]
            sxx = sum((value * value for value in deviations), Fraction(0))

            if e == 0 or sxx == 0:
                geary_fraction = Fraction(1)
                p_fraction = Fraction(1)
            else:
                edge_sum = sum(
                    (
                        (deviations[a] - deviations[b])
                        * (deviations[a] - deviations[b])
                        for a, b in bucket_edges
                    ),
                    Fraction(0),
                )
                geary_fraction = Fraction(n - 1) * edge_sum / (2 * e * sxx)
                distance = abs(geary_fraction - 1)
                hits = 0
                for perm in permutations(d):
                    perm_edge_sum = sum(
                        (
                            (perm[a] - perm[b]) * (perm[a] - perm[b])
                            for a, b in bucket_edges
                        ),
                        Fraction(0),
                    )
                    perm_geary = Fraction(n - 1) * perm_edge_sum / (2 * e * sxx)
                    if abs(perm_geary - 1) >= distance:
                        hits += 1
                p_fraction = Fraction(hits, math.factorial(n))

            groups.append(
                '{"key":' + str(bucket)
                + ',"n":' + str(n)
                + ',"geary":' + _format6(_fraction_to_decimal(geary_fraction))
                + ',"p":' + _format6(_fraction_to_decimal(p_fraction))
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


_LOCAL_MORAN_MAX_N = 8


def _fraction_to_decimal(value: Fraction) -> Decimal:
    """Exact-rational to ``Decimal`` conversion under the active context."""
    return Decimal(value.numerator) / Decimal(value.denominator)


def effect_matrix_local_moran_report(
    details: list,
    neighbors: list,
    *,
    minutes: int = 60,
) -> str:
    """Aggregate scenario deltas per bucket and emit a local Moran's I JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``neighbors`` is a list of ``(a, b)`` two-tuples
    describing an undirected adjacency: ``a`` and ``b`` must be distinct
    non-empty cell id strings occurring in ``details``; self-loops and
    repeated edges (in either orientation) are illegal. ``minutes`` must be a
    non-boolean integer in ``1..1440`` that divides 1440.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within each
    bucket/cell pair are averaged. Each bucket holds the vector ``d`` of its
    per-cell mean deltas in ascending cell id order; a bucket with more than
    8 cells raises ``ValueError``. With ``n`` the size of ``d``,
    ``x_i = d_i - mean(d)`` and ``S = sum(x_i ** 2)``, the local Moran
    statistic of cell ``c`` is
    ``L_c = n * x_c * sum(x_j over the neighbors of c inside the bucket) /
    S``. When ``S`` is 0 or ``c`` has no neighbor inside the bucket,
    ``L_c`` is 0 and ``p`` is 1; otherwise ``p`` is the exact permutation
    p-value: all ``n!`` permutations of ``d`` are enumerated in
    lexicographic order (duplicate values not deduplicated), the local
    statistic is recomputed for each and ``p`` is the proportion with
    ``|L_perm| >= |L|``, compared on the unquantized values via exact
    rational arithmetic so theoretically-equal statistics are never split
    by rounding.

    Numbers enter as ``Decimal(str(x))`` and bucket accumulation happens
    under a precision-1000, ROUND_HALF_EVEN local context; the per-bucket
    means are then lifted to exact fractions for the statistic and its
    permutation test. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups``, each group object uses the key order
    ``key, n, cells`` (groups in ascending bucket order) and each cell object
    the key order ``key, local, p`` with cells in ascending cell id order.
    ``local`` and ``p`` are rendered with exactly six decimals, negative
    zero normalized to ``0.000000``; bucket keys and ``n`` are integers and
    cell ids are JSON-escaped with Unicode preserved. An empty ``details``
    yields ``{"minutes":60,"groups":[]}``. ``details`` or ``neighbors`` not
    being a list raises ``TypeError``; every other contract violation raises
    ``ValueError``.
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
            if n > _LOCAL_MORAN_MAX_N:
                raise ValueError(
                    f"bucket {bucket} has {n} cells; local Moran report "
                    f"requires at most {_LOCAL_MORAN_MAX_N} cells per bucket"
                )
            d = []
            for cell_id in cells:
                total, count = buckets[bucket][cell_id]
                d.append(Fraction(total) / count)
            position = {cell_id: index for index, cell_id in enumerate(cells)}
            adjacency: list[list[int]] = [[] for _ in range(n)]
            for a, b in sorted(edges):
                if a in position and b in position:
                    i, j = position[a], position[b]
                    adjacency[i].append(j)
                    adjacency[j].append(i)
            for neighbors_of in adjacency:
                neighbors_of.sort()

            mean_d = sum(d, Fraction(0)) / n
            deviations = [value - mean_d for value in d]
            sxx = sum((value * value for value in deviations), Fraction(0))

            if sxx == 0:
                observed = [Fraction(0)] * n
                p_values = [Fraction(1)] * n
            else:
                observed = [
                    Fraction(n)
                    * deviations[index]
                    * sum((deviations[j] for j in adjacency[index]), Fraction(0))
                    / sxx
                    for index in range(n)
                ]
                active = [bool(neighbors_of) for neighbors_of in adjacency]
                # Write d_i = p_i / D with a shared denominator D and put
                # z_i = n*p_i - T where T = sum(p_k). Then, for every
                # permutation,
                #   L_i = (n^2/D) * z_i * sum_{j~i} z_j /
                #         ((1/D) * sum_k z_k**2);
                # the positive factor n**2 / sum_k z_k**2 is permutation
                # invariant, so |L_perm| >= |L| is decided by the exact
                # integer products z_i * sum_{j~i} z_j.
                denominator = 1
                for value in d:
                    denominator = denominator * value.denominator // math.gcd(
                        denominator, value.denominator
                    )
                p_values = [Fraction(1)] * n
                if any(active):
                    numerators = [
                        value.numerator * (denominator // value.denominator)
                        for value in d
                    ]
                    total_numerators = sum(numerators)
                    z = [
                        n * numerator - total_numerators
                        for numerator in numerators
                    ]
                    observed_scores = [
                        z[index]
                        * sum((z[j] for j in adjacency[index]), 0)
                        for index in range(n)
                    ]
                    hits = [0] * n
                    factorial = math.factorial(n)
                    for perm in permutations(z):
                        for index in range(n):
                            if not active[index]:
                                continue
                            perm_score = perm[index] * sum(
                                (perm[j] for j in adjacency[index]), 0
                            )
                            if abs(perm_score) >= abs(observed_scores[index]):
                                hits[index] += 1
                    for index in range(n):
                        if active[index]:
                            p_values[index] = Fraction(hits[index], factorial)

            cell_items = []
            for index, cell_id in enumerate(cells):
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"local":' + _format6(_fraction_to_decimal(observed[index]))
                    + ',"p":' + _format6(_fraction_to_decimal(p_values[index]))
                    + '}'
                )

            groups.append(
                '{"key":' + str(bucket)
                + ',"n":' + str(n)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def effect_matrix_hotspot_report(
    details: list,
    neighbors: list,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Aggregate scenario deltas per bucket into a local-Moran hotspot report.

    ``details`` and ``neighbors`` follow the ``effect_matrix_local_moran_report``
    contract: ``details`` is a list of ``scenario`` eight-tuples
    ``(timestamp, cell_id, base, post, delta, cg, cr, cm)`` with a
    non-boolean non-negative integer timestamp, a non-empty string cell id,
    finite non-boolean numeric fields and unique ``(timestamp, cell_id)``
    pairs; ``neighbors`` is a list of distinct undirected ``(a, b)``
    two-tuples of distinct non-empty cell id strings occurring in
    ``details``. ``minutes`` must be a non-boolean integer in ``1..1440``
    that divides 1440; ``alpha`` is a non-boolean finite number with
    ``0 < alpha <= 1``.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within each
    bucket/cell pair are averaged. Each bucket holds the vector ``d`` of its
    per-cell mean deltas in ascending cell id order; a bucket with more than
    8 cells raises ``ValueError``. With ``n`` the size of ``d``,
    ``x_i = d_i - mean(d)`` and ``S = sum(x_i ** 2)``, the local Moran
    statistic of cell ``c`` is
    ``L_c = n * x_c * sum(x_j over the neighbors of c inside the bucket) /
    S``. When ``S`` is 0 or ``c`` has no neighbor inside the bucket,
    ``L_c`` is 0 and ``p`` is 1; otherwise ``p`` is the exact permutation
    p-value: all ``n!`` permutations of ``d`` are enumerated in
    lexicographic order (duplicate values not deduplicated), the local
    statistic is recomputed for each and ``p`` is the proportion with
    ``|L_perm| >= |L|``, compared on the unquantized values via exact
    rational arithmetic so theoretically-equal statistics are never split
    by rounding.

    With ``N`` the total number of bucket/cell cells, all cells are ranked
    ascending by ``(p, bucket, cell)`` and each rank ``j`` (1-based) gets the
    Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    bucket/cell; ``reject`` is ``q <= alpha``, compared on the unquantized
    exact values. Each rejected cell is classified from the signs of
    ``x_c`` and ``v_c = sum(x_j over its in-bucket neighbors)``:
    ``(+, +)`` is ``HH``, ``(-, -)`` is ``LL``, ``(+, -)`` is ``HL`` and
    ``(-, +)`` is ``LH``; non-rejected cells and cells with ``x_c = 0`` or
    ``v_c = 0`` are ``NS``.

    Numbers enter as ``Decimal(str(x))`` and bucket accumulation happens
    under a precision-1000, ROUND_HALF_EVEN local context; the per-bucket
    means are then lifted to exact fractions for the statistic, its
    permutation test and the q-values. Returns a compact UTF-8 JSON string
    with no spaces and no trailing newline; the top-level key order is
    ``minutes, alpha, groups``, each group object uses the key order
    ``key, n, cells`` (groups in ascending bucket order) and each cell
    object the key order ``key, local, p, q, reject, kind`` with cells in
    ascending cell id order. ``alpha``, ``local``, ``p`` and ``q`` are
    rendered with exactly six decimals, negative zero normalized to
    ``0.000000``; bucket keys and ``n`` are integers and cell ids are
    JSON-escaped with Unicode preserved. An empty ``details`` yields
    ``{"minutes":60,"alpha":0.050000,"groups":[]}``. ``details`` or
    ``neighbors`` not being a list raises ``TypeError``; every other
    contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if not isinstance(neighbors, list):
        raise TypeError("neighbors must be a list")
    minutes = _validate_minutes(minutes)
    alpha_value = _validate_finite_number(alpha, "alpha")
    if alpha_value <= 0 or alpha_value > 1:
        raise ValueError("alpha must be greater than 0 and at most 1")

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

        # Flat records in ascending (bucket, cell) output order:
        # ``[bucket, n, cell_id, local, p, x_c, v_c]`` with fractions.
        records: list[list] = []
        for bucket in sorted(buckets):
            cells = sorted(buckets[bucket])
            n = len(cells)
            if n > _LOCAL_MORAN_MAX_N:
                raise ValueError(
                    f"bucket {bucket} has {n} cells; hotspot report "
                    f"requires at most {_LOCAL_MORAN_MAX_N} cells per bucket"
                )
            d = []
            for cell_id in cells:
                total, count = buckets[bucket][cell_id]
                d.append(Fraction(total) / count)
            position = {cell_id: index for index, cell_id in enumerate(cells)}
            adjacency: list[list[int]] = [[] for _ in range(n)]
            for a, b in sorted(edges):
                if a in position and b in position:
                    i, j = position[a], position[b]
                    adjacency[i].append(j)
                    adjacency[j].append(i)
            for neighbors_of in adjacency:
                neighbors_of.sort()

            mean_d = sum(d, Fraction(0)) / n
            deviations = [value - mean_d for value in d]
            sxx = sum((value * value for value in deviations), Fraction(0))
            neighbor_sums = [
                sum((deviations[j] for j in neighbors_of), Fraction(0))
                for neighbors_of in adjacency
            ]

            if sxx == 0:
                observed = [Fraction(0)] * n
                p_values = [Fraction(1)] * n
            else:
                observed = [
                    Fraction(n)
                    * deviations[index]
                    * neighbor_sums[index]
                    / sxx
                    for index in range(n)
                ]
                active = [bool(neighbors_of) for neighbors_of in adjacency]
                # As in effect_matrix_local_moran_report, write d_i = p_i / D
                # with a shared denominator D and z_i = n*p_i - T; the
                # permutation-invariant factor drops out of |L_perm| >= |L|,
                # so the test is decided by the exact integer products
                # z_i * sum_{j~i} z_j.
                denominator = 1
                for value in d:
                    denominator = denominator * value.denominator // math.gcd(
                        denominator, value.denominator
                    )
                p_values = [Fraction(1)] * n
                if any(active):
                    numerators = [
                        value.numerator * (denominator // value.denominator)
                        for value in d
                    ]
                    total_numerators = sum(numerators)
                    z = [
                        n * numerator - total_numerators
                        for numerator in numerators
                    ]
                    observed_scores = [
                        z[index] * sum((z[j] for j in adjacency[index]), 0)
                        for index in range(n)
                    ]
                    hits = [0] * n
                    factorial = math.factorial(n)
                    for perm in permutations(z):
                        for index in range(n):
                            if not active[index]:
                                continue
                            perm_score = perm[index] * sum(
                                (perm[j] for j in adjacency[index]), 0
                            )
                            if abs(perm_score) >= abs(observed_scores[index]):
                                hits[index] += 1
                    for index in range(n):
                        if active[index]:
                            p_values[index] = Fraction(hits[index], factorial)

            for index, cell_id in enumerate(cells):
                records.append(
                    [
                        bucket,
                        n,
                        cell_id,
                        observed[index],
                        p_values[index],
                        deviations[index],
                        neighbor_sums[index],
                    ]
                )

        # Benjamini-Hochberg q-values across ALL bucket/cell cells: rank
        # ascending by (p, bucket, cell), then accumulate the running minimum
        # of N * p_l / l from the top rank down, mapping q back to each cell.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (records[idx][4], records[idx][0], records[idx][2]),
        )
        q_values: list[Fraction | None] = [None] * count
        running = Fraction(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Fraction(count) * records[idx][4] / rank
            if candidate < running:
                running = candidate
            q_values[idx] = running
        alpha_fraction = Fraction(alpha_value)

        groups = []
        cell_items = []
        current_bucket = None
        current_n = 0
        for idx, (bucket, n, cell_id, local, p_value, x_c, v_c) in enumerate(records):
            if current_bucket is not None and bucket != current_bucket:
                groups.append(
                    '{"key":' + str(current_bucket)
                    + ',"n":' + str(current_n)
                    + ',"cells":[' + ",".join(cell_items) + ']}'
                )
                cell_items = []
            current_bucket = bucket
            current_n = n
            q_value = q_values[idx]
            reject = q_value <= alpha_fraction
            if reject and x_c != 0 and v_c != 0:
                if x_c > 0:
                    kind = "HH" if v_c > 0 else "HL"
                else:
                    kind = "LH" if v_c > 0 else "LL"
            else:
                kind = "NS"
            cell_items.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"local":' + _format6(_fraction_to_decimal(local))
                + ',"p":' + _format6(_fraction_to_decimal(p_value))
                + ',"q":' + _format6(_fraction_to_decimal(q_value))
                + ',"reject":' + ("true" if reject else "false")
                + ',"kind":' + json.dumps(kind)
                + '}'
            )
        if current_bucket is not None:
            groups.append(
                '{"key":' + str(current_bucket)
                + ',"n":' + str(current_n)
                + ',"cells":[' + ",".join(cell_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


_SPATIAL_LAG_MAX_N = 8


def effect_matrix_spatial_lag_report(
    details: list,
    neighbors: list,
    *,
    minutes: int = 60,
) -> str:
    """Aggregate scenario deltas per bucket and emit a spatial-lag JSON report.

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
    8 cells raises ``ValueError``. With ``x_i = d_i - mean(d)``, the spatial
    lag of cell ``c`` is the mean deviation of its neighbors inside the
    bucket, ``lag_c = sum(x_j over the neighbors of c) / k_c`` (0 when ``c``
    has no in-bucket neighbor), and ``local_c = x_c * lag_c``. When the
    neighbor list is empty (no edge is active inside a bucket) or all
    ``d`` are equal, every cell gets ``lag = local = 0`` and ``p = 1``.
    Otherwise ``p`` is the exact permutation p-value: all ``n!``
    permutations of ``d`` are enumerated in lexicographic order (duplicate
    values not deduplicated), ``local`` is recomputed for each and ``p`` is
    the proportion with ``|local_perm| >= |local|``, compared on the
    unquantized values via exact rational arithmetic so theoretically-equal
    statistics are never split by rounding.

    Numbers enter as ``Decimal(str(x))`` and bucket accumulation happens
    under a precision-1000, ROUND_HALF_EVEN local context; the per-bucket
    means are then lifted to exact fractions for the statistics and their
    permutation test. Returns a compact UTF-8 JSON string with no spaces and
    no trailing newline; the top-level key order is ``minutes, groups``,
    each group object uses the key order ``key, cells`` (groups in ascending
    bucket order) and each cell object the key order ``key, lag, local, p``
    with cells in ascending cell id order. ``lag``, ``local`` and ``p`` are
    rendered with exactly six decimals, negative zero normalized to
    ``0.000000``; bucket keys are integers and cell ids are JSON-escaped
    with Unicode preserved. An empty ``details`` yields
    ``{"minutes":60,"groups":[]}``. ``details`` or ``neighbors`` not being a
    list raises ``TypeError``; every other contract violation raises
    ``ValueError``.
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
            if n > _SPATIAL_LAG_MAX_N:
                raise ValueError(
                    f"bucket {bucket} has {n} cells; spatial lag report "
                    f"requires at most {_SPATIAL_LAG_MAX_N} cells per bucket"
                )
            d = []
            for cell_id in cells:
                total, count = buckets[bucket][cell_id]
                d.append(Fraction(total) / count)
            position = {cell_id: index for index, cell_id in enumerate(cells)}
            adjacency: list[list[int]] = [[] for _ in range(n)]
            for a, b in sorted(edges):
                if a in position and b in position:
                    i, j = position[a], position[b]
                    adjacency[i].append(j)
                    adjacency[j].append(i)
            for neighbors_of in adjacency:
                neighbors_of.sort()
            degrees = [len(neighbors_of) for neighbors_of in adjacency]

            mean_d = sum(d, Fraction(0)) / n
            deviations = [value - mean_d for value in d]
            sxx = sum((value * value for value in deviations), Fraction(0))

            observed_lag: list[Fraction] = []
            observed_local: list[Fraction] = []
            for index in range(n):
                if degrees[index] == 0 or sxx == 0:
                    observed_lag.append(Fraction(0))
                    observed_local.append(Fraction(0))
                else:
                    lag = (
                        sum((deviations[j] for j in adjacency[index]), Fraction(0))
                        / degrees[index]
                    )
                    observed_lag.append(lag)
                    observed_local.append(deviations[index] * lag)

            if sxx == 0 or not any(degrees):
                p_values = [Fraction(1)] * n
            else:
                active = [degree > 0 for degree in degrees]
                # As in effect_matrix_local_moran_report, write d_i = p_i / D
                # with a shared denominator D and z_i = n*p_i - T; then
                # local_c = z_c * sum_{j~c} z_j / (k_c * (nD) ** 2), so the
                # positive per-cell factor drops out of |local_perm| >=
                # |local| and the test is decided by exact integer products.
                denominator = 1
                for value in d:
                    denominator = denominator * value.denominator // math.gcd(
                        denominator, value.denominator
                    )
                numerators = [
                    value.numerator * (denominator // value.denominator)
                    for value in d
                ]
                z = [
                    n * numerator - sum(numerators)
                    for numerator in numerators
                ]
                observed_scores = [
                    z[index] * sum((z[j] for j in adjacency[index]), 0)
                    for index in range(n)
                ]
                hits = [0] * n
                factorial = math.factorial(n)
                for perm in permutations(z):
                    for index in range(n):
                        if not active[index]:
                            continue
                        perm_score = perm[index] * sum(
                            (perm[j] for j in adjacency[index]), 0
                        )
                        if abs(perm_score) >= abs(observed_scores[index]):
                            hits[index] += 1
                p_values = [Fraction(1)] * n
                for index in range(n):
                    if active[index]:
                        p_values[index] = Fraction(hits[index], factorial)

            cell_items = []
            for index, cell_id in enumerate(cells):
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"lag":' + _format6(_fraction_to_decimal(observed_lag[index]))
                    + ',"local":'
                    + _format6(_fraction_to_decimal(observed_local[index]))
                    + ',"p":' + _format6(_fraction_to_decimal(p_values[index]))
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


_SPATIOTEMPORAL_MAX_N = 16


def effect_matrix_spatiotemporal_report(
    details: list,
    neighbors: list,
    *,
    minutes: int = 60,
    lag: int = 1,
    z: float = 1.96,
) -> str:
    """Pair neighbor deltas across a temporal lag per bucket and emit JSON.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``neighbors`` is a list of ``(a, b)`` two-tuples
    describing an undirected adjacency: ``a`` and ``b`` must be distinct
    non-empty cell id strings occurring in ``details``; self-loops and
    repeated edges (in either orientation) are illegal. ``minutes`` must be
    a non-boolean integer in ``1..1440`` that divides 1440; ``lag`` must be
    a positive non-boolean integer; ``z`` is a non-boolean finite number
    greater than or equal to 0.

    Rows are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within
    each ``(B, c)`` pair are averaged. For each bucket ``B``, the edges are
    visited in lexicographic order and an edge ``(a, b)`` contributes a
    pairing only when both endpoints have a value at ``B`` and at
    ``B - lag * minutes * 60``; with ``d_{B, c}`` the bucket/cell mean and
    ``Delta_c = d_{B, c} - d_{B - lag * minutes * 60, c}``, the paired value
    is ``x = Delta_a - Delta_b``. A bucket with no pairings is omitted; a
    bucket with more than 16 pairings raises ``ValueError``.

    With ``n`` the number of pairings, ``mean = sum(x) / n``; ``se`` is 0
    when ``n <= 1`` and otherwise
    ``sqrt(sum((x - mean) ** 2) / (n * (n - 1)))``. ``p`` is the exact
    two-sided sign-flip p-value: all ``2 ** n`` sign vectors ``s_i`` in
    ``{-1, 1}`` are enumerated and
    ``p = 2 ** -n * #{|sum(s_i * x_i) / n| >= |mean|}``; ``lower``/``upper``
    are ``mean - z * se`` / ``mean + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, lag, z, groups`` and each group object uses the key
    order ``key, n, mean, p, se, lower, upper`` with groups in ascending
    bucket order. ``z`` and every numeric result are rendered with exactly
    six decimals, negative zero normalized to ``0.000000``; bucket keys and
    ``n`` are integers. An empty ``details`` yields
    ``{"minutes":60,"lag":1,"z":1.960000,"groups":[]}``. ``details`` or
    ``neighbors`` not being a list raises ``TypeError``; every other
    contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if not isinstance(neighbors, list):
        raise TypeError("neighbors must be a list")
    minutes = _validate_minutes(minutes)
    if isinstance(lag, bool) or not isinstance(lag, int):
        raise ValueError("lag must be an integer")
    if lag < 1:
        raise ValueError("lag must be a positive integer")
    z_value = _validate_finite_number(z, "z")
    if z_value < 0:
        raise ValueError("z must be non-negative")

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

        bucket_means: dict[int, dict[str, Decimal]] = {
            bucket: {
                cell_id: total / count
                for cell_id, (total, count) in cells.items()
            }
            for bucket, cells in buckets.items()
        }

        items = []
        offset_buckets = lag * minutes * 60
        for bucket in sorted(bucket_means):
            previous = bucket_means.get(bucket - offset_buckets)
            if previous is None:
                continue
            current = bucket_means[bucket]
            paired: list[Decimal] = []
            for endpoint_a, endpoint_b in sorted(edges):
                if (
                    endpoint_a in current
                    and endpoint_b in current
                    and endpoint_a in previous
                    and endpoint_b in previous
                ):
                    delta_a = current[endpoint_a] - previous[endpoint_a]
                    delta_b = current[endpoint_b] - previous[endpoint_b]
                    paired.append(delta_a - delta_b)
            if not paired:
                continue
            n = len(paired)
            if n > _SPATIOTEMPORAL_MAX_N:
                raise ValueError(
                    f"bucket {bucket} has {n} pairings; spatiotemporal report "
                    f"requires at most {_SPATIOTEMPORAL_MAX_N} pairings per bucket"
                )

            total = Decimal(0)
            for value in paired:
                total += value
            mean = total / n

            if n > 1:
                squared = Decimal(0)
                for value in paired:
                    deviation = value - mean
                    squared += deviation * deviation
                se = (squared / (n * (n - 1))).sqrt()
            else:
                se = Decimal(0)

            # |sum(s_i * x_i) / n| >= |mean| is equivalent (n > 0) to
            # |sum(s_i * x_i)| >= |total|; compare the raw sums so exact
            # ties are decided without any division rounding.
            hits = 0
            for mask in range(1 << n):
                signed_sum = Decimal(0)
                for index, value in enumerate(paired):
                    if (mask >> index) & 1:
                        signed_sum -= value
                    else:
                        signed_sum += value
                if abs(signed_sum) >= abs(total):
                    hits += 1
            p_value = Decimal(hits) / Decimal(1 << n)

            lower = mean - z_value * se
            upper = mean + z_value * se

            items.append(
                '{"key":' + str(bucket)
                + ',"n":' + str(n)
                + ',"mean":' + _format6(mean)
                + ',"p":' + _format6(p_value)
                + ',"se":' + _format6(se)
                + ',"lower":' + _format6(lower)
                + ',"upper":' + _format6(upper)
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"lag":' + str(lag)
            + ',"z":' + _format6(z_value)
            + ',"groups":[' + ",".join(items) + ']}'
        )


def effect_matrix_spatiotemporal_fdr_report(
    details: list,
    neighbors: list,
    *,
    minutes: int = 60,
    lag: int = 1,
    alpha: float = 0.05,
) -> str:
    """Pair neighbor deltas across a temporal lag per bucket and BH-adjust.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``neighbors`` is a list of ``(a, b)`` two-tuples
    describing an undirected adjacency: ``a`` and ``b`` must be distinct
    non-empty cell id strings occurring in ``details``; self-loops and
    repeated edges (in either orientation) are illegal. ``minutes`` must be
    a non-boolean integer in ``1..1440`` that divides 1440; ``lag`` must be
    a positive non-boolean integer; ``alpha`` is a non-boolean finite number
    with ``0 < alpha <= 1``.

    Rows are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within
    each ``(B, c)`` pair are averaged. For each bucket ``B``, the edges are
    visited in lexicographic order and an edge ``(a, b)`` contributes a
    pairing only when both endpoints have a value at ``B`` and at
    ``B - lag * minutes * 60``; with ``d_{B, c}`` the bucket/cell mean and
    ``Delta_c = d_{B, c} - d_{B - lag * minutes * 60, c}``, the paired value
    is ``x = Delta_a - Delta_b``. A bucket with no pairings is omitted; a
    bucket with more than 16 pairings raises ``ValueError``.

    With ``n`` the number of pairings in a bucket, ``mean = sum(x) / n`` and
    ``p`` is the exact two-sided sign-flip p-value: the proportion of all
    ``2 ** n`` sign vectors ``s_i`` in ``{-1, 1}`` for which
    ``|sum(s_i * x_i) / n| >= |mean|``. With ``N`` the number of retained
    buckets, buckets are ranked ascending by ``(p, B)`` and each rank ``j``
    (1-based) gets the Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    bucket; ``reject`` is ``q <= alpha``, compared on the unquantized
    values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, lag, alpha, groups`` and each group object uses the
    key order ``key, n, mean, p, q, reject`` with groups in ascending bucket
    order. ``alpha``, ``mean``, ``p`` and ``q`` are rendered with exactly
    six decimals, negative zero normalized to ``0.000000``; bucket keys and
    ``n`` are integers. An empty result yields
    ``{"minutes":60,"lag":1,"alpha":0.050000,"groups":[]}``. ``details`` or
    ``neighbors`` not being a list raises ``TypeError``; every other
    contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if not isinstance(neighbors, list):
        raise TypeError("neighbors must be a list")
    minutes = _validate_minutes(minutes)
    if isinstance(lag, bool) or not isinstance(lag, int):
        raise ValueError("lag must be an integer")
    if lag < 1:
        raise ValueError("lag must be a positive integer")
    alpha_value = _validate_finite_number(alpha, "alpha")
    if alpha_value <= 0 or alpha_value > 1:
        raise ValueError("alpha must be greater than 0 and at most 1")

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

        bucket_means: dict[int, dict[str, Decimal]] = {
            bucket: {
                cell_id: total / count
                for cell_id, (total, count) in cells.items()
            }
            for bucket, cells in buckets.items()
        }

        # One record per retained bucket: ``[bucket, n, mean, p, q]`` with q
        # filled in by the Benjamini-Hochberg step below.
        records: list[list] = []
        offset_buckets = lag * minutes * 60
        for bucket in sorted(bucket_means):
            previous = bucket_means.get(bucket - offset_buckets)
            if previous is None:
                continue
            current = bucket_means[bucket]
            paired: list[Decimal] = []
            for endpoint_a, endpoint_b in sorted(edges):
                if (
                    endpoint_a in current
                    and endpoint_b in current
                    and endpoint_a in previous
                    and endpoint_b in previous
                ):
                    delta_a = current[endpoint_a] - previous[endpoint_a]
                    delta_b = current[endpoint_b] - previous[endpoint_b]
                    paired.append(delta_a - delta_b)
            if not paired:
                continue
            n = len(paired)
            if n > _SPATIOTEMPORAL_MAX_N:
                raise ValueError(
                    f"bucket {bucket} has {n} pairings; spatiotemporal report "
                    f"requires at most {_SPATIOTEMPORAL_MAX_N} pairings per bucket"
                )

            total = Decimal(0)
            for value in paired:
                total += value
            mean = total / n

            # |sum(s_i * x_i) / n| >= |mean| is equivalent (n > 0) to
            # |sum(s_i * x_i)| >= |total|; compare the raw sums so exact
            # ties are decided without any division rounding.
            hits = 0
            for mask in range(1 << n):
                signed_sum = Decimal(0)
                for index, value in enumerate(paired):
                    if (mask >> index) & 1:
                        signed_sum -= value
                    else:
                        signed_sum += value
                if abs(signed_sum) >= abs(total):
                    hits += 1
            p_value = Decimal(hits) / Decimal(1 << n)

            records.append([bucket, n, mean, p_value, None])

        # Benjamini-Hochberg q-values across all retained buckets: rank
        # ascending by (p, bucket), then accumulate the running minimum of
        # N * p_l / l from the top rank down, mapping q back to each bucket.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (records[idx][3], records[idx][0]),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Decimal(count) * records[idx][3] / rank
            if candidate < running:
                running = candidate
            records[idx][4] = running

        items = []
        for bucket, n, mean, p_value, q_value in records:
            reject = q_value <= alpha_value
            items.append(
                '{"key":' + str(bucket)
                + ',"n":' + str(n)
                + ',"mean":' + _format6(mean)
                + ',"p":' + _format6(p_value)
                + ',"q":' + _format6(q_value)
                + ',"reject":' + ("true" if reject else "false")
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"lag":' + str(lag)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(items) + ']}'
        )


_TEMPORAL_LAG_MAX_N = 8


def effect_matrix_temporal_lag_report(
    rows: list,
    *,
    minutes: int = 60,
    lag: int = 1,
) -> str:
    """Pair per-cell bucket means across a temporal lag and emit a JSON report.

    ``rows`` is a list of ``(timestamp, cell_id, value)`` three-tuples:
    ``timestamp`` must be a non-boolean non-negative integer, ``cell_id`` a
    non-empty string and ``value`` a finite non-boolean int/float;
    ``(timestamp, cell_id)`` pairs must be unique. ``minutes`` must be a
    non-boolean integer in ``1..1440`` that divides 1440; ``lag`` must be a
    positive non-boolean integer.

    Rows are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)`` and the values within
    each ``(B, c)`` pair are averaged. For each cell ``c``, every bucket
    ``B`` with a mean at both ``B`` and ``B - lag * minutes * 60``
    contributes one pairing with ``x`` the current mean and ``y`` the lagged
    mean. A cell with fewer than 2 pairings is omitted; a cell with more
    than 8 pairings raises ``ValueError``.

    With ``n`` the number of pairings, ``Sxx = sum((x - mean(x)) ** 2)``,
    ``Syy = sum((y - mean(y)) ** 2)`` and
    ``Sxy = sum((x - mean(x)) * (y - mean(y)))``. When ``Sxx`` or ``Syy``
    is 0, ``slope = corr = 0`` and ``p = 1``; otherwise
    ``slope = Sxy / Sxx``, ``corr = Sxy / sqrt(Sxx * Syy)`` and ``p`` is the
    exact permutation p-value: all ``n!`` permutations of ``y`` are
    enumerated and ``p`` is the proportion with ``|Sxy_perm| >= |Sxy|``,
    compared on the unquantized values via exact rational arithmetic so
    theoretically-equal statistics are never split by rounding.

    Numbers enter as ``Decimal(str(x))`` and bucket accumulation happens
    under a precision-1000, ROUND_HALF_EVEN local context; the per-bucket
    means are then lifted to exact fractions for the statistics and their
    permutation test. Returns a compact UTF-8 JSON string with no spaces and
    no trailing newline; the top-level key order is ``minutes, lag, groups``
    and each group object uses the key order ``key, n, slope, corr, p``
    with ``key`` the cell id and groups in ascending cell id order.
    ``slope``, ``corr`` and ``p`` are rendered with exactly six decimals,
    negative zero normalized to ``0.000000``; ``n`` is an integer and cell
    ids are JSON-escaped with Unicode preserved. An empty ``rows`` yields
    ``{"minutes":60,"lag":1,"groups":[]}``. ``rows`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    minutes = _validate_minutes(minutes)
    if isinstance(lag, bool) or not isinstance(lag, int):
        raise ValueError("lag must be an integer")
    if lag < 1:
        raise ValueError("lag must be a positive integer")

    parsed = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(
                "each row must be a (timestamp, cell_id, value) three-tuple"
            )
        timestamp, cell_id, value = row
        _validate_timestamp(timestamp)
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell_id must be a non-empty string")
        validated_value = _validate_finite_number(value, "value")
        key = (timestamp, cell_id)
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append((timestamp, cell_id, validated_value))

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # cell_id -> bucket start -> [value sum, row count]
        cells: dict[str, dict[int, list]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for timestamp, cell_id, value in parsed:
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                acc = cells.setdefault(cell_id, {}).setdefault(
                    bucket, [Decimal(0), 0]
                )
                acc[0] += value
                acc[1] += 1

        offset_buckets = lag * minutes * 60
        groups = []
        for cell_id in sorted(cells):
            means = {
                bucket: Fraction(total) / count
                for bucket, (total, count) in cells[cell_id].items()
            }
            pairs = [
                (means[bucket], means[bucket - offset_buckets])
                for bucket in sorted(means)
                if bucket - offset_buckets in means
            ]
            n = len(pairs)
            if n < 2:
                continue
            if n > _TEMPORAL_LAG_MAX_N:
                raise ValueError(
                    f"cell {cell_id!r} has {n} pairings; temporal lag report "
                    f"requires at most {_TEMPORAL_LAG_MAX_N} pairings per cell"
                )

            x = [pair[0] for pair in pairs]
            y = [pair[1] for pair in pairs]
            mean_x = sum(x, Fraction(0)) / n
            mean_y = sum(y, Fraction(0)) / n
            dx = [value - mean_x for value in x]
            dy = [value - mean_y for value in y]
            sxx = sum((deviation * deviation for deviation in dx), Fraction(0))
            syy = sum((deviation * deviation for deviation in dy), Fraction(0))
            sxy = sum(
                (dx[index] * dy[index] for index in range(n)), Fraction(0)
            )

            if sxx == 0 or syy == 0:
                slope = Decimal(0)
                corr = Decimal(0)
                p_value = Decimal(1)
            else:
                slope = _fraction_to_decimal(sxy / sxx)
                corr = _fraction_to_decimal(sxy) / (
                    _fraction_to_decimal(sxx) * _fraction_to_decimal(syy)
                ).sqrt()
                # Permuting y permutes the dy deviations; since the dx
                # deviations sum to zero, Sxy_perm = sum(dx_i * dy_perm[i])
                # and exact Fraction comparison decides ties without any
                # division rounding.
                hits = 0
                factorial = math.factorial(n)
                for perm in permutations(dy):
                    perm_sxy = sum(
                        (dx[index] * perm[index] for index in range(n)),
                        Fraction(0),
                    )
                    if abs(perm_sxy) >= abs(sxy):
                        hits += 1
                p_value = _fraction_to_decimal(Fraction(hits, factorial))

            groups.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"slope":' + _format6(slope)
                + ',"corr":' + _format6(corr)
                + ',"p":' + _format6(p_value)
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"lag":' + str(lag)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


_THEILSEN_MAX_N = 8


def _s_score(values: list[Decimal]) -> int:
    """Kendall score ``S = sum_{i < j} sign(values[j] - values[i])`` with
    ``sign`` in ``{-1, 0, 1}``."""
    score = 0
    size = len(values)
    for i in range(size):
        for j in range(i + 1, size):
            if values[j] > values[i]:
                score += 1
            elif values[j] < values[i]:
                score -= 1
    return score


def effect_matrix_theilsen_report(
    details: list,
    *,
    minutes: int = 60,
    min_points: int = 3,
) -> str:
    """Aggregate scenario deltas into a per-cell Theil-Sen / Kendall trend report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440; ``min_points`` must be a non-boolean
    integer greater than or equal to 3.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within each
    ``(bucket, cell_id)`` pair are averaged. For each cell id ``c`` the
    occupied buckets, in ascending bucket order, give the points
    ``(x, y) = (bucket, mean delta)``; cells with fewer than
    ``min_points`` occupied buckets are omitted and a cell with more than 8
    occupied buckets raises ``ValueError``. The kept cells are emitted in
    ascending cell id order.

    With ``n`` the number of points, ``slope`` is the Theil-Sen estimator:
    the median of the ``C(n, 2)`` pairwise slopes
    ``(y_j - y_i) / (x_j - x_i)`` for ``i < j`` (the interpolated median of
    the sorted slopes when their count is even). ``S`` is the Kendall score
    ``sum_{i < j} sign(y_j - y_i)`` with ``sign`` in ``{-1, 0, 1}`` and
    ``tau = S / C(n, 2)``. ``p`` is the exact two-sided permutation
    p-value: all ``n!`` permutations of ``y`` are enumerated in lexicographic
    order (duplicate values not deduplicated), ``S'`` is recomputed for each
    and ``p`` is the proportion with ``|S'| >= |S|``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups`` and each group object uses the key order
    ``key, n, slope, tau, p`` with ``key`` the cell id. Every numeric result
    is rendered with exactly six decimals, negative zero normalized to
    ``0.000000``; ``n`` is an integer and cell ids are JSON-escaped with
    Unicode preserved. An empty ``details`` yields
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

        groups = []
        for cell_id in sorted(cells):
            buckets = cells[cell_id]
            n = len(buckets)
            if n < min_points:
                continue
            if n > _THEILSEN_MAX_N:
                raise ValueError(
                    f"cell {cell_id!r} has {n} occupied buckets; Theil-Sen "
                    f"report requires at most {_THEILSEN_MAX_N} buckets per cell"
                )
            points = [
                (Decimal(bucket), total / count)
                for bucket, (total, count) in sorted(buckets.items())
            ]
            ys = [y for _, y in points]

            pairwise_slopes = []
            for i in range(n):
                x_i, y_i = points[i]
                for j in range(i + 1, n):
                    x_j, y_j = points[j]
                    pairwise_slopes.append((y_j - y_i) / (x_j - x_i))
            pairwise_slopes.sort()
            mid = len(pairwise_slopes) // 2
            if len(pairwise_slopes) % 2:
                slope = pairwise_slopes[mid]
            else:
                slope = (pairwise_slopes[mid - 1] + pairwise_slopes[mid]) / 2

            s_value = _s_score(ys)
            pair_count = n * (n - 1) // 2
            tau = Decimal(s_value) / pair_count

            abs_s = abs(s_value)
            hits = 0
            for perm in permutations(ys):
                if abs(_s_score(list(perm))) >= abs_s:
                    hits += 1
            p_value = Decimal(hits) / Decimal(math.factorial(n))

            groups.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"slope":' + _format6(slope)
                + ',"tau":' + _format6(tau)
                + ',"p":' + _format6(p_value)
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


_WILCOXON_MAX_M = 16


def _average_ranks(values: list[Fraction]) -> list[Fraction]:
    """Ranks (1-based) of ``values`` in ascending order, ties sharing the
    average of the ranks they span."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [Fraction(0)] * len(values)
    low = 0
    while low < len(order):
        high = low
        while high + 1 < len(order) and values[order[high + 1]] == values[order[low]]:
            high += 1
        average = Fraction(low + 1 + high + 1, 2)
        for position in range(low, high + 1):
            ranks[order[position]] = average
        low = high + 1
    return ranks


def effect_matrix_wilcoxon_report(
    before: list,
    after: list,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Pair two scenario-detail tables into a Wilcoxon signed-rank FDR report.

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
    string order. With ``n`` the number of pairs in a bucket/cell cell, each
    pair contributes the change ``x = after.delta - before.delta`` and
    ``change`` is ``sum(x) / n``. The Wilcoxon signed-rank test is applied
    to the ``m`` nonzero changes (a cell with ``m > 16`` raises
    ``ValueError``): the nonzero ``|x|`` are ranked ascending with ties
    sharing average ranks, ``w`` is the sum of the ranks of the positive
    changes and ``p`` is the exact two-sided p-value from enumerating all
    ``2 ** m`` sign vectors ``s_i`` in ``{-1, 1}``: with ``W'`` the rank sum
    of the positively signed entries,
    ``p = min(1, 2 * min(Pr(W' <= w), Pr(W' >= w)))``; ``p`` is 1 when
    ``m`` is 0.

    With ``N`` the number of bucket/cell cells, all cells are ranked
    ascending by ``(p, bucket, cell)`` and each rank ``j`` (1-based) gets
    the Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    bucket/cell; ``reject`` is ``q <= alpha``, compared on the unquantized
    values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context; ranks, ``w`` and the
    sign-enumeration tail counts are exact rational arithmetic. Returns a
    compact UTF-8 JSON string with no spaces and no trailing newline; the
    top-level key order is ``minutes, alpha, groups``, each group object
    uses the key order ``key, cells`` and each cell object the key order
    ``key, n, m, change, w, p, q, reject`` with ``key`` the cell id.
    ``alpha`` and every numeric result are rendered with exactly six
    decimals, negative zero normalized to ``0.000000``; bucket keys, cell
    sizes and ``m`` are integers and cell ids are JSON-escaped with Unicode
    preserved. ``before`` or ``after`` not being a list raises ``TypeError``;
    every other contract violation raises ``ValueError``.
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

        # (bucket start, cell_id) -> list of paired changes
        cells: dict[tuple[int, str], list[Decimal]] = {}
        if before_map:
            bucket_seconds = minutes * 60
            for (timestamp, cell_id), before_delta in before_map.items():
                after_delta = after_map[(timestamp, cell_id)]
                bucket = (timestamp // bucket_seconds) * bucket_seconds
                cells.setdefault((bucket, cell_id), []).append(
                    after_delta - before_delta
                )

        # One record per bucket/cell in ascending (bucket, cell) output
        # order: ``[bucket, cell_id, n, m, change, w, p, q]`` with q filled
        # in below.
        records: list[list] = []
        for bucket, cell_id in sorted(cells):
            changes = cells[(bucket, cell_id)]
            n = len(changes)
            nonzero = [change for change in changes if change != 0]
            m = len(nonzero)
            if m > _WILCOXON_MAX_M:
                raise ValueError(
                    f"cell ({bucket}, {cell_id!r}) has {m} nonzero changes; "
                    f"Wilcoxon report requires at most {_WILCOXON_MAX_M} "
                    "nonzero changes per bucket/cell"
                )
            total = Decimal(0)
            for change in changes:
                total += change
            change_mean = total / n

            if m == 0:
                w_value = Fraction(0)
                p_fraction = Fraction(1)
            else:
                ranks = _average_ranks([Fraction(abs(x)) for x in nonzero])
                w_value = sum(
                    (rank for rank, x in zip(ranks, nonzero) if x > 0),
                    Fraction(0),
                )
                lower = 0
                upper = 0
                for mask in range(1 << m):
                    signed_sum = Fraction(0)
                    for index, rank in enumerate(ranks):
                        if (mask >> index) & 1:
                            signed_sum += rank
                    if signed_sum <= w_value:
                        lower += 1
                    if signed_sum >= w_value:
                        upper += 1
                p_fraction = min(
                    Fraction(1),
                    2 * min(lower, upper) / Fraction(1 << m),
                )
            records.append(
                [
                    bucket,
                    cell_id,
                    n,
                    m,
                    change_mean,
                    _fraction_to_decimal(w_value),
                    _fraction_to_decimal(p_fraction),
                    None,
                ]
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
        for bucket, cell_id, n, m, change_mean, w_value, p_value, q_value in records:
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
                + ',"m":' + str(m)
                + ',"change":' + _format6(change_mean)
                + ',"w":' + _format6(w_value)
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


def _validate_ventilation_record(
    record: object,
) -> tuple[int, str, Decimal, Decimal, Decimal, Decimal]:
    if not isinstance(record, Mapping):
        raise ValueError("each record must be a mapping")
    if set(record.keys()) != _VENTILATION_KEYS:
        raise ValueError(
            "each record must contain exactly the keys 'timestamp', "
            "'cell_id', 'wind_u', 'wind_v', 'height' and 'density'"
        )

    timestamp = _validate_timestamp(record["timestamp"])

    cell_id = record["cell_id"]
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")

    wind_u = _validate_finite_number(record["wind_u"], "wind_u")
    wind_v = _validate_finite_number(record["wind_v"], "wind_v")
    height = _validate_finite_number(record["height"], "height")
    density = _validate_finite_number(record["density"], "density")
    if height < 0:
        raise ValueError("height must be non-negative")
    if not 0 <= density <= 1:
        raise ValueError("density must be in [0, 1]")
    return timestamp, cell_id, wind_u, wind_v, height, density


def ventilation_report(records: list, *, minutes: int = 60) -> str:
    """Aggregate wind records into a time-bucket x cell ventilation report.

    ``records`` is a list of mappings, each with exactly the keys
    ``timestamp``, ``cell_id``, ``wind_u``, ``wind_v``, ``height`` and
    ``density``: ``timestamp`` must be a non-boolean non-negative integer,
    ``cell_id`` a non-empty string and the other four fields finite
    non-boolean int/float values with ``height >= 0`` and
    ``0 <= density <= 1``. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string order.
    Each cell object carries the group size ``n`` and the within-group
    arithmetic means ``wind_u``, ``wind_v``, ``height`` and ``density``
    (``u``, ``v``, ``h`` and ``d`` below), plus
    ``speed = sqrt(u ** 2 + v ** 2)`` and
    ``ventilation = speed * (1 - d) / (1 + h / 10)``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups``, each group object uses the key order
    ``key, cells`` and each cell object uses the key order
    ``key, n, wind_u, wind_v, speed, height, density, ventilation`` with
    ``key`` the cell id. An empty ``records`` yields
    ``{"minutes":60,"groups":[]}``. Bucket keys and ``n`` are integers and
    every other numeric result is rendered with exactly six decimals,
    negative zero normalized to ``0.000000``. ``records`` not being a list
    raises ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    minutes = _validate_minutes(minutes)

    parsed = [_validate_ventilation_record(record) for record in records]

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # bucket start -> cell_id -> list of (wind_u, wind_v, height, density)
        buckets: dict[int, dict[str, list[tuple[Decimal, ...]]]] = {}
        for timestamp, cell_id, wind_u, wind_v, height, density in parsed:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(
                (wind_u, wind_v, height, density)
            )

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                rows = buckets[bucket][cell_id]
                n = len(rows)
                sums = [Decimal(0), Decimal(0), Decimal(0), Decimal(0)]
                for values in rows:
                    for index in range(4):
                        sums[index] += values[index]
                u = sums[0] / n
                v = sums[1] / n
                h = sums[2] / n
                d = sums[3] / n
                speed = (u * u + v * v).sqrt()
                ventilation = speed * (1 - d) / (1 + h / 10)
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"wind_u":' + _format6(u)
                    + ',"wind_v":' + _format6(v)
                    + ',"speed":' + _format6(speed)
                    + ',"height":' + _format6(h)
                    + ',"density":' + _format6(d)
                    + ',"ventilation":' + _format6(ventilation)
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


def _validate_coupling_record(
    record: object,
) -> tuple[int, str, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]:
    if not isinstance(record, Mapping):
        raise ValueError("each record must be a mapping")
    if set(record.keys()) != _COUPLING_KEYS:
        raise ValueError(
            "each record must contain exactly the keys 'timestamp', "
            "'cell_id', 'residual', 'temp', 'wind_u', 'wind_v', 'height' "
            "and 'density'"
        )

    timestamp = _validate_timestamp(record["timestamp"])

    cell_id = record["cell_id"]
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")

    residual = _validate_finite_number(record["residual"], "residual")
    temp = _validate_finite_number(record["temp"], "temp")
    wind_u = _validate_finite_number(record["wind_u"], "wind_u")
    wind_v = _validate_finite_number(record["wind_v"], "wind_v")
    height = _validate_finite_number(record["height"], "height")
    density = _validate_finite_number(record["density"], "density")
    if height < 0:
        raise ValueError("height must be non-negative")
    if not 0 <= density <= 1:
        raise ValueError("density must be in [0, 1]")
    return timestamp, cell_id, residual, temp, wind_u, wind_v, height, density


def microclimate_coupling_report(rows: list, *, minutes: int = 60) -> str:
    """Aggregate residual/temp/wind rows into a time-bucket x cell coupling report.

    ``rows`` is a list of mappings, each with exactly the keys ``timestamp``,
    ``cell_id``, ``residual``, ``temp``, ``wind_u``, ``wind_v``, ``height``
    and ``density``: ``timestamp`` must be a non-boolean non-negative
    integer, ``cell_id`` a non-empty string and the other six fields finite
    non-boolean int/float values with ``height >= 0`` and
    ``0 <= density <= 1``. ``(timestamp, cell_id)`` pairs must be unique.
    ``minutes`` must be a non-boolean integer in ``1..1440`` that divides
    1440.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; buckets are emitted in
    ascending order and, within each bucket, cells in ascending string
    order. Each cell object carries the group size ``n`` and ``coupling``
    computed from the within-group arithmetic means (``r``, ``x``, ``u``,
    ``v``, ``h`` and ``d`` below) as
    ``r * x * sqrt(u ** 2 + v ** 2) * (1 - d) / (1 + h / 10)``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups``, each group object uses the key order
    ``key, cells`` and each cell object uses the key order
    ``key, n, coupling`` with ``key`` the cell id. An empty ``rows`` yields
    ``{"minutes":60,"groups":[]}``. Bucket keys and ``n`` are integers and
    ``coupling`` is rendered as a string with exactly six decimals, negative
    zero normalized to ``0.000000``. ``rows`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    minutes = _validate_minutes(minutes)

    parsed = [_validate_coupling_record(record) for record in rows]
    seen = set()
    for timestamp, cell_id, *_ in parsed:
        key = (timestamp, cell_id)
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # bucket start -> cell_id -> list of
        # (residual, temp, wind_u, wind_v, height, density)
        buckets: dict[int, dict[str, list[tuple[Decimal, ...]]]] = {}
        for timestamp, cell_id, residual, temp, wind_u, wind_v, height, density in parsed:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(
                (residual, temp, wind_u, wind_v, height, density)
            )

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                cell_rows = buckets[bucket][cell_id]
                n = len(cell_rows)
                sums = [Decimal(0)] * 6
                for values in cell_rows:
                    for index in range(6):
                        sums[index] += values[index]
                r = sums[0] / n
                x = sums[1] / n
                u = sums[2] / n
                v = sums[3] / n
                h = sums[4] / n
                d = sums[5] / n
                speed = (u * u + v * v).sqrt()
                coupling = r * x * speed * (1 - d) / (1 + h / 10)
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"coupling":"' + _format6(coupling) + '"'
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


def microclimate_coupling_uhi_report(
    rows: list,
    zones: dict,
    *,
    minutes: int = 60,
) -> str:
    """Aggregate residual/temp/wind rows into a per-zone UHI coupling report.

    ``rows`` is a list of mappings, each with exactly the keys ``timestamp``,
    ``cell_id``, ``residual``, ``temp``, ``wind_u``, ``wind_v``, ``height``
    and ``density``: ``timestamp`` must be a non-boolean non-negative
    integer, ``cell_id`` a non-empty string and the other six fields finite
    non-boolean int/float values with ``height >= 0`` and
    ``0 <= density <= 1``. ``(timestamp, cell_id)`` pairs must be unique.
    ``zones`` maps non-empty grid cell ID strings to ``'urban'`` or
    ``'rural'`` and its keys must be exactly the cell IDs occurring in
    ``rows`` (the empty ``rows``/empty ``zones`` pair is allowed and yields
    ``{"minutes":60,"groups":[]}``). ``minutes`` must be a non-boolean
    integer in ``1..1440`` that divides 1440.

    Rows are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)``. Within each
    ``(B, c)`` pair the six fields are averaged (means ``r``, ``x``, ``u``,
    ``v``, ``h`` and ``d`` below) and the cell coupling is
    ``q = r * x * sqrt(u ** 2 + v ** 2) * (1 - d) / (1 + h / 10)``. Within
    each bucket the cell couplings are averaged per zone; a bucket is
    dropped when either zone has no contributing cell. ``n_urban`` and
    ``n_rural`` are the contributing cell counts and ``uhi`` is the urban
    mean minus the rural mean.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups`` and each group object uses the key order
    ``key, n_urban, n_rural, urban, rural, uhi`` with groups in ascending
    bucket order. Bucket keys and the two counts are integers; ``urban``,
    ``rural`` and ``uhi`` are strings with exactly six decimals, negative
    zero normalized to ``"0.000000"``. ``rows`` not being a list or
    ``zones`` not being a dict raises ``TypeError``; every other contract
    violation raises ``ValueError``.
    """
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    zones = _validate_fusion_zones(zones)
    minutes = _validate_minutes(minutes)

    parsed = [_validate_coupling_record(record) for record in rows]
    seen = set()
    cell_ids = set()
    for timestamp, cell_id, *_ in parsed:
        key = (timestamp, cell_id)
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        cell_ids.add(cell_id)

    if set(zones) != cell_ids:
        raise ValueError(
            "zones keys must be exactly the cell IDs occurring in rows"
        )

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # bucket start -> cell_id -> [sum_r, sum_x, sum_u, sum_v, sum_h, sum_d, n]
        buckets: dict[int, dict[str, list]] = {}
        for timestamp, cell_id, residual, temp, wind_u, wind_v, height, density in parsed:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = buckets.setdefault(bucket, {}).setdefault(
                cell_id, [Decimal(0)] * 6 + [0]
            )
            for index, value in enumerate(
                (residual, temp, wind_u, wind_v, height, density)
            ):
                acc[index] += value
            acc[6] += 1

        groups = []
        for bucket in sorted(buckets):
            zone_sums: dict[str, list] = {}
            for cell_id, acc in buckets[bucket].items():
                n = acc[6]
                r = acc[0] / n
                x = acc[1] / n
                u = acc[2] / n
                v = acc[3] / n
                h = acc[4] / n
                d = acc[5] / n
                speed = (u * u + v * v).sqrt()
                q = r * x * speed * (1 - d) / (1 + h / 10)
                zone_acc = zone_sums.setdefault(zones[cell_id], [Decimal(0), 0])
                zone_acc[0] += q
                zone_acc[1] += 1
            urban = zone_sums.get("urban")
            rural = zone_sums.get("rural")
            if not urban or not rural:
                continue
            urban_mean = urban[0] / urban[1]
            rural_mean = rural[0] / rural[1]
            groups.append(
                '{"key":' + str(bucket)
                + ',"n_urban":' + str(urban[1])
                + ',"n_rural":' + str(rural[1])
                + ',"urban":"' + _format6(urban_mean) + '"'
                + ',"rural":"' + _format6(rural_mean) + '"'
                + ',"uhi":"' + _format6(urban_mean - rural_mean) + '"'
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def microclimate_coupling_trend_report(
    rows: list,
    zones: dict,
    *,
    minutes: int = 60,
    min_points: int = 2,
) -> str:
    """Fit a time trend to the per-bucket UHI coupling delta and emit JSON.

    ``rows`` is a list of mappings, each with exactly the keys ``timestamp``,
    ``cell_id``, ``residual``, ``temp``, ``wind_u``, ``wind_v``, ``height``
    and ``density``: ``timestamp`` must be a non-boolean non-negative
    integer, ``cell_id`` a non-empty string and the other six fields finite
    non-boolean int/float values with ``height >= 0`` and
    ``0 <= density <= 1``. ``(timestamp, cell_id)`` pairs must be unique.
    ``zones`` maps non-empty grid cell ID strings to ``'urban'`` or
    ``'rural'``; its keys must be exactly the cell IDs occurring in ``rows``
    and both classes must be non-empty. ``minutes`` must be a non-boolean
    integer in ``1..1440`` that divides 1440 and ``min_points`` a
    non-boolean integer greater than or equal to 2.

    Rows are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)``. Within each
    ``(B, c)`` pair the six fields are averaged (means ``r``, ``x``, ``u``,
    ``v``, ``h`` and ``d`` below) and the cell coupling is
    ``q = r * x * sqrt(u ** 2 + v ** 2) * (1 - d) / (1 + h / 10)``. Within
    each bucket the cell couplings are averaged per zone; a bucket is
    dropped when either zone has no contributing cell, and ``delta`` is the
    urban mean minus the rural mean.

    When fewer than ``min_points`` buckets remain, ``groups`` is empty.
    Otherwise an ordinary least-squares slope is fitted to the points
    ``(x, y)`` with ``x`` the bucket start and ``y`` the bucket ``delta``:
    ``slope = sum((x - x_bar) * (y - y_bar)) / sum((x - x_bar) ** 2)`` and
    ``groups`` carries a single object with ``key`` ``"uhi"``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, min_points, groups`` and the group object uses the
    key order ``key, n, slope`` with ``n`` the number of fitted buckets.
    ``n`` is an integer and ``slope`` is a string with exactly six decimals,
    negative zero normalized to ``"0.000000"``. ``rows`` not being a list or
    ``zones`` not being a dict raises ``TypeError``; every other contract
    violation raises ``ValueError``.
    """
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    zones = _validate_fusion_zones(zones)
    minutes = _validate_minutes(minutes)
    min_points = _validate_min_points(min_points)

    parsed = [_validate_coupling_record(record) for record in rows]
    seen = set()
    cell_ids = set()
    for timestamp, cell_id, *_ in parsed:
        key = (timestamp, cell_id)
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        cell_ids.add(cell_id)

    if set(zones) != cell_ids:
        raise ValueError(
            "zones keys must be exactly the cell IDs occurring in rows"
        )
    if set(zones.values()) != _ZONES:
        raise ValueError(
            "zones must contain at least one 'urban' and one 'rural' cell"
        )

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # bucket start -> cell_id -> [sum_r, sum_x, sum_u, sum_v, sum_h, sum_d, n]
        buckets: dict[int, dict[str, list]] = {}
        for timestamp, cell_id, residual, temp, wind_u, wind_v, height, density in parsed:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = buckets.setdefault(bucket, {}).setdefault(
                cell_id, [Decimal(0)] * 6 + [0]
            )
            for index, value in enumerate(
                (residual, temp, wind_u, wind_v, height, density)
            ):
                acc[index] += value
            acc[6] += 1

        # bucket start -> urban mean coupling minus rural mean coupling
        points: list[tuple[int, Decimal]] = []
        for bucket in sorted(buckets):
            zone_sums: dict[str, list] = {}
            for cell_id, acc in buckets[bucket].items():
                n = acc[6]
                r = acc[0] / n
                x = acc[1] / n
                u = acc[2] / n
                v = acc[3] / n
                h = acc[4] / n
                d = acc[5] / n
                speed = (u * u + v * v).sqrt()
                q = r * x * speed * (1 - d) / (1 + h / 10)
                zone_acc = zone_sums.setdefault(zones[cell_id], [Decimal(0), 0])
                zone_acc[0] += q
                zone_acc[1] += 1
            urban = zone_sums.get("urban")
            rural = zone_sums.get("rural")
            if not urban or not rural:
                continue
            points.append(
                (bucket, urban[0] / urban[1] - rural[0] / rural[1])
            )

        groups = []
        if len(points) >= min_points:
            n = len(points)
            x_total = Decimal(0)
            y_total = Decimal(0)
            for bucket, delta in points:
                x_total += bucket
                y_total += delta
            x_bar = x_total / n
            y_bar = y_total / n
            sxx = Decimal(0)
            sxy = Decimal(0)
            for bucket, delta in points:
                dx = bucket - x_bar
                sxx += dx * dx
                sxy += dx * (delta - y_bar)
            slope = sxy / sxx
            groups.append(
                '{"key":"uhi","n":' + str(n)
                + ',"slope":"' + _format6(slope) + '"}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"min_points":' + str(min_points)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def microclimate_coupling_trend_significance_report(
    rows: list,
    zones: dict,
    *,
    minutes: int = 60,
    min_points: int = 3,
    z: float = 1.96,
) -> str:
    """Fit a permutation-tested UHI coupling trend and emit JSON.

    ``rows``/``zones`` follow the ``microclimate_coupling_trend_report``
    contract: ``rows`` is a list of mappings, each with exactly the keys
    ``timestamp``, ``cell_id``, ``residual``, ``temp``, ``wind_u``,
    ``wind_v``, ``height`` and ``density`` with ``timestamp`` a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values (``height >= 0`` and
    ``0 <= density <= 1``); ``(timestamp, cell_id)`` pairs must be unique.
    ``zones`` maps non-empty grid cell ID strings to ``'urban'`` or
    ``'rural'``, its keys must be exactly the cell IDs occurring in ``rows``
    and both classes must be non-empty. ``minutes`` must be a non-boolean
    integer in ``1..1440`` that divides 1440, ``min_points`` a non-boolean
    integer in ``3..8`` and ``z`` a non-boolean finite number with
    ``z >= 0``.

    Bucketing and per-zone aggregation are identical to
    ``microclimate_coupling_trend_report``: bucket key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)``, per-cell coupling
    ``q`` averaged inside each ``(B, c)`` pair, per-zone means per bucket and
    ``delta`` the urban mean minus the rural mean, dropping buckets where
    either zone is absent. When fewer than ``min_points`` buckets remain,
    ``groups`` is empty; when more than 8 remain, ``ValueError`` is raised.

    For the ascending points ``(B, delta)`` with means ``B_bar``/``delta_bar``
    an ordinary least-squares line is fitted:
    ``Sxx = sum((B - B_bar) ** 2)``,
    ``slope = sum((B - B_bar) * (delta - delta_bar)) / Sxx`` and
    ``se = sqrt(sum((delta - delta_bar - slope * (B - B_bar)) ** 2)
    / ((n - 2) * Sxx))``. The exact permutation p-value is the proportion of
    the ``n!`` positional permutations of the ``delta`` values (duplicates not
    deduplicated) whose recomputed slope ``slope'`` satisfies
    ``|slope'| >= |slope|``; ``lower``/``upper`` are
    ``slope - z * se`` / ``slope + z * se``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, min_points, z, groups`` and the group object uses the
    key order ``key, n, slope, se, p, lower, upper`` with ``key`` ``"uhi"``.
    ``n`` is an integer and ``z`` and every numeric result are strings with
    exactly six decimals, negative zero normalized to ``"0.000000"``.
    ``rows`` not being a list or ``zones`` not being a dict raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    zones = _validate_fusion_zones(zones)
    minutes = _validate_minutes(minutes)
    min_points = _validate_trend_significance_min_points(min_points)
    z_value = _validate_finite_number(z, "z")
    if z_value < 0:
        raise ValueError("z must be non-negative")

    parsed = [_validate_coupling_record(record) for record in rows]
    seen = set()
    cell_ids = set()
    for timestamp, cell_id, *_ in parsed:
        key = (timestamp, cell_id)
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        cell_ids.add(cell_id)

    if set(zones) != cell_ids:
        raise ValueError(
            "zones keys must be exactly the cell IDs occurring in rows"
        )
    if set(zones.values()) != _ZONES:
        raise ValueError(
            "zones must contain at least one 'urban' and one 'rural' cell"
        )

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # bucket start -> cell_id -> [sum_r, sum_x, sum_u, sum_v, sum_h, sum_d, n]
        buckets: dict[int, dict[str, list]] = {}
        for timestamp, cell_id, residual, temp, wind_u, wind_v, height, density in parsed:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = buckets.setdefault(bucket, {}).setdefault(
                cell_id, [Decimal(0)] * 6 + [0]
            )
            for index, value in enumerate(
                (residual, temp, wind_u, wind_v, height, density)
            ):
                acc[index] += value
            acc[6] += 1

        # bucket start -> urban mean coupling minus rural mean coupling
        points: list[tuple[int, Decimal]] = []
        for bucket in sorted(buckets):
            zone_sums: dict[str, list] = {}
            for cell_id, acc in buckets[bucket].items():
                n = acc[6]
                r = acc[0] / n
                x = acc[1] / n
                u = acc[2] / n
                v = acc[3] / n
                h = acc[4] / n
                d = acc[5] / n
                speed = (u * u + v * v).sqrt()
                q = r * x * speed * (1 - d) / (1 + h / 10)
                zone_acc = zone_sums.setdefault(zones[cell_id], [Decimal(0), 0])
                zone_acc[0] += q
                zone_acc[1] += 1
            urban = zone_sums.get("urban")
            rural = zone_sums.get("rural")
            if not urban or not rural:
                continue
            points.append(
                (bucket, urban[0] / urban[1] - rural[0] / rural[1])
            )

        groups = []
        n = len(points)
        if n > _TREND_SIGNIFICANCE_MAX_N:
            raise ValueError(
                f"{n} buckets remain; coupling trend significance report "
                f"requires at most {_TREND_SIGNIFICANCE_MAX_N} buckets"
            )
        if n >= min_points:
            x_total = Decimal(0)
            y_total = Decimal(0)
            for bucket, delta in points:
                x_total += bucket
                y_total += delta
            x_bar = x_total / n
            y_bar = y_total / n
            sxx = Decimal(0)
            sxy = Decimal(0)
            for bucket, delta in points:
                dx = bucket - x_bar
                sxx += dx * dx
                sxy += dx * (delta - y_bar)
            slope = sxy / sxx
            sse = Decimal(0)
            for bucket, delta in points:
                residual = (delta - y_bar) - slope * (bucket - x_bar)
                sse += residual * residual
            se = (sse / (Decimal(n - 2) * sxx)).sqrt()
            ys = [delta for _, delta in points]
            hits = 0
            for permuted in permutations(ys):
                permuted_sxy = Decimal(0)
                for (bucket, _), delta in zip(points, permuted):
                    permuted_sxy += (bucket - x_bar) * (delta - y_bar)
                permuted_slope = permuted_sxy / sxx
                if abs(permuted_slope) >= abs(slope):
                    hits += 1
            p_value = Decimal(hits) / Decimal(math.factorial(n))
            lower = slope - z_value * se
            upper = slope + z_value * se
            groups.append(
                '{"key":"uhi","n":' + str(n)
                + ',"slope":"' + _format6(slope) + '"'
                + ',"se":"' + _format6(se) + '"'
                + ',"p":"' + _format6(p_value) + '"'
                + ',"lower":"' + _format6(lower) + '"'
                + ',"upper":"' + _format6(upper) + '"'
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"min_points":' + str(min_points)
            + ',"z":"' + _format6(z_value) + '"'
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def _validate_airflow_row(row: object) -> tuple[int, str, Decimal]:
    """Validate one ``(t, c, v)`` airflow three-tuple."""
    if not isinstance(row, tuple) or len(row) != 3:
        raise ValueError(
            "each airflow row must be a (timestamp, cell_id, v) three-tuple"
        )
    timestamp, cell_id, value = row
    _validate_timestamp(timestamp)
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")
    value_d = _validate_finite_number(value, "v")
    return timestamp, cell_id, value_d


_VENT_EFFECT_MAX_N = 8


def vent_effect_report(details: list, airflow: list, *, minutes: int = 60) -> str:
    """Regress scenario deltas on airflow per cell and emit a compact JSON report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)`` with finite non-boolean numeric fields
    and unique ``(timestamp, cell_id)`` pairs. ``airflow`` is a list of
    ``(timestamp, cell_id, v)`` three-tuples where ``timestamp`` is a
    non-boolean non-negative integer, ``cell_id`` a non-empty string and
    ``v`` a finite non-boolean int/float. ``minutes`` must be a non-boolean
    integer in ``1..1440`` that divides 1440.

    Both inputs are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``. Within each
    ``(bucket, cell_id)`` pair the ``delta`` values of ``details`` and the
    ``v`` values of ``airflow`` are averaged separately, and only pairs
    present in both inputs are kept. For each cell id ``c`` (emitted in
    ascending order) the kept pairs give points ``(x, y)`` with ``x`` the
    mean airflow and ``y`` the mean delta; cells with fewer than 2 points
    are omitted and a cell with more than 8 points raises ``ValueError``.

    With ``x_bar``/``y_bar`` the point means, ``Sxx = sum((x - x_bar) ** 2)``
    and ``Sxy = sum((x - x_bar) * (y - y_bar))``: when ``Sxx`` is 0 the
    ``slope`` is 0 and ``p`` is 1; otherwise ``slope = Sxy / Sxx`` and ``p``
    is the exact permutation p-value — the proportion of the ``n!``
    positional permutations of the ``y`` values whose permuted slope
    ``slope'`` satisfies ``|slope'| >= |slope|``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups`` and each group object uses the key order
    ``key, n, slope, p`` with ``key`` the cell id and ``n`` the number of
    points. ``n`` is an integer and every other numeric result is rendered
    with exactly six decimals, negative zero normalized to ``0.000000``.
    An empty result yields ``{"minutes":60,"groups":[]}``. ``details`` or
    ``airflow`` not being a list raises ``TypeError``; every other contract
    violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if not isinstance(airflow, list):
        raise TypeError("airflow must be a list")
    minutes = _validate_minutes(minutes)

    parsed_details = []
    seen: set[tuple[int, str]] = set()
    for row in details:
        validated = _validate_detail_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed_details.append(validated)

    parsed_airflow = [_validate_airflow_row(row) for row in airflow]

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # (bucket, cell_id) -> [sum, count]
        delta_acc: dict[tuple[int, str], list] = {}
        for validated in parsed_details:
            timestamp, cell_id, delta = validated[0], validated[1], validated[4]
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = delta_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += delta
            acc[1] += 1

        airflow_acc: dict[tuple[int, str], list] = {}
        for timestamp, cell_id, value in parsed_airflow:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = airflow_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += value
            acc[1] += 1

        # cell_id -> list of (mean airflow, mean delta) over shared buckets
        cells: dict[str, list[tuple[Decimal, Decimal]]] = {}
        for key, (delta_total, delta_count) in delta_acc.items():
            airflow_entry = airflow_acc.get(key)
            if airflow_entry is None:
                continue
            airflow_total, airflow_count = airflow_entry
            cells.setdefault(key[1], []).append(
                (airflow_total / airflow_count, delta_total / delta_count)
            )

        items = []
        for cell_id in sorted(cells):
            points = cells[cell_id]
            n = len(points)
            if n < 2:
                continue
            if n > _VENT_EFFECT_MAX_N:
                raise ValueError(
                    f"cell {cell_id!r} has {n} points; vent effect report "
                    f"requires at most {_VENT_EFFECT_MAX_N} points per cell"
                )
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
            if sxx == 0:
                slope = Decimal(0)
                p = Decimal(1)
            else:
                slope = sxy / sxx
                ys = [y for _, y in points]
                hits = 0
                for permuted in permutations(ys):
                    permuted_sxy = Decimal(0)
                    for (x, _), y in zip(points, permuted):
                        permuted_sxy += (x - x_bar) * (y - y_bar)
                    if abs(permuted_sxy) >= abs(sxy):
                        hits += 1
                p = Decimal(hits) / Decimal(math.factorial(n))

            items.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"slope":' + _format6(slope)
                + ',"p":' + _format6(p)
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(items) + ']}'
        )


def effect_matrix_ventilation_effect_report(
    details: list, airflow: list, *, minutes: int = 60
) -> str:
    """Regress scenario deltas on per-record ventilation and emit JSON.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)`` with finite non-boolean numeric fields
    and unique ``(timestamp, cell_id)`` pairs. ``airflow`` is a list of
    ``ventilation_report`` records: mappings with exactly the keys
    ``timestamp``, ``cell_id``, ``wind_u``, ``wind_v``, ``height`` and
    ``density``, where ``timestamp`` is a non-boolean non-negative integer,
    ``cell_id`` a non-empty string and the other four fields finite
    non-boolean int/float values with ``height >= 0`` and
    ``0 <= density <= 1``. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440.

    Both inputs are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``. Within each
    ``(bucket, cell_id)`` pair the ``delta`` values of ``details`` are
    averaged, and each ``airflow`` record contributes
    ``v = sqrt(wind_u ** 2 + wind_v ** 2) * (1 - density) / (1 + height / 10)``
    with the ``v`` values averaged over the pair; only pairs present in both
    inputs are kept. For each cell id ``c`` (emitted in ascending order) the
    kept pairs give points ``(x, y)`` with ``x`` the mean ventilation and
    ``y`` the mean delta; cells with fewer than 2 points are omitted and a
    cell with more than 8 points raises ``ValueError``.

    With ``x_bar``/``y_bar`` the point means, ``Sxx = sum((x - x_bar) ** 2)``
    and ``Sxy = sum((x - x_bar) * (y - y_bar))``: when ``Sxx`` is 0 the
    ``slope`` is 0 and ``p`` is 1; otherwise ``slope = Sxy / Sxx`` and ``p``
    is the exact permutation p-value — the proportion of the ``n!``
    positional permutations of the ``y`` values whose permuted slope
    ``slope'`` satisfies ``|slope'| >= |slope|``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups`` and each group object uses the key order
    ``key, n, slope, p`` with ``key`` the cell id and ``n`` the number of
    points. ``n`` is an integer and every other numeric result is rendered
    with exactly six decimals, negative zero normalized to ``0.000000``.
    An empty result yields ``{"minutes":60,"groups":[]}``. ``details`` or
    ``airflow`` not being a list raises ``TypeError``; every other contract
    violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if not isinstance(airflow, list):
        raise TypeError("airflow must be a list")
    minutes = _validate_minutes(minutes)

    parsed_details = []
    seen: set[tuple[int, str]] = set()
    for row in details:
        validated = _validate_detail_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed_details.append(validated)

    parsed_airflow = [_validate_ventilation_record(record) for record in airflow]

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # (bucket, cell_id) -> [sum, count]
        delta_acc: dict[tuple[int, str], list] = {}
        for validated in parsed_details:
            timestamp, cell_id, delta = validated[0], validated[1], validated[4]
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = delta_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += delta
            acc[1] += 1

        airflow_acc: dict[tuple[int, str], list] = {}
        for timestamp, cell_id, wind_u, wind_v, height, density in parsed_airflow:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            speed = (wind_u * wind_u + wind_v * wind_v).sqrt()
            value = speed * (1 - density) / (1 + height / 10)
            acc = airflow_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += value
            acc[1] += 1

        # cell_id -> list of (mean ventilation, mean delta) over shared buckets
        cells: dict[str, list[tuple[Decimal, Decimal]]] = {}
        for key, (delta_total, delta_count) in delta_acc.items():
            airflow_entry = airflow_acc.get(key)
            if airflow_entry is None:
                continue
            airflow_total, airflow_count = airflow_entry
            cells.setdefault(key[1], []).append(
                (airflow_total / airflow_count, delta_total / delta_count)
            )

        items = []
        for cell_id in sorted(cells):
            points = cells[cell_id]
            n = len(points)
            if n < 2:
                continue
            if n > _VENT_EFFECT_MAX_N:
                raise ValueError(
                    f"cell {cell_id!r} has {n} points; ventilation effect "
                    f"report requires at most {_VENT_EFFECT_MAX_N} points per cell"
                )
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
            if sxx == 0:
                slope = Decimal(0)
                p = Decimal(1)
            else:
                slope = sxy / sxx
                ys = [y for _, y in points]
                hits = 0
                for permuted in permutations(ys):
                    permuted_sxy = Decimal(0)
                    for (x, _), y in zip(points, permuted):
                        permuted_sxy += (x - x_bar) * (y - y_bar)
                    if abs(permuted_sxy) >= abs(sxy):
                        hits += 1
                p = Decimal(hits) / Decimal(math.factorial(n))

            items.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"slope":' + _format6(slope)
                + ',"p":' + _format6(p)
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(items) + ']}'
        )


_CLUSTER_MAX_K = 12


def effect_matrix_cluster_report(
    details: list,
    neighbors: list,
    *,
    minutes: int = 60,
    threshold: float = 0.0,
) -> str:
    """Cluster active bucket/cell mean deltas into edge-connected components.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``neighbors`` is a list of ``(a, b)`` two-tuples
    describing an undirected adjacency: ``a`` and ``b`` must be distinct
    non-empty cell id strings occurring in ``details``; self-loops and
    repeated edges (in either orientation) are illegal. ``minutes`` must be
    a non-boolean integer in ``1..1440`` that divides 1440; ``threshold``
    is a non-boolean finite number greater than or equal to 0.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within each
    bucket/cell pair are averaged. A bucket/cell pair is active when its
    mean delta satisfies ``|mean| >= threshold``; only active cells are
    kept. Per bucket, the active cells are split into connected components
    of the neighbor graph (an active cell without an active neighbor forms
    its own cluster). Groups are emitted in ascending bucket order, clusters
    in ascending first-cell order and each cluster's cells in ascending
    order; buckets without active cells are omitted.

    With ``k`` the number of cells in a cluster (``k > 12`` raises
    ``ValueError``), ``mean`` is the mean of the cluster's cell means,
    ``peak`` is the cell mean with the largest absolute value (ties broken
    by the smaller cell id) and ``kind`` is ``"hot"``, ``"cold"`` or
    ``"mixed"`` when ``mean`` is positive, negative or zero. ``p`` is the
    exact sign-flip share: all ``2 ** k`` sign vectors ``s_i`` in
    ``{-1, 1}`` are enumerated and ``p`` is the proportion with
    ``|sum(s_i * d_i) / k| >= |mean|``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, threshold, groups``, each group object uses the key
    order ``key, clusters`` and each cluster object uses the key order
    ``key, cells, n, mean, peak, p, kind``. Group keys are integer bucket
    starts, cluster keys are the first cell id string, ``cells`` is a list
    of cell id strings and ``n`` is an integer. ``threshold``, ``mean``,
    ``peak`` and ``p`` are rendered with exactly six decimals, negative zero
    normalized to ``0.000000``. An empty ``details`` yields
    ``{"minutes":60,"threshold":0.000000,"groups":[]}``. ``details`` or
    ``neighbors`` not being a list raises ``TypeError``; every other
    contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if not isinstance(neighbors, list):
        raise TypeError("neighbors must be a list")
    minutes = _validate_minutes(minutes)
    threshold_value = _validate_finite_number(threshold, "threshold")
    if threshold_value < 0:
        raise ValueError("threshold must be non-negative")

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
            # cell_id -> mean delta, active cells only
            active: dict[str, Decimal] = {}
            for cell_id, (total, count) in buckets[bucket].items():
                cell_mean = total / count
                if abs(cell_mean) >= threshold_value:
                    active[cell_id] = cell_mean
            if not active:
                continue

            adjacency: dict[str, list[str]] = {cell_id: [] for cell_id in active}
            for endpoint_a, endpoint_b in edges:
                if endpoint_a in active and endpoint_b in active:
                    adjacency[endpoint_a].append(endpoint_b)
                    adjacency[endpoint_b].append(endpoint_a)

            clusters = []
            unseen = set(active)
            while unseen:
                seed = min(unseen)
                unseen.discard(seed)
                component = [seed]
                cursor = 0
                while cursor < len(component):
                    for adjacent in adjacency[component[cursor]]:
                        if adjacent in unseen:
                            unseen.discard(adjacent)
                            component.append(adjacent)
                    cursor += 1
                clusters.append(sorted(component))

            cluster_items = []
            for cells in clusters:
                k = len(cells)
                if k > _CLUSTER_MAX_K:
                    raise ValueError(
                        f"cluster {cells[0]!r} has {k} cells; cluster report "
                        f"requires at most {_CLUSTER_MAX_K} cells per cluster"
                    )
                means = [active[cell_id] for cell_id in cells]
                total = Decimal(0)
                for cell_mean in means:
                    total += cell_mean
                cluster_mean = total / k

                # cells ascend, so a strict comparison keeps the smallest
                # cell id among equally large absolute means.
                peak = means[0]
                for cell_mean in means:
                    if abs(cell_mean) > abs(peak):
                        peak = cell_mean

                if cluster_mean > 0:
                    kind = "hot"
                elif cluster_mean < 0:
                    kind = "cold"
                else:
                    kind = "mixed"

                # |sum(s_i * d_i) / k| >= |mean| is equivalent (k > 0) to
                # |sum(s_i * d_i)| >= |sum(d_i)|; compare the raw sums so
                # exact ties are decided without any division rounding.
                hits = 0
                for mask in range(1 << k):
                    signed_sum = Decimal(0)
                    for index, cell_mean in enumerate(means):
                        if (mask >> index) & 1:
                            signed_sum -= cell_mean
                        else:
                            signed_sum += cell_mean
                    if abs(signed_sum) >= abs(total):
                        hits += 1
                p = Decimal(hits) / Decimal(1 << k)

                cluster_items.append(
                    '{"key":' + json.dumps(cells[0], ensure_ascii=False)
                    + ',"cells":'
                    + json.dumps(cells, ensure_ascii=False, separators=(",", ":"))
                    + ',"n":' + str(k)
                    + ',"mean":' + _format6(cluster_mean)
                    + ',"peak":' + _format6(peak)
                    + ',"p":' + _format6(p)
                    + ',"kind":' + json.dumps(kind)
                    + '}'
                )

            groups.append(
                '{"key":' + str(bucket)
                + ',"clusters":[' + ",".join(cluster_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"threshold":' + _format6(threshold_value)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def _validate_energy_balance_row(
    row: object,
) -> tuple[int, str, Decimal, Decimal, Decimal, Decimal]:
    """Validate one energy balance six-tuple."""
    if not isinstance(row, tuple) or len(row) != 6:
        raise ValueError(
            "each energy balance row must be a (timestamp, cell_id, "
            "net_rad, sensible, latent, storage) six-tuple"
        )
    timestamp, cell_id, net_rad, sensible, latent, storage = row
    _validate_timestamp(timestamp)
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")
    net_rad = _validate_finite_number(net_rad, "net_rad")
    sensible = _validate_finite_number(sensible, "sensible")
    latent = _validate_finite_number(latent, "latent")
    storage = _validate_finite_number(storage, "storage")
    return timestamp, cell_id, net_rad, sensible, latent, storage


def energy_balance_report(records: list, *, minutes: int = 60) -> str:
    """Aggregate energy-balance rows into a time-bucket x cell report.

    ``records`` is a list of six-tuples ``(timestamp, cell_id, net_rad,
    sensible, latent, storage)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other
    four fields finite non-boolean int/float values; ``(timestamp,
    cell_id)`` pairs must be unique. ``minutes`` must be a non-boolean
    integer in ``1..1440`` that divides 1440.

    Rows are bucketed by Unix epoch with key
    ``floor(t / (minutes * 60)) * (minutes * 60)``; every occupied
    bucket/cell pair is kept, buckets emitted in ascending order and
    cells within each bucket in ascending string order. Each cell object
    carries the group size ``n`` and the within-group arithmetic means of
    ``net_rad``, ``sensible``, ``latent`` and ``storage`` plus
    ``residual = net_rad - sensible - latent - storage`` computed from
    those means.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact
    UTF-8 JSON string with no spaces and no trailing newline; the
    top-level key order is ``minutes, groups``, each group object uses
    the key order ``key, cells`` and each cell object uses the key order
    ``key, n, net_rad, sensible, latent, storage, residual`` with
    ``key`` the cell id. An empty ``records`` yields
    ``{"minutes":60,"groups":[]}``. Bucket keys and ``n`` are integers
    and every other numeric result is rendered with exactly six
    decimals, negative zero normalized to ``0.000000``. ``records`` not
    being a list raises ``TypeError``; every other contract violation
    raises ``ValueError``.
    """
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    minutes = _validate_minutes(minutes)

    parsed = []
    seen: set[tuple[int, str]] = set()
    for row in records:
        validated = _validate_energy_balance_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append(validated)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # bucket start -> cell_id -> list of (net_rad, sensible, latent, storage)
        buckets: dict[int, dict[str, list[tuple[Decimal, ...]]]] = {}
        for timestamp, cell_id, net_rad, sensible, latent, storage in parsed:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(
                (net_rad, sensible, latent, storage)
            )

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                rows = buckets[bucket][cell_id]
                n = len(rows)
                sums = [Decimal(0), Decimal(0), Decimal(0), Decimal(0)]
                for values in rows:
                    for index in range(4):
                        sums[index] += values[index]
                net_rad = sums[0] / n
                sensible = sums[1] / n
                latent = sums[2] / n
                storage = sums[3] / n
                residual = net_rad - sensible - latent - storage
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"net_rad":' + _format6(net_rad)
                    + ',"sensible":' + _format6(sensible)
                    + ',"latent":' + _format6(latent)
                    + ',"storage":' + _format6(storage)
                    + ',"residual":' + _format6(residual)
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


def _validate_energy_balance_action(
    action: object, known_cells: set
) -> tuple[str, Decimal, Decimal, Decimal, Decimal]:
    """Validate one ``(cell_id, dn, ds, dl, dst)`` action five-tuple."""
    if not isinstance(action, tuple) or len(action) != 5:
        raise ValueError(
            "each action must be a (cell_id, dn, ds, dl, dst) five-tuple"
        )
    cell_id, dn, ds, dl, dst = action
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("action cell_id must be a non-empty string")
    if cell_id not in known_cells:
        raise ValueError(f"action cell {cell_id!r} is not present in records")
    dn = _validate_finite_number(dn, "dn")
    ds = _validate_finite_number(ds, "ds")
    dl = _validate_finite_number(dl, "dl")
    dst = _validate_finite_number(dst, "dst")
    return cell_id, dn, ds, dl, dst


def energy_balance_scenario_report(
    records: list, actions: list, *, minutes: int = 60
) -> str:
    """Aggregate energy-balance rows with per-cell increments into buckets.

    ``records`` is a list of six-tuples ``(timestamp, cell_id, net_rad,
    sensible, latent, storage)`` with the same contract as
    :func:`energy_balance_report`: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string, the other four
    fields finite non-boolean int/float values and ``(timestamp,
    cell_id)`` pairs unique. ``actions`` is a list of strict
    ``(cell_id, dn, ds, dl, dst)`` five-tuples: ``cell_id`` must occur in
    ``records``, each ``cell_id`` may appear at most once and the four
    increments are finite non-boolean int/float values (any sign); a
    cell's increments apply to every one of its rows and cells without an
    action get zero increments. ``minutes`` must be a non-boolean integer
    in ``1..1440`` that divides 1440.

    Rows are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)``. For every
    occupied ``(B, cell_id)`` pair the four quantities and the increments
    are averaged as ``Decimal(str(x))`` under a precision-1000,
    ROUND_HALF_EVEN local context, and the pair carries ``n`` the number
    of rows, ``n_actions`` (1 when the cell has an action, otherwise 0),
    ``base_residual = mean(n - s - l - st)``,
    ``post_residual = base_residual + dn - ds - dl - dst`` using the
    averaged increments and ``delta_residual = post_residual -
    base_residual``.

    Returns a compact UTF-8 JSON string with no spaces and no trailing
    newline; the top-level key order is ``minutes, groups``, each group
    object uses the key order ``key, cells`` and each cell object uses the
    key order ``key, n, n_actions, base_residual, post_residual,
    delta_residual`` with ``key`` the cell id. Buckets and cells are
    emitted in ascending order. Bucket keys, ``n`` and ``n_actions`` are
    integers and the three residuals are rendered with exactly six
    decimals, negative zero normalized to ``0.000000``. Empty ``records``
    (and ``actions``) yields ``{"minutes":60,"groups":[]}``. ``records``
    or ``actions`` not being a list raises ``TypeError``; every other
    contract violation raises ``ValueError``.
    """
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    if not isinstance(actions, list):
        raise TypeError("actions must be a list")
    minutes = _validate_minutes(minutes)

    parsed = []
    seen: set[tuple[int, str]] = set()
    for row in records:
        validated = _validate_energy_balance_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append(validated)

    action_map: dict[str, tuple[Decimal, Decimal, Decimal, Decimal]] = {}
    known_cells = {cell_id for _, cell_id, *_ in parsed}
    for action in actions:
        cell_id, dn, ds, dl, dst = _validate_energy_balance_action(
            action, known_cells
        )
        if cell_id in action_map:
            raise ValueError(f"duplicate action for cell id: {cell_id!r}")
        action_map[cell_id] = (dn, ds, dl, dst)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zero = Decimal(0)
        bucket_seconds = minutes * 60
        # bucket start -> cell_id -> list of (net_rad, sensible, latent, storage)
        buckets: dict[int, dict[str, list[tuple[Decimal, ...]]]] = {}
        for timestamp, cell_id, net_rad, sensible, latent, storage in parsed:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(
                (net_rad, sensible, latent, storage)
            )

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                rows = buckets[bucket][cell_id]
                n = len(rows)
                sums = [zero, zero, zero, zero]
                for values in rows:
                    for index in range(4):
                        sums[index] += values[index]
                means = [total / n for total in sums]
                dn, ds, dl, dst = action_map.get(cell_id, (zero, zero, zero, zero))
                base_residual = means[0] - means[1] - means[2] - means[3]
                post_residual = base_residual + dn - ds - dl - dst
                delta_residual = post_residual - base_residual
                n_actions = 1 if cell_id in action_map else 0
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"n_actions":' + str(n_actions)
                    + ',"base_residual":"' + _format6(base_residual) + '"'
                    + ',"post_residual":"' + _format6(post_residual) + '"'
                    + ',"delta_residual":"' + _format6(delta_residual) + '"'
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


def _validate_temp_row(row: object) -> tuple[int, str, Decimal]:
    """Validate one ``(t, c, v)`` temperature three-tuple."""
    if not isinstance(row, tuple) or len(row) != 3:
        raise ValueError(
            "each temp row must be a (timestamp, cell_id, value) three-tuple"
        )
    timestamp, cell_id, value = row
    _validate_timestamp(timestamp)
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")
    value_d = _validate_finite_number(value, "value")
    return timestamp, cell_id, value_d


def energy_temperature_report(
    energy: list, temp: list, *, minutes: int = 60
) -> str:
    """Join energy-balance residuals and temperatures per bucket and cell.

    ``energy`` is a list of strict six-tuples ``(timestamp, cell_id,
    net_rad, sensible, latent, storage)`` and ``temp`` a list of strict
    three-tuples ``(timestamp, cell_id, value)``: ``timestamp`` must be a
    non-boolean non-negative integer, ``cell_id`` a non-empty string and
    the numeric fields finite non-boolean int/float values; ``(timestamp,
    cell_id)`` pairs must be unique within each input. ``minutes`` must be
    a non-boolean integer in ``1..1440`` that divides 1440.

    Both inputs are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)``. Within each
    ``(B, cell_id)`` pair the per-row energy residuals
    ``r = net_rad - sensible - latent - storage`` and the ``temp`` values
    are averaged separately, and only pairs present in both inputs are
    kept. Each kept pair carries ``n_energy``/``n_temp`` (the row counts),
    the mean ``residual``, the mean ``temp`` and
    ``coupling = residual * temp``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact
    UTF-8 JSON string with no spaces and no trailing newline; the
    top-level key order is ``minutes, groups``, each group object uses
    the key order ``key, cells`` and each cell object uses the key order
    ``key, n_energy, n_temp, residual, temp, coupling`` with ``key`` the
    cell id. Groups are emitted in ascending bucket order and cells
    within each group in ascending string order. Bucket keys,
    ``n_energy`` and ``n_temp`` are integers and ``residual``, ``temp``
    and ``coupling`` are strings with exactly six decimals, negative zero
    normalized to ``0.000000``. An empty result yields
    ``{"minutes":60,"groups":[]}``. ``energy`` or ``temp`` not being a
    list raises ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    if not isinstance(energy, list):
        raise TypeError("energy must be a list")
    if not isinstance(temp, list):
        raise TypeError("temp must be a list")
    minutes = _validate_minutes(minutes)

    parsed_energy = []
    seen: set[tuple[int, str]] = set()
    for row in energy:
        validated = _validate_energy_balance_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed_energy.append(validated)

    parsed_temp = []
    seen = set()
    for row in temp:
        validated = _validate_temp_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed_temp.append(validated)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # (bucket, cell_id) -> [residual sum, row count]
        energy_acc: dict[tuple[int, str], list] = {}
        for row in parsed_energy:
            timestamp, cell_id = row[0], row[1]
            net_rad, sensible, latent, storage = row[2], row[3], row[4], row[5]
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = energy_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += net_rad - sensible - latent - storage
            acc[1] += 1

        # (bucket, cell_id) -> [temp sum, row count]
        temp_acc: dict[tuple[int, str], list] = {}
        for timestamp, cell_id, value in parsed_temp:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = temp_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += value
            acc[1] += 1

        # bucket -> cell_id -> (n_energy, n_temp, residual, temp)
        buckets: dict[int, dict[str, tuple[int, int, Decimal, Decimal]]] = {}
        for key, (residual_total, energy_count) in energy_acc.items():
            temp_entry = temp_acc.get(key)
            if temp_entry is None:
                continue
            temp_total, temp_count = temp_entry
            buckets.setdefault(key[0], {})[key[1]] = (
                energy_count,
                temp_count,
                residual_total / energy_count,
                temp_total / temp_count,
            )

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                n_energy, n_temp, residual, temp_mean = buckets[bucket][cell_id]
                coupling = residual * temp_mean
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n_energy":' + str(n_energy)
                    + ',"n_temp":' + str(n_temp)
                    + ',"residual":"' + _format6(residual) + '"'
                    + ',"temp":"' + _format6(temp_mean) + '"'
                    + ',"coupling":"' + _format6(coupling) + '"'
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


def energy_temperature_scenario_report(
    energy: list, temp: list, actions: list, *, minutes: int = 60
) -> str:
    """Join energy residuals and temperatures with per-cell actions.

    ``energy`` is a list of strict six-tuples ``(timestamp, cell_id,
    net_rad, sensible, latent, storage)``, ``temp`` a list of strict
    three-tuples ``(timestamp, cell_id, value)`` and ``actions`` a list
    of strict five-tuples ``(cell_id, dn, ds, dl, dst)``: timestamps
    must be non-boolean non-negative integers, ``cell_id`` a non-empty
    string and the numeric fields finite non-boolean int/float values;
    ``(timestamp, cell_id)`` pairs must be unique within each input and
    each action ``cell_id`` must occur in ``energy`` at most once.
    ``minutes`` must be a non-boolean integer in ``1..1440`` that divides
    1440.

    Both inputs are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)``. Within each
    ``(B, cell_id)`` pair the per-row energy residuals
    ``r = net_rad - sensible - latent - storage`` and the ``temp`` values
    are averaged separately as ``Decimal(str(x))``, and only pairs
    present in both inputs are kept. For each kept pair the action
    increments give ``a = dn - ds - dl - dst`` (zero when the cell has no
    action), ``post_residual = base_residual + a`` and
    ``coupling = residual * temp`` for the base, post and delta values.

    Returns a compact UTF-8 JSON string with no spaces and no trailing
    newline; the top-level key order is ``minutes, groups``, each group
    object uses the key order ``key, cells`` and each cell object uses
    the key order ``key, n_energy, n_temp, base_residual, post_residual,
    temp, base_coupling, post_coupling, delta_coupling`` with ``key`` the
    cell id. Groups are emitted in ascending bucket order and cells
    within each group in ascending string order. Bucket keys,
    ``n_energy`` and ``n_temp`` are integers and every other numeric
    result is a string with exactly six decimals, negative zero
    normalized to ``"0.000000"``. An empty result yields
    ``{"minutes":60,"groups":[]}``. ``energy``, ``temp`` or ``actions``
    not being a list raises ``TypeError``; every other contract
    violation raises ``ValueError``.
    """
    if not isinstance(energy, list):
        raise TypeError("energy must be a list")
    if not isinstance(temp, list):
        raise TypeError("temp must be a list")
    if not isinstance(actions, list):
        raise TypeError("actions must be a list")
    minutes = _validate_minutes(minutes)

    parsed_energy = []
    seen: set[tuple[int, str]] = set()
    for row in energy:
        validated = _validate_energy_balance_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed_energy.append(validated)

    parsed_temp = []
    seen = set()
    for row in temp:
        validated = _validate_temp_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed_temp.append(validated)

    action_map: dict[str, tuple[Decimal, Decimal, Decimal, Decimal]] = {}
    known_cells = {cell_id for _, cell_id, *_ in parsed_energy}
    for action in actions:
        cell_id, dn, ds, dl, dst = _validate_energy_balance_action(
            action, known_cells
        )
        if cell_id in action_map:
            raise ValueError(f"duplicate action for cell id: {cell_id!r}")
        action_map[cell_id] = (dn, ds, dl, dst)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zero = Decimal(0)
        bucket_seconds = minutes * 60
        # (bucket, cell_id) -> [residual sum, row count]
        energy_acc: dict[tuple[int, str], list] = {}
        for row in parsed_energy:
            timestamp, cell_id = row[0], row[1]
            net_rad, sensible, latent, storage = row[2], row[3], row[4], row[5]
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = energy_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += net_rad - sensible - latent - storage
            acc[1] += 1

        # (bucket, cell_id) -> [temp sum, row count]
        temp_acc: dict[tuple[int, str], list] = {}
        for timestamp, cell_id, value in parsed_temp:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = temp_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += value
            acc[1] += 1

        # bucket -> cell_id -> (n_energy, n_temp, base_residual, temp)
        buckets: dict[int, dict[str, tuple[int, int, Decimal, Decimal]]] = {}
        for key, (residual_total, energy_count) in energy_acc.items():
            temp_entry = temp_acc.get(key)
            if temp_entry is None:
                continue
            temp_total, temp_count = temp_entry
            buckets.setdefault(key[0], {})[key[1]] = (
                energy_count,
                temp_count,
                residual_total / energy_count,
                temp_total / temp_count,
            )

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                n_energy, n_temp, base_residual, temp_mean = (
                    buckets[bucket][cell_id]
                )
                dn, ds, dl, dst = action_map.get(
                    cell_id, (zero, zero, zero, zero)
                )
                delta_residual = dn - ds - dl - dst
                post_residual = base_residual + delta_residual
                base_coupling = base_residual * temp_mean
                post_coupling = post_residual * temp_mean
                delta_coupling = post_coupling - base_coupling
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n_energy":' + str(n_energy)
                    + ',"n_temp":' + str(n_temp)
                    + ',"base_residual":"' + _format6(base_residual) + '"'
                    + ',"post_residual":"' + _format6(post_residual) + '"'
                    + ',"temp":"' + _format6(temp_mean) + '"'
                    + ',"base_coupling":"' + _format6(base_coupling) + '"'
                    + ',"post_coupling":"' + _format6(post_coupling) + '"'
                    + ',"delta_coupling":"' + _format6(delta_coupling) + '"'
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


def energy_temperature_uhi_report(
    energy: list, temp: list, zones: dict, *, minutes: int = 60
) -> str:
    """Join energy residuals and temperatures into a per-zone UHI report.

    ``energy`` is a list of strict six-tuples ``(timestamp, cell_id,
    net_rad, sensible, latent, storage)`` and ``temp`` a list of strict
    three-tuples ``(timestamp, cell_id, value)``: ``timestamp`` must be a
    non-boolean non-negative integer, ``cell_id`` a non-empty string and
    the numeric fields finite non-boolean int/float values; ``(timestamp,
    cell_id)`` pairs must be unique within each input. ``zones`` maps
    non-empty cell id strings to ``'urban'`` or ``'rural'`` and must
    cover every cell id occurring in ``energy`` and ``temp``. ``minutes``
    must be a non-boolean integer in ``1..1440`` that divides 1440.

    Both inputs are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)``. Within each
    ``(B, cell_id)`` pair the per-row energy residuals
    ``r = net_rad - sensible - latent - storage`` and the ``temp`` values
    are averaged separately, and only pairs present in both inputs are
    kept. Within each bucket the kept cell residuals and temperatures are
    averaged per zone; a bucket is dropped when either zone has no
    contributing cell. ``residual_uhi`` is the urban mean residual minus
    the rural mean residual and ``temp_uhi`` the urban mean temperature
    minus the rural mean temperature.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact
    UTF-8 JSON string with no spaces and no trailing newline; the
    top-level key order is ``minutes, groups`` and each group object uses
    the key order ``key, n_urban, n_rural, residual_uhi, temp_uhi`` with
    groups in ascending bucket order. Bucket keys, ``n_urban`` and
    ``n_rural`` (the per-zone cell counts) are integers and
    ``residual_uhi`` and ``temp_uhi`` are strings with exactly six
    decimals, negative zero normalized to ``"0.000000"``. An empty result
    yields ``{"minutes":60,"groups":[]}``. ``energy`` or ``temp`` not
    being a list or ``zones`` not being a dict raises ``TypeError``;
    every other contract violation raises ``ValueError``.
    """
    if not isinstance(energy, list):
        raise TypeError("energy must be a list")
    if not isinstance(temp, list):
        raise TypeError("temp must be a list")
    zones = _validate_fusion_zones(zones)
    minutes = _validate_minutes(minutes)

    parsed_energy = []
    seen: set[tuple[int, str]] = set()
    for row in energy:
        validated = _validate_energy_balance_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed_energy.append(validated)

    parsed_temp = []
    seen = set()
    for row in temp:
        validated = _validate_temp_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed_temp.append(validated)

    for row in parsed_energy:
        if row[1] not in zones:
            raise ValueError(f"zones is missing cell: {row[1]!r}")
    for _, cell_id, _ in parsed_temp:
        if cell_id not in zones:
            raise ValueError(f"zones is missing cell: {cell_id!r}")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # (bucket, cell_id) -> [residual sum, row count]
        energy_acc: dict[tuple[int, str], list] = {}
        for row in parsed_energy:
            timestamp, cell_id = row[0], row[1]
            net_rad, sensible, latent, storage = row[2], row[3], row[4], row[5]
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = energy_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += net_rad - sensible - latent - storage
            acc[1] += 1

        # (bucket, cell_id) -> [temp sum, row count]
        temp_acc: dict[tuple[int, str], list] = {}
        for timestamp, cell_id, value in parsed_temp:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = temp_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += value
            acc[1] += 1

        # bucket -> zone -> [residual sum, temp sum, cell count]
        zone_acc: dict[int, dict[str, list]] = {}
        for key, (residual_total, energy_count) in energy_acc.items():
            temp_entry = temp_acc.get(key)
            if temp_entry is None:
                continue
            temp_total, temp_count = temp_entry
            bucket, cell_id = key
            acc = zone_acc.setdefault(bucket, {}).setdefault(
                zones[cell_id], [Decimal(0), Decimal(0), 0]
            )
            acc[0] += residual_total / energy_count
            acc[1] += temp_total / temp_count
            acc[2] += 1

        groups = []
        for bucket in sorted(zone_acc):
            per_zone = zone_acc[bucket]
            urban = per_zone.get("urban")
            rural = per_zone.get("rural")
            if not urban or not rural:
                continue
            residual_uhi = urban[0] / urban[2] - rural[0] / rural[2]
            temp_uhi = urban[1] / urban[2] - rural[1] / rural[2]
            groups.append(
                '{"key":' + str(bucket)
                + ',"n_urban":' + str(urban[2])
                + ',"n_rural":' + str(rural[2])
                + ',"residual_uhi":"' + _format6(residual_uhi) + '"'
                + ',"temp_uhi":"' + _format6(temp_uhi) + '"'
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def energy_temperature_scenario_uhi_report(
    energy: list, temp: list, zones: dict, actions: list, *, minutes: int = 60
) -> str:
    """Join energy residuals and temperatures with actions into a UHI report.

    ``energy`` is a list of strict six-tuples ``(timestamp, cell_id,
    net_rad, sensible, latent, storage)``, ``temp`` a list of strict
    three-tuples ``(timestamp, cell_id, value)`` and ``actions`` a list
    of strict five-tuples ``(cell_id, dn, ds, dl, dst)``: timestamps
    must be non-boolean non-negative integers, ``cell_id`` a non-empty
    string and the numeric fields finite non-boolean int/float values;
    ``(timestamp, cell_id)`` pairs must be unique within each input,
    each action ``cell_id`` must occur in ``energy`` and action cell ids
    must be unique. ``zones`` maps non-empty cell id strings to
    ``'urban'`` or ``'rural'`` and must cover every cell id occurring in
    ``energy`` and ``temp``. ``minutes`` must be a non-boolean integer in
    ``1..1440`` that divides 1440.

    Both inputs are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)``. Within each
    ``(B, cell_id)`` pair the per-row energy residuals
    ``r = net_rad - sensible - latent - storage`` and the ``temp`` values
    are averaged separately as ``Decimal(str(x))``, and only pairs
    present in both inputs are kept. For each kept pair the action
    increments give ``a = dn - ds - dl - dst`` (zero when the cell has no
    action) and ``post = r + a``. Within each bucket the kept cell base
    residuals, post residuals and temperatures are averaged per zone; a
    bucket is dropped when either zone has no contributing cell. The
    three UHI values are the urban mean minus the rural mean of the base
    residual, the post residual and the temperature, in that order.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact
    UTF-8 JSON string with no spaces and no trailing newline; the
    top-level key order is ``minutes, groups`` and each group object uses
    the key order ``key, base, post, uhi`` with groups in ascending
    bucket order. Bucket keys are integers and ``base``, ``post`` and
    ``uhi`` are strings with exactly six decimals, negative zero
    normalized to ``"0.000000"``. An empty result yields
    ``{"minutes":60,"groups":[]}``. ``energy``, ``temp`` or ``actions``
    not being a list or ``zones`` not being a dict raises ``TypeError``;
    every other contract violation raises ``ValueError``.
    """
    if not isinstance(energy, list):
        raise TypeError("energy must be a list")
    if not isinstance(temp, list):
        raise TypeError("temp must be a list")
    if not isinstance(zones, dict):
        raise TypeError("zones must be a dict")
    if not isinstance(actions, list):
        raise TypeError("actions must be a list")
    zones = _validate_fusion_zones(zones)
    minutes = _validate_minutes(minutes)

    parsed_energy = []
    seen: set[tuple[int, str]] = set()
    for row in energy:
        validated = _validate_energy_balance_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed_energy.append(validated)

    parsed_temp = []
    seen = set()
    for row in temp:
        validated = _validate_temp_row(row)
        key = (validated[0], validated[1])
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed_temp.append(validated)

    for row in parsed_energy:
        if row[1] not in zones:
            raise ValueError(f"zones is missing cell: {row[1]!r}")
    for _, cell_id, _ in parsed_temp:
        if cell_id not in zones:
            raise ValueError(f"zones is missing cell: {cell_id!r}")

    action_map: dict[str, tuple[Decimal, Decimal, Decimal, Decimal]] = {}
    known_cells = {cell_id for _, cell_id, *_ in parsed_energy}
    for action in actions:
        cell_id, dn, ds, dl, dst = _validate_energy_balance_action(
            action, known_cells
        )
        if cell_id in action_map:
            raise ValueError(f"duplicate action for cell id: {cell_id!r}")
        action_map[cell_id] = (dn, ds, dl, dst)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zero = Decimal(0)
        bucket_seconds = minutes * 60
        # (bucket, cell_id) -> [residual sum, row count]
        energy_acc: dict[tuple[int, str], list] = {}
        for row in parsed_energy:
            timestamp, cell_id = row[0], row[1]
            net_rad, sensible, latent, storage = row[2], row[3], row[4], row[5]
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = energy_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += net_rad - sensible - latent - storage
            acc[1] += 1

        # (bucket, cell_id) -> [temp sum, row count]
        temp_acc: dict[tuple[int, str], list] = {}
        for timestamp, cell_id, value in parsed_temp:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            acc = temp_acc.setdefault((bucket, cell_id), [Decimal(0), 0])
            acc[0] += value
            acc[1] += 1

        # bucket -> zone -> [base sum, post sum, temp sum, cell count]
        zone_acc: dict[int, dict[str, list]] = {}
        for key, (residual_total, energy_count) in energy_acc.items():
            temp_entry = temp_acc.get(key)
            if temp_entry is None:
                continue
            temp_total, temp_count = temp_entry
            bucket, cell_id = key
            base = residual_total / energy_count
            temp_mean = temp_total / temp_count
            dn, ds, dl, dst = action_map.get(
                cell_id, (zero, zero, zero, zero)
            )
            post = base + dn - ds - dl - dst
            acc = zone_acc.setdefault(bucket, {}).setdefault(
                zones[cell_id], [Decimal(0), Decimal(0), Decimal(0), 0]
            )
            acc[0] += base
            acc[1] += post
            acc[2] += temp_mean
            acc[3] += 1

        groups = []
        for bucket in sorted(zone_acc):
            per_zone = zone_acc[bucket]
            urban = per_zone.get("urban")
            rural = per_zone.get("rural")
            if not urban or not rural:
                continue
            base_uhi = urban[0] / urban[3] - rural[0] / rural[3]
            post_uhi = urban[1] / urban[3] - rural[1] / rural[3]
            temp_uhi = urban[2] / urban[3] - rural[2] / rural[3]
            groups.append(
                '{"key":' + str(bucket)
                + ',"base":"' + _format6(base_uhi) + '"'
                + ',"post":"' + _format6(post_uhi) + '"'
                + ',"uhi":"' + _format6(temp_uhi) + '"'
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def effect_matrix_exposure_report(
    details: list, population: dict, *, minutes: int = 60
) -> str:
    """Aggregate scenario deltas into a population-weighted exposure report.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``population`` is a mapping of non-empty cell id
    strings to finite non-boolean int/float values greater than or equal to
    0; every cell id occurring in ``details`` must be present. ``minutes``
    must be a non-boolean integer in ``1..1440`` that divides 1440.

    Rows are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within
    each ``(B, c)`` pair are averaged. Each cell also reports ``n`` the
    number of detail rows in the pair, ``population`` the mapped value and
    ``exposure = delta * population``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``minutes, groups``, each group object uses the key order
    ``key, cells`` (groups in ascending bucket order) and each cell object
    uses the key order ``key, n, delta, population, exposure`` with cells in
    ascending cell id order. Bucket keys and ``n`` are integers and every
    other numeric result is rendered with exactly six decimals, negative
    zero normalized to ``0.000000``. An empty ``details`` yields
    ``{"minutes":60,"groups":[]}``. ``details`` not being a list or
    ``population`` not being a dict raises ``TypeError``; every other
    contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if not isinstance(population, dict):
        raise TypeError("population must be a dict")
    minutes = _validate_minutes(minutes)

    parsed = _validate_detail_rows(details)

    populations: dict[str, Decimal] = {}
    for cell_id, value in population.items():
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("population keys must be non-empty cell id strings")
        pop_value = _validate_finite_number(value, "population value")
        if pop_value < 0:
            raise ValueError("population values must be non-negative")
        populations[cell_id] = pop_value

    with localcontext() as ctx:
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # bucket start -> cell_id -> [deltas..., ]: raw Decimals are stored
        # unrounded; the arithmetic context is widened below so finite
        # arbitrarily large integers keep every digit (the fixed precision
        # of 1000 would otherwise truncate their low integer digits).
        buckets: dict[int, dict[str, list]] = {}
        for validated in parsed:
            timestamp, cell_id, delta = validated[0], validated[1], validated[4]
            if cell_id not in populations:
                raise ValueError(f"missing population for cell id: {cell_id!r}")
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            buckets.setdefault(bucket, {}).setdefault(cell_id, []).append(delta)

        if parsed:
            # Width in integer places, also correct for values stored in
            # scientific notation such as Decimal("1E+308").
            inputs = [validated[4] for validated in parsed]
            inputs.extend(populations.values())
            integer_places = max(
                (max(0, value.adjusted() + 1) for value in inputs), default=0
            )
            # A sum adds at most ceil(log10(n)) integer places and a product
            # at most the two operands' worth; the guard keeps non-terminating
            # divisions correctly rounded at six fractional places.
            row_digits = len(str(len(parsed)))
            ctx.prec = max(
                _MODEL_PRECISION, integer_places * 2 + row_digits + 64
            )

        groups = []
        for bucket in sorted(buckets):
            cell_items = []
            for cell_id in sorted(buckets[bucket]):
                deltas = buckets[bucket][cell_id]
                n = len(deltas)
                delta_total = sum(deltas, Decimal(0))
                delta_mean = delta_total / n
                pop_value = populations[cell_id]
                exposure = delta_mean * pop_value
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"delta":"' + _format6(delta_mean) + '"'
                    + ',"population":"' + _format6(pop_value) + '"'
                    + ',"exposure":"' + _format6(exposure) + '"'
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


def _validate_fusion_cells(cells: object) -> dict:
    if not isinstance(cells, dict):
        raise TypeError(
            "cells must be a dict mapping station IDs to grid cell IDs"
        )
    for station_id, cell_id in cells.items():
        if not isinstance(station_id, str) or not station_id:
            raise ValueError(
                f"cells keys must be non-empty station ID strings, "
                f"got {station_id!r}"
            )
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(
                f"station {station_id!r} must map to a non-empty grid cell ID "
                f"string, got {cell_id!r}"
            )
    return cells


def _validate_fusion_station_record(
    record: object, cells: dict
) -> tuple[str, int, Decimal]:
    if not isinstance(record, Mapping):
        raise ValueError("each station record must be a mapping")
    if set(record.keys()) != _RECORD_KEYS:
        raise ValueError(
            "each station record must contain exactly the keys "
            "'station_id', 'timestamp' and 'temp_c'"
        )

    station_id = record["station_id"]
    if not isinstance(station_id, str) or not station_id:
        raise ValueError(
            f"station_id must be a non-empty string, got {station_id!r}"
        )
    if station_id not in cells:
        raise ValueError(f"unknown station_id: {station_id!r}")

    timestamp = _validate_timestamp(record["timestamp"])
    temp = _validate_temperature(record["temp_c"], "temp_c")
    return station_id, timestamp, temp


def _validate_fusion_satellite_record(record: object) -> tuple[str, int, Decimal]:
    if not isinstance(record, Mapping):
        raise ValueError("each satellite record must be a mapping")
    if set(record.keys()) != _SATELLITE_KEYS:
        raise ValueError(
            "each satellite record must contain exactly the keys "
            "'cell_id', 'timestamp' and 'lst_c'"
        )

    cell_id = record["cell_id"]
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError(
            f"cell_id must be a non-empty string, got {cell_id!r}"
        )

    timestamp = _validate_timestamp(record["timestamp"])
    temp = _validate_temperature(record["lst_c"], "lst_c")
    return cell_id, timestamp, temp


def temperature_fusion_report(
    station: list,
    satellite: list,
    cells: dict,
    *,
    minutes: int = 60,
    min_count: int = 1,
) -> str:
    """Fuse station and satellite temperatures into a bias/RMSE report.

    ``station`` records must be mappings with exactly the keys
    ``station_id`` (a non-empty string), ``timestamp`` (a non-boolean
    non-negative integer) and ``temp_c`` (a finite non-boolean
    int/float); ``satellite`` records must be mappings with exactly the
    keys ``cell_id`` (a non-empty string), ``timestamp`` and ``lst_c``
    under the same rules. ``cells`` maps non-empty station ID strings to
    non-empty grid cell ID strings and must contain every station ID
    occurring in ``station``. ``minutes`` must be a non-boolean integer
    in ``1..1440`` that divides 1440 and ``min_count`` a non-boolean
    positive integer.

    Both sources are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)``. Within each
    ``(B, c)`` pair station temperatures are averaged per station first
    and satellite temperatures are averaged per cell; a pair is kept
    only when both sources are present and at least ``min_count``
    distinct stations contribute, otherwise it is dropped. For each
    kept pair and each contributing station ``d`` is the station mean
    minus the satellite mean; the cell reports ``n_station`` (the
    number of contributing stations), ``n_satellite`` (the number of
    satellite records), ``bias = mean(d)`` and
    ``rmse = sqrt(mean(d ** 2))``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact
    UTF-8 JSON string with no spaces and no trailing newline; the
    top-level key order is ``minutes, groups``, each group object uses
    the key order ``key, cells`` (groups in ascending bucket order) and
    each cell object uses the key order ``key, n_station, n_satellite,
    bias, rmse`` with cells in ascending cell id order. Bucket keys and
    the two counts are integers; ``bias`` and ``rmse`` are strings with
    exactly six decimals, negative zero normalized to ``"0.000000"``.
    An empty result yields ``{"minutes":60,"groups":[]}``. ``station``
    or ``satellite`` not being a list or ``cells`` not being a dict
    raises ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    if not isinstance(station, list):
        raise TypeError("station must be a list")
    if not isinstance(satellite, list):
        raise TypeError("satellite must be a list")
    cells = _validate_fusion_cells(cells)
    minutes = _validate_minutes(minutes)
    min_count = _validate_min_count(min_count)

    station_rows = [_validate_fusion_station_record(record, cells) for record in station]
    satellite_rows = [_validate_fusion_satellite_record(record) for record in satellite]

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # bucket -> cell_id -> station_id -> [sum, count]
        station_buckets: dict[int, dict[str, dict[str, list]]] = {}
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
        satellite_buckets: dict[int, dict[str, list]] = {}
        for cell_id, timestamp, temp in satellite_rows:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            cell_acc = satellite_buckets.setdefault(bucket, {}).setdefault(
                cell_id, [Decimal(0), 0]
            )
            cell_acc[0] += temp
            cell_acc[1] += 1

        groups = []
        for bucket in sorted(station_buckets):
            sat_cells = satellite_buckets.get(bucket)
            if not sat_cells:
                continue
            cell_items = []
            for cell_id in sorted(station_buckets[bucket]):
                if cell_id not in sat_cells:
                    continue
                per_station = station_buckets[bucket][cell_id]
                n_station = len(per_station)
                if n_station < min_count:
                    continue
                sat_total, sat_count = sat_cells[cell_id]
                lst_mean = sat_total / sat_count
                d_sum = Decimal(0)
                d_squared = Decimal(0)
                for total, count in per_station.values():
                    d = total / count - lst_mean
                    d_sum += d
                    d_squared += d * d
                bias = d_sum / n_station
                rmse = (d_squared / n_station).sqrt()
                cell_items.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n_station":' + str(n_station)
                    + ',"n_satellite":' + str(sat_count)
                    + ',"bias":"' + _format6(bias) + '"'
                    + ',"rmse":"' + _format6(rmse) + '"'
                    + '}'
                )
            if cell_items:
                groups.append(
                    '{"key":' + str(bucket)
                    + ',"cells":[' + ",".join(cell_items) + ']}'
                )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def _validate_fusion_zones(zones: object) -> dict:
    if not isinstance(zones, dict):
        raise TypeError(
            "zones must be a dict mapping grid cell IDs to 'urban' or 'rural'"
        )
    for cell_id, zone in zones.items():
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError(
                f"zones keys must be non-empty grid cell ID strings, "
                f"got {cell_id!r}"
            )
        if zone not in _ZONES:
            raise ValueError(
                f"grid cell {cell_id!r} must map to 'urban' or 'rural', "
                f"got {zone!r}"
            )
    return zones


def temperature_fusion_uhi_report(
    station: list,
    satellite: list,
    cells: dict,
    zones: dict,
    *,
    minutes: int = 60,
    min_count: int = 1,
) -> str:
    """Fuse station and satellite temperatures into a per-zone UHI report.

    ``station`` records must be mappings with exactly the keys
    ``station_id`` (a non-empty string), ``timestamp`` (a non-boolean
    non-negative integer) and ``temp_c`` (a finite non-boolean
    int/float); ``satellite`` records must be mappings with exactly the
    keys ``cell_id`` (a non-empty string), ``timestamp`` and ``lst_c``
    under the same rules. ``cells`` maps non-empty station ID strings to
    non-empty grid cell ID strings and must contain every station ID
    occurring in ``station``. ``zones`` maps non-empty grid cell ID
    strings to ``'urban'`` or ``'rural'`` and must cover every grid cell
    occurring in ``cells`` values and in ``satellite`` records.
    ``minutes`` must be a non-boolean integer in ``1..1440`` that
    divides 1440 and ``min_count`` a non-boolean positive integer.

    Both sources are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)``. Within each
    ``(B, c)`` pair station temperatures are averaged per station first
    and then across stations, and satellite temperatures are averaged
    per cell; a pair is kept only when both sources are present and at
    least ``min_count`` distinct stations contribute, otherwise it is
    dropped. The cell bias is the across-station mean minus the
    satellite mean. Within each bucket the kept cell biases are averaged
    per zone; a bucket is dropped when either zone has no contributing
    cell. ``uhi`` is the urban mean bias minus the rural mean bias.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact
    UTF-8 JSON string with no spaces and no trailing newline; the
    top-level key order is ``minutes, groups`` and each group object
    uses the key order ``key, n_urban, n_rural, urban_bias, rural_bias,
    uhi`` with groups in ascending bucket order. Bucket keys and the two
    counts are integers; ``urban_bias``, ``rural_bias`` and ``uhi`` are
    strings with exactly six decimals, negative zero normalized to
    ``"0.000000"``. An empty result yields
    ``{"minutes":60,"groups":[]}``. ``station`` or ``satellite`` not
    being a list or ``cells`` or ``zones`` not being a dict raises
    ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    if not isinstance(station, list):
        raise TypeError("station must be a list")
    if not isinstance(satellite, list):
        raise TypeError("satellite must be a list")
    cells = _validate_fusion_cells(cells)
    zones = _validate_fusion_zones(zones)
    minutes = _validate_minutes(minutes)
    min_count = _validate_min_count(min_count)

    station_rows = [_validate_fusion_station_record(record, cells) for record in station]
    satellite_rows = [_validate_fusion_satellite_record(record) for record in satellite]

    for cell_id in cells.values():
        if cell_id not in zones:
            raise ValueError(f"zones is missing grid cell: {cell_id!r}")
    for cell_id, _, _ in satellite_rows:
        if cell_id not in zones:
            raise ValueError(f"zones is missing grid cell: {cell_id!r}")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        bucket_seconds = minutes * 60
        # bucket -> cell_id -> station_id -> [sum, count]
        station_buckets: dict[int, dict[str, dict[str, list]]] = {}
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
        satellite_buckets: dict[int, dict[str, list]] = {}
        for cell_id, timestamp, temp in satellite_rows:
            bucket = (timestamp // bucket_seconds) * bucket_seconds
            cell_acc = satellite_buckets.setdefault(bucket, {}).setdefault(
                cell_id, [Decimal(0), 0]
            )
            cell_acc[0] += temp
            cell_acc[1] += 1

        groups = []
        for bucket in sorted(station_buckets):
            sat_cells = satellite_buckets.get(bucket)
            if not sat_cells:
                continue
            zone_sums: dict[str, list] = {}
            for cell_id in station_buckets[bucket]:
                if cell_id not in sat_cells:
                    continue
                per_station = station_buckets[bucket][cell_id]
                n_station = len(per_station)
                if n_station < min_count:
                    continue
                sat_total, sat_count = sat_cells[cell_id]
                lst_mean = sat_total / sat_count
                station_mean = sum(
                    (total / count for total, count in per_station.values()),
                    Decimal(0),
                ) / n_station
                zone_acc = zone_sums.setdefault(zones[cell_id], [Decimal(0), 0])
                zone_acc[0] += station_mean - lst_mean
                zone_acc[1] += 1
            urban = zone_sums.get("urban")
            rural = zone_sums.get("rural")
            if not urban or not rural:
                continue
            urban_bias = urban[0] / urban[1]
            rural_bias = rural[0] / rural[1]
            groups.append(
                '{"key":' + str(bucket)
                + ',"n_urban":' + str(urban[1])
                + ',"n_rural":' + str(rural[1])
                + ',"urban_bias":"' + _format6(urban_bias) + '"'
                + ',"rural_bias":"' + _format6(rural_bias) + '"'
                + ',"uhi":"' + _format6(urban_bias - rural_bias) + '"'
                + '}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def model_residual_report(
    model: tuple,
    rows: list,
    *,
    by: str = "time",
    minutes: int = 60,
) -> str:
    """Group fitted-model residuals and report bias, MAE and RMSE per group.

    ``model`` is a ``fit_uhi_model`` return value
    ``(n, b0, bh, bb, bi, bg, r2)``: ``n`` must be a non-boolean positive
    integer and the other six fields finite non-boolean int/float values.
    ``rows`` follows the ``fit_uhi_model`` contract: a list of nine-tuples
    ``(timestamp, cell_id, station_c, satellite_c, d, h, b, i, g)`` with
    unique ``(timestamp, cell_id)`` pairs; an empty list yields empty
    ``groups``. ``by`` is ``"time"`` (rows are bucketed by Unix epoch,
    floored to ``minutes``-sized buckets) or ``"cell"`` (rows are grouped
    by cell id); ``minutes`` must be a non-boolean integer in ``1..1440``
    that divides 1440.

    For each row the prediction is ``p = b0 + bh*h + bb*b + bi*i + bg*g``
    and the residual ``e = d - p``. Groups are emitted in ascending key
    order (bucket-start seconds
    ``floor(t / (minutes * 60)) * (minutes * 60)`` for ``time``, cell id
    strings for ``cell``); each group reports ``n`` the group size,
    ``bias = mean(e)``, ``mae = mean(|e|)`` and
    ``rmse = sqrt(mean(e ** 2))``.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and no trailing newline; the top-level key
    order is ``by, minutes, groups`` and each group object uses the key
    order ``key, n, bias, mae, rmse``. Bucket keys and ``n`` are integers
    and every other numeric result is a fixed six-decimal JSON string,
    negative zero normalized to ``"0.000000"``. ``model`` not being a
    tuple or ``rows`` not being a list raises ``TypeError``; every other
    contract violation raises ``ValueError``.
    """
    if not isinstance(model, tuple):
        raise TypeError("model must be a tuple")
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    if by not in ("time", "cell"):
        raise ValueError("by must be 'time' or 'cell'")
    minutes = _validate_minutes(minutes)

    if len(model) != 7:
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

    parsed = []
    seen: set[tuple[int, str]] = set()
    for row in rows:
        timestamp, cell_id, height, build, imp, green, diff = _validate_feature_row(
            row
        )
        key = (timestamp, cell_id)
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)
        parsed.append((timestamp, cell_id, height, build, imp, green, diff))

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        groups: dict[object, list[Decimal]] = {}
        if parsed:
            bucket_seconds = minutes * 60
            for timestamp, cell_id, height, build, imp, green, diff in parsed:
                residual = diff - (
                    b0 + bh * height + bb * build + bi * imp + bg * green
                )
                if by == "time":
                    group_key = (timestamp // bucket_seconds) * bucket_seconds
                else:
                    group_key = cell_id
                groups.setdefault(group_key, []).append(residual)

        items = []
        for group_key in sorted(groups):
            residuals = groups[group_key]
            n = len(residuals)
            residual_sum = Decimal(0)
            absolute_sum = Decimal(0)
            squared_sum = Decimal(0)
            for residual in residuals:
                residual_sum += residual
                absolute_sum += abs(residual)
                squared_sum += residual * residual
            bias = residual_sum / n
            mae = absolute_sum / n
            rmse = (squared_sum / n).sqrt()
            if by == "time":
                key_json = str(group_key)
            else:
                key_json = json.dumps(group_key, ensure_ascii=False)
            items.append(
                '{"key":' + key_json
                + ',"n":' + str(n)
                + ',"bias":"' + _format6(bias) + '"'
                + ',"mae":"' + _format6(mae) + '"'
                + ',"rmse":"' + _format6(rmse) + '"'
                + '}'
            )

        return (
            '{"by":' + json.dumps(by)
            + ',"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(items) + ']}'
        )


_MORPHOLOGY_KINDS = ("building", "impervious", "green", "other")


def surface_morphology_report(items: list) -> str:
    """Summarize per-cell surface morphology as a compact JSON report.

    ``items`` is a list of ``(cell_id, kind, height, area, grid_area)``
    five-tuples following the ``grid_features`` item contract: ``cell_id``
    a non-empty string, ``kind`` one of ``building``, ``impervious``,
    ``green`` or ``other``, ``height``/``area`` non-negative finite
    non-boolean numbers, ``grid_area`` positive, non-building heights 0 and
    a consistent ``grid_area`` per cell (compared via ``Decimal(str(x))``).
    Each cell must contain at least one building item and at least one
    coverage (non-building) item; the total building area must not exceed
    the grid area and the coverage areas must sum exactly to the grid area.

    For every cell the report carries the area-weighted mean building
    height ``mu = sum(h*a)/A_b`` (0 when the total building area ``A_b`` is
    0), the area-weighted population standard deviation
    ``sqrt(sum(a*(h-mu)**2)/A_b)`` (0 when ``A_b`` is 0), the four kind
    area fractions ``A_k/g`` and the dominant kind (largest ``A_k``, ties
    broken building > impervious > green > other).

    All arithmetic uses ``Decimal(str(x))`` under a precision-1000,
    ROUND_HALF_EVEN local context. Cells are emitted in ascending cell id
    order. Returns a compact JSON string with no spaces and no trailing
    newline; the top-level key is ``cells`` and each cell object uses the
    key order ``key, h, sd, b, i, g, o, dom`` with ``key``/``dom`` strings
    and the six numeric fields rendered as fixed six-decimal strings,
    negative zero normalized to ``0.000000``. An empty ``items`` yields
    ``{"cells":[]}``. ``items`` not being a list raises ``TypeError``;
    every other contract violation raises ``ValueError``.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    parsed = [_validate_grid_item(item) for item in items]

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zero = Decimal(0)
        cells: dict[str, dict] = {}
        for cell_id, kind, height, area, grid_area in parsed:
            entry = cells.get(cell_id)
            if entry is None:
                entry = {
                    "g": grid_area,
                    "buildings": [],
                    "areas": dict.fromkeys(_MORPHOLOGY_KINDS, zero),
                    "n_build": 0,
                    "n_cov": 0,
                }
                cells[cell_id] = entry
            elif entry["g"] != grid_area:
                raise ValueError(
                    f"grid_area for cell {cell_id!r} must be consistent across items"
                )
            entry["areas"][kind] += area
            if kind == "building":
                entry["n_build"] += 1
                entry["buildings"].append((height, area))
            else:
                entry["n_cov"] += 1

        rendered = []
        for cell_id in sorted(cells):
            entry = cells[cell_id]
            if entry["n_build"] < 1:
                raise ValueError(
                    f"cell {cell_id!r} must contain at least one building item"
                )
            if entry["n_cov"] < 1:
                raise ValueError(
                    f"cell {cell_id!r} must contain at least one coverage item"
                )
            grid_area = entry["g"]
            areas = entry["areas"]
            building_area = areas["building"]
            if building_area > grid_area:
                raise ValueError(
                    f"total building area for cell {cell_id!r} exceeds grid_area"
                )
            coverage_area = areas["impervious"] + areas["green"] + areas["other"]
            if coverage_area != grid_area:
                raise ValueError(
                    f"coverage areas for cell {cell_id!r} must sum to grid_area"
                )

            if building_area == 0:
                mu = zero
                sd = zero
            else:
                weighted = zero
                for height, area in entry["buildings"]:
                    weighted += height * area
                mu = weighted / building_area
                squared = zero
                for height, area in entry["buildings"]:
                    deviation = height - mu
                    squared += area * deviation * deviation
                sd = (squared / building_area).sqrt()

            dominant = "building"
            dominant_area = building_area
            for kind in ("impervious", "green", "other"):
                if areas[kind] > dominant_area:
                    dominant = kind
                    dominant_area = areas[kind]

            rendered.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"h":"' + _format6(mu) + '"'
                + ',"sd":"' + _format6(sd) + '"'
                + ',"b":"' + _format6(building_area / grid_area) + '"'
                + ',"i":"' + _format6(areas["impervious"] / grid_area) + '"'
                + ',"g":"' + _format6(areas["green"] / grid_area) + '"'
                + ',"o":"' + _format6(areas["other"] / grid_area) + '"'
                + ',"dom":' + json.dumps(dominant)
                + '}'
            )

        return '{"cells":[' + ",".join(rendered) + ']}'


_THERMAL_KINDS = frozenset({"building", "roof", "impervious", "green", "other"})
_THERMAL_COVERAGE_KINDS = ("roof", "impervious", "green", "other")


def _validate_thermal_number(
    value: object, name: str, *, minimum: Decimal, maximum: Decimal,
    exclusive_min: bool,
) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite int or float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    decimal_value = Decimal(str(value))
    if exclusive_min:
        if decimal_value <= minimum:
            raise ValueError(f"{name} must be greater than {minimum}")
    elif decimal_value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if decimal_value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return decimal_value


def _validate_thermal_item(
    item: object,
) -> tuple[str, str, Decimal, Decimal, Decimal, Decimal, Decimal]:
    if not isinstance(item, tuple) or len(item) != 7:
        raise ValueError(
            "each item must be a (cell_id, kind, height, area, grid_area,"
            " albedo, emissivity) tuple"
        )
    cell_id, kind, height, area, grid_area, albedo, emissivity = item
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")
    _check_hashable(kind, "kind")
    if kind not in _THERMAL_KINDS:
        raise ValueError(
            "kind must be one of 'building', 'roof', 'impervious', 'green'"
            " or 'other'"
        )
    zero = Decimal(0)
    one = Decimal(1)
    height_d = _validate_thermal_number(
        height, "height", minimum=zero, maximum=Decimal("Infinity"),
        exclusive_min=False,
    )
    area_d = _validate_thermal_number(
        area, "area", minimum=zero, maximum=Decimal("Infinity"),
        exclusive_min=False,
    )
    grid_area_d = _validate_thermal_number(
        grid_area, "grid_area", minimum=zero, maximum=Decimal("Infinity"),
        exclusive_min=True,
    )
    albedo_d = _validate_thermal_number(
        albedo, "albedo", minimum=zero, maximum=one, exclusive_min=False,
    )
    emissivity_d = _validate_thermal_number(
        emissivity, "emissivity", minimum=zero, maximum=one, exclusive_min=True,
    )
    if kind != "building" and height_d != 0:
        raise ValueError("height of non-building items must be 0")
    return cell_id, kind, height_d, area_d, grid_area_d, albedo_d, emissivity_d


def surface_thermal_report(items: list) -> str:
    """Summarize per-cell surface thermal properties as a compact JSON report.

    ``items`` is a list of ``(cell_id, kind, height, area, grid_area,
    albedo, emissivity)`` seven-tuples: ``cell_id`` a non-empty string,
    ``kind`` one of ``building``, ``roof``, ``impervious``, ``green`` or
    ``other``, ``height``/``area`` non-negative finite non-boolean numbers,
    ``grid_area`` positive, ``albedo`` in ``[0, 1]``, ``emissivity`` in
    ``(0, 1]``, non-building heights 0 and a consistent ``grid_area`` per
    cell (compared via ``Decimal(str(x))``). Each cell must contain at
    least one building item and at least one coverage (non-building) item;
    the total building area must not exceed the grid area and the four
    coverage kind areas must sum exactly to the grid area.

    For every cell the report carries the area-weighted mean building
    height ``mu = sum(h*a)/sum(a)`` (0 when the total building area is 0),
    the area-weighted population standard deviation
    ``sqrt(sum(a*(h-mu)**2)/sum(a))`` (0 when the total building area is
    0), the four coverage kind area fractions ``sum(a)/grid_area`` (roof,
    impervious, green, other), the coverage area-weighted mean albedo
    ``sum(a*A)/grid_area``, the coverage area-weighted mean emissivity
    ``sum(a*E)/grid_area``, the thermal load
    ``sum(a*(1-A)*E)/grid_area`` and the dominant coverage kind (largest
    area, ties broken roof > impervious > green > other).

    All arithmetic uses ``Decimal(str(x))`` under a precision-1000,
    ROUND_HALF_EVEN local context. Cells are emitted in ascending cell id
    order. Returns a compact JSON string with no spaces and no trailing
    newline; the top-level key is ``cells`` and each cell object uses the
    key order ``key, h, sd, r, i, g, o, A, E, load, dom`` with
    ``key``/``dom`` strings and the nine numeric fields rendered as fixed
    six-decimal strings, negative zero normalized to ``0.000000``. An
    empty ``items`` yields ``{"cells":[]}``. ``items`` not being a list
    raises ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    parsed = [_validate_thermal_item(item) for item in items]

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zero = Decimal(0)
        one = Decimal(1)
        cells: dict[str, dict] = {}
        for cell_id, kind, height, area, grid_area, albedo, emissivity in parsed:
            entry = cells.get(cell_id)
            if entry is None:
                entry = {
                    "g": grid_area,
                    "buildings": [],
                    "areas": dict.fromkeys(_THERMAL_COVERAGE_KINDS, zero),
                    "albedo": zero,
                    "emissivity": zero,
                    "load": zero,
                    "n_build": 0,
                    "n_cov": 0,
                    "building_area": zero,
                }
                cells[cell_id] = entry
            elif entry["g"] != grid_area:
                raise ValueError(
                    f"grid_area for cell {cell_id!r} must be consistent across items"
                )
            if kind == "building":
                entry["n_build"] += 1
                entry["building_area"] += area
                entry["buildings"].append((height, area))
            else:
                entry["n_cov"] += 1
                entry["areas"][kind] += area
                entry["albedo"] += area * albedo
                entry["emissivity"] += area * emissivity
                entry["load"] += area * (one - albedo) * emissivity

        rendered = []
        for cell_id in sorted(cells):
            entry = cells[cell_id]
            if entry["n_build"] < 1:
                raise ValueError(
                    f"cell {cell_id!r} must contain at least one building item"
                )
            if entry["n_cov"] < 1:
                raise ValueError(
                    f"cell {cell_id!r} must contain at least one coverage item"
                )
            grid_area = entry["g"]
            building_area = entry["building_area"]
            if building_area > grid_area:
                raise ValueError(
                    f"total building area for cell {cell_id!r} exceeds grid_area"
                )
            areas = entry["areas"]
            coverage_area = zero
            for kind in _THERMAL_COVERAGE_KINDS:
                coverage_area += areas[kind]
            if coverage_area != grid_area:
                raise ValueError(
                    f"coverage areas for cell {cell_id!r} must sum to grid_area"
                )

            if building_area == 0:
                mu = zero
                sd = zero
            else:
                weighted = zero
                for height, area in entry["buildings"]:
                    weighted += height * area
                mu = weighted / building_area
                squared = zero
                for height, area in entry["buildings"]:
                    deviation = height - mu
                    squared += area * deviation * deviation
                sd = (squared / building_area).sqrt()

            dominant = "roof"
            dominant_area = areas["roof"]
            for kind in ("impervious", "green", "other"):
                if areas[kind] > dominant_area:
                    dominant = kind
                    dominant_area = areas[kind]

            rendered.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"h":"' + _format6(mu) + '"'
                + ',"sd":"' + _format6(sd) + '"'
                + ',"r":"' + _format6(areas["roof"] / grid_area) + '"'
                + ',"i":"' + _format6(areas["impervious"] / grid_area) + '"'
                + ',"g":"' + _format6(areas["green"] / grid_area) + '"'
                + ',"o":"' + _format6(areas["other"] / grid_area) + '"'
                + ',"A":"' + _format6(entry["albedo"] / grid_area) + '"'
                + ',"E":"' + _format6(entry["emissivity"] / grid_area) + '"'
                + ',"load":"' + _format6(entry["load"] / grid_area) + '"'
                + ',"dom":' + json.dumps(dominant)
                + '}'
            )

        return '{"cells":[' + ",".join(rendered) + ']}'


def surface_thermal_scenario_report(items: list, actions: list) -> str:
    """Apply per-kind albedo/emissivity deltas and report the thermal effect.

    ``items`` follows the same ``(cell_id, kind, height, area, grid_area,
    albedo, emissivity)`` seven-tuple contract as
    :func:`surface_thermal_report`. ``actions`` is a list of strict
    ``(cell_id, kind, d_albedo, d_emissivity)`` four-tuples: ``cell_id``
    must occur in ``items``, ``kind`` must be one of ``roof``,
    ``impervious``, ``green`` or ``other`` and the cell must contain a
    coverage item of that kind, ``d_albedo``/``d_emissivity`` must be
    non-boolean finite int/float, and each ``(cell_id, kind)`` pair may
    appear at most once. A delta is applied to every coverage item of the
    named kind in the named cell; afterwards each affected item must still
    satisfy ``albedo`` in ``[0, 1]`` and ``emissivity`` in ``(0, 1]``.

    For every cell the report carries the coverage area-weighted mean
    albedo ``sum(a*A)/grid_area``, the coverage area-weighted mean
    emissivity ``sum(a*E)/grid_area`` and the thermal load
    ``sum(a*(1-A)*E)/grid_area``, each computed before (``base_``) and
    after (``post_``) applying the actions, plus ``delta_load`` (post
    minus base) and ``n_actions`` (the number of actions targeting the
    cell). Cells without actions are unchanged.

    All arithmetic uses ``Decimal(str(x))`` under a precision-1000,
    ROUND_HALF_EVEN local context. Cells are emitted in ascending cell id
    order. Returns a compact JSON string with no spaces and no trailing
    newline; the top-level key is ``cells`` and each cell object uses the
    key order ``key, n_actions, base_A, post_A, base_E, post_E, base_load,
    post_load, delta_load`` with ``key`` a string, ``n_actions`` an
    integer and the seven numeric fields rendered as fixed six-decimal
    strings, negative zero normalized to ``0.000000``. Empty ``items``
    (and ``actions``) yields ``{"cells":[]}``. ``items`` or ``actions``
    not being a list raises ``TypeError``; every other contract violation
    raises ``ValueError``.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if not isinstance(actions, list):
        raise TypeError("actions must be a list")
    parsed = [_validate_thermal_item(item) for item in items]

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zero = Decimal(0)
        one = Decimal(1)
        cells: dict[str, dict] = {}
        for cell_id, kind, height, area, grid_area, albedo, emissivity in parsed:
            entry = cells.get(cell_id)
            if entry is None:
                entry = {
                    "g": grid_area,
                    "n_build": 0,
                    "n_cov": 0,
                    "building_area": zero,
                    "areas": dict.fromkeys(_THERMAL_COVERAGE_KINDS, zero),
                    "coverage": [],
                }
                cells[cell_id] = entry
            elif entry["g"] != grid_area:
                raise ValueError(
                    f"grid_area for cell {cell_id!r} must be consistent across items"
                )
            if kind == "building":
                entry["n_build"] += 1
                entry["building_area"] += area
            else:
                entry["n_cov"] += 1
                entry["areas"][kind] += area
                entry["coverage"].append((kind, area, albedo, emissivity))

        for cell_id, entry in cells.items():
            if entry["n_build"] < 1:
                raise ValueError(
                    f"cell {cell_id!r} must contain at least one building item"
                )
            if entry["n_cov"] < 1:
                raise ValueError(
                    f"cell {cell_id!r} must contain at least one coverage item"
                )
            if entry["building_area"] > entry["g"]:
                raise ValueError(
                    f"total building area for cell {cell_id!r} exceeds grid_area"
                )
            coverage_area = zero
            for kind in _THERMAL_COVERAGE_KINDS:
                coverage_area += entry["areas"][kind]
            if coverage_area != entry["g"]:
                raise ValueError(
                    f"coverage areas for cell {cell_id!r} must sum to grid_area"
                )

        infinity = Decimal("Infinity")
        deltas: dict[str, dict[str, tuple[Decimal, Decimal]]] = {}
        for action in actions:
            if not isinstance(action, tuple) or len(action) != 4:
                raise ValueError(
                    "each action must be a (cell_id, kind, d_albedo,"
                    " d_emissivity) tuple"
                )
            cell_id, kind, d_albedo, d_emissivity = action
            if not isinstance(cell_id, str) or not cell_id:
                raise ValueError("action cell_id must be a non-empty string")
            entry = cells.get(cell_id)
            if entry is None:
                raise ValueError(
                    f"action cell {cell_id!r} is not present in items"
                )
            _check_hashable(kind, "kind")
            if kind not in _THERMAL_COVERAGE_KINDS:
                raise ValueError(
                    "action kind must be one of 'roof', 'impervious', 'green'"
                    " or 'other'"
                )
            if not any(
                item_kind == kind for item_kind, _, _, _ in entry["coverage"]
            ):
                raise ValueError(
                    f"cell {cell_id!r} contains no coverage item of kind"
                    f" {kind!r}"
                )
            cell_deltas = deltas.setdefault(cell_id, {})
            if kind in cell_deltas:
                raise ValueError(
                    f"duplicate action for cell {cell_id!r} kind {kind!r}"
                )
            d_albedo_d = _validate_thermal_number(
                d_albedo, "d_albedo", minimum=-infinity, maximum=infinity,
                exclusive_min=False,
            )
            d_emissivity_d = _validate_thermal_number(
                d_emissivity, "d_emissivity", minimum=-infinity,
                maximum=infinity, exclusive_min=False,
            )
            cell_deltas[kind] = (d_albedo_d, d_emissivity_d)

        rendered = []
        for cell_id in sorted(cells):
            entry = cells[cell_id]
            grid_area = entry["g"]
            cell_deltas = deltas.get(cell_id, {})
            base_albedo = zero
            base_emissivity = zero
            base_load = zero
            post_albedo = zero
            post_emissivity = zero
            post_load = zero
            for kind, area, albedo, emissivity in entry["coverage"]:
                base_albedo += area * albedo
                base_emissivity += area * emissivity
                base_load += area * (one - albedo) * emissivity
                delta = cell_deltas.get(kind)
                if delta is not None:
                    new_albedo = albedo + delta[0]
                    new_emissivity = emissivity + delta[1]
                    if new_albedo < zero or new_albedo > one:
                        raise ValueError(
                            f"action on cell {cell_id!r} kind {kind!r} pushes"
                            " albedo out of [0, 1]"
                        )
                    if new_emissivity <= zero or new_emissivity > one:
                        raise ValueError(
                            f"action on cell {cell_id!r} kind {kind!r} pushes"
                            " emissivity out of (0, 1]"
                        )
                    albedo = new_albedo
                    emissivity = new_emissivity
                post_albedo += area * albedo
                post_emissivity += area * emissivity
                post_load += area * (one - albedo) * emissivity

            base_a = base_albedo / grid_area
            post_a = post_albedo / grid_area
            base_e = base_emissivity / grid_area
            post_e = post_emissivity / grid_area
            base_l = base_load / grid_area
            post_l = post_load / grid_area
            rendered.append(
                '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                + ',"n_actions":' + str(len(cell_deltas))
                + ',"base_A":"' + _format6(base_a) + '"'
                + ',"post_A":"' + _format6(post_a) + '"'
                + ',"base_E":"' + _format6(base_e) + '"'
                + ',"post_E":"' + _format6(post_e) + '"'
                + ',"base_load":"' + _format6(base_l) + '"'
                + ',"post_load":"' + _format6(post_l) + '"'
                + ',"delta_load":"' + _format6(post_l - base_l) + '"'
                + '}'
            )

        return '{"cells":[' + ",".join(rendered) + ']}'


def surface_thermal_attribution_report(items: list, actions: list) -> str:
    """Attribute per-action thermal-load changes as a compact JSON report.

    ``items`` follows the same ``(cell_id, kind, height, area, grid_area,
    albedo, emissivity)`` seven-tuple contract as
    :func:`surface_thermal_report`. ``actions`` follows the same strict
    ``(cell_id, kind, d_albedo, d_emissivity)`` four-tuple contract as
    :func:`surface_thermal_scenario_report`: each ``(cell_id, kind)`` pair
    may appear at most once, the cell must contain a coverage item of the
    named kind and applying the deltas to every such item must leave
    ``albedo`` in ``[0, 1]`` and ``emissivity`` in ``(0, 1]``.

    Each action acts on every coverage item of its kind in its cell. The
    report entry carries the kind area fraction ``fraction = sum(a)/g``,
    the base thermal load ``base = sum(a*(1-A)*E)/g`` over those items,
    the same quantity after applying the action ``post`` and
    ``delta = post - base``, where ``g`` is the cell grid area.

    All arithmetic uses ``Decimal(str(x))`` under a precision-1000,
    ROUND_HALF_EVEN local context. Entries are emitted in ascending cell
    id order with action kinds ordered roof, impervious, green, other.
    Returns a compact JSON string with no spaces and no trailing newline;
    the top-level key is ``actions`` and each object uses the key order
    ``cell, kind, fraction, base, post, delta`` with ``cell``/``kind``
    strings and the four numeric fields rendered as fixed six-decimal
    strings, negative zero normalized to ``0.000000``. Empty ``actions``
    yields ``{"actions":[]}``. ``items`` or ``actions`` not being a list
    raises ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if not isinstance(actions, list):
        raise TypeError("actions must be a list")
    parsed = [_validate_thermal_item(item) for item in items]

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zero = Decimal(0)
        one = Decimal(1)
        cells: dict[str, dict] = {}
        for cell_id, kind, height, area, grid_area, albedo, emissivity in parsed:
            entry = cells.get(cell_id)
            if entry is None:
                entry = {
                    "g": grid_area,
                    "n_build": 0,
                    "n_cov": 0,
                    "building_area": zero,
                    "areas": dict.fromkeys(_THERMAL_COVERAGE_KINDS, zero),
                    "coverage": [],
                }
                cells[cell_id] = entry
            elif entry["g"] != grid_area:
                raise ValueError(
                    f"grid_area for cell {cell_id!r} must be consistent across items"
                )
            if kind == "building":
                entry["n_build"] += 1
                entry["building_area"] += area
            else:
                entry["n_cov"] += 1
                entry["areas"][kind] += area
                entry["coverage"].append((kind, area, albedo, emissivity))

        for cell_id, entry in cells.items():
            if entry["n_build"] < 1:
                raise ValueError(
                    f"cell {cell_id!r} must contain at least one building item"
                )
            if entry["n_cov"] < 1:
                raise ValueError(
                    f"cell {cell_id!r} must contain at least one coverage item"
                )
            if entry["building_area"] > entry["g"]:
                raise ValueError(
                    f"total building area for cell {cell_id!r} exceeds grid_area"
                )
            coverage_area = zero
            for kind in _THERMAL_COVERAGE_KINDS:
                coverage_area += entry["areas"][kind]
            if coverage_area != entry["g"]:
                raise ValueError(
                    f"coverage areas for cell {cell_id!r} must sum to grid_area"
                )

        infinity = Decimal("Infinity")
        deltas: dict[str, dict[str, tuple[Decimal, Decimal]]] = {}
        for action in actions:
            if not isinstance(action, tuple) or len(action) != 4:
                raise ValueError(
                    "each action must be a (cell_id, kind, d_albedo,"
                    " d_emissivity) tuple"
                )
            cell_id, kind, d_albedo, d_emissivity = action
            if not isinstance(cell_id, str) or not cell_id:
                raise ValueError("action cell_id must be a non-empty string")
            entry = cells.get(cell_id)
            if entry is None:
                raise ValueError(
                    f"action cell {cell_id!r} is not present in items"
                )
            _check_hashable(kind, "kind")
            if kind not in _THERMAL_COVERAGE_KINDS:
                raise ValueError(
                    "action kind must be one of 'roof', 'impervious', 'green'"
                    " or 'other'"
                )
            if not any(
                item_kind == kind for item_kind, _, _, _ in entry["coverage"]
            ):
                raise ValueError(
                    f"cell {cell_id!r} contains no coverage item of kind"
                    f" {kind!r}"
                )
            cell_deltas = deltas.setdefault(cell_id, {})
            if kind in cell_deltas:
                raise ValueError(
                    f"duplicate action for cell {cell_id!r} kind {kind!r}"
                )
            d_albedo_d = _validate_thermal_number(
                d_albedo, "d_albedo", minimum=-infinity, maximum=infinity,
                exclusive_min=False,
            )
            d_emissivity_d = _validate_thermal_number(
                d_emissivity, "d_emissivity", minimum=-infinity,
                maximum=infinity, exclusive_min=False,
            )
            cell_deltas[kind] = (d_albedo_d, d_emissivity_d)

        rendered = []
        for cell_id in sorted(cells):
            entry = cells[cell_id]
            grid_area = entry["g"]
            cell_deltas = deltas.get(cell_id)
            if not cell_deltas:
                continue
            for kind in _THERMAL_COVERAGE_KINDS:
                delta = cell_deltas.get(kind)
                if delta is None:
                    continue
                kind_area = zero
                base_load = zero
                post_load = zero
                for item_kind, area, albedo, emissivity in entry["coverage"]:
                    if item_kind != kind:
                        continue
                    kind_area += area
                    base_load += area * (one - albedo) * emissivity
                    new_albedo = albedo + delta[0]
                    new_emissivity = emissivity + delta[1]
                    if new_albedo < zero or new_albedo > one:
                        raise ValueError(
                            f"action on cell {cell_id!r} kind {kind!r} pushes"
                            " albedo out of [0, 1]"
                        )
                    if new_emissivity <= zero or new_emissivity > one:
                        raise ValueError(
                            f"action on cell {cell_id!r} kind {kind!r} pushes"
                            " emissivity out of (0, 1]"
                        )
                    post_load += area * (one - new_albedo) * new_emissivity

                fraction = kind_area / grid_area
                base = base_load / grid_area
                post = post_load / grid_area
                rendered.append(
                    '{"cell":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"kind":' + json.dumps(kind)
                    + ',"fraction":"' + _format6(fraction) + '"'
                    + ',"base":"' + _format6(base) + '"'
                    + ',"post":"' + _format6(post) + '"'
                    + ',"delta":"' + _format6(post - base) + '"'
                    + '}'
                )

        return '{"actions":[' + ",".join(rendered) + ']}'


def surface_thermal_zone_scenario_report(
    items: list, zones: dict, actions: list
) -> str:
    """Report per-zone thermal-load effects of albedo/emissivity actions.

    ``items`` follows the same ``(cell_id, kind, height, area, grid_area,
    albedo, emissivity)`` seven-tuple contract and validation as
    :func:`surface_thermal_report`; duplicate items are allowed. ``zones``
    maps cell ids to ``'urban'`` or ``'rural'``; its keys must be exactly
    the cell ids occurring in ``items`` and both zones must be non-empty.
    ``actions`` follows the same strict ``(cell_id, kind, d_albedo,
    d_emissivity)`` four-tuple contract as
    :func:`surface_thermal_scenario_report`: the cell must occur in
    ``items`` and contain a coverage item of the named kind, each
    ``(cell_id, kind)`` pair may appear at most once, the deltas must be
    non-boolean finite int/float values and applying them to every
    coverage item of the named kind in the named cell must leave
    ``albedo`` in ``[0, 1]`` and ``emissivity`` in ``(0, 1]``.

    For every cell the thermal load is ``sum(a*(1-A)*E)/g`` over its
    coverage items, computed before (``base``) and after (``post``)
    applying the actions, with ``delta = post - base``; ``g`` is the cell
    grid area. Each zone group carries ``n`` (the number of cells in the
    zone) and the per-cell means of ``base``, ``post`` and ``delta``. The
    ``uhi`` object carries the urban minus rural mean of each of the
    three quantities.

    All arithmetic uses ``Decimal(str(x))`` under a precision-1000,
    ROUND_HALF_EVEN local context. Returns a compact UTF-8 JSON string
    with no spaces and no trailing newline; the top-level key order is
    ``groups, uhi`` with groups ordered urban then rural, each group
    object using the key order ``zone, n, base, post, delta`` and the
    ``uhi`` object using the key order ``base, post, delta``. ``n`` is an
    integer and every other numeric result is rendered as a fixed
    six-decimal string, negative zero normalized to ``"0.000000"``. Empty
    inputs are only legal as ``items=[]``, ``zones={}`` and
    ``actions=[]`` together and yield ``{"groups":[],"uhi":null}``.
    ``items`` or ``actions`` not being a list or ``zones`` not being a
    dict raises ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    if not isinstance(items, list):
        raise TypeError("items must be a list")
    if not isinstance(zones, dict):
        raise TypeError("zones must be a dict")
    if not isinstance(actions, list):
        raise TypeError("actions must be a list")
    parsed = [_validate_thermal_item(item) for item in items]

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zero = Decimal(0)
        one = Decimal(1)
        cells: dict[str, dict] = {}
        for cell_id, kind, height, area, grid_area, albedo, emissivity in parsed:
            entry = cells.get(cell_id)
            if entry is None:
                entry = {
                    "g": grid_area,
                    "n_build": 0,
                    "n_cov": 0,
                    "building_area": zero,
                    "areas": dict.fromkeys(_THERMAL_COVERAGE_KINDS, zero),
                    "coverage": [],
                }
                cells[cell_id] = entry
            elif entry["g"] != grid_area:
                raise ValueError(
                    f"grid_area for cell {cell_id!r} must be consistent across items"
                )
            if kind == "building":
                entry["n_build"] += 1
                entry["building_area"] += area
            else:
                entry["n_cov"] += 1
                entry["areas"][kind] += area
                entry["coverage"].append((kind, area, albedo, emissivity))

        for cell_id, entry in cells.items():
            if entry["n_build"] < 1:
                raise ValueError(
                    f"cell {cell_id!r} must contain at least one building item"
                )
            if entry["n_cov"] < 1:
                raise ValueError(
                    f"cell {cell_id!r} must contain at least one coverage item"
                )
            if entry["building_area"] > entry["g"]:
                raise ValueError(
                    f"total building area for cell {cell_id!r} exceeds grid_area"
                )
            coverage_area = zero
            for kind in _THERMAL_COVERAGE_KINDS:
                coverage_area += entry["areas"][kind]
            if coverage_area != entry["g"]:
                raise ValueError(
                    f"coverage areas for cell {cell_id!r} must sum to grid_area"
                )

        if set(zones) != set(cells):
            raise ValueError("zones keys must be exactly the cell ids in items")
        for cell_id, zone in zones.items():
            _check_hashable(zone, "zone")
            if zone not in _ZONES:
                raise ValueError(
                    f"grid cell {cell_id!r} must map to 'urban' or 'rural', "
                    f"got {zone!r}"
                )
        if cells and set(zones.values()) != _ZONES:
            raise ValueError("zones must contain both 'urban' and 'rural' cells")

        infinity = Decimal("Infinity")
        deltas: dict[str, dict[str, tuple[Decimal, Decimal]]] = {}
        for action in actions:
            if not isinstance(action, tuple) or len(action) != 4:
                raise ValueError(
                    "each action must be a (cell_id, kind, d_albedo,"
                    " d_emissivity) tuple"
                )
            cell_id, kind, d_albedo, d_emissivity = action
            if not isinstance(cell_id, str) or not cell_id:
                raise ValueError("action cell_id must be a non-empty string")
            entry = cells.get(cell_id)
            if entry is None:
                raise ValueError(
                    f"action cell {cell_id!r} is not present in items"
                )
            _check_hashable(kind, "kind")
            if kind not in _THERMAL_COVERAGE_KINDS:
                raise ValueError(
                    "action kind must be one of 'roof', 'impervious', 'green'"
                    " or 'other'"
                )
            if not any(
                item_kind == kind for item_kind, _, _, _ in entry["coverage"]
            ):
                raise ValueError(
                    f"cell {cell_id!r} contains no coverage item of kind"
                    f" {kind!r}"
                )
            cell_deltas = deltas.setdefault(cell_id, {})
            if kind in cell_deltas:
                raise ValueError(
                    f"duplicate action for cell {cell_id!r} kind {kind!r}"
                )
            d_albedo_d = _validate_thermal_number(
                d_albedo, "d_albedo", minimum=-infinity, maximum=infinity,
                exclusive_min=False,
            )
            d_emissivity_d = _validate_thermal_number(
                d_emissivity, "d_emissivity", minimum=-infinity,
                maximum=infinity, exclusive_min=False,
            )
            cell_deltas[kind] = (d_albedo_d, d_emissivity_d)

        if not cells:
            return '{"groups":[],"uhi":null}'

        # zone -> [base sum, post sum, cell count]
        zone_acc: dict[str, list] = {
            zone: [zero, zero, 0] for zone in ("urban", "rural")
        }
        for cell_id, entry in cells.items():
            grid_area = entry["g"]
            cell_deltas = deltas.get(cell_id, {})
            base_load = zero
            post_load = zero
            for kind, area, albedo, emissivity in entry["coverage"]:
                base_load += area * (one - albedo) * emissivity
                delta = cell_deltas.get(kind)
                if delta is not None:
                    new_albedo = albedo + delta[0]
                    new_emissivity = emissivity + delta[1]
                    if new_albedo < zero or new_albedo > one:
                        raise ValueError(
                            f"action on cell {cell_id!r} kind {kind!r} pushes"
                            " albedo out of [0, 1]"
                        )
                    if new_emissivity <= zero or new_emissivity > one:
                        raise ValueError(
                            f"action on cell {cell_id!r} kind {kind!r} pushes"
                            " emissivity out of (0, 1]"
                        )
                    albedo = new_albedo
                    emissivity = new_emissivity
                post_load += area * (one - albedo) * emissivity

            acc = zone_acc[zones[cell_id]]
            acc[0] += base_load / grid_area
            acc[1] += post_load / grid_area
            acc[2] += 1

        means: dict[str, tuple[Decimal, Decimal, Decimal]] = {}
        groups = []
        for zone in ("urban", "rural"):
            base_total, post_total, n = zone_acc[zone]
            base_mean = base_total / n
            post_mean = post_total / n
            delta_mean = post_mean - base_mean
            means[zone] = (base_mean, post_mean, delta_mean)
            groups.append(
                '{"zone":' + json.dumps(zone)
                + ',"n":' + str(n)
                + ',"base":"' + _format6(base_mean) + '"'
                + ',"post":"' + _format6(post_mean) + '"'
                + ',"delta":"' + _format6(delta_mean) + '"'
                + '}'
            )

        urban = means["urban"]
        rural = means["rural"]
        return (
            '{"groups":[' + ",".join(groups) + ']'
            + ',"uhi":{"base":"' + _format6(urban[0] - rural[0]) + '"'
            + ',"post":"' + _format6(urban[1] - rural[1]) + '"'
            + ',"delta":"' + _format6(urban[2] - rural[2]) + '"'
            + '}}'
        )
