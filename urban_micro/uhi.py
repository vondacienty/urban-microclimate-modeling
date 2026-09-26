"""Urban heat island (UHI) intensity computation."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from itertools import combinations, permutations
import json
import math
import re

__all__ = [
    "compute_uhi",
    "align_temp",
    "grid_features",
    "fit_uhi_model",
    "scenario",
    "scenario_rank",
    "scenario_score",
    "scenario_sensitivity",
    "score_test",
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
    "effect_matrix_lags_fdr_report",
    "effect_matrix_lags_group_fdr_report",
    "effect_matrix_lags_group_compare_report",
    "window_compare",
    "window_shift",
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
    "decision_priority_shift",
    "decision_priority_consensus",
    "portfolio",
    "portfolio_robustness",
    "portfolio_attribution",
    "panel_priority",
    "panel_report",
    "frontier_sensitivity",
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
    """Pair neighbor deltas across a temporal lag per bucket and emit FDR JSON.

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
    ``p`` is the exact two-sided sign-flip p-value: all ``2 ** n`` sign
    vectors ``s_i`` in ``{-1, 1}`` are enumerated and
    ``p = 2 ** -n * #{|sum(s_i * x_i) / n| >= |mean|}``.

    With ``N`` the number of buckets, all buckets are ranked ascending by
    ``(p, B)`` and each rank ``j`` (1-based) gets the Benjamini-Hochberg
    q-value ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to
    its bucket; ``reject`` is ``q <= alpha``, compared on the unquantized
    values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and exactly one trailing newline; the
    top-level key
    order is ``minutes, lag, alpha, groups`` and each group object uses the
    key order ``key, n, mean, p, q, reject`` with groups in ascending bucket
    order. ``alpha`` and every numeric result are rendered with exactly six
    decimals, negative zero normalized to ``0.000000``; bucket keys and ``n``
    are integers. An empty ``details`` yields
    ``{"minutes":60,"lag":1,"alpha":0.050000,"groups":[]}`` (plus the
    trailing newline). ``details`` or
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

        # One record per emitted bucket: ``[B, n, mean, p, q]`` with q
        # filled in below.
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
                    f"bucket {bucket} has {n} pairings; spatiotemporal fdr "
                    f"report requires at most {_SPATIOTEMPORAL_MAX_N} "
                    f"pairings per bucket"
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

        # Benjamini-Hochberg q-values across ALL buckets: rank ascending by
        # (p, B), then accumulate the running minimum of N * p_l / l from
        # the top rank down, mapping q back to each bucket.
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
            + "\n"
        )


def effect_matrix_lags_fdr_report(
    details: list,
    neighbors: list,
    lags: list,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Pair neighbor deltas across several temporal lags and emit FDR JSON.

    ``details`` is a list of ``scenario`` eight-tuples ``(timestamp, cell_id,
    base, post, delta, cg, cr, cm)``: ``timestamp`` must be a non-boolean
    non-negative integer, ``cell_id`` a non-empty string and the other six
    fields finite non-boolean int/float values; ``(timestamp, cell_id)``
    pairs must be unique. ``neighbors`` is a list of ``(a, b)`` two-tuples
    describing an undirected adjacency: ``a`` and ``b`` must be distinct
    non-empty cell id strings occurring in ``details``; self-loops and
    repeated edges (in either orientation) are illegal. ``lags`` is a list
    of distinct positive non-boolean integers (an empty list is legal).
    ``minutes`` must be a non-boolean integer in ``1..1440`` that divides
    1440; ``alpha`` is a non-boolean finite number with ``0 < alpha <= 1``.

    Rows are bucketed by Unix epoch with key
    ``B = floor(t / (minutes * 60)) * (minutes * 60)`` and the deltas within
    each ``(B, c)`` pair are averaged. For each lag (in ascending order) and
    each bucket ``B``, the edges are visited in lexicographic order and an
    edge ``(a, b)`` contributes a pairing only when both endpoints have a
    value at ``B`` and at ``B - lag * minutes * 60``; with ``d_{B, c}`` the
    bucket/cell mean and ``Delta_c = d_{B, c} - d_{B - lag * minutes * 60,
    c}``, the paired value is ``x = Delta_a - Delta_b``. A bucket with no
    pairings is omitted; a bucket with more than 16 pairings raises
    ``ValueError``.

    With ``n`` the number of pairings in a bucket, ``mean = sum(x) / n`` and
    ``p`` is the exact two-sided sign-flip p-value: all ``2 ** n`` sign
    vectors ``s_i`` in ``{-1, 1}`` are enumerated and
    ``p = 2 ** -n * #{|sum(s_i * x_i) / n| >= |mean|}``.

    With ``N`` the total number of ``(lag, B)`` tests, all tests are ranked
    ascending by ``(p, lag, B)`` and each rank ``j`` (1-based) gets the
    Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    test; ``reject`` is ``q <= alpha``, compared on the unquantized values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and exactly one trailing newline; the
    top-level key order is ``minutes, alpha, groups``, each group object
    uses the key order ``lag, tests`` and each test object the key order
    ``key, n, mean, p, q, reject``, with groups in ascending lag order and
    tests in ascending bucket order. ``alpha`` and every numeric result are
    rendered with exactly six decimals, negative zero normalized to
    ``0.000000``; bucket keys, lags and ``n`` are integers and ``reject`` is
    a boolean. An empty ``details`` or an empty ``lags`` yields
    ``{"minutes":60,"alpha":0.050000,"groups":[]}`` (plus the trailing
    newline). ``details``, ``neighbors`` or ``lags`` not being a list raises
    ``TypeError``; every other contract violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if not isinstance(neighbors, list):
        raise TypeError("neighbors must be a list")
    if not isinstance(lags, list):
        raise TypeError("lags must be a list")
    minutes = _validate_minutes(minutes)
    seen_lags: set[int] = set()
    for lag in lags:
        if isinstance(lag, bool) or not isinstance(lag, int):
            raise ValueError("each lag must be an integer")
        if lag < 1:
            raise ValueError("each lag must be a positive integer")
        if lag in seen_lags:
            raise ValueError(f"duplicate lag: {lag!r}")
        seen_lags.add(lag)
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

        # One record per emitted (lag, bucket) test:
        # ``[lag, B, n, mean, p, q]`` with q filled in below.
        grouped: dict[int, list[list]] = {}
        records: list[list] = []
        for lag in sorted(seen_lags):
            lag_records: list[list] = []
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
                        f"lag {lag} bucket {bucket} has {n} pairings; lags "
                        f"fdr report requires at most {_SPATIOTEMPORAL_MAX_N} "
                        f"pairings per bucket"
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

                record = [lag, bucket, n, mean, p_value, None]
                lag_records.append(record)
                records.append(record)
            grouped[lag] = lag_records

        # Benjamini-Hochberg q-values across ALL (lag, bucket) tests: rank
        # ascending by (p, lag, B), then accumulate the running minimum of
        # N * p_l / l from the top rank down, mapping q back to each test.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (
                records[idx][4],
                records[idx][0],
                records[idx][1],
            ),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Decimal(count) * records[idx][4] / rank
            if candidate < running:
                running = candidate
            records[idx][5] = running

        groups = []
        if parsed:
            for lag in sorted(seen_lags):
                tests = []
                for _, bucket, n, mean, p_value, q_value in grouped[lag]:
                    reject = q_value <= alpha_value
                    tests.append(
                        '{"key":' + str(bucket)
                        + ',"n":' + str(n)
                        + ',"mean":' + _format6(mean)
                        + ',"p":' + _format6(p_value)
                        + ',"q":' + _format6(q_value)
                        + ',"reject":' + ("true" if reject else "false")
                        + '}'
                    )
                groups.append(
                    '{"lag":' + str(lag)
                    + ',"tests":[' + ",".join(tests) + ']}'
                )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(groups) + ']}'
            + "\n"
        )


def _lags_group_x_values(
    details: list,
    neighbors: list,
    lags: list,
    labels: dict,
    minutes: int,
    alpha: float,
) -> tuple[int, list[int], Decimal, list, dict]:
    """Validate the shared lags/group inputs and compute same-label x values.

    This is the common core of :func:`effect_matrix_lags_group_fdr_report`
    and :func:`effect_matrix_lags_group_compare_report`. Returns
    ``(minutes, lags, alpha, parsed, xmap)`` where ``lags`` is the sorted
    list of distinct lags, ``alpha`` the validated alpha as a ``Decimal``,
    ``parsed`` the validated detail rows and ``xmap`` maps
    ``label -> lag -> bucket start -> [x, ...]`` holding only non-empty
    same-label edge pairings (``x = Delta_a - Delta_b`` as in
    :func:`effect_matrix_lags_fdr_report`, edges in lexicographic order).
    All Decimal work runs under a precision-1000, ROUND_HALF_EVEN local
    context. ``details``, ``neighbors`` or ``lags`` not being a list, or
    ``labels`` not being a dict, raises ``TypeError``; every other contract
    violation raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if not isinstance(neighbors, list):
        raise TypeError("neighbors must be a list")
    if not isinstance(lags, list):
        raise TypeError("lags must be a list")
    minutes = _validate_minutes(minutes)
    seen_lags: set[int] = set()
    for lag in lags:
        if isinstance(lag, bool) or not isinstance(lag, int):
            raise ValueError("each lag must be an integer")
        if lag < 1:
            raise ValueError("each lag must be a positive integer")
        if lag in seen_lags:
            raise ValueError(f"duplicate lag: {lag!r}")
        seen_lags.add(lag)
    alpha_value = _validate_finite_number(alpha, "alpha")
    if alpha_value <= 0 or alpha_value > 1:
        raise ValueError("alpha must be greater than 0 and at most 1")

    parsed = _validate_detail_rows(details)

    if not isinstance(labels, dict):
        raise TypeError("labels must be a dict")
    cell_ids = {cell_id for _, cell_id, *_ in parsed}
    if set(labels) != cell_ids:
        raise ValueError("labels keys must be exactly the details cell ids")
    for cell_id, label in labels.items():
        if not isinstance(label, str) or not label:
            raise ValueError("each label must be a non-empty string")

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

    # label -> sorted list of same-label edges participating in its tests
    label_edges: dict[str, list[tuple[str, str]]] = {}
    for edge in sorted(edges):
        label_a = labels[edge[0]]
        label_b = labels[edge[1]]
        if label_a == label_b:
            label_edges.setdefault(label_a, []).append(edge)

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

        xmap: dict[str, dict[int, dict[int, list[Decimal]]]] = {}
        for label in sorted(label_edges):
            lag_map: dict[int, dict[int, list[Decimal]]] = {}
            group_edges = label_edges[label]
            for lag in sorted(seen_lags):
                bucket_map: dict[int, list[Decimal]] = {}
                offset_buckets = lag * minutes * 60
                for bucket in sorted(bucket_means):
                    previous = bucket_means.get(bucket - offset_buckets)
                    if previous is None:
                        continue
                    current = bucket_means[bucket]
                    paired: list[Decimal] = []
                    for endpoint_a, endpoint_b in group_edges:
                        if (
                            endpoint_a in current
                            and endpoint_b in current
                            and endpoint_a in previous
                            and endpoint_b in previous
                        ):
                            delta_a = current[endpoint_a] - previous[endpoint_a]
                            delta_b = current[endpoint_b] - previous[endpoint_b]
                            paired.append(delta_a - delta_b)
                    if paired:
                        bucket_map[bucket] = paired
                lag_map[lag] = bucket_map
            xmap[label] = lag_map

    return minutes, sorted(seen_lags), alpha_value, parsed, xmap


def effect_matrix_lags_group_fdr_report(
    details: list,
    neighbors: list,
    lags: list,
    labels: dict,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Pair same-label neighbor deltas across temporal lags and emit FDR JSON.

    This is the label-grouped counterpart of
    :func:`effect_matrix_lags_fdr_report`: ``details``, ``neighbors``,
    ``lags``, ``minutes`` and ``alpha`` follow exactly the same validation,
    bucketing, pairing, mean and exact two-sided sign-flip p-value rules.

    ``labels`` must be a dict whose keys are exactly the ``cell_id`` values
    occurring in ``details`` and whose values are non-empty strings; passing a
    non-dict raises ``TypeError`` and any key/value mismatch raises
    ``ValueError``. Only edges whose two endpoints share a label participate,
    and each ``(label, lag, B)`` triple with at least one pairing forms one
    test whose ``n`` pairings are the same-label edges in lexicographic order;
    a test with more than 16 pairings raises ``ValueError``.

    With ``N`` the total number of ``(label, lag, B)`` tests, all tests are
    ranked ascending by ``(p, label, lag, B)`` and each rank ``j`` (1-based)
    gets the Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its test;
    ``reject`` is ``q <= alpha``, compared on the unquantized values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and exactly one trailing newline; the top-level
    key order is ``minutes, alpha, groups``, each group object uses the key
    order ``key, lags`` (with ``key`` the label), each lag object the key order
    ``lag, tests`` and each test object the key order
    ``key, n, mean, p, q, reject`` (with ``key`` the bucket start ``B``).
    Groups, lag objects and tests are in ascending label, lag and bucket order
    respectively; labels without any test are omitted. Labels render as JSON
    strings, lags, buckets and ``n`` as integers, ``reject`` as a boolean and
    ``alpha``, ``mean``, ``p`` and ``q`` with exactly six decimals, negative
    zero normalized to ``0.000000``. An empty ``details``, an empty ``lags``
    or the absence of any test yields
    ``{"minutes":60,"alpha":0.050000,"groups":[]}`` (plus the trailing
    newline).
    """
    minutes, lag_list, alpha_value, parsed, xmap = _lags_group_x_values(
        details, neighbors, lags, labels, minutes, alpha
    )

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # One record per emitted (label, lag, bucket) test:
        # ``[label, lag, B, n, mean, p, q]`` with q filled in below.
        grouped: dict[str, dict[int, list[list]]] = {}
        records: list[list] = []
        for label in sorted(xmap):
            lag_groups: dict[int, list[list]] = {}
            for lag in lag_list:
                lag_records: list[list] = []
                for bucket in sorted(xmap[label].get(lag, {})):
                    paired = xmap[label][lag][bucket]
                    n = len(paired)
                    if n > _SPATIOTEMPORAL_MAX_N:
                        raise ValueError(
                            f"label {label!r} lag {lag} bucket {bucket} has "
                            f"{n} pairings; lags group fdr report requires at "
                            f"most {_SPATIOTEMPORAL_MAX_N} pairings per test"
                        )

                    total = Decimal(0)
                    for value in paired:
                        total += value
                    mean = total / n

                    # |sum(s_i * x_i) / n| >= |mean| is equivalent (n > 0) to
                    # |sum(s_i * x_i)| >= |total|; compare the raw sums so
                    # exact ties are decided without any division rounding.
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

                    record = [label, lag, bucket, n, mean, p_value, None]
                    lag_records.append(record)
                    records.append(record)
                lag_groups[lag] = lag_records
            grouped[label] = lag_groups

        # Benjamini-Hochberg q-values across ALL (label, lag, bucket) tests:
        # rank ascending by (p, label, lag, B), then accumulate the running
        # minimum of N * p_l / l from the top rank down, mapping q back.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (
                records[idx][5],
                records[idx][0],
                records[idx][1],
                records[idx][2],
            ),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Decimal(count) * records[idx][5] / rank
            if candidate < running:
                running = candidate
            records[idx][6] = running

        groups = []
        if parsed:
            for label in sorted(grouped):
                lag_items = []
                for lag in lag_list:
                    tests = []
                    for (
                        _,
                        _,
                        bucket,
                        n,
                        mean,
                        p_value,
                        q_value,
                    ) in grouped[label][lag]:
                        reject = q_value <= alpha_value
                        tests.append(
                            '{"key":' + str(bucket)
                            + ',"n":' + str(n)
                            + ',"mean":' + _format6(mean)
                            + ',"p":' + _format6(p_value)
                            + ',"q":' + _format6(q_value)
                            + ',"reject":' + ("true" if reject else "false")
                            + '}'
                        )
                    if tests:
                        lag_items.append(
                            '{"lag":' + str(lag)
                            + ',"tests":[' + ",".join(tests) + ']}'
                        )
                if lag_items:
                    groups.append(
                        '{"key":' + json.dumps(label, ensure_ascii=False)
                        + ',"lags":[' + ",".join(lag_items) + ']}'
                    )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(groups) + ']}'
            + "\n"
        )


def effect_matrix_lags_group_compare_report(
    details: list,
    neighbors: list,
    lags: list,
    labels: dict,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Compare same-label neighbor-delta groups pairwise across lags.

    This is the label-pair comparison counterpart of
    :func:`effect_matrix_lags_group_fdr_report`: ``details``, ``neighbors``,
    ``lags``, ``labels``, ``minutes`` and ``alpha`` follow exactly the same
    validation, bucketing and same-label edge pairing rules, so each
    ``(label, lag, B)`` triple yields the same list of ``x`` values.
    ``details``, ``neighbors`` or ``lags`` not being a list, or ``labels``
    not being a dict, raises ``TypeError``; every other contract violation
    raises ``ValueError``.

    Distinct labels are compared pairwise with ``a < b``. For each label
    pair, each lag (ascending) and each bucket ``B`` (ascending), a test is
    emitted only when both labels have at least one ``x`` value at that
    ``(lag, B)``; with ``n_a``/``n_b`` the two sample sizes, a test with
    ``n_a + n_b > 16`` raises ``ValueError``. The statistic is
    ``diff = mean(a) - mean(b)`` and ``p`` is the exact two-sided
    permutation p-value: the pooled sample is enumerated over all
    ``C(n_a + n_b, n_a)`` position assignments and
    ``p = #{|diff'| >= |diff|} / C(n_a + n_b, n_a)``.

    With ``N`` the total number of ``(a, b, lag, B)`` tests, all tests are
    ranked ascending by ``(p, a, b, lag, B)`` and each rank ``j`` (1-based)
    gets the Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    test; ``reject`` is ``q <= alpha``, compared on the unquantized values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and exactly one trailing newline; the
    top-level key order is ``minutes, alpha, groups``, each group object
    uses the key order ``a, b, lags``, each lag object the key order
    ``lag, tests`` and each test object the key order
    ``key, n_a, n_b, diff, p, q, reject`` (with ``key`` the bucket start
    ``B``). Groups, lag objects and tests are in ascending ``(a, b)``, lag
    and bucket order respectively; pairs without any test are omitted.
    Labels render as JSON strings, lags, buckets and sample sizes as
    integers, ``reject`` as a boolean and ``alpha``, ``diff``, ``p`` and
    ``q`` with exactly six decimals, negative zero normalized to
    ``0.000000``. An empty ``details``, an empty ``lags`` or the absence of
    any comparison yields ``{"minutes":60,"alpha":0.050000,"groups":[]}``
    (plus the trailing newline).
    """
    minutes, lag_list, alpha_value, parsed, xmap = _lags_group_x_values(
        details, neighbors, lags, labels, minutes, alpha
    )

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # One record per emitted (a, b, lag, bucket) test:
        # ``[a, b, lag, B, n_a, n_b, diff, p, q]`` with q filled in below.
        grouped: dict[tuple[str, str], dict[int, list[list]]] = {}
        records: list[list] = []
        pair_labels = sorted(xmap)
        for index_a, label_a in enumerate(pair_labels):
            for label_b in pair_labels[index_a + 1:]:
                lag_groups: dict[int, list[list]] = {}
                for lag in lag_list:
                    lag_records: list[list] = []
                    buckets_a = xmap[label_a].get(lag, {})
                    buckets_b = xmap[label_b].get(lag, {})
                    for bucket in sorted(buckets_a):
                        sample_a = buckets_a[bucket]
                        sample_b = buckets_b.get(bucket)
                        if not sample_b:
                            continue
                        n_a = len(sample_a)
                        n_b = len(sample_b)
                        if n_a + n_b > _SPATIOTEMPORAL_MAX_N:
                            raise ValueError(
                                f"labels {label_a!r}/{label_b!r} lag {lag} "
                                f"bucket {bucket} have {n_a}+{n_b} pairings; "
                                f"lags group compare report requires at most "
                                f"{_SPATIOTEMPORAL_MAX_N} combined pairings "
                                f"per test"
                            )

                        sum_a = Decimal(0)
                        for value in sample_a:
                            sum_a += value
                        sum_b = Decimal(0)
                        for value in sample_b:
                            sum_b += value
                        diff = sum_a / n_a - sum_b / n_b

                        # |s_a'/n_a - s_b'/n_b| >= |diff| is equivalent
                        # (n_a, n_b > 0) to
                        # |s_a'*n_b - s_b'*n_a| >= |sum_a*n_b - sum_b*n_a|;
                        # compare the scaled sums so exact ties are decided
                        # without any division rounding.
                        threshold = abs(sum_a * n_b - sum_b * n_a)
                        pooled = sample_a + sample_b
                        pooled_sum = sum_a + sum_b
                        assignments = math.comb(n_a + n_b, n_a)
                        hits = 0
                        for combo in combinations(range(n_a + n_b), n_a):
                            perm_a = Decimal(0)
                            for position in combo:
                                perm_a += pooled[position]
                            perm_b = pooled_sum - perm_a
                            if abs(perm_a * n_b - perm_b * n_a) >= threshold:
                                hits += 1
                        p_value = Decimal(hits) / Decimal(assignments)

                        record = [
                            label_a, label_b, lag, bucket,
                            n_a, n_b, diff, p_value, None,
                        ]
                        lag_records.append(record)
                        records.append(record)
                    lag_groups[lag] = lag_records
                grouped[(label_a, label_b)] = lag_groups

        # Benjamini-Hochberg q-values across ALL (a, b, lag, bucket) tests:
        # rank ascending by (p, a, b, lag, B), then accumulate the running
        # minimum of N * p_l / l from the top rank down, mapping q back.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (
                records[idx][7],
                records[idx][0],
                records[idx][1],
                records[idx][2],
                records[idx][3],
            ),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Decimal(count) * records[idx][7] / rank
            if candidate < running:
                running = candidate
            records[idx][8] = running

        groups = []
        if parsed:
            for label_a, label_b in sorted(grouped):
                lag_items = []
                for lag in lag_list:
                    tests = []
                    for (
                        _,
                        _,
                        _,
                        bucket,
                        n_a,
                        n_b,
                        diff,
                        p_value,
                        q_value,
                    ) in grouped[(label_a, label_b)][lag]:
                        reject = q_value <= alpha_value
                        tests.append(
                            '{"key":' + str(bucket)
                            + ',"n_a":' + str(n_a)
                            + ',"n_b":' + str(n_b)
                            + ',"diff":' + _format6(diff)
                            + ',"p":' + _format6(p_value)
                            + ',"q":' + _format6(q_value)
                            + ',"reject":' + ("true" if reject else "false")
                            + '}'
                        )
                    if tests:
                        lag_items.append(
                            '{"lag":' + str(lag)
                            + ',"tests":[' + ",".join(tests) + ']}'
                        )
                if lag_items:
                    groups.append(
                        '{"a":' + json.dumps(label_a, ensure_ascii=False)
                        + ',"b":' + json.dumps(label_b, ensure_ascii=False)
                        + ',"lags":[' + ",".join(lag_items) + ']}'
                    )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(groups) + ']}'
            + "\n"
        )


def window_compare(
    details: list,
    neighbors: list,
    lags: list,
    labels: dict,
    windows: dict,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Compare same-label neighbor-delta samples between named time windows.

    ``details``, ``neighbors``, ``lags``, ``labels``, ``minutes`` and
    ``alpha`` follow exactly the same validation, bucketing and same-label
    edge pairing rules as
    :func:`effect_matrix_lags_group_compare_report`, so each
    ``(label, lag, B)`` triple yields the same list of ``x`` values.

    ``windows`` must be a dict whose keys are exactly all bucket starts
    ``B`` derived from ``details`` under ``minutes`` (independently of lag
    availability) and whose values are non-empty strings naming the window
    the bucket belongs to; a non-dict raises ``TypeError`` and any key or
    value mismatch raises ``ValueError``. Buckets assigned to the same
    window name form one window.

    For each window, label and lag, the ``x`` values are concatenated over
    the window's buckets in ascending ``B`` order, themselves taken in
    ascending same-label edge order. Two distinct windows with names
    ``a < b`` are compared only when both concatenated samples are
    non-empty; with ``n_a``/``n_b`` the two sample sizes, a comparison with
    ``n_a + n_b > 16`` raises ``ValueError``. The statistic is
    ``diff = mean(a) - mean(b)`` and ``p`` is the exact two-sided
    permutation p-value: the pooled sample is enumerated over all
    ``C(n_a + n_b, n_a)`` equal-size position assignments and
    ``p = #{|diff'| >= |diff|} / C(n_a + n_b, n_a)``.

    With ``N`` the total number of ``(window, a, b, lag)`` comparisons, all
    comparisons are ranked ascending by ``(p, window, a, b, lag)`` and each
    rank ``j`` (1-based) gets the Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    comparison; ``reject`` is ``q <= alpha``, compared on the unquantized
    values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and exactly one trailing newline; the
    top-level key order is ``minutes, alpha, groups``, each group object
    uses the key order ``key, comparisons`` (with ``key`` the window name),
    each comparison object the key order ``a, b, lags`` and each lag object
    the key order ``lag, n_a, n_b, diff, p, q, reject``. Groups,
    comparisons and lag objects are in ascending window name,
    ``(a, b)`` and lag order respectively; windows without any comparison
    are omitted and no comparisons at all yields ``groups`` empty.
    Window names and labels render as JSON strings, lags and sample sizes
    as integers, ``reject`` as a boolean and ``alpha``, ``diff``, ``p`` and
    ``q`` with exactly six decimals, negative zero normalized to
    ``0.000000``.
    """
    minutes, lag_list, alpha_value, parsed, xmap = _lags_group_x_values(
        details, neighbors, lags, labels, minutes, alpha
    )

    if not isinstance(windows, dict):
        raise TypeError("windows must be a dict")

    # The full bucket set is the set of bucket starts that the parsed
    # detail rows fall into under ``minutes``, whether or not any lag
    # pairing exists there.
    all_buckets: set[int] = set()
    if parsed:
        bucket_seconds = minutes * 60
        for validated in parsed:
            timestamp = validated[0]
            all_buckets.add((timestamp // bucket_seconds) * bucket_seconds)
    if set(windows) != all_buckets:
        raise ValueError(
            "windows keys must be exactly the details-derived bucket starts"
        )
    for bucket, name in windows.items():
        if not isinstance(name, str) or not name:
            raise ValueError("each window name must be a non-empty string")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # window -> label -> lag -> concatenated [x, ...] over the window's
        # buckets in ascending B order (each bucket's values already in
        # ascending edge order).
        window_samples: dict[str, dict[str, dict[int, list[Decimal]]]] = {}
        for bucket in sorted(all_buckets):
            name = windows[bucket]
            label_map = window_samples.setdefault(name, {})
            for label in sorted(xmap):
                lag_map = label_map.setdefault(label, {})
                for lag in lag_list:
                    values = xmap[label].get(lag, {}).get(bucket)
                    if values:
                        lag_map.setdefault(lag, []).extend(values)

        # One record per emitted (window, a, b, lag) comparison:
        # ``[window, a, b, lag, n_a, n_b, diff, p, q]`` with q filled in
        # below.
        grouped: dict[str, dict[tuple[str, str], dict[int, list]]] = {}
        records: list[list] = []
        for name in sorted(window_samples):
            pair_groups: dict[tuple[str, str], dict[int, list]] = {}
            window_labels = window_samples[name]
            pair_labels = sorted(window_labels)
            for index_a, label_a in enumerate(pair_labels):
                for label_b in pair_labels[index_a + 1:]:
                    lag_groups: dict[int, list] = {}
                    for lag in lag_list:
                        sample_a = window_labels[label_a].get(lag)
                        sample_b = window_labels[label_b].get(lag)
                        if not sample_a or not sample_b:
                            continue
                        n_a = len(sample_a)
                        n_b = len(sample_b)
                        if n_a + n_b > _SPATIOTEMPORAL_MAX_N:
                            raise ValueError(
                                f"window {name!r} labels {label_a!r}/"
                                f"{label_b!r} lag {lag} have {n_a}+{n_b} "
                                f"values; window compare requires at most "
                                f"{_SPATIOTEMPORAL_MAX_N} combined values per "
                                f"comparison"
                            )

                        sum_a = Decimal(0)
                        for value in sample_a:
                            sum_a += value
                        sum_b = Decimal(0)
                        for value in sample_b:
                            sum_b += value
                        diff = sum_a / n_a - sum_b / n_b

                        # |s_a'/n_a - s_b'/n_b| >= |diff| is equivalent
                        # (n_a, n_b > 0) to
                        # |s_a'*n_b - s_b'*n_a| >= |sum_a*n_b - sum_b*n_a|;
                        # compare the scaled sums so exact ties are decided
                        # without any division rounding.
                        threshold = abs(sum_a * n_b - sum_b * n_a)
                        pooled = sample_a + sample_b
                        pooled_sum = sum_a + sum_b
                        assignments = math.comb(n_a + n_b, n_a)
                        hits = 0
                        for combo in combinations(range(n_a + n_b), n_a):
                            perm_a = Decimal(0)
                            for position in combo:
                                perm_a += pooled[position]
                            perm_b = pooled_sum - perm_a
                            if abs(perm_a * n_b - perm_b * n_a) >= threshold:
                                hits += 1
                        p_value = Decimal(hits) / Decimal(assignments)

                        record = [
                            name, label_a, label_b, lag,
                            n_a, n_b, diff, p_value, None,
                        ]
                        lag_groups[lag] = record
                        records.append(record)
                    if lag_groups:
                        pair_groups[(label_a, label_b)] = lag_groups
            grouped[name] = pair_groups

        # Benjamini-Hochberg q-values across ALL (window, a, b, lag)
        # comparisons: rank ascending by (p, window, a, b, lag), then
        # accumulate the running minimum of N * p_l / l from the top rank
        # down, mapping q back.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (
                records[idx][7],
                records[idx][0],
                records[idx][1],
                records[idx][2],
                records[idx][3],
            ),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Decimal(count) * records[idx][7] / rank
            if candidate < running:
                running = candidate
            records[idx][8] = running

        groups = []
        if parsed:
            for name in sorted(grouped):
                comparison_items = []
                for label_a, label_b in sorted(grouped[name]):
                    lag_items = []
                    for lag in lag_list:
                        record = grouped[name][(label_a, label_b)].get(lag)
                        if record is None:
                            continue
                        (
                            _,
                            _,
                            _,
                            record_lag,
                            n_a,
                            n_b,
                            diff,
                            p_value,
                            q_value,
                        ) = record
                        reject = q_value <= alpha_value
                        lag_items.append(
                            '{"lag":' + str(record_lag)
                            + ',"n_a":' + str(n_a)
                            + ',"n_b":' + str(n_b)
                            + ',"diff":' + _format6(diff)
                            + ',"p":' + _format6(p_value)
                            + ',"q":' + _format6(q_value)
                            + ',"reject":' + ("true" if reject else "false")
                            + '}'
                        )
                    if lag_items:
                        comparison_items.append(
                            '{"a":' + json.dumps(label_a, ensure_ascii=False)
                            + ',"b":' + json.dumps(label_b, ensure_ascii=False)
                            + ',"lags":[' + ",".join(lag_items) + ']}'
                        )
                if comparison_items:
                    groups.append(
                        '{"key":' + json.dumps(name, ensure_ascii=False)
                        + ',"comparisons":['
                        + ",".join(comparison_items) + ']}'
                    )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(groups) + ']}'
            + "\n"
        )


def window_shift(
    details: list,
    neighbors: list,
    lags: list,
    labels: dict,
    windows: dict,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Compare same-label neighbor-delta samples between named time windows.

    ``details``, ``neighbors``, ``lags``, ``labels``, ``minutes`` and
    ``alpha`` follow exactly the same validation, bucketing and same-label
    edge pairing rules as :func:`window_compare`, so each
    ``(label, lag, B)`` triple yields the same list of ``x`` values
    (``x = Delta_a - Delta_b``).

    ``windows`` follows the same contract as in :func:`window_compare`: a
    dict whose keys are exactly all bucket starts ``B`` derived from
    ``details`` under ``minutes`` (independently of lag availability) and
    whose values are non-empty strings naming the window the bucket belongs
    to; a non-dict raises ``TypeError`` and any key or value mismatch raises
    ``ValueError``.

    For each label and lag, the ``x`` values are concatenated over each
    window's buckets in ascending ``B`` order, themselves taken in
    ascending same-label edge order. Two distinct windows with names
    ``a < b`` are compared only when both concatenated samples are
    non-empty; with ``n_a``/``n_b`` the two sample sizes, a comparison with
    ``n_a + n_b > 16`` raises ``ValueError``. The statistic is
    ``diff = mean(a) - mean(b)`` and ``p`` is the exact two-sided
    permutation p-value: the pooled sample is enumerated over all
    ``C(n_a + n_b, n_a)`` equal-size position assignments and
    ``p = #{|diff'| >= |diff|} / C(n_a + n_b, n_a)``.

    With ``N`` the total number of ``(label, a, b, lag)`` comparisons, all
    comparisons are ranked ascending by ``(p, label, a, b, lag)`` and each
    rank ``j`` (1-based) gets the Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    comparison; ``reject`` is ``q <= alpha``, compared on the unquantized
    values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and exactly one trailing newline; the
    top-level key order is ``minutes, alpha, groups``, each group object
    uses the key order ``key, comparisons`` (with ``key`` the label), each
    comparison object the key order ``a, b, lags`` and each lag object the
    key order ``lag, n_a, n_b, diff, p, q, reject``. Groups, comparisons
    and lag objects are in ascending label, ``(a, b)`` and lag order
    respectively; labels without any comparison are omitted and no
    comparisons at all yields ``groups`` empty. Window names and labels
    render as JSON strings, lags and sample sizes as integers, ``reject`` as
    a boolean and ``alpha``, ``diff``, ``p`` and ``q`` with exactly six
    decimals, negative zero normalized to ``0.000000``.
    """
    minutes, lag_list, alpha_value, parsed, xmap = _lags_group_x_values(
        details, neighbors, lags, labels, minutes, alpha
    )
    return _window_shift_report(
        parsed, xmap, windows, minutes, lag_list, alpha_value, "window shift"
    )


def _window_shift_report(
    parsed: list,
    xmap: dict,
    windows: dict,
    minutes: int,
    lag_list: list,
    alpha_value: Decimal,
    comparison_name: str,
) -> str:
    """Shared body of :func:`window_shift` and :func:`scenario_shift`.

    ``parsed`` holds the validated detail rows used to derive the full
    bucket set and ``xmap`` maps ``label -> lag -> bucket start`` to the
    per-bucket value lists (for :func:`scenario_shift` the after-minus-
    before ``e`` values). ``windows`` follows the :func:`window_compare`
    contract and ``comparison_name`` is the phrase used in the
    combined-size error message.
    """
    if not isinstance(windows, dict):
        raise TypeError("windows must be a dict")

    # The full bucket set is the set of bucket starts that the parsed
    # detail rows fall into under ``minutes``, whether or not any lag
    # pairing exists there.
    all_buckets: set[int] = set()
    if parsed:
        bucket_seconds = minutes * 60
        for validated in parsed:
            timestamp = validated[0]
            all_buckets.add((timestamp // bucket_seconds) * bucket_seconds)
    if set(windows) != all_buckets:
        raise ValueError(
            "windows keys must be exactly the details-derived bucket starts"
        )
    for bucket, name in windows.items():
        if not isinstance(name, str) or not name:
            raise ValueError("each window name must be a non-empty string")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # label -> window -> lag -> concatenated [x, ...] over the window's
        # buckets in ascending B order (each bucket's values already in
        # ascending edge order).
        label_window_samples: dict[
            str, dict[str, dict[int, list[Decimal]]]
        ] = {}
        for bucket in sorted(all_buckets):
            name = windows[bucket]
            for label in sorted(xmap):
                window_map = label_window_samples.setdefault(label, {})
                lag_map = window_map.setdefault(name, {})
                for lag in lag_list:
                    values = xmap[label].get(lag, {}).get(bucket)
                    if values:
                        lag_map.setdefault(lag, []).extend(values)

        # One record per emitted (label, a, b, lag) comparison:
        # ``[label, a, b, lag, n_a, n_b, diff, p, q]`` with q filled in
        # below.
        grouped: dict[str, dict[tuple[str, str], dict[int, list]]] = {}
        records: list[list] = []
        for label in sorted(label_window_samples):
            pair_groups: dict[tuple[str, str], dict[int, list]] = {}
            window_map = label_window_samples[label]
            window_names = sorted(window_map)
            for index_a, window_a in enumerate(window_names):
                for window_b in window_names[index_a + 1:]:
                    lag_groups: dict[int, list] = {}
                    for lag in lag_list:
                        sample_a = window_map[window_a].get(lag)
                        sample_b = window_map[window_b].get(lag)
                        if not sample_a or not sample_b:
                            continue
                        n_a = len(sample_a)
                        n_b = len(sample_b)
                        if n_a + n_b > _SPATIOTEMPORAL_MAX_N:
                            raise ValueError(
                                f"label {label!r} windows {window_a!r}/"
                                f"{window_b!r} lag {lag} have {n_a}+{n_b} "
                                f"values; {comparison_name} requires at most "
                                f"{_SPATIOTEMPORAL_MAX_N} combined values per "
                                f"comparison"
                            )

                        sum_a = Decimal(0)
                        for value in sample_a:
                            sum_a += value
                        sum_b = Decimal(0)
                        for value in sample_b:
                            sum_b += value
                        diff = sum_a / n_a - sum_b / n_b

                        # |s_a'/n_a - s_b'/n_b| >= |diff| is equivalent
                        # (n_a, n_b > 0) to
                        # |s_a'*n_b - s_b'*n_a| >= |sum_a*n_b - sum_b*n_a|;
                        # compare the scaled sums so exact ties are decided
                        # without any division rounding.
                        threshold = abs(sum_a * n_b - sum_b * n_a)
                        pooled = sample_a + sample_b
                        pooled_sum = sum_a + sum_b
                        assignments = math.comb(n_a + n_b, n_a)
                        hits = 0
                        for combo in combinations(range(n_a + n_b), n_a):
                            perm_a = Decimal(0)
                            for position in combo:
                                perm_a += pooled[position]
                            perm_b = pooled_sum - perm_a
                            if abs(perm_a * n_b - perm_b * n_a) >= threshold:
                                hits += 1
                        p_value = Decimal(hits) / Decimal(assignments)

                        record = [
                            label, window_a, window_b, lag,
                            n_a, n_b, diff, p_value, None,
                        ]
                        lag_groups[lag] = record
                        records.append(record)
                    if lag_groups:
                        pair_groups[(window_a, window_b)] = lag_groups
            if pair_groups:
                grouped[label] = pair_groups

        # Benjamini-Hochberg q-values across ALL (label, a, b, lag)
        # comparisons: rank ascending by (p, label, a, b, lag), then
        # accumulate the running minimum of N * p_l / l from the top rank
        # down, mapping q back.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (
                records[idx][7],
                records[idx][0],
                records[idx][1],
                records[idx][2],
                records[idx][3],
            ),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Decimal(count) * records[idx][7] / rank
            if candidate < running:
                running = candidate
            records[idx][8] = running

        groups = []
        if parsed:
            for label in sorted(grouped):
                comparison_items = []
                for window_a, window_b in sorted(grouped[label]):
                    lag_items = []
                    for lag in lag_list:
                        record = grouped[label][(window_a, window_b)].get(lag)
                        if record is None:
                            continue
                        (
                            _,
                            _,
                            _,
                            record_lag,
                            n_a,
                            n_b,
                            diff,
                            p_value,
                            q_value,
                        ) = record
                        reject = q_value <= alpha_value
                        lag_items.append(
                            '{"lag":' + str(record_lag)
                            + ',"n_a":' + str(n_a)
                            + ',"n_b":' + str(n_b)
                            + ',"diff":' + _format6(diff)
                            + ',"p":' + _format6(p_value)
                            + ',"q":' + _format6(q_value)
                            + ',"reject":' + ("true" if reject else "false")
                            + '}'
                        )
                    if lag_items:
                        comparison_items.append(
                            '{"a":' + json.dumps(window_a, ensure_ascii=False)
                            + ',"b":' + json.dumps(window_b, ensure_ascii=False)
                            + ',"lags":[' + ",".join(lag_items) + ']}'
                        )
                if comparison_items:
                    groups.append(
                        '{"key":' + json.dumps(label, ensure_ascii=False)
                        + ',"comparisons":['
                        + ",".join(comparison_items) + ']}'
                    )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(groups) + ']}'
            + "\n"
        )


def scenario_shift(
    before: list,
    after: list,
    neighbors: list,
    lags: list,
    labels: dict,
    windows: dict,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Compare after-minus-before neighbor-delta shifts between windows.

    ``before`` and ``after`` are two scenario-detail tables, each following
    the ``details`` contract of :func:`window_shift`, and both must share
    exactly the same set of ``(timestamp, cell_id)`` pairs; a non-list
    table raises ``TypeError`` and a key-set mismatch raises ``ValueError``.
    ``neighbors``, ``lags``, ``labels``, ``windows``, ``minutes`` and
    ``alpha`` follow exactly the same validation, bucketing, same-label
    edge pairing, edge order and exception rules as :func:`window_shift`.

    For each table the same ``x = Delta_a - Delta_b`` values as in
    :func:`window_shift` are computed per ``(label, lag, B)`` triple and
    paired element-wise (the identical key sets guarantee identical
    pairings); the shift sample is ``e = x_after - x_before``. For each
    label and lag, the ``e`` values are concatenated over each window's
    buckets in ascending ``B`` order, themselves taken in ascending
    same-label edge order. Two distinct windows with names ``a < b`` are
    compared only when both concatenated samples are non-empty; with
    ``n_a``/``n_b`` the two sample sizes, a comparison with
    ``n_a + n_b > 16`` raises ``ValueError``. The statistic is
    ``diff = mean(a) - mean(b)`` and ``p`` is the exact two-sided
    permutation p-value: the pooled sample is enumerated over all
    ``C(n_a + n_b, n_a)`` equal-size position assignments and
    ``p = #{|diff'| >= |diff|} / C(n_a + n_b, n_a)``.

    With ``N`` the total number of ``(label, a, b, lag)`` comparisons, all
    comparisons are ranked ascending by ``(p, label, a, b, lag)`` and each
    rank ``j`` (1-based) gets the Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    comparison; ``reject`` is ``q <= alpha``, compared on the unquantized
    values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and exactly one trailing newline; the
    top-level key order is ``minutes, alpha, groups``, each group object
    uses the key order ``key, comparisons`` (with ``key`` the label), each
    comparison object the key order ``a, b, lags`` and each lag object the
    key order ``lag, n_a, n_b, diff, p, q, reject``. Groups, comparisons
    and lag objects are in ascending label, ``(a, b)`` and lag order
    respectively; labels without any comparison are omitted and no
    comparisons at all yields ``groups`` empty. Window names and labels
    render as JSON strings, lags and sample sizes as integers, ``reject``
    as a boolean and ``alpha``, ``diff``, ``p`` and ``q`` with exactly six
    decimals, negative zero normalized to ``0.000000``.
    """
    if not isinstance(before, list):
        raise TypeError("before must be a list")
    if not isinstance(after, list):
        raise TypeError("after must be a list")
    minutes, lag_list, alpha_value, parsed_before, xmap_before = (
        _lags_group_x_values(before, neighbors, lags, labels, minutes, alpha)
    )
    _, _, _, parsed_after, xmap_after = _lags_group_x_values(
        after, neighbors, lags, labels, minutes, alpha
    )
    keys_before = {(row[0], row[1]) for row in parsed_before}
    keys_after = {(row[0], row[1]) for row in parsed_after}
    if keys_before != keys_after:
        raise ValueError(
            "before and after must share the same (timestamp, cell_id) pairs"
        )

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # Identical (timestamp, cell_id) key sets make the two xmaps
        # structurally identical, so the per-(label, lag, B) value lists
        # pair up element-wise in the same ascending edge order.
        emap: dict[str, dict[int, dict[int, list[Decimal]]]] = {}
        for label in sorted(xmap_before):
            lag_map: dict[int, dict[int, list[Decimal]]] = {}
            for lag in lag_list:
                before_buckets = xmap_before[label].get(lag, {})
                after_buckets = xmap_after[label].get(lag, {})
                bucket_map: dict[int, list[Decimal]] = {}
                for bucket in sorted(before_buckets):
                    before_values = before_buckets[bucket]
                    after_values = after_buckets[bucket]
                    bucket_map[bucket] = [
                        after_value - before_value
                        for after_value, before_value in zip(
                            after_values, before_values
                        )
                    ]
                lag_map[lag] = bucket_map
            emap[label] = lag_map

    return _window_shift_report(
        parsed_before, emap, windows, minutes, lag_list, alpha_value,
        "scenario shift",
    )


def scenario_rank(
    base: list,
    sets: dict,
    edges: list,
    lags: list,
    labels: dict,
    windows: dict,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Rank scenario sets by mean after-minus-base shift and compare pairs.

    ``base`` and every table in ``sets`` follow the ``details`` contract of
    :func:`scenario_shift`, and all tables must share exactly the same set of
    ``(timestamp, cell_id)`` pairs; a key-set mismatch raises ``ValueError``.
    ``edges``, ``lags``, ``labels``, ``windows``, ``minutes`` and ``alpha``
    follow exactly the same validation, bucketing, same-label edge pairing,
    edge order and exception rules as in :func:`scenario_shift`.

    ``sets`` must be a dict with at least two non-empty string keys, each
    mapping to a list table; passing a non-dict raises ``TypeError`` and any
    key or value mismatch raises ``ValueError``.

    For each set ``key`` the same ``x`` values as in :func:`window_shift` are
    computed per ``(label, lag, B)`` triple for both tables and paired
    element-wise (the identical key sets guarantee identical pairings),
    yielding ``e = x_set - x_base``. Within each ``(label, window, lag)``
    group the ``e`` values are concatenated over the window's buckets in
    ascending ``B`` order, themselves taken in ascending same-label edge
    order. Sets whose concatenated sample is non-empty are ranked ascending
    by ``(mean, key)`` and receive ``rank = 1..k``; sets with an empty sample
    are omitted from the ranks.

    Two distinct ranked sets with keys ``a < b`` are compared only when both
    samples are non-empty; with ``n_a``/``n_b`` the two sample sizes, a
    comparison with ``n_a + n_b > 16`` raises ``ValueError``. The statistic
    is ``diff = mean(a) - mean(b)`` and ``p`` is the exact two-sided
    permutation p-value: the pooled sample is enumerated over all
    ``C(n_a + n_b, n_a)`` equal-size position assignments and
    ``p = #{|diff'| >= |diff|} / C(n_a + n_b, n_a)``.

    With ``N`` the total number of ``(label, window, lag, a, b)``
    comparisons, all comparisons are ranked ascending by
    ``(p, label, window, lag, a, b)`` and each rank ``j`` (1-based) gets the
    Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``, mapped back to its
    comparison; ``reject`` is ``q <= alpha``, compared on the unquantized
    values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and exactly one trailing newline; the
    top-level key order is ``minutes, alpha, groups``, each group object
    uses the key order ``label, window, lag, ranks, tests``, each rank
    object the key order ``key, mean, rank`` and each test object the key
    order ``a, b, diff, p, q, reject``. Groups, ranks and tests are in
    ascending ``(label, window, lag)``, ``(rank, key)`` and ``(a, b)`` order
    respectively; ``(label, window, lag)`` triples without any ranked set
    are omitted and no groups at all yields ``groups`` empty. Set keys,
    labels and window names render as JSON strings, lags and ranks as
    integers, ``reject`` as a boolean and ``alpha``, ``mean``, ``diff``,
    ``p`` and ``q`` with exactly six decimals, negative zero normalized to
    ``0.000000``.
    """
    if not isinstance(sets, dict):
        raise TypeError("sets must be a dict")
    if len(sets) < 2:
        raise ValueError("sets must contain at least two entries")
    for key, table in sets.items():
        if not isinstance(key, str) or not key:
            raise ValueError("each sets key must be a non-empty string")
        if not isinstance(table, list):
            raise ValueError("each sets value must be a list")

    minutes, lag_list, alpha_value, parsed_base, xmap_base = (
        _lags_group_x_values(base, edges, lags, labels, minutes, alpha)
    )

    base_keys = {(row[0], row[1]) for row in parsed_base}

    # key -> label -> lag -> bucket start -> [e, ...] (after-minus-base).
    emaps: dict[str, dict] = {}
    for key in sorted(sets):
        _, _, _, parsed_set, xmap_set = _lags_group_x_values(
            sets[key], edges, lags, labels, minutes, alpha
        )
        set_keys = {(row[0], row[1]) for row in parsed_set}
        if set_keys != base_keys:
            raise ValueError(
                f"sets table {key!r} must share the same (timestamp, cell_id) "
                "pairs as base"
            )

        with localcontext() as ctx:
            ctx.prec = _MODEL_PRECISION
            ctx.rounding = ROUND_HALF_EVEN

            # Identical (timestamp, cell_id) key sets make the two xmaps
            # structurally identical, so the per-(label, lag, B) value lists
            # pair up element-wise in the same ascending edge order.
            emap: dict[str, dict[int, dict[int, list[Decimal]]]] = {}
            for label in sorted(xmap_base):
                lag_map: dict[int, dict[int, list[Decimal]]] = {}
                for lag in lag_list:
                    base_buckets = xmap_base[label].get(lag, {})
                    set_buckets = xmap_set[label].get(lag, {})
                    bucket_map: dict[int, list[Decimal]] = {}
                    for bucket in sorted(base_buckets):
                        base_values = base_buckets[bucket]
                        set_values = set_buckets[bucket]
                        bucket_map[bucket] = [
                            set_value - base_value
                            for set_value, base_value in zip(
                                set_values, base_values
                            )
                        ]
                    lag_map[lag] = bucket_map
                emap[label] = lag_map
        emaps[key] = emap

    # Resolve the windows contract against the base table, exactly as
    # _window_shift_report does for scenario_shift.
    if not isinstance(windows, dict):
        raise TypeError("windows must be a dict")
    all_buckets: set[int] = set()
    if parsed_base:
        bucket_seconds = minutes * 60
        for validated in parsed_base:
            timestamp = validated[0]
            all_buckets.add((timestamp // bucket_seconds) * bucket_seconds)
    if set(windows) != all_buckets:
        raise ValueError(
            "windows keys must be exactly the details-derived bucket starts"
        )
    for bucket, name in windows.items():
        if not isinstance(name, str) or not name:
            raise ValueError("each window name must be a non-empty string")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # (label, window, lag) -> set key -> concatenated [e, ...] over the
        # window's buckets in ascending B order (each bucket's values already
        # in ascending edge order).
        triple_samples: dict[
            tuple[str, str, int], dict[str, list[Decimal]]
        ] = {}
        for bucket in sorted(all_buckets):
            name = windows[bucket]
            for key in sorted(emaps):
                emap = emaps[key]
                for label in sorted(emap):
                    for lag in lag_list:
                        values = emap[label].get(lag, {}).get(bucket)
                        if values:
                            triple_samples.setdefault(
                                (label, name, lag), {}
                            ).setdefault(key, []).extend(values)

        # One record per emitted (label, window, lag, a, b) comparison:
        # ``[label, window, lag, a, b, diff, p, q]`` with q filled in below.
        # ranks maps each triple to ``[key, mean, rank]`` rows.
        grouped: dict[
            tuple[str, str, int],
            tuple[list[list], dict[tuple[str, str], list]],
        ] = {}
        records: list[list] = []
        for triple in sorted(triple_samples):
            samples = triple_samples[triple]
            means: dict[str, Decimal] = {}
            sums: dict[str, Decimal] = {}
            for key, sample in samples.items():
                total = Decimal(0)
                for value in sample:
                    total += value
                sums[key] = total
                means[key] = total / len(sample)

            ranked_keys = sorted(means, key=lambda key: (means[key], key))
            rank_rows: list[list] = []
            for index, key in enumerate(ranked_keys):
                rank_rows.append([key, means[key], index + 1])

            # Comparisons pair distinct ranked sets in ascending KEY order
            # (a < b), independently of the mean-based ranking.
            pair_groups: dict[tuple[str, str], list] = {}
            present_keys = sorted(samples)
            for index_a, key_a in enumerate(present_keys):
                for key_b in present_keys[index_a + 1:]:
                    sample_a = samples[key_a]
                    sample_b = samples[key_b]
                    n_a = len(sample_a)
                    n_b = len(sample_b)
                    if n_a + n_b > _SPATIOTEMPORAL_MAX_N:
                        label, name, lag = triple
                        raise ValueError(
                            f"label {label!r} window {name!r} lag {lag} sets "
                            f"{key_a!r}/{key_b!r} have {n_a}+{n_b} values; "
                            f"scenario rank requires at most "
                            f"{_SPATIOTEMPORAL_MAX_N} combined values per "
                            f"comparison"
                        )

                    sum_a = sums[key_a]
                    sum_b = sums[key_b]
                    diff = means[key_a] - means[key_b]

                    # |s_a'/n_a - s_b'/n_b| >= |diff| is equivalent
                    # (n_a, n_b > 0) to
                    # |s_a'*n_b - s_b'*n_a| >= |sum_a*n_b - sum_b*n_a|;
                    # compare the scaled sums so exact ties are decided
                    # without any division rounding.
                    threshold = abs(sum_a * n_b - sum_b * n_a)
                    pooled = sample_a + sample_b
                    pooled_sum = sum_a + sum_b
                    assignments = math.comb(n_a + n_b, n_a)
                    hits = 0
                    for combo in combinations(range(n_a + n_b), n_a):
                        perm_a = Decimal(0)
                        for position in combo:
                            perm_a += pooled[position]
                        perm_b = pooled_sum - perm_a
                        if abs(perm_a * n_b - perm_b * n_a) >= threshold:
                            hits += 1
                    p_value = Decimal(hits) / Decimal(assignments)

                    record = [
                        triple[0], triple[1], triple[2],
                        key_a, key_b, diff, p_value, None,
                    ]
                    pair_groups[(key_a, key_b)] = record
                    records.append(record)

            if rank_rows:
                grouped[triple] = (rank_rows, pair_groups)

        # Benjamini-Hochberg q-values across ALL (label, window, lag, a, b)
        # comparisons: rank ascending by (p, label, window, lag, a, b), then
        # accumulate the running minimum of N * p_l / l from the top rank
        # down, mapping q back.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (
                records[idx][6],
                records[idx][0],
                records[idx][1],
                records[idx][2],
                records[idx][3],
                records[idx][4],
            ),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Decimal(count) * records[idx][6] / rank
            if candidate < running:
                running = candidate
            records[idx][7] = running

        groups = []
        for label, name, lag in sorted(grouped):
            rank_rows, pair_groups = grouped[(label, name, lag)]
            rank_items = []
            for key, mean, rank in rank_rows:
                rank_items.append(
                    '{"key":' + json.dumps(key, ensure_ascii=False)
                    + ',"mean":' + _format6(mean)
                    + ',"rank":' + str(rank)
                    + '}'
                )
            test_items = []
            for key_a, key_b in sorted(pair_groups):
                record = pair_groups[(key_a, key_b)]
                diff = record[5]
                p_value = record[6]
                q_value = record[7]
                reject = q_value <= alpha_value
                test_items.append(
                    '{"a":' + json.dumps(key_a, ensure_ascii=False)
                    + ',"b":' + json.dumps(key_b, ensure_ascii=False)
                    + ',"diff":' + _format6(diff)
                    + ',"p":' + _format6(p_value)
                    + ',"q":' + _format6(q_value)
                    + ',"reject":' + ("true" if reject else "false")
                    + '}'
                )
            groups.append(
                '{"label":' + json.dumps(label, ensure_ascii=False)
                + ',"window":' + json.dumps(name, ensure_ascii=False)
                + ',"lag":' + str(lag)
                + ',"ranks":[' + ",".join(rank_items) + ']'
                + ',"tests":[' + ",".join(test_items) + ']}'
            )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"groups":[' + ",".join(groups) + ']}'
            + "\n"
        )


def scenario_score(
    base: list,
    sets: dict,
    edges: list,
    lags: list,
    labels: dict,
    windows: dict,
    weights: dict,
    *,
    minutes: int = 60,
) -> str:
    """Score scenario sets with per-group weights and tally pairwise wins.

    ``base``, every table in ``sets``, ``edges``, ``lags``, ``labels``,
    ``windows`` and ``minutes`` follow exactly the same validation,
    bucketing, same-label edge pairing, edge order, exception rules and
    ``(timestamp, cell_id)`` key-set equality as :func:`scenario_rank`; as
    there, each set ``key`` gets an unquantized ``mean`` of its
    after-minus-base ``e`` sample within every non-empty
    ``(label, window, lag)`` group (the sample concatenates the window's
    buckets in ascending ``B`` order, each bucket's values in ascending
    same-label edge order).

    ``weights`` must be a dict; passing a non-dict raises ``TypeError``.
    Its keys must be exactly the ``(label, window, lag)`` triples that have
    a non-empty sample and its values positive, finite, non-boolean
    int/float numbers; a key-set mismatch, a wrongly shaped key or an
    invalid value raises ``ValueError``. When no group has a non-empty
    sample the only valid input is ``weights={}``. Within each group, any
    pair of sets whose combined sample sizes exceed 16 raises
    ``ValueError``.

    With ``W`` the sum of the weights, each set's score is
    ``score_s = sum_g (w_g * mean_s,g) / W`` over the groups it appears in,
    computed as ``Decimal(str(...))`` under a precision-1000,
    ROUND_HALF_EVEN local context. Sets are ranked ascending by
    ``(score, key)`` and receive ``rank = 1..k`` (1-based). Each pair of
    distinct sets ``a < b`` is compared group by group on the unquantized
    means: the set with the smaller mean wins the group's weight and an
    exact tie splits it evenly. ``a_win``, ``b_win`` and ``tie`` are the
    corresponding accumulated weights divided by ``W``. With no groups,
    ``W`` is 0 and both ``ranks`` and ``pairs`` are empty.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is
    ``minutes, total_weight, ranks, pairs``, each rank object uses the key
    order ``key, score, rank`` and each pair object the key order
    ``a, b, a_win, b_win, tie``. Ranks are in ascending ``(rank, key)`` and
    pairs in ascending ``(a, b)`` order. Set keys render as JSON strings,
    ``minutes`` and ``rank`` as integers and ``total_weight``, ``score``,
    ``a_win``, ``b_win`` and ``tie`` with exactly six decimals, negative
    zero normalized to ``0.000000``; the win shares are compared and
    accumulated on the unquantized values.
    """
    if not isinstance(sets, dict):
        raise TypeError("sets must be a dict")
    if len(sets) < 2:
        raise ValueError("sets must contain at least two entries")
    for key, table in sets.items():
        if not isinstance(key, str) or not key:
            raise ValueError("each sets key must be a non-empty string")
        if not isinstance(table, list):
            raise ValueError("each sets value must be a list")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")

    minutes, lag_list, _, parsed_base, xmap_base = _lags_group_x_values(
        base, edges, lags, labels, minutes, 1.0
    )

    base_keys = {(row[0], row[1]) for row in parsed_base}

    # key -> label -> lag -> bucket start -> [e, ...] (after-minus-base).
    emaps: dict[str, dict] = {}
    for key in sorted(sets):
        _, _, _, parsed_set, xmap_set = _lags_group_x_values(
            sets[key], edges, lags, labels, minutes, 1.0
        )
        set_keys = {(row[0], row[1]) for row in parsed_set}
        if set_keys != base_keys:
            raise ValueError(
                f"sets table {key!r} must share the same (timestamp, cell_id) "
                "pairs as base"
            )

        with localcontext() as ctx:
            ctx.prec = _MODEL_PRECISION
            ctx.rounding = ROUND_HALF_EVEN

            # Identical (timestamp, cell_id) key sets make the two xmaps
            # structurally identical, so the per-(label, lag, B) value lists
            # pair up element-wise in the same ascending edge order.
            emap: dict[str, dict[int, dict[int, list[Decimal]]]] = {}
            for label in sorted(xmap_base):
                lag_map: dict[int, dict[int, list[Decimal]]] = {}
                for lag in lag_list:
                    base_buckets = xmap_base[label].get(lag, {})
                    set_buckets = xmap_set[label].get(lag, {})
                    bucket_map: dict[int, list[Decimal]] = {}
                    for bucket in sorted(base_buckets):
                        base_values = base_buckets[bucket]
                        set_values = set_buckets[bucket]
                        bucket_map[bucket] = [
                            set_value - base_value
                            for set_value, base_value in zip(
                                set_values, base_values
                            )
                        ]
                    lag_map[lag] = bucket_map
                emap[label] = lag_map
        emaps[key] = emap

    # Resolve the windows contract against the base table, exactly as
    # scenario_rank does.
    if not isinstance(windows, dict):
        raise TypeError("windows must be a dict")
    all_buckets: set[int] = set()
    if parsed_base:
        bucket_seconds = minutes * 60
        for validated in parsed_base:
            timestamp = validated[0]
            all_buckets.add((timestamp // bucket_seconds) * bucket_seconds)
    if set(windows) != all_buckets:
        raise ValueError(
            "windows keys must be exactly the details-derived bucket starts"
        )
    for bucket, name in windows.items():
        if not isinstance(name, str) or not name:
            raise ValueError("each window name must be a non-empty string")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # (label, window, lag) -> set key -> concatenated [e, ...] over the
        # window's buckets in ascending B order (each bucket's values already
        # in ascending edge order).
        triple_samples: dict[
            tuple[str, str, int], dict[str, list[Decimal]]
        ] = {}
        for bucket in sorted(all_buckets):
            name = windows[bucket]
            for key in sorted(emaps):
                emap = emaps[key]
                for label in sorted(emap):
                    for lag in lag_list:
                        values = emap[label].get(lag, {}).get(bucket)
                        if values:
                            triple_samples.setdefault(
                                (label, name, lag), {}
                            ).setdefault(key, []).extend(values)

        # Only triples with a non-empty sample carry a weight.
        groups = sorted(triple_samples)
        if set(weights) != set(groups):
            raise ValueError(
                "weights keys must be exactly the non-empty "
                "(label, window, lag) triples"
            )
        weight_values: dict[tuple[str, str, int], Decimal] = {}
        for triple, weight in weights.items():
            if (
                not isinstance(triple, tuple)
                or len(triple) != 3
                or not isinstance(triple[0], str)
                or not isinstance(triple[1], str)
                or isinstance(triple[2], bool)
                or not isinstance(triple[2], int)
            ):
                raise ValueError(
                    "each weights key must be a (label, window, lag) triple "
                    "with two strings and an integer lag"
                )
            if isinstance(weight, bool) or not isinstance(
                weight, (int, float)
            ):
                raise ValueError("each weight must be a finite int or float")
            if isinstance(weight, float) and not math.isfinite(weight):
                raise ValueError("each weight must be finite")
            decimal_weight = Decimal(str(weight))
            if decimal_weight <= 0:
                raise ValueError("each weight must be positive")
            weight_values[triple] = decimal_weight

        total_weight = Decimal(0)
        for triple in groups:
            total_weight += weight_values[triple]

        # Per-set unquantized means for every group the set has a sample in.
        group_means: dict[
            tuple[str, str, int], dict[str, Decimal]
        ] = {}
        for triple in groups:
            samples = triple_samples[triple]
            present_keys = sorted(samples)
            for index_a, key_a in enumerate(present_keys):
                for key_b in present_keys[index_a + 1:]:
                    n_a = len(samples[key_a])
                    n_b = len(samples[key_b])
                    if n_a + n_b > _SPATIOTEMPORAL_MAX_N:
                        label, name, lag = triple
                        raise ValueError(
                            f"label {label!r} window {name!r} lag {lag} sets "
                            f"{key_a!r}/{key_b!r} have {n_a}+{n_b} values; "
                            f"scenario score requires at most "
                            f"{_SPATIOTEMPORAL_MAX_N} combined values per "
                            f"comparison"
                        )
            means: dict[str, Decimal] = {}
            for key, sample in samples.items():
                total = Decimal(0)
                for value in sample:
                    total += value
                means[key] = total / len(sample)
            group_means[triple] = means

        rank_items: list[str] = []
        pair_items: list[str] = []
        if total_weight > 0:
            weighted: dict[str, Decimal] = {}
            for triple in groups:
                weight = weight_values[triple]
                for key, mean in group_means[triple].items():
                    weighted[key] = weighted.get(key, Decimal(0)) + (
                        weight * mean
                    )
            scores = {
                key: value / total_weight for key, value in weighted.items()
            }
            ranked_keys = sorted(scores, key=lambda key: (scores[key], key))
            for index, key in enumerate(ranked_keys):
                rank_items.append(
                    '{"key":' + json.dumps(key, ensure_ascii=False)
                    + ',"score":' + _format6(scores[key])
                    + ',"rank":' + str(index + 1)
                    + '}'
                )

            present_keys = sorted(weighted)
            for index_a, key_a in enumerate(present_keys):
                for key_b in present_keys[index_a + 1:]:
                    wins_a = Decimal(0)
                    wins_b = Decimal(0)
                    ties = Decimal(0)
                    for triple in groups:
                        means = group_means[triple]
                        mean_a = means.get(key_a)
                        mean_b = means.get(key_b)
                        if mean_a is None or mean_b is None:
                            continue
                        weight = weight_values[triple]
                        if mean_a < mean_b:
                            wins_a += weight
                        elif mean_b < mean_a:
                            wins_b += weight
                        else:
                            ties += weight
                    pair_items.append(
                        '{"a":' + json.dumps(key_a, ensure_ascii=False)
                        + ',"b":' + json.dumps(key_b, ensure_ascii=False)
                        + ',"a_win":' + _format6(wins_a / total_weight)
                        + ',"b_win":' + _format6(wins_b / total_weight)
                        + ',"tie":' + _format6(ties / total_weight)
                        + '}'
                    )

        return (
            '{"minutes":' + str(minutes)
            + ',"total_weight":' + _format6(total_weight)
            + ',"ranks":[' + ",".join(rank_items) + ']'
            + ',"pairs":[' + ",".join(pair_items) + ']}'
            + "\n"
        )


_SCORE_TEST_MAX_GROUPS = 16


def score_test(
    base: list,
    sets: dict,
    edges: list,
    lags: list,
    labels: dict,
    windows: dict,
    weights: dict,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Test weighted scenario score differences with sign permutations.

    ``base``, every table in ``sets``, ``edges``, ``lags``, ``labels``,
    ``windows``, ``weights`` and ``minutes`` follow exactly the same
    validation, bucketing, same-label edge pairing, edge order, exception
    rules, ``(timestamp, cell_id)`` key-set equality, group means, weights
    contract and score computation as :func:`scenario_score` (including the
    combined-sample-size limit of 16 per in-group pair of sets). ``alpha``
    must be a finite non-boolean int/float in ``(0, 1]``; any other value
    raises ``ValueError``.

    With ``G`` the number of groups, ``G > 16`` raises ``ValueError``. Each
    set's leave-one-group-out stability is checked by deleting every group
    in turn and re-ranking the remaining weighted scores ascending by
    ``(score, key)``; ``stable`` is ``true`` when the set's rank never
    changes, and always ``true`` when ``G <= 1``.

    Each pair of distinct sets ``a < b`` takes the per-group difference
    ``d_g = mean_a,g - mean_b,g`` and ``diff = sum(w_g * d_g) / sum(w_g)``.
    The exact two-sided sign-permutation p-value enumerates all ``2 ** G``
    sign assignments ``s`` and ``p`` is the proportion with
    ``|sum(s_g * w_g * d_g) / sum(w_g)| >= |diff|``, compared on the
    unquantized values. With ``N`` the number of pairs, all pairs are ranked
    ascending by ``(p, a, b)`` and each rank ``j`` (1-based) gets the
    Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``; ``reject`` is
    ``q <= alpha``, compared on the unquantized values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and exactly one trailing newline; the
    top-level key order is ``minutes, alpha, total_weight, ranks, pairs``,
    each rank object uses the key order ``key, score, rank, stable`` and
    each pair object the key order ``a, b, diff, p, q, reject``. Ranks are
    in ascending ``(rank, key)`` and pairs in ascending ``(a, b)`` order.
    Set keys render as JSON strings, ``minutes`` and ``rank`` as integers,
    ``stable`` and ``reject`` as booleans and ``alpha``, ``total_weight``,
    ``score``, ``diff``, ``p`` and ``q`` with exactly six decimals, negative
    zero normalized to ``0.000000``. With no groups ``total_weight`` is
    ``0.000000``, both arrays are empty and the only valid input is
    ``weights={}``.
    """
    if not isinstance(sets, dict):
        raise TypeError("sets must be a dict")
    if len(sets) < 2:
        raise ValueError("sets must contain at least two entries")
    for key, table in sets.items():
        if not isinstance(key, str) or not key:
            raise ValueError("each sets key must be a non-empty string")
        if not isinstance(table, list):
            raise ValueError("each sets value must be a list")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")

    minutes, lag_list, alpha_value, parsed_base, xmap_base = (
        _lags_group_x_values(base, edges, lags, labels, minutes, alpha)
    )

    base_keys = {(row[0], row[1]) for row in parsed_base}

    # key -> label -> lag -> bucket start -> [e, ...] (after-minus-base).
    emaps: dict[str, dict] = {}
    for key in sorted(sets):
        _, _, _, parsed_set, xmap_set = _lags_group_x_values(
            sets[key], edges, lags, labels, minutes, alpha
        )
        set_keys = {(row[0], row[1]) for row in parsed_set}
        if set_keys != base_keys:
            raise ValueError(
                f"sets table {key!r} must share the same (timestamp, cell_id) "
                "pairs as base"
            )

        with localcontext() as ctx:
            ctx.prec = _MODEL_PRECISION
            ctx.rounding = ROUND_HALF_EVEN

            # Identical (timestamp, cell_id) key sets make the two xmaps
            # structurally identical, so the per-(label, lag, B) value lists
            # pair up element-wise in the same ascending edge order.
            emap: dict[str, dict[int, dict[int, list[Decimal]]]] = {}
            for label in sorted(xmap_base):
                lag_map: dict[int, dict[int, list[Decimal]]] = {}
                for lag in lag_list:
                    base_buckets = xmap_base[label].get(lag, {})
                    set_buckets = xmap_set[label].get(lag, {})
                    bucket_map: dict[int, list[Decimal]] = {}
                    for bucket in sorted(base_buckets):
                        base_values = base_buckets[bucket]
                        set_values = set_buckets[bucket]
                        bucket_map[bucket] = [
                            set_value - base_value
                            for set_value, base_value in zip(
                                set_values, base_values
                            )
                        ]
                    lag_map[lag] = bucket_map
                emap[label] = lag_map
        emaps[key] = emap

    # Resolve the windows contract against the base table, exactly as
    # scenario_score does.
    if not isinstance(windows, dict):
        raise TypeError("windows must be a dict")
    all_buckets: set[int] = set()
    if parsed_base:
        bucket_seconds = minutes * 60
        for validated in parsed_base:
            timestamp = validated[0]
            all_buckets.add((timestamp // bucket_seconds) * bucket_seconds)
    if set(windows) != all_buckets:
        raise ValueError(
            "windows keys must be exactly the details-derived bucket starts"
        )
    for bucket, name in windows.items():
        if not isinstance(name, str) or not name:
            raise ValueError("each window name must be a non-empty string")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # (label, window, lag) -> set key -> concatenated [e, ...] over the
        # window's buckets in ascending B order (each bucket's values already
        # in ascending edge order).
        triple_samples: dict[
            tuple[str, str, int], dict[str, list[Decimal]]
        ] = {}
        for bucket in sorted(all_buckets):
            name = windows[bucket]
            for key in sorted(emaps):
                emap = emaps[key]
                for label in sorted(emap):
                    for lag in lag_list:
                        values = emap[label].get(lag, {}).get(bucket)
                        if values:
                            triple_samples.setdefault(
                                (label, name, lag), {}
                            ).setdefault(key, []).extend(values)

        # Only triples with a non-empty sample carry a weight.
        groups = sorted(triple_samples)
        if set(weights) != set(groups):
            raise ValueError(
                "weights keys must be exactly the non-empty "
                "(label, window, lag) triples"
            )
        weight_values: dict[tuple[str, str, int], Decimal] = {}
        for triple, weight in weights.items():
            if (
                not isinstance(triple, tuple)
                or len(triple) != 3
                or not isinstance(triple[0], str)
                or not isinstance(triple[1], str)
                or isinstance(triple[2], bool)
                or not isinstance(triple[2], int)
            ):
                raise ValueError(
                    "each weights key must be a (label, window, lag) triple "
                    "with two strings and an integer lag"
                )
            if isinstance(weight, bool) or not isinstance(
                weight, (int, float)
            ):
                raise ValueError("each weight must be a finite int or float")
            if isinstance(weight, float) and not math.isfinite(weight):
                raise ValueError("each weight must be finite")
            decimal_weight = Decimal(str(weight))
            if decimal_weight <= 0:
                raise ValueError("each weight must be positive")
            weight_values[triple] = decimal_weight

        group_count = len(groups)
        if group_count > _SCORE_TEST_MAX_GROUPS:
            raise ValueError(
                f"there are {group_count} groups; score test requires at "
                f"most {_SCORE_TEST_MAX_GROUPS} groups"
            )

        total_weight = Decimal(0)
        for triple in groups:
            total_weight += weight_values[triple]

        # Per-set unquantized means for every group the set has a sample in.
        group_means: dict[
            tuple[str, str, int], dict[str, Decimal]
        ] = {}
        for triple in groups:
            samples = triple_samples[triple]
            present = sorted(samples)
            for index_a, key_a in enumerate(present):
                for key_b in present[index_a + 1:]:
                    n_a = len(samples[key_a])
                    n_b = len(samples[key_b])
                    if n_a + n_b > _SPATIOTEMPORAL_MAX_N:
                        label, name, lag = triple
                        raise ValueError(
                            f"label {label!r} window {name!r} lag {lag} sets "
                            f"{key_a!r}/{key_b!r} have {n_a}+{n_b} values; "
                            f"score test requires at most "
                            f"{_SPATIOTEMPORAL_MAX_N} combined values per "
                            f"comparison"
                        )
            means: dict[str, Decimal] = {}
            for key, sample in samples.items():
                total = Decimal(0)
                for value in sample:
                    total += value
                means[key] = total / len(sample)
            group_means[triple] = means

        rank_items: list[str] = []
        pair_items: list[str] = []
        if total_weight > 0:
            weighted: dict[str, Decimal] = {}
            for triple in groups:
                weight = weight_values[triple]
                for key, mean in group_means[triple].items():
                    weighted[key] = weighted.get(key, Decimal(0)) + (
                        weight * mean
                    )
            scores = {
                key: value / total_weight for key, value in weighted.items()
            }
            ranked_keys = sorted(scores, key=lambda key: (scores[key], key))
            for index, key in enumerate(ranked_keys):
                rank = index + 1
                # Leave-one-group-out stability: re-rank the scores with
                # each single group deleted; the set is stable when its
                # rank never changes (always true with at most one group).
                stable = True
                if group_count > 1:
                    for dropped in groups:
                        reduced_total = total_weight - weight_values[dropped]
                        reduced: dict[str, Decimal] = {}
                        for triple in groups:
                            if triple == dropped:
                                continue
                            weight = weight_values[triple]
                            for other, mean in group_means[triple].items():
                                reduced[other] = reduced.get(
                                    other, Decimal(0)
                                ) + (weight * mean)
                        reduced_scores = {
                            other: value / reduced_total
                            for other, value in reduced.items()
                        }
                        reduced_ranked = sorted(
                            reduced_scores,
                            key=lambda other: (
                                reduced_scores[other],
                                other,
                            ),
                        )
                        if reduced_ranked.index(key) + 1 != rank:
                            stable = False
                            break
                rank_items.append(
                    '{"key":' + json.dumps(key, ensure_ascii=False)
                    + ',"score":' + _format6(scores[key])
                    + ',"rank":' + str(rank)
                    + ',"stable":' + ("true" if stable else "false")
                    + '}'
                )

            # One record per emitted (a, b) pair: ``[a, b, diff, p, q]``
            # with q filled in below.
            records: list[list] = []
            present_keys = sorted(weighted)
            for index_a, key_a in enumerate(present_keys):
                for key_b in present_keys[index_a + 1:]:
                    # w_g * d_g per group; the common positive denominator
                    # sum(w_g) cancels in the |.| comparison, so the sign
                    # enumeration compares the unscaled sums exactly.
                    scaled: list[Decimal] = []
                    for triple in groups:
                        means = group_means[triple]
                        scaled.append(
                            weight_values[triple]
                            * (means[key_a] - means[key_b])
                        )
                    total_diff = Decimal(0)
                    for value in scaled:
                        total_diff += value
                    diff = total_diff / total_weight

                    threshold = abs(total_diff)
                    assignments = 1 << group_count
                    hits = 0
                    for mask in range(assignments):
                        signed = Decimal(0)
                        for position in range(group_count):
                            if mask >> position & 1:
                                signed += scaled[position]
                            else:
                                signed -= scaled[position]
                        if abs(signed) >= threshold:
                            hits += 1
                    p_value = Decimal(hits) / Decimal(assignments)
                    records.append([key_a, key_b, diff, p_value, None])

            # Benjamini-Hochberg q-values across all pairs: rank ascending
            # by (p, a, b), then accumulate the running minimum of
            # N * p_l / l from the top rank down, mapping q back.
            count = len(records)
            ranked = sorted(
                range(count),
                key=lambda idx: (
                    records[idx][3],
                    records[idx][0],
                    records[idx][1],
                ),
            )
            running = Decimal(1)
            for rank in range(count, 0, -1):
                idx = ranked[rank - 1]
                candidate = Decimal(count) * records[idx][3] / rank
                if candidate < running:
                    running = candidate
                records[idx][4] = running

            for key_a, key_b, diff, p_value, q_value in records:
                reject = q_value <= alpha_value
                pair_items.append(
                    '{"a":' + json.dumps(key_a, ensure_ascii=False)
                    + ',"b":' + json.dumps(key_b, ensure_ascii=False)
                    + ',"diff":' + _format6(diff)
                    + ',"p":' + _format6(p_value)
                    + ',"q":' + _format6(q_value)
                    + ',"reject":' + ("true" if reject else "false")
                    + '}'
                )

        return (
            '{"minutes":' + str(minutes)
            + ',"alpha":' + _format6(alpha_value)
            + ',"total_weight":' + _format6(total_weight)
            + ',"ranks":[' + ",".join(rank_items) + ']'
            + ',"pairs":[' + ",".join(pair_items) + ']}'
            + "\n"
        )


_SCENARIO_SENSITIVITY_KINDS = frozenset({"green", "roof", "material"})


def scenario_sensitivity(
    base: list,
    sets: dict,
    edges: list,
    lags: list,
    labels: dict,
    windows: dict,
    weights: dict,
    kinds: dict,
    *,
    minutes: int = 60,
    alpha: float = 0.05,
) -> str:
    """Rank share-adjusted scenario scores and test pair differences.

    ``base``, every table in ``sets``, ``edges``, ``lags``, ``labels``,
    ``windows``, ``weights`` and ``minutes`` follow exactly the same
    validation, bucketing, same-label edge pairing, edge order, exception
    rules, ``(timestamp, cell_id)`` key-set equality, group means, weights
    contract and score computation as :func:`score_test` (including the
    combined-sample-size limit of 16 per in-group pair of sets). ``alpha``
    must be a finite non-boolean int/float in ``(0, 1]``; any other value
    raises ``ValueError``.

    ``kinds`` must be a dict whose key set is exactly the ``sets`` key set;
    passing a non-dict raises ``TypeError`` and a key-set mismatch raises
    ``ValueError``. Each value must be a ``(kind, share)`` two-tuple with
    ``kind`` one of ``"green"``, ``"roof"`` or ``"material"`` and ``share``
    a finite non-boolean int/float in ``(0, 1]``; any other value raises
    ``ValueError``.

    Within each group ``g`` every set ``s`` takes the share-adjusted value
    ``v_g = mean_s,g / share_s`` and the weighted score
    ``score = sum(w_g * v_g) / sum(w_g)``; sets are ranked ascending by
    ``(score, key)``. With ``G`` the number of groups, ``G > 16`` raises
    ``ValueError``. Each set's leave-one-group-out stability is checked by
    deleting every group in turn and re-ranking the remaining weighted
    scores ascending by ``(score, key)``; ``stable`` is ``true`` when the
    set's rank never changes, and always ``true`` when ``G <= 1``.

    Each pair of distinct sets ``a < b`` takes the per-group difference
    ``d_g = v_a,g - v_b,g`` and ``diff = sum(w_g * d_g) / sum(w_g)``. The
    exact two-sided sign-permutation p-value enumerates all ``2 ** G``
    sign assignments ``s`` and ``p`` is the proportion with
    ``|sum(s_g * w_g * d_g) / sum(w_g)| >= |diff|``, compared on the
    unquantized values. With ``N`` the number of pairs, all pairs are
    ranked ascending by ``(p, a, b)`` and each rank ``j`` (1-based) gets
    the Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``; ``reject`` is
    ``q <= alpha``, compared on the unquantized values.

    All numbers enter the computation as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN local context. Returns a compact UTF-8
    JSON string with no spaces and exactly one trailing newline; the
    top-level key order is ``ranks, pairs``, each rank object uses the key
    order ``key, kind, share, score, rank, stable`` and each pair object
    the key order ``a, b, diff, p, q, reject``. Ranks are in ascending
    ``(rank, key)`` and pairs in ascending ``(a, b)`` order. Set keys and
    kinds render as JSON strings, ``stable`` and ``reject`` as booleans
    and ``share``, ``score``, ``rank``, ``diff``, ``p`` and ``q`` with
    exactly six decimals, negative zero normalized to ``0.000000``. With
    no groups the only valid input is ``weights={}`` and both arrays are
    empty.
    """
    if not isinstance(sets, dict):
        raise TypeError("sets must be a dict")
    if len(sets) < 2:
        raise ValueError("sets must contain at least two entries")
    for key, table in sets.items():
        if not isinstance(key, str) or not key:
            raise ValueError("each sets key must be a non-empty string")
        if not isinstance(table, list):
            raise ValueError("each sets value must be a list")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")
    if not isinstance(kinds, dict):
        raise TypeError("kinds must be a dict")
    if set(kinds) != set(sets):
        raise ValueError("kinds keys must be exactly the sets keys")
    kind_values: dict[str, str] = {}
    share_values: dict[str, Decimal] = {}
    for key, entry in kinds.items():
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise ValueError("each kinds value must be a (kind, share) tuple")
        kind, share = entry
        if not isinstance(kind, str) or kind not in (
            _SCENARIO_SENSITIVITY_KINDS
        ):
            raise ValueError(
                "each kind must be one of 'green', 'roof' or 'material'"
            )
        if isinstance(share, bool) or not isinstance(share, (int, float)):
            raise ValueError("each share must be a finite int or float")
        if isinstance(share, float) and not math.isfinite(share):
            raise ValueError("each share must be finite")
        decimal_share = Decimal(str(share))
        if decimal_share <= 0 or decimal_share > 1:
            raise ValueError("each share must be in (0, 1]")
        kind_values[key] = kind
        share_values[key] = decimal_share

    minutes, lag_list, alpha_value, parsed_base, xmap_base = (
        _lags_group_x_values(base, edges, lags, labels, minutes, alpha)
    )

    base_keys = {(row[0], row[1]) for row in parsed_base}

    # key -> label -> lag -> bucket start -> [e, ...] (after-minus-base).
    emaps: dict[str, dict] = {}
    for key in sorted(sets):
        _, _, _, parsed_set, xmap_set = _lags_group_x_values(
            sets[key], edges, lags, labels, minutes, alpha
        )
        set_keys = {(row[0], row[1]) for row in parsed_set}
        if set_keys != base_keys:
            raise ValueError(
                f"sets table {key!r} must share the same (timestamp, cell_id) "
                "pairs as base"
            )

        with localcontext() as ctx:
            ctx.prec = _MODEL_PRECISION
            ctx.rounding = ROUND_HALF_EVEN

            # Identical (timestamp, cell_id) key sets make the two xmaps
            # structurally identical, so the per-(label, lag, B) value lists
            # pair up element-wise in the same ascending edge order.
            emap: dict[str, dict[int, dict[int, list[Decimal]]]] = {}
            for label in sorted(xmap_base):
                lag_map: dict[int, dict[int, list[Decimal]]] = {}
                for lag in lag_list:
                    base_buckets = xmap_base[label].get(lag, {})
                    set_buckets = xmap_set[label].get(lag, {})
                    bucket_map: dict[int, list[Decimal]] = {}
                    for bucket in sorted(base_buckets):
                        base_values = base_buckets[bucket]
                        set_values = set_buckets[bucket]
                        bucket_map[bucket] = [
                            set_value - base_value
                            for set_value, base_value in zip(
                                set_values, base_values
                            )
                        ]
                    lag_map[lag] = bucket_map
                emap[label] = lag_map
        emaps[key] = emap

    # Resolve the windows contract against the base table, exactly as
    # scenario_score does.
    if not isinstance(windows, dict):
        raise TypeError("windows must be a dict")
    all_buckets: set[int] = set()
    if parsed_base:
        bucket_seconds = minutes * 60
        for validated in parsed_base:
            timestamp = validated[0]
            all_buckets.add((timestamp // bucket_seconds) * bucket_seconds)
    if set(windows) != all_buckets:
        raise ValueError(
            "windows keys must be exactly the details-derived bucket starts"
        )
    for bucket, name in windows.items():
        if not isinstance(name, str) or not name:
            raise ValueError("each window name must be a non-empty string")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # (label, window, lag) -> set key -> concatenated [e, ...] over the
        # window's buckets in ascending B order (each bucket's values already
        # in ascending edge order).
        triple_samples: dict[
            tuple[str, str, int], dict[str, list[Decimal]]
        ] = {}
        for bucket in sorted(all_buckets):
            name = windows[bucket]
            for key in sorted(emaps):
                emap = emaps[key]
                for label in sorted(emap):
                    for lag in lag_list:
                        values = emap[label].get(lag, {}).get(bucket)
                        if values:
                            triple_samples.setdefault(
                                (label, name, lag), {}
                            ).setdefault(key, []).extend(values)

        # Only triples with a non-empty sample carry a weight.
        groups = sorted(triple_samples)
        if set(weights) != set(groups):
            raise ValueError(
                "weights keys must be exactly the non-empty "
                "(label, window, lag) triples"
            )
        weight_values: dict[tuple[str, str, int], Decimal] = {}
        for triple, weight in weights.items():
            if (
                not isinstance(triple, tuple)
                or len(triple) != 3
                or not isinstance(triple[0], str)
                or not isinstance(triple[1], str)
                or isinstance(triple[2], bool)
                or not isinstance(triple[2], int)
            ):
                raise ValueError(
                    "each weights key must be a (label, window, lag) triple "
                    "with two strings and an integer lag"
                )
            if isinstance(weight, bool) or not isinstance(
                weight, (int, float)
            ):
                raise ValueError("each weight must be a finite int or float")
            if isinstance(weight, float) and not math.isfinite(weight):
                raise ValueError("each weight must be finite")
            decimal_weight = Decimal(str(weight))
            if decimal_weight <= 0:
                raise ValueError("each weight must be positive")
            weight_values[triple] = decimal_weight

        group_count = len(groups)
        if group_count > _SCORE_TEST_MAX_GROUPS:
            raise ValueError(
                f"there are {group_count} groups; scenario sensitivity "
                f"requires at most {_SCORE_TEST_MAX_GROUPS} groups"
            )

        total_weight = Decimal(0)
        for triple in groups:
            total_weight += weight_values[triple]

        # Per-set unquantized means for every group the set has a sample in.
        group_means: dict[
            tuple[str, str, int], dict[str, Decimal]
        ] = {}
        for triple in groups:
            samples = triple_samples[triple]
            present = sorted(samples)
            for index_a, key_a in enumerate(present):
                for key_b in present[index_a + 1:]:
                    n_a = len(samples[key_a])
                    n_b = len(samples[key_b])
                    if n_a + n_b > _SPATIOTEMPORAL_MAX_N:
                        label, name, lag = triple
                        raise ValueError(
                            f"label {label!r} window {name!r} lag {lag} sets "
                            f"{key_a!r}/{key_b!r} have {n_a}+{n_b} values; "
                            f"scenario sensitivity requires at most "
                            f"{_SPATIOTEMPORAL_MAX_N} combined values per "
                            f"comparison"
                        )
            means: dict[str, Decimal] = {}
            for key, sample in samples.items():
                total = Decimal(0)
                for value in sample:
                    total += value
                means[key] = total / len(sample)
            group_means[triple] = means

        # Share-adjusted per-set values v = mean / share for every group.
        group_values: dict[
            tuple[str, str, int], dict[str, Decimal]
        ] = {}
        for triple in groups:
            group_values[triple] = {
                key: mean / share_values[key]
                for key, mean in group_means[triple].items()
            }

        rank_items: list[str] = []
        pair_items: list[str] = []
        if total_weight > 0:
            weighted: dict[str, Decimal] = {}
            for triple in groups:
                weight = weight_values[triple]
                for key, value in group_values[triple].items():
                    weighted[key] = weighted.get(key, Decimal(0)) + (
                        weight * value
                    )
            scores = {
                key: value / total_weight for key, value in weighted.items()
            }
            ranked_keys = sorted(scores, key=lambda key: (scores[key], key))
            for index, key in enumerate(ranked_keys):
                rank = index + 1
                # Leave-one-group-out stability: re-rank the scores with
                # each single group deleted; the set is stable when its
                # rank never changes (always true with at most one group).
                stable = True
                if group_count > 1:
                    for dropped in groups:
                        reduced_total = total_weight - weight_values[dropped]
                        reduced: dict[str, Decimal] = {}
                        for triple in groups:
                            if triple == dropped:
                                continue
                            weight = weight_values[triple]
                            for other, value in group_values[triple].items():
                                reduced[other] = reduced.get(
                                    other, Decimal(0)
                                ) + (weight * value)
                        reduced_scores = {
                            other: value / reduced_total
                            for other, value in reduced.items()
                        }
                        reduced_ranked = sorted(
                            reduced_scores,
                            key=lambda other: (
                                reduced_scores[other],
                                other,
                            ),
                        )
                        if reduced_ranked.index(key) + 1 != rank:
                            stable = False
                            break
                rank_items.append(
                    '{"key":' + json.dumps(key, ensure_ascii=False)
                    + ',"kind":'
                    + json.dumps(kind_values[key], ensure_ascii=False)
                    + ',"share":' + _format6(share_values[key])
                    + ',"score":' + _format6(scores[key])
                    + ',"rank":' + _format6(Decimal(rank))
                    + ',"stable":' + ("true" if stable else "false")
                    + '}'
                )

            # One record per emitted (a, b) pair: ``[a, b, diff, p, q]``
            # with q filled in below.
            records: list[list] = []
            present_keys = sorted(weighted)
            for index_a, key_a in enumerate(present_keys):
                for key_b in present_keys[index_a + 1:]:
                    # w_g * d_g per group; the common positive denominator
                    # sum(w_g) cancels in the |.| comparison, so the sign
                    # enumeration compares the unscaled sums exactly.
                    scaled: list[Decimal] = []
                    for triple in groups:
                        values = group_values[triple]
                        scaled.append(
                            weight_values[triple]
                            * (values[key_a] - values[key_b])
                        )
                    total_diff = Decimal(0)
                    for value in scaled:
                        total_diff += value
                    diff = total_diff / total_weight

                    threshold = abs(total_diff)
                    assignments = 1 << group_count
                    hits = 0
                    for mask in range(assignments):
                        signed = Decimal(0)
                        for position in range(group_count):
                            if mask >> position & 1:
                                signed += scaled[position]
                            else:
                                signed -= scaled[position]
                        if abs(signed) >= threshold:
                            hits += 1
                    p_value = Decimal(hits) / Decimal(assignments)
                    records.append([key_a, key_b, diff, p_value, None])

            # Benjamini-Hochberg q-values across all pairs: rank ascending
            # by (p, a, b), then accumulate the running minimum of
            # N * p_l / l from the top rank down, mapping q back.
            count = len(records)
            ranked = sorted(
                range(count),
                key=lambda idx: (
                    records[idx][3],
                    records[idx][0],
                    records[idx][1],
                ),
            )
            running = Decimal(1)
            for rank in range(count, 0, -1):
                idx = ranked[rank - 1]
                candidate = Decimal(count) * records[idx][3] / rank
                if candidate < running:
                    running = candidate
                records[idx][4] = running

            for key_a, key_b, diff, p_value, q_value in records:
                reject = q_value <= alpha_value
                pair_items.append(
                    '{"a":' + json.dumps(key_a, ensure_ascii=False)
                    + ',"b":' + json.dumps(key_b, ensure_ascii=False)
                    + ',"diff":' + _format6(diff)
                    + ',"p":' + _format6(p_value)
                    + ',"q":' + _format6(q_value)
                    + ',"reject":' + ("true" if reject else "false")
                    + '}'
                )

        return (
            '{"ranks":[' + ",".join(rank_items) + ']'
            + ',"pairs":[' + ",".join(pair_items) + ']}'
            + "\n"
        )


def _scenario_decision_parse_constant(value: str) -> Decimal:
    raise ValueError(
        "each reports value must be a scenario_sensitivity JSON output"
    )


def _scenario_decision_inputs(
    reports: dict, weights: dict
) -> tuple | None:
    """Validate and parse the shared ``scenario_decision*`` inputs.

    Performs every input check of :func:`scenario_decision` and
    returns ``None`` when ``reports`` is empty (the only valid empty
    case); otherwise returns ``(weight_values, panels, scenario_keys,
    kind_values, share_values)`` with ``weight_values`` the per-panel
    ``Decimal`` weights, ``panels`` mapping each ``(region, window)``
    panel key to ``{"ranks": key -> (kind, share, score, rank,
    stable), "pairs": [(a, b, diff, reject), ...]}`` and
    ``kind_values``/``share_values`` the per-scenario values that are
    required to agree across all panels.
    """
    if not isinstance(reports, dict):
        raise TypeError("reports must be a dict")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")
    for panel_key in reports:
        if (
            not isinstance(panel_key, tuple)
            or len(panel_key) != 2
            or not isinstance(panel_key[0], str)
            or not panel_key[0]
            or not isinstance(panel_key[1], str)
            or not panel_key[1]
        ):
            raise ValueError(
                "each reports key must be a (region, window) tuple of two "
                "non-empty strings"
            )
    if set(weights) != set(reports):
        raise ValueError("weights keys must be exactly the reports keys")
    if not reports:
        return None

    weight_values: dict[tuple[str, str], Decimal] = {}
    for panel_key, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("each weight must be a finite int or float")
        if isinstance(weight, float) and not math.isfinite(weight):
            raise ValueError("each weight must be finite")
        decimal_weight = Decimal(str(weight))
        if decimal_weight <= 0:
            raise ValueError("each weight must be positive")
        weight_values[panel_key] = decimal_weight

    # panel key -> {"ranks": scenario key -> (kind, share, score, rank,
    # stable), "pairs": [(a, b, diff, reject), ...]}
    panels: dict[tuple[str, str], dict] = {}
    scenario_keys: set[str] | None = None
    for panel_key in sorted(reports):
        raw = reports[panel_key]
        if not isinstance(raw, str):
            raise ValueError(
                "each reports value must be a scenario_sensitivity JSON "
                "output"
            )
        try:
            data = json.loads(
                raw,
                parse_float=Decimal,
                parse_int=Decimal,
                parse_constant=_scenario_decision_parse_constant,
            )
        except ValueError:
            raise ValueError(
                "each reports value must be a scenario_sensitivity JSON "
                "output"
            ) from None
        if not isinstance(data, dict) or set(data) != {"ranks", "pairs"}:
            raise ValueError(
                "each reports value must be a scenario_sensitivity JSON "
                "output"
            )
        ranks_raw = data["ranks"]
        pairs_raw = data["pairs"]
        if not isinstance(ranks_raw, list) or not isinstance(pairs_raw, list):
            raise ValueError(
                "each reports value must be a scenario_sensitivity JSON "
                "output"
            )

        rank_map: dict[str, tuple] = {}
        for item in ranks_raw:
            if not isinstance(item, dict) or set(item) != {
                "key", "kind", "share", "score", "rank", "stable",
            }:
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            key = item["key"]
            kind = item["kind"]
            share = item["share"]
            score = item["score"]
            rank = item["rank"]
            stable = item["stable"]
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            if (
                not isinstance(kind, str)
                or kind not in _SCENARIO_SENSITIVITY_KINDS
            ):
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            if (
                not isinstance(share, Decimal)
                or not share.is_finite()
                or share <= 0
                or share > 1
            ):
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            if not isinstance(score, Decimal) or not score.is_finite():
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            if (
                not isinstance(rank, Decimal)
                or not rank.is_finite()
                or rank != rank.to_integral_value()
                or rank < 1
            ):
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            if not isinstance(stable, bool):
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            if key in rank_map:
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            rank_map[key] = (kind, share, score, rank, stable)

        pair_list: list[tuple] = []
        for item in pairs_raw:
            if not isinstance(item, dict) or set(item) != {
                "a", "b", "diff", "p", "q", "reject",
            }:
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            key_a = item["a"]
            key_b = item["b"]
            diff = item["diff"]
            p_value = item["p"]
            q_value = item["q"]
            reject = item["reject"]
            if (
                not isinstance(key_a, str)
                or not key_a
                or not isinstance(key_b, str)
                or not key_b
            ):
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            for number in (diff, p_value, q_value):
                if not isinstance(number, Decimal) or not number.is_finite():
                    raise ValueError(
                        "each reports value must be a scenario_sensitivity "
                        "JSON output"
                    )
            if not isinstance(reject, bool):
                raise ValueError(
                    "each reports value must be a scenario_sensitivity JSON "
                    "output"
                )
            pair_list.append((key_a, key_b, diff, reject))

        if len(rank_map) < 2:
            raise ValueError(
                "each report must rank at least two scenarios"
            )
        if scenario_keys is None:
            scenario_keys = set(rank_map)
        elif set(rank_map) != scenario_keys:
            raise ValueError(
                "all reports must rank the same scenario keys"
            )
        panels[panel_key] = {"ranks": rank_map, "pairs": pair_list}

    # A scenario's kind and share must agree across every panel.
    kind_values: dict[str, str] = {}
    share_values: dict[str, Decimal] = {}
    for key in scenario_keys:
        first_kind, first_share = panels[sorted(reports)[0]]["ranks"][key][:2]
        for panel_key in panels:
            kind, share = panels[panel_key]["ranks"][key][:2]
            if kind != first_kind or share != first_share:
                raise ValueError(
                    "each scenario's kind and share must agree across all "
                    "reports"
                )
        kind_values[key] = first_kind
        share_values[key] = first_share
    return (
        weight_values, panels, scenario_keys,
        kind_values, share_values
    )


def scenario_decision(reports: dict, weights: dict) -> str:
    """Combine per-panel scenario sensitivity reports into one decision.

    ``reports`` must be a dict whose keys are ``(region, window)``
    two-tuples of non-empty strings and whose values are
    :func:`scenario_sensitivity` JSON outputs; passing a non-dict raises
    ``TypeError`` and any other violation raises ``ValueError``. Every
    report must rank the same set of at least two scenario keys, and a
    scenario's ``kind`` and ``share`` must agree across all reports.
    ``weights`` must be a dict with exactly the ``reports`` key set whose
    values are positive finite non-boolean int/float weights; a non-dict
    raises ``TypeError`` and any other violation raises ``ValueError``.

    Within each panel the candidate is the scenario with the smallest
    ``rank``; the panel recommends its candidate only when the candidate's
    ``stable`` is true, every pair involving it has ``reject`` true, and
    the pair ``diff`` is negative when the candidate is ``a`` and positive
    when it is ``b``. With ``W`` the sum of all weights, each scenario's
    ``score`` is the weight-weighted mean of its per-panel scores, its
    ``support`` is the fraction of ``W`` held by panels recommending it,
    and its ``stable`` is true only when it is stable in every panel.
    Scenarios are ranked ascending by ``(score, key)`` with integer ranks
    starting at 1. ``recommend`` is the first-ranked scenario's key only
    when that scenario is stable and its support is exactly 1, and
    ``null`` otherwise.

    All numbers enter the computation as ``Decimal(str(x))`` (report
    numbers are parsed straight from the JSON text) under a precision-1000,
    ROUND_HALF_EVEN local context. Returns a compact UTF-8 JSON string
    with no spaces and exactly one trailing newline; the top-level key
    order is ``total_weight, recommend, ranks`` and each rank object uses
    the key order ``key, kind, share, score, support, rank, stable``.
    Ranks are in ascending ``(rank, key)`` order. ``rank`` renders as a
    JSON integer, ``stable`` as a boolean and ``recommend`` as a JSON
    string or null; every other number renders with exactly six decimals,
    negative zero normalized to ``0.000000``. With no panels the only
    valid input is ``reports={}`` and ``weights={}``, yielding
    ``total_weight`` ``0.000000``, ``recommend`` ``null`` and an empty
    ``ranks`` array.
    """
    parsed = _scenario_decision_inputs(reports, weights)
    if parsed is None:
        return (
            '{"total_weight":0.000000,"recommend":null,"ranks":[]}'
            + "\n"
        )
    (
        weight_values, panels, scenario_keys,
        kind_values, share_values
    ) = parsed

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        total_weight = Decimal(0)
        for panel_key in panels:
            total_weight += weight_values[panel_key]

        # Each panel recommends its smallest-rank candidate only when the
        # candidate is stable, every pair involving it rejects, and the
        # pair diff points away from it (negative as ``a``, positive as
        # ``b``).
        panel_choice: dict[tuple[str, str], str | None] = {}
        for panel_key in sorted(panels):
            rank_map = panels[panel_key]["ranks"]
            candidate = min(
                rank_map, key=lambda key: (rank_map[key][3], key)
            )
            recommended = rank_map[candidate][4]
            if recommended:
                for key_a, key_b, diff, reject in panels[panel_key]["pairs"]:
                    if candidate != key_a and candidate != key_b:
                        continue
                    if not reject:
                        recommended = False
                        break
                    if candidate == key_a and not diff < 0:
                        recommended = False
                        break
                    if candidate == key_b and not diff > 0:
                        recommended = False
                        break
            panel_choice[panel_key] = candidate if recommended else None

        decision_items: list[dict] = []
        for key in sorted(scenario_keys):
            weighted_score = Decimal(0)
            support_weight = Decimal(0)
            stable = True
            for panel_key in panels:
                weight = weight_values[panel_key]
                weighted_score += weight * panels[panel_key]["ranks"][key][2]
                if panel_choice[panel_key] == key:
                    support_weight += weight
                if not panels[panel_key]["ranks"][key][4]:
                    stable = False
            decision_items.append(
                {
                    "key": key,
                    "score": weighted_score / total_weight,
                    "support": support_weight / total_weight,
                    "stable": stable,
                }
            )
        decision_items.sort(key=lambda item: (item["score"], item["key"]))
        for index, item in enumerate(decision_items):
            item["rank"] = index + 1

        first = decision_items[0]
        recommend = (
            first["key"]
            if first["stable"] and first["support"] == 1
            else None
        )

        rank_items: list[str] = []
        for item in decision_items:
            key = item["key"]
            rank_items.append(
                '{"key":' + json.dumps(key, ensure_ascii=False)
                + ',"kind":'
                + json.dumps(kind_values[key], ensure_ascii=False)
                + ',"share":' + _format6(share_values[key])
                + ',"score":' + _format6(item["score"])
                + ',"support":' + _format6(item["support"])
                + ',"rank":' + str(item["rank"])
                + ',"stable":' + ("true" if item["stable"] else "false")
                + '}'
            )

        return (
            '{"total_weight":' + _format6(total_weight)
            + ',"recommend":'
            + (
                json.dumps(recommend, ensure_ascii=False)
                if recommend is not None
                else "null"
            )
            + ',"ranks":[' + ",".join(rank_items) + ']}'
            + "\n"
        )


def scenario_decision_attribution(reports: dict, weights: dict) -> str:
    """Attribute a combined scenario decision back to its panels.

    Inputs, validation and ``Decimal`` computation follow
    :func:`scenario_decision`: ``reports`` maps ``(region, window)``
    two-tuples of non-empty strings to :func:`scenario_sensitivity` JSON
    outputs and ``weights`` holds one positive finite non-boolean
    int/float weight per panel; a non-dict argument raises ``TypeError``
    and any other violation raises ``ValueError``.

    The attributed ``key`` is the first-ranked scenario of the combined
    decision (scenarios ranked ascending by ``(score, key)`` with
    ``score`` the weight-weighted mean of the per-panel scores). Within
    each panel the ``candidate`` is the scenario with the smallest
    ``(rank, key)`` and its ``reason`` is the first matching of:
    ``unstable`` when the candidate's ``stable`` is false,
    ``nonsignificant`` when a pair involving the candidate has ``reject``
    false, ``direction`` when the candidate is ``a`` with ``diff >= 0``
    or ``b`` with ``diff <= 0`` in a pair involving it, and ``recommend``
    otherwise. The panel's ``decision`` is its candidate only when the
    reason is ``recommend`` and ``null`` otherwise. With ``W`` the sum of
    all weights, the panel ``score`` is ``weight`` times the attributed
    key's score in that panel divided by ``W``, and ``support`` is
    ``weight / W`` when the panel's ``decision`` is the attributed key
    and ``0`` otherwise.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``total_weight, key,
    panels`` and each panel object uses the key order ``region, window,
    weight, candidate, decision, reason, score, support``. Panels are in
    ascending ``(region, window)`` order. ``key``, ``candidate`` and
    ``reason`` render as JSON strings, ``decision`` as a JSON string or
    null, and every number renders with exactly six decimals, negative
    zero normalized to ``0.000000``. With no panels the only valid input
    is ``reports={}`` and ``weights={}``, yielding
    ``{"total_weight":0.000000,"key":null,"panels":[]}`` followed by a
    newline.
    """
    parsed = _scenario_decision_inputs(reports, weights)
    if parsed is None:
        return '{"total_weight":0.000000,"key":null,"panels":[]}\n'
    (
        weight_values, panels, scenario_keys,
        kind_values, share_values,
    ) = parsed

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        total_weight = Decimal(0)
        for panel_key in panels:
            total_weight += weight_values[panel_key]

        # The attributed key is the first-ranked scenario of the combined
        # decision: ascending (score, key) with score the weight-weighted
        # mean of the per-panel scores.
        scenario_scores: dict[str, Decimal] = {}
        for scenario_key in scenario_keys:
            weighted_score = Decimal(0)
            for panel_key in panels:
                weighted_score += (
                    weight_values[panel_key]
                    * panels[panel_key]["ranks"][scenario_key][2]
                )
            scenario_scores[scenario_key] = weighted_score / total_weight
        key = min(
            scenario_scores,
            key=lambda name: (scenario_scores[name], name),
        )

        panel_items: list[str] = []
        for panel_key in sorted(panels):
            rank_map = panels[panel_key]["ranks"]
            candidate = min(
                rank_map, key=lambda name: (rank_map[name][3], name)
            )
            reason = "recommend"
            if not rank_map[candidate][4]:
                reason = "unstable"
            else:
                for key_a, key_b, diff, reject in panels[panel_key]["pairs"]:
                    if candidate != key_a and candidate != key_b:
                        continue
                    if not reject:
                        reason = "nonsignificant"
                        break
                if reason == "recommend":
                    for key_a, key_b, diff, reject in panels[panel_key][
                        "pairs"
                    ]:
                        if candidate != key_a and candidate != key_b:
                            continue
                        if (candidate == key_a and diff >= 0) or (
                            candidate == key_b and diff <= 0
                        ):
                            reason = "direction"
                            break
            decision = candidate if reason == "recommend" else None
            weight = weight_values[panel_key]
            score = weight * rank_map[key][2] / total_weight
            support = (
                weight / total_weight if decision == key else Decimal(0)
            )
            region, window = panel_key
            panel_items.append(
                '{"region":' + json.dumps(region, ensure_ascii=False)
                + ',"window":' + json.dumps(window, ensure_ascii=False)
                + ',"weight":' + _format6(weight)
                + ',"candidate":'
                + json.dumps(candidate, ensure_ascii=False)
                + ',"decision":'
                + (
                    json.dumps(decision, ensure_ascii=False)
                    if decision is not None
                    else "null"
                )
                + ',"reason":' + json.dumps(reason, ensure_ascii=False)
                + ',"score":' + _format6(score)
                + ',"support":' + _format6(support)
                + '}'
            )

        return (
            '{"total_weight":' + _format6(total_weight)
            + ',"key":' + json.dumps(key, ensure_ascii=False)
            + ',"panels":[' + ",".join(panel_items) + ']}'
            + "\n"
        )


def decision_summary(reports: dict, weights: dict, *, by: str = "region") -> str:
    """Summarize per-panel scenario decisions grouped by region or window.

    Inputs, validation and ``Decimal`` computation follow
    :func:`scenario_decision_attribution`: ``reports`` maps
    ``(region, window)`` two-tuples of non-empty strings to
    :func:`scenario_sensitivity` JSON outputs and ``weights`` holds one
    positive finite non-boolean int/float weight per panel; a non-dict
    argument raises ``TypeError`` and any other violation raises
    ``ValueError``. ``by`` must be ``"region"`` or ``"window"`` and any
    other value raises ``ValueError``.

    Panels are grouped by their region when ``by`` is ``"region"`` and by
    their window when ``by`` is ``"window"``; groups are rendered in
    ascending group key order and each group lists every scenario in
    ascending key order. The per-panel judgment follows
    :func:`scenario_decision_attribution`: within each panel the
    ``candidate`` is the scenario with the smallest ``(rank, key)``, its
    ``reason`` is the first matching of ``unstable``, ``nonsignificant``,
    ``direction`` and ``recommend``, and the panel's ``decision`` is its
    candidate only when the reason is ``recommend``.

    With ``W`` the sum of all weights across every panel and the group
    ``weight`` the sum of its panels' weights, each scenario's numbers
    within a group are: ``contribution`` the sum of ``weight`` times the
    scenario's per-panel ``score`` divided by ``W``; ``support`` the sum
    of the weights of the group's panels whose ``decision`` is the
    scenario divided by ``W``; and ``unstable``, ``nonsignificant`` and
    ``direction`` the sums of the weights of the group's panels whose
    ``candidate`` is the scenario with the like-named ``reason``, each
    divided by ``W``.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``by, total_weight,
    groups``, each group object uses the key order ``key, weight,
    candidates`` and each candidate object uses the key order ``key,
    kind, share, contribution, support, unstable, nonsignificant,
    direction``. ``groups`` and ``candidates`` are JSON arrays; ``by``
    and each ``key`` and ``kind`` render as JSON strings; ``kind`` and
    ``share`` carry the report values. Every number renders with exactly
    six decimals, negative zero normalized to ``0.000000``. With no
    panels the only valid input is ``reports={}`` and ``weights={}``,
    yielding ``total_weight`` ``0.000000`` and an empty ``groups``
    array.
    """
    if by not in ("region", "window"):
        raise ValueError("by must be 'region' or 'window'")
    parsed = _scenario_decision_inputs(reports, weights)
    if parsed is None:
        return (
            '{"by":' + json.dumps(by, ensure_ascii=False)
            + ',"total_weight":0.000000,"groups":[]}'
            + "\n"
        )
    (
        weight_values, panels, scenario_keys,
        kind_values, share_values,
    ) = parsed

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        total_weight = Decimal(0)
        for panel_key in panels:
            total_weight += weight_values[panel_key]

        # Per-panel judgment as in scenario_decision_attribution: the
        # candidate is the smallest-(rank, key) scenario, its reason the
        # first matching of unstable, nonsignificant, direction and
        # recommend, and the decision the candidate only on recommend.
        panel_candidate: dict[tuple[str, str], str] = {}
        panel_reason: dict[tuple[str, str], str] = {}
        panel_decision: dict[tuple[str, str], str | None] = {}
        for panel_key in sorted(panels):
            rank_map = panels[panel_key]["ranks"]
            candidate = min(
                rank_map, key=lambda name: (rank_map[name][3], name)
            )
            reason = "recommend"
            if not rank_map[candidate][4]:
                reason = "unstable"
            else:
                for key_a, key_b, diff, reject in panels[panel_key]["pairs"]:
                    if candidate != key_a and candidate != key_b:
                        continue
                    if not reject:
                        reason = "nonsignificant"
                        break
                if reason == "recommend":
                    for key_a, key_b, diff, reject in panels[panel_key][
                        "pairs"
                    ]:
                        if candidate != key_a and candidate != key_b:
                            continue
                        if (candidate == key_a and diff >= 0) or (
                            candidate == key_b and diff <= 0
                        ):
                            reason = "direction"
                            break
            panel_candidate[panel_key] = candidate
            panel_reason[panel_key] = reason
            panel_decision[panel_key] = (
                candidate if reason == "recommend" else None
            )

        group_index = 0 if by == "region" else 1
        group_members: dict[str, list[tuple[str, str]]] = {}
        for panel_key in panels:
            group_members.setdefault(panel_key[group_index], []).append(
                panel_key
            )

        group_items: list[str] = []
        for group_key in sorted(group_members):
            members = group_members[group_key]
            group_weight = Decimal(0)
            for panel_key in members:
                group_weight += weight_values[panel_key]
            candidate_items: list[str] = []
            for scenario_key in sorted(scenario_keys):
                contribution = Decimal(0)
                support = Decimal(0)
                unstable = Decimal(0)
                nonsignificant = Decimal(0)
                direction = Decimal(0)
                for panel_key in members:
                    weight = weight_values[panel_key]
                    contribution += (
                        weight * panels[panel_key]["ranks"][scenario_key][2]
                    )
                    if panel_decision[panel_key] == scenario_key:
                        support += weight
                    if panel_candidate[panel_key] == scenario_key:
                        reason = panel_reason[panel_key]
                        if reason == "unstable":
                            unstable += weight
                        elif reason == "nonsignificant":
                            nonsignificant += weight
                        elif reason == "direction":
                            direction += weight
                candidate_items.append(
                    '{"key":'
                    + json.dumps(scenario_key, ensure_ascii=False)
                    + ',"kind":'
                    + json.dumps(kind_values[scenario_key], ensure_ascii=False)
                    + ',"share":' + _format6(share_values[scenario_key])
                    + ',"contribution":'
                    + _format6(contribution / total_weight)
                    + ',"support":' + _format6(support / total_weight)
                    + ',"unstable":' + _format6(unstable / total_weight)
                    + ',"nonsignificant":'
                    + _format6(nonsignificant / total_weight)
                    + ',"direction":' + _format6(direction / total_weight)
                    + '}'
                )
            group_items.append(
                '{"key":' + json.dumps(group_key, ensure_ascii=False)
                + ',"weight":' + _format6(group_weight)
                + ',"candidates":[' + ",".join(candidate_items) + ']}'
            )

        return (
            '{"by":' + json.dumps(by, ensure_ascii=False)
            + ',"total_weight":' + _format6(total_weight)
            + ',"groups":[' + ",".join(group_items) + ']}'
            + "\n"
        )


def decision_priority_matrix(reports: dict, weights: dict) -> str:
    """Rank scenarios within every region and window group by priority.

    Inputs, validation and ``Decimal`` computation follow
    :func:`decision_summary`: ``reports`` maps ``(region, window)``
    two-tuples of non-empty strings to :func:`scenario_sensitivity`
    JSON outputs and ``weights`` holds one positive finite non-boolean
    int/float weight per panel; a non-dict argument raises ``TypeError``
    and any other violation raises ``ValueError``.

    Panels are grouped twice, first by region then by window, using the
    same per-panel judgment as :func:`decision_summary`: each scenario's
    within-group ``contribution``, ``support``, ``unstable``,
    ``nonsignificant`` and ``direction`` are the summary values for that
    group. Its ``priority`` is
    ``contribution - support + unstable + nonsignificant + direction``;
    scenarios are ranked ascending by ``(priority, key)`` with integer
    ranks starting at 1. Across every group of both groupings each
    scenario's ``min_rank`` and ``max_rank`` are its smallest and largest
    ranks and ``gap`` is ``max_rank - min_rank``.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is
    ``total_weight, groups, spreads``. Each group object uses the key
    order ``by, key, candidates`` with ``by`` ``"region"`` for the region
    groups followed by ``"window"`` for the window groups, group keys
    ascending within each; each candidate object uses the key order
    ``key, priority, rank`` with candidates in ascending ``(rank, key)``
    order; each spread object uses the key order
    ``key, min_rank, max_rank, gap`` with spreads in ascending key
    order. Ranks and rank statistics render as JSON integers; every other
    number renders with exactly six decimals, negative zero normalized to
    ``0.000000``. With no panels the only valid input is ``reports={}``
    and ``weights={}``, yielding ``total_weight`` ``0.000000`` and empty
    ``groups`` and ``spreads`` arrays.
    """
    parsed = _scenario_decision_inputs(reports, weights)
    if parsed is None:
        return (
            '{"total_weight":0.000000,"groups":[],"spreads":[]}'
            + "\n"
        )
    (
        weight_values, panels, scenario_keys,
        _kind_values, _share_values,
    ) = parsed

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        total_weight = Decimal(0)
        for panel_key in panels:
            total_weight += weight_values[panel_key]

        # Per-panel judgment as in decision_summary: the candidate is the
        # smallest-(rank, key) scenario, its reason the first matching of
        # unstable, nonsignificant, direction and recommend, and the
        # decision the candidate only on recommend.
        panel_candidate: dict[tuple[str, str], str] = {}
        panel_reason: dict[tuple[str, str], str] = {}
        panel_decision: dict[tuple[str, str], str | None] = {}
        for panel_key in sorted(panels):
            rank_map = panels[panel_key]["ranks"]
            candidate = min(
                rank_map, key=lambda name: (rank_map[name][3], name)
            )
            reason = "recommend"
            if not rank_map[candidate][4]:
                reason = "unstable"
            else:
                for key_a, key_b, diff, reject in panels[panel_key]["pairs"]:
                    if candidate != key_a and candidate != key_b:
                        continue
                    if not reject:
                        reason = "nonsignificant"
                        break
                if reason == "recommend":
                    for key_a, key_b, diff, reject in panels[panel_key][
                        "pairs"
                    ]:
                        if candidate != key_a and candidate != key_b:
                            continue
                        if (candidate == key_a and diff >= 0) or (
                            candidate == key_b and diff <= 0
                        ):
                            reason = "direction"
                            break
            panel_candidate[panel_key] = candidate
            panel_reason[panel_key] = reason
            panel_decision[panel_key] = (
                candidate if reason == "recommend" else None
            )

        min_rank: dict[str, int] = {}
        max_rank: dict[str, int] = {}

        def render_groups(by: str) -> list[str]:
            group_index = 0 if by == "region" else 1
            group_members: dict[str, list[tuple[str, str]]] = {}
            for panel_key in panels:
                group_members.setdefault(panel_key[group_index], []).append(
                    panel_key
                )

            group_items: list[str] = []
            for group_key in sorted(group_members):
                members = group_members[group_key]
                priorities: list[tuple[str, Decimal]] = []
                for scenario_key in sorted(scenario_keys):
                    contribution = Decimal(0)
                    support = Decimal(0)
                    unstable = Decimal(0)
                    nonsignificant = Decimal(0)
                    direction = Decimal(0)
                    for panel_key in members:
                        weight = weight_values[panel_key]
                        contribution += (
                            weight
                            * panels[panel_key]["ranks"][scenario_key][2]
                        )
                        if panel_decision[panel_key] == scenario_key:
                            support += weight
                        if panel_candidate[panel_key] == scenario_key:
                            member_reason = panel_reason[panel_key]
                            if member_reason == "unstable":
                                unstable += weight
                            elif member_reason == "nonsignificant":
                                nonsignificant += weight
                            elif member_reason == "direction":
                                direction += weight
                    contribution /= total_weight
                    support /= total_weight
                    unstable /= total_weight
                    nonsignificant /= total_weight
                    direction /= total_weight
                    priority = (
                        contribution - support + unstable
                        + nonsignificant + direction
                    )
                    priorities.append((scenario_key, priority))
                priorities.sort(key=lambda item: (item[1], item[0]))

                candidate_items: list[str] = []
                for rank, (scenario_key, priority) in enumerate(
                    priorities, start=1
                ):
                    if scenario_key not in min_rank or rank < min_rank[
                        scenario_key
                    ]:
                        min_rank[scenario_key] = rank
                    if scenario_key not in max_rank or rank > max_rank[
                        scenario_key
                    ]:
                        max_rank[scenario_key] = rank
                    candidate_items.append(
                        '{"key":'
                        + json.dumps(scenario_key, ensure_ascii=False)
                        + ',"priority":' + _format6(priority)
                        + ',"rank":' + str(rank)
                        + '}'
                    )
                group_items.append(
                    '{"by":' + json.dumps(by, ensure_ascii=False)
                    + ',"key":' + json.dumps(group_key, ensure_ascii=False)
                    + ',"candidates":[' + ",".join(candidate_items) + ']}'
                )
            return group_items

        group_items = render_groups("region") + render_groups("window")

        spread_items: list[str] = []
        for scenario_key in sorted(scenario_keys):
            low = min_rank[scenario_key]
            high = max_rank[scenario_key]
            spread_items.append(
                '{"key":'
                + json.dumps(scenario_key, ensure_ascii=False)
                + ',"min_rank":' + str(low)
                + ',"max_rank":' + str(high)
                + ',"gap":' + str(high - low)
                + '}'
            )

        return (
            '{"total_weight":' + _format6(total_weight)
            + ',"groups":[' + ",".join(group_items) + ']'
            + ',"spreads":[' + ",".join(spread_items) + ']}'
            + "\n"
        )


_PRIORITY_MATRIX_ERROR = (
    "each value must be a decision_priority_matrix JSON output"
)
_PRIORITY_MATRIX_INT_RE = re.compile(r"(?:0|[1-9][0-9]*)")
_PRIORITY_MATRIX_DECIMAL_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)\.[0-9]{6}"
)


class _PriorityMatrixInt(Decimal):
    """Marker for a canonical unsigned JSON integer token."""


class _PriorityMatrixDecimal(Decimal):
    """Marker for a canonical fixed-six-decimal JSON number token."""


def _priority_matrix_signed_zero(value: Decimal) -> bool:
    return value == 0 and value.is_signed()


def _decision_priority_matrix_parse_constant(value: str) -> Decimal:
    raise ValueError(_PRIORITY_MATRIX_ERROR)


def _decision_priority_matrix_groups(raw: object) -> dict:
    """Parse one :func:`decision_priority_matrix` JSON output.

    Returns a dict mapping each ``(by, key)`` group pair to its candidate
    ranks (candidate key -> rank). Any structural deviation from the
    canonical output raises ``ValueError``.
    """
    if not isinstance(raw, str):
        raise ValueError(_PRIORITY_MATRIX_ERROR)
    if not raw.endswith("\n") or raw.endswith("\n\n"):
        raise ValueError(_PRIORITY_MATRIX_ERROR)
    payload = raw[:-1]

    def _parse_integer(value: str) -> Decimal:
        if not _PRIORITY_MATRIX_INT_RE.fullmatch(value):
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        return _PriorityMatrixInt(value)

    def _parse_decimal(value: str) -> Decimal:
        if not _PRIORITY_MATRIX_DECIMAL_RE.fullmatch(value):
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        return _PriorityMatrixDecimal(value)

    try:
        data = json.loads(
            payload,
            parse_float=_parse_decimal,
            parse_int=_parse_integer,
            parse_constant=_decision_priority_matrix_parse_constant,
        )
    except ValueError:
        raise ValueError(_PRIORITY_MATRIX_ERROR) from None
    if not isinstance(data, dict) or set(data) != {
        "total_weight", "groups", "spreads",
    }:
        raise ValueError(_PRIORITY_MATRIX_ERROR)
    total_weight = data["total_weight"]
    groups_raw = data["groups"]
    spreads_raw = data["spreads"]
    if (
        not isinstance(total_weight, _PriorityMatrixDecimal)
        or total_weight < 0
        or _priority_matrix_signed_zero(total_weight)
        or not isinstance(groups_raw, list)
        or not isinstance(spreads_raw, list)
    ):
        raise ValueError(_PRIORITY_MATRIX_ERROR)

    def _valid_rank(value: object) -> bool:
        return (
            isinstance(value, _PriorityMatrixInt)
            and value == value.to_integral_value()
            and value >= 1
        )

    groups: dict[tuple[str, str], dict[str, int]] = {}
    last_by = "region"
    last_key: str | None = None
    for item in groups_raw:
        if not isinstance(item, dict) or set(item) != {
            "by", "key", "candidates",
        }:
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        by = item["by"]
        key = item["key"]
        candidates_raw = item["candidates"]
        if (
            by not in ("region", "window")
            or not isinstance(key, str)
            or not key
            or not isinstance(candidates_raw, list)
        ):
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        # Canonical order: region groups first, then window groups, group
        # keys ascending within each.
        if by == "region":
            if last_by == "window":
                raise ValueError(_PRIORITY_MATRIX_ERROR)
        elif last_by == "region":
            last_key = None
        if last_key is not None and key <= last_key:
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        last_by = by
        last_key = key
        if (by, key) in groups:
            raise ValueError(_PRIORITY_MATRIX_ERROR)

        ranks: dict[str, int] = {}
        for candidate in candidates_raw:
            if not isinstance(candidate, dict) or set(candidate) != {
                "key", "priority", "rank",
            }:
                raise ValueError(_PRIORITY_MATRIX_ERROR)
            candidate_key = candidate["key"]
            priority = candidate["priority"]
            rank = candidate["rank"]
            if (
                not isinstance(candidate_key, str)
                or not candidate_key
                or not isinstance(priority, _PriorityMatrixDecimal)
                or _priority_matrix_signed_zero(priority)
                or not _valid_rank(rank)
            ):
                raise ValueError(_PRIORITY_MATRIX_ERROR)
            if candidate_key in ranks:
                raise ValueError(_PRIORITY_MATRIX_ERROR)
            rank_value = int(rank)
            # Canonical candidates are ranked 1..n in ascending (rank, key),
            # so the rank at each position is exactly that position.
            if rank_value != len(ranks) + 1:
                raise ValueError(_PRIORITY_MATRIX_ERROR)
            ranks[candidate_key] = rank_value
        if len(ranks) < 2:
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        groups[(by, key)] = ranks

    spread_rows: dict[str, tuple[int, int, int]] = {}
    last_spread: str | None = None
    for item in spreads_raw:
        if not isinstance(item, dict) or set(item) != {
            "key", "min_rank", "max_rank", "gap",
        }:
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        key = item["key"]
        min_rank = item["min_rank"]
        max_rank = item["max_rank"]
        gap = item["gap"]
        if (
            not isinstance(key, str)
            or not key
            or not _valid_rank(min_rank)
            or not _valid_rank(max_rank)
            or not isinstance(gap, _PriorityMatrixInt)
            or gap != gap.to_integral_value()
            or gap < 0
        ):
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        if key in spread_rows or (
            last_spread is not None and key <= last_spread
        ):
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        last_spread = key
        spread_rows[key] = (int(min_rank), int(max_rank), int(gap))

    if groups:
        if total_weight <= 0:
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        candidate_keys: set[str] | None = None
        for by in ("region", "window"):
            if not any(group_by == by for group_by, _ in groups):
                raise ValueError(_PRIORITY_MATRIX_ERROR)
        for ranks in groups.values():
            if candidate_keys is None:
                candidate_keys = set(ranks)
            elif set(ranks) != candidate_keys:
                raise ValueError(_PRIORITY_MATRIX_ERROR)
        if set(spread_rows) != candidate_keys:
            raise ValueError(_PRIORITY_MATRIX_ERROR)
        # Every spread must match the candidate's ranks across the groups.
        for candidate_key in candidate_keys:
            observed = [
                ranks[candidate_key] for ranks in groups.values()
            ]
            low, high, gap = spread_rows[candidate_key]
            if (
                low != min(observed)
                or high != max(observed)
                or gap != high - low
            ):
                raise ValueError(_PRIORITY_MATRIX_ERROR)
    elif spread_rows or total_weight != 0:
        raise ValueError(_PRIORITY_MATRIX_ERROR)
    return groups


def decision_priority_shift(base: str, snapshots: dict) -> str:
    """Compare snapshot priority ranks against a baseline matrix.

    ``base`` must be a :func:`decision_priority_matrix` JSON output and
    ``snapshots`` a non-empty dict mapping non-empty string snapshot names
    to :func:`decision_priority_matrix` JSON outputs; a non-string
    ``base`` or a non-dict ``snapshots`` raises ``TypeError`` and any
    other contract violation raises ``ValueError``. Every snapshot must
    share the baseline's set of ``(by, key)`` groups and, within each
    group, the same set of candidate keys; a mismatch raises
    ``ValueError``.

    For every snapshot, group and candidate, ``delta`` is the snapshot
    rank minus the baseline rank; ``direction`` is ``"rise"``,
    ``"fall"`` or ``"same"`` for a negative, positive or zero delta.
    Across all snapshots and groups each candidate's ``rise``, ``fall``
    and ``same`` count the directions, ``net`` is the sum of the deltas,
    ``max_abs`` the largest absolute delta and ``stable`` is true
    exactly when ``max_abs`` is zero.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``changes, summary``.
    Each change object uses the key order ``snapshot, by, key,
    candidates`` and each candidate object the key order ``key,
    base_rank, rank, delta, direction``; each summary object uses the
    key order ``key, rise, fall, same, net, max_abs, stable``. Changes
    are ordered by snapshot name, region groups before window groups and
    ascending group key; candidates and summary entries are in ascending
    key order. Ranks and statistics render as JSON integers and
    ``stable`` as a boolean.
    """
    if not isinstance(base, str):
        raise TypeError("base must be a str")
    if not isinstance(snapshots, dict):
        raise TypeError("snapshots must be a dict")
    if not snapshots:
        raise ValueError("snapshots must be a non-empty dict")
    for name in snapshots:
        if not isinstance(name, str) or not name:
            raise ValueError(
                "each snapshots key must be a non-empty string"
            )

    base_groups = _decision_priority_matrix_groups(base)
    snapshot_groups: dict[str, dict] = {}
    for name in sorted(snapshots):
        groups = _decision_priority_matrix_groups(snapshots[name])
        if set(groups) != set(base_groups):
            raise ValueError(
                "each snapshot must share the baseline's (by, key) groups"
            )
        for group_key, ranks in groups.items():
            if set(ranks) != set(base_groups[group_key]):
                raise ValueError(
                    "each snapshot group must share the baseline's "
                    "candidate keys"
                )
        snapshot_groups[name] = groups

    # candidate key -> [rise, fall, same, net, max_abs]
    stats: dict[str, list] = {}
    change_items: list[str] = []
    for name in sorted(snapshot_groups):
        groups = snapshot_groups[name]
        for by in ("region", "window"):
            group_keys = sorted(
                key for group_by, key in groups if group_by == by
            )
            for group_key in group_keys:
                base_ranks = base_groups[(by, group_key)]
                ranks = groups[(by, group_key)]
                candidate_items: list[str] = []
                for candidate_key in sorted(ranks):
                    base_rank = base_ranks[candidate_key]
                    rank = ranks[candidate_key]
                    delta = rank - base_rank
                    if delta < 0:
                        direction = "rise"
                    elif delta > 0:
                        direction = "fall"
                    else:
                        direction = "same"
                    entry = stats.setdefault(
                        candidate_key, [0, 0, 0, 0, 0]
                    )
                    if delta < 0:
                        entry[0] += 1
                    elif delta > 0:
                        entry[1] += 1
                    else:
                        entry[2] += 1
                    entry[3] += delta
                    if abs(delta) > entry[4]:
                        entry[4] = abs(delta)
                    candidate_items.append(
                        '{"key":'
                        + json.dumps(candidate_key, ensure_ascii=False)
                        + ',"base_rank":' + str(base_rank)
                        + ',"rank":' + str(rank)
                        + ',"delta":' + str(delta)
                        + ',"direction":' + json.dumps(direction)
                        + '}'
                    )
                change_items.append(
                    '{"snapshot":' + json.dumps(name, ensure_ascii=False)
                    + ',"by":' + json.dumps(by, ensure_ascii=False)
                    + ',"key":' + json.dumps(group_key, ensure_ascii=False)
                    + ',"candidates":[' + ",".join(candidate_items) + ']}'
                )

    summary_items: list[str] = []
    for candidate_key in sorted(stats):
        rise, fall, same, net, max_abs = stats[candidate_key]
        summary_items.append(
            '{"key":' + json.dumps(candidate_key, ensure_ascii=False)
            + ',"rise":' + str(rise)
            + ',"fall":' + str(fall)
            + ',"same":' + str(same)
            + ',"net":' + str(net)
            + ',"max_abs":' + str(max_abs)
            + ',"stable":' + ("true" if max_abs == 0 else "false")
            + '}'
        )

    return (
        '{"changes":[' + ",".join(change_items) + ']'
        + ',"summary":[' + ",".join(summary_items) + ']}'
        + "\n"
    )


def decision_priority_consensus(
    base: str, snapshots: dict, evidence: dict
) -> str:
    """Aggregate snapshot rank shifts with supporting evidence.

    ``base`` must be a :func:`decision_priority_matrix` JSON output and
    ``snapshots`` a non-empty dict mapping non-empty string snapshot
    names to :func:`decision_priority_matrix` JSON outputs whose groups
    and candidate sets match the baseline's. ``evidence`` must be a dict
    whose keys are exactly the full set of
    ``(snapshot, by, group, candidate)`` four-tuples covered by the
    snapshots and whose values are booleans. A non-string ``base`` or a
    non-dict ``snapshots`` or ``evidence`` raises ``TypeError``; every
    other contract violation raises ``ValueError``.

    For each candidate, ``delta`` is the snapshot rank minus the
    baseline rank at every ``(snapshot, by, group)`` observation:
    ``direction`` is ``"rise"`` when every delta is non-positive and at
    least one is negative, ``"fall"`` when every delta is non-negative
    and at least one is positive, ``"same"`` when every delta is zero
    and ``"mixed"`` otherwise. ``support`` is the number of true
    evidence values divided by the number of observations, quantized to
    six decimals under precision 1000 with ROUND_HALF_EVEN (negative
    zero normalized to ``0.000000``). ``stable`` is true exactly when
    every delta is zero and ``recommend`` is true exactly when the
    candidate ranks first in every snapshot observation and every
    evidence value is true.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``candidates,
    recommend``. Candidates are in ascending key order and each object
    uses the key order ``key, direction, support, stable, recommend``;
    ``stable`` and ``recommend`` render as JSON booleans. The top-level
    ``recommend`` is the ascending list of recommended candidate keys.
    """
    if not isinstance(base, str):
        raise TypeError("base must be a str")
    if not isinstance(snapshots, dict):
        raise TypeError("snapshots must be a dict")
    if not isinstance(evidence, dict):
        raise TypeError("evidence must be a dict")
    if not snapshots:
        raise ValueError("snapshots must be a non-empty dict")
    for name in snapshots:
        if not isinstance(name, str) or not name:
            raise ValueError(
                "each snapshots key must be a non-empty string"
            )

    base_groups = _decision_priority_matrix_groups(base)
    snapshot_groups: dict[str, dict] = {}
    candidate_keys: set[str] = set()
    if base_groups:
        candidate_keys = set(next(iter(base_groups.values())))
    for name in sorted(snapshots):
        groups = _decision_priority_matrix_groups(snapshots[name])
        if set(groups) != set(base_groups):
            raise ValueError(
                "each snapshot must share the baseline's (by, key) groups"
            )
        for group_key, ranks in groups.items():
            if set(ranks) != set(base_groups[group_key]):
                raise ValueError(
                    "each snapshot group must share the baseline's "
                    "candidate keys"
                )
        snapshot_groups[name] = groups

    expected_evidence = {
        (name, by, group_key, candidate_key)
        for name in snapshot_groups
        for by, group_key in base_groups
        for candidate_key in candidate_keys
    }
    if set(evidence) != expected_evidence:
        raise ValueError(
            "evidence keys must be the full (snapshot, by, group, "
            "candidate) set with no missing or extra tuples"
        )
    for value in evidence.values():
        if not isinstance(value, bool):
            raise ValueError("each evidence value must be a bool")

    observations = len(snapshot_groups) * len(base_groups)
    candidate_items: list[str] = []
    recommended: list[str] = []
    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN
        for candidate_key in sorted(candidate_keys):
            negative = 0
            positive = 0
            all_zero = True
            all_first = True
            true_evidence = 0
            all_evidence = True
            for name in sorted(snapshot_groups):
                groups = snapshot_groups[name]
                for group_id in sorted(groups):
                    by, group_key = group_id
                    delta = (
                        groups[group_id][candidate_key]
                        - base_groups[group_id][candidate_key]
                    )
                    if delta < 0:
                        negative += 1
                        all_zero = False
                    elif delta > 0:
                        positive += 1
                        all_zero = False
                    if groups[group_id][candidate_key] != 1:
                        all_first = False
                    if evidence[(name, by, group_key, candidate_key)]:
                        true_evidence += 1
                    else:
                        all_evidence = False
            if negative and not positive:
                direction = "rise"
            elif positive and not negative:
                direction = "fall"
            elif all_zero:
                direction = "same"
            else:
                direction = "mixed"
            stable = all_zero
            is_recommended = observations > 0 and all_first and all_evidence
            if is_recommended:
                recommended.append(candidate_key)
            support = (
                Decimal(true_evidence) / Decimal(observations)
                if observations
                else Decimal(0)
            )
            candidate_items.append(
                '{"key":'
                + json.dumps(candidate_key, ensure_ascii=False)
                + ',"direction":' + json.dumps(direction)
                + ',"support":' + _format6(support)
                + ',"stable":' + ("true" if stable else "false")
                + ',"recommend":'
                + ("true" if is_recommended else "false")
                + '}'
            )

    return (
        '{"candidates":[' + ",".join(candidate_items) + ']'
        + ',"recommend":'
        + json.dumps(recommended, ensure_ascii=False, separators=(",", ":"))
        + "}\n"
    )


_CONSENSUS_ERROR = "report must be a decision_priority_consensus JSON output"
_CONSENSUS_SUPPORT_RE = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{6}")


class _ConsensusSupport(Decimal):
    """Marker for a canonical fixed-six-decimal support token."""


def _consensus_parse_int(value: str) -> Decimal:
    raise ValueError(_CONSENSUS_ERROR)


def _consensus_parse_constant(value: str) -> Decimal:
    raise ValueError(_CONSENSUS_ERROR)


def _consensus_candidates(raw: object) -> list[tuple[str, Decimal]]:
    """Parse one :func:`decision_priority_consensus` JSON output.

    Returns ``(key, support)`` pairs in ascending key order. The input
    must be byte-for-byte identical to a canonical output (key order,
    escaping, spacing and the six-decimal support tokens included);
    any deviation raises ``ValueError``.
    """
    if not isinstance(raw, str):
        raise ValueError(_CONSENSUS_ERROR)
    if not raw.endswith("\n") or raw.endswith("\n\n"):
        raise ValueError(_CONSENSUS_ERROR)
    payload = raw[:-1]

    def _parse_support(value: str) -> Decimal:
        if not _CONSENSUS_SUPPORT_RE.fullmatch(value):
            raise ValueError(_CONSENSUS_ERROR)
        return _ConsensusSupport(value)

    try:
        data = json.loads(
            payload,
            parse_float=_parse_support,
            parse_int=_consensus_parse_int,
            parse_constant=_consensus_parse_constant,
        )
    except ValueError:
        raise ValueError(_CONSENSUS_ERROR) from None
    if not isinstance(data, dict) or set(data) != {"candidates", "recommend"}:
        raise ValueError(_CONSENSUS_ERROR)
    candidates_raw = data["candidates"]
    recommended_raw = data["recommend"]
    if not isinstance(candidates_raw, list) or not isinstance(
        recommended_raw, list
    ):
        raise ValueError(_CONSENSUS_ERROR)

    pairs: list[tuple[str, Decimal]] = []
    seen: set[str] = set()
    candidate_recommend: set[str] = set()
    last_key: str | None = None
    candidate_tokens: list[str] = []
    for item in candidates_raw:
        if not isinstance(item, dict) or set(item) != {
            "key", "direction", "support", "stable", "recommend",
        }:
            raise ValueError(_CONSENSUS_ERROR)
        key = item["key"]
        direction = item["direction"]
        support = item["support"]
        stable = item["stable"]
        recommend = item["recommend"]
        if (
            not isinstance(key, str)
            or not key
            or direction not in ("rise", "fall", "same", "mixed")
            or not isinstance(support, _ConsensusSupport)
            or not isinstance(stable, bool)
            or not isinstance(recommend, bool)
        ):
            raise ValueError(_CONSENSUS_ERROR)
        if key in seen or (last_key is not None and key <= last_key):
            raise ValueError(_CONSENSUS_ERROR)
        seen.add(key)
        last_key = key
        if recommend:
            candidate_recommend.add(key)
        # Support is a proportion in [0, 1]; negative zero never serializes.
        if support < 0 or support > 1:
            raise ValueError(_CONSENSUS_ERROR)
        pairs.append((key, support))
        candidate_tokens.append(
            '{"key":'
            + json.dumps(key, ensure_ascii=False)
            + ',"direction":' + json.dumps(direction, ensure_ascii=False)
            + ',"support":' + str(support)
            + ',"stable":' + ("true" if stable else "false")
            + ',"recommend":' + ("true" if recommend else "false")
            + "}"
        )

    recommended: set[str] = set()
    recommended_keys: list[str] = []
    last_recommended: str | None = None
    for key in recommended_raw:
        if not isinstance(key, str) or not key or key not in seen:
            raise ValueError(_CONSENSUS_ERROR)
        if key in recommended or (
            last_recommended is not None and key <= last_recommended
        ):
            raise ValueError(_CONSENSUS_ERROR)
        recommended.add(key)
        recommended_keys.append(key)
        last_recommended = key
    # The top-level list must match the per-candidate recommend flags.
    if recommended != candidate_recommend:
        raise ValueError(_CONSENSUS_ERROR)

    # Structural validation alone accepts equivalent re-serializations
    # (whitespace, escaping, key order); the payload must reproduce the
    # canonical output byte-for-byte.
    canonical = (
        '{"candidates":[' + ",".join(candidate_tokens) + '],"recommend":'
        + json.dumps(
            recommended_keys, ensure_ascii=False, separators=(",", ":")
        )
        + "}"
    )
    if payload != canonical:
        raise ValueError(_CONSENSUS_ERROR)
    return pairs


def _portfolio_number(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite int or float")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return Decimal(str(value))


def portfolio(
    report: str, cost: dict, gain: dict, limit: float, rules: dict
) -> str:
    """Select the best candidate subset from a consensus report.

    ``report`` must be a :func:`decision_priority_consensus` JSON output;
    ``cost`` and ``gain`` are dicts whose key sets are exactly the
    report's candidate keys, with finite non-boolean int/float values
    (strictly positive costs, non-negative gains); ``limit`` is a
    non-negative finite non-boolean int/float; and ``rules`` has exactly
    the keys ``"mutex"`` and ``"requires"``, each a list of two-tuples of
    distinct candidate keys. Mutex pairs are undirected and
    deduplicated; a ``requires`` pair ``(a, b)`` means selecting ``a``
    forces ``b`` to be selected and pairs are deduplicated in order. A
    non-string ``report`` or a non-dict ``cost``, ``gain`` or ``rules``
    raises ``TypeError``; every other contract violation raises
    ``ValueError``.

    Each candidate's score is ``gain * support`` where ``support`` is
    the report value, computed as ``Decimal(str(x))`` under a
    precision-1000, ROUND_HALF_EVEN context. Subsets are enumerated
    exhaustively, keeping those whose total cost is at most ``limit``,
    which never select both ends of a mutex pair and which are closed
    under every ``requires`` edge. The winner maximizes total score,
    then minimizes total cost, then minimizes the lexicographically
    ascending selected-key list.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the key order is ``cost, score, pick, skip``.
    ``cost`` and ``score`` are the selected totals rendered with six
    decimals (negative zero normalized to ``0.000000``); ``pick`` is the
    ascending selected-key array and ``skip`` is an ascending
    ``[key, reason]`` array. Each unselected candidate is tagged, in
    priority order, ``"requires"`` (a required candidate is absent),
    ``"mutex"`` (it conflicts with a selection), ``"budget"`` (adding
    it exceeds the limit) or ``"dominated"`` (otherwise).
    """
    if not isinstance(report, str):
        raise TypeError("report must be a str")
    if not isinstance(cost, dict):
        raise TypeError("cost must be a dict")
    if not isinstance(gain, dict):
        raise TypeError("gain must be a dict")
    if not isinstance(rules, dict):
        raise TypeError("rules must be a dict")

    pairs = _consensus_candidates(report)
    keys = [key for key, _ in pairs]
    supports = [support for _, support in pairs]
    candidate_set = set(keys)
    if set(cost) != candidate_set:
        raise ValueError("cost keys must be exactly the candidate keys")
    if set(gain) != candidate_set:
        raise ValueError("gain keys must be exactly the candidate keys")
    limit_value = _portfolio_number(limit, "limit")
    if limit_value < 0:
        raise ValueError("limit must be non-negative")
    if set(rules) != {"mutex", "requires"}:
        raise ValueError("rules keys must be exactly 'mutex' and 'requires'")
    mutex_raw = rules["mutex"]
    requires_raw = rules["requires"]
    if not isinstance(mutex_raw, list) or not isinstance(requires_raw, list):
        raise ValueError("rules values must be lists")

    index = {key: i for i, key in enumerate(keys)}
    costs: list[Decimal] = []
    gains: list[Decimal] = []
    for key in keys:
        cost_value = _portfolio_number(cost[key], "cost")
        gain_value = _portfolio_number(gain[key], "gain")
        if cost_value <= 0:
            raise ValueError("each cost must be positive")
        if gain_value < 0:
            raise ValueError("each gain must be non-negative")
        costs.append(cost_value)
        gains.append(gain_value)

    def _rule_pair(item: object) -> tuple[int, int]:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("each rule must be a two-tuple of candidate keys")
        left, right = item
        if (
            not isinstance(left, str)
            or not isinstance(right, str)
            or left not in candidate_set
            or right not in candidate_set
            or left == right
        ):
            raise ValueError(
                "each rule must pair two distinct candidate keys"
            )
        return index[left], index[right]

    mutex_edges: set[frozenset[int]] = set()
    for item in mutex_raw:
        edge = frozenset(_rule_pair(item))
        if edge in mutex_edges:
            raise ValueError("mutex pairs must not repeat")
        mutex_edges.add(edge)

    requires_edges: set[tuple[int, int]] = set()
    for item in requires_raw:
        pair = _rule_pair(item)
        if pair in requires_edges:
            raise ValueError("requires pairs must not repeat")
        requires_edges.add(pair)

    n = len(keys)
    requires_by_key: list[list[int]] = [[] for _ in range(n)]
    for left, right in requires_edges:
        requires_by_key[left].append(right)
    mutex_neighbors: list[set[int]] = [set() for _ in range(n)]
    for edge in mutex_edges:
        left, right = tuple(edge)
        mutex_neighbors[left].add(right)
        mutex_neighbors[right].add(left)

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN
        scores = [gains[i] * supports[i] for i in range(n)]

        def _closure(chosen: frozenset) -> frozenset | None:
            closure = set(chosen)
            while True:
                added = {
                    required
                    for member in closure
                    for required in requires_by_key[member]
                    if required not in closure
                }
                if not added:
                    break
                closure |= added
            if sum((costs[i] for i in closure), Decimal(0)) > limit_value:
                return None
            for edge in mutex_edges:
                left, right = tuple(edge)
                if left in closure and right in closure:
                    return None
            return frozenset(closure)

        best: tuple[Decimal, Decimal, tuple[str, ...], frozenset] | None = None
        for size in range(n + 1):
            for idxs in combinations(range(n), size):
                closure = _closure(frozenset(idxs))
                if closure is None:
                    continue
                total_score = sum((scores[i] for i in closure), Decimal(0))
                total_cost = sum((costs[i] for i in closure), Decimal(0))
                chosen_keys = tuple(keys[i] for i in sorted(closure))
                if (
                    best is None
                    or total_score > best[0]
                    or (
                        total_score == best[0]
                        and (
                            total_cost < best[1]
                            or (
                                total_cost == best[1]
                                and chosen_keys < best[2]
                            )
                        )
                    )
                ):
                    best = (total_score, total_cost, chosen_keys, closure)

        assert best is not None  # the empty subset is always feasible
        total_score, total_cost, chosen_keys, chosen = best
        chosen_indexes = set(chosen)

        skip_items: list[str] = []
        for i, key in enumerate(keys):
            if i in chosen_indexes:
                continue
            if any(
                required not in chosen_indexes
                for required in requires_by_key[i]
            ):
                reason = "requires"
            elif mutex_neighbors[i] & chosen_indexes:
                reason = "mutex"
            elif total_cost + costs[i] > limit_value:
                reason = "budget"
            else:
                reason = "dominated"
            skip_items.append(
                "["
                + json.dumps(key, ensure_ascii=False)
                + ","
                + json.dumps(reason, ensure_ascii=False)
                + "]"
            )

    pick_json = json.dumps(
        list(chosen_keys), ensure_ascii=False, separators=(",", ":")
    )
    return (
        '{"cost":' + _format6(total_cost)
        + ',"score":' + _format6(total_score)
        + ',"pick":' + pick_json
        + ',"skip":[' + ",".join(skip_items) + "]}\n"
    )


def _portfolio_rule_pairs(
    rules: object, candidate_set: set[str], index: dict[str, int]
) -> tuple[set[frozenset[int]], set[tuple[int, int]]]:
    """Validate a ``rules`` mapping as in :func:`portfolio`."""
    if not isinstance(rules, dict):
        raise TypeError("rules must be a dict")
    if set(rules) != {"mutex", "requires"}:
        raise ValueError("rules keys must be exactly 'mutex' and 'requires'")
    mutex_raw = rules["mutex"]
    requires_raw = rules["requires"]
    if not isinstance(mutex_raw, list) or not isinstance(requires_raw, list):
        raise ValueError("rules values must be lists")

    def _pair(item: object) -> tuple[int, int]:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("each rule must be a two-tuple of candidate keys")
        left, right = item
        if (
            not isinstance(left, str)
            or not isinstance(right, str)
            or left not in candidate_set
            or right not in candidate_set
            or left == right
        ):
            raise ValueError("each rule must pair two distinct candidate keys")
        return index[left], index[right]

    mutex_edges: set[frozenset[int]] = set()
    for item in mutex_raw:
        edge = frozenset(_pair(item))
        if edge in mutex_edges:
            raise ValueError("mutex pairs must not repeat")
        mutex_edges.add(edge)

    requires_edges: set[tuple[int, int]] = set()
    for item in requires_raw:
        pair = _pair(item)
        if pair in requires_edges:
            raise ValueError("requires pairs must not repeat")
        requires_edges.add(pair)
    return mutex_edges, requires_edges


def _portfolio_scenario_data(
    reports: dict, weights: dict, gains: dict
) -> tuple[
    list[str],
    list[str],
    list[list[Decimal]],
    list[Decimal],
    Decimal,
    list[list[Decimal]],
    set[str],
]:
    """Validate the shared ``reports``/``weights``/``gains`` contract of
    :func:`portfolio_robustness` and :func:`portfolio_attribution`.

    Returns ``(scenario_names, keys, scenario_supports, weight_values,
    total_weight, gains_by_scenario, candidate_set)`` with scenarios in
    ascending name order; ``keys`` is the shared candidate key list in
    ascending order and every per-scenario row aligns with it.
    """
    if not isinstance(reports, dict):
        raise TypeError("reports must be a dict")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")
    if not isinstance(gains, dict):
        raise TypeError("gains must be a dict")

    if len(reports) < 2:
        raise ValueError("reports must contain at least two scenarios")
    scenario_names: list[str] = []
    for name in reports:
        if not isinstance(name, str) or not name:
            raise ValueError("each reports key must be a non-empty string")
        scenario_names.append(name)
    scenario_names.sort()

    scenario_keys: list[list[str]] = []
    scenario_supports: list[list[Decimal]] = []
    candidate_set: set[str] | None = None
    for name in scenario_names:
        pairs = _consensus_candidates(reports[name])
        keys = [key for key, _ in pairs]
        if candidate_set is None:
            candidate_set = set(keys)
        elif set(keys) != candidate_set:
            raise ValueError("every report must share the same candidate keys")
        scenario_keys.append(keys)
        scenario_supports.append([support for _, support in pairs])
    assert candidate_set is not None

    if set(weights) != set(scenario_names):
        raise ValueError("weights keys must be exactly the scenario keys")
    weight_values: list[Decimal] = []
    total_weight = Decimal(0)
    for name in scenario_names:
        weight_value = _portfolio_number(weights[name], "weight")
        if weight_value <= 0:
            raise ValueError("each weight must be positive")
        weight_values.append(weight_value)
        total_weight += weight_value

    if set(gains) != set(scenario_names):
        raise ValueError("gains keys must be exactly the scenario keys")
    gains_by_scenario: list[list[Decimal]] = []
    for s, name in enumerate(scenario_names):
        scenario_gain = gains[name]
        if not isinstance(scenario_gain, dict):
            raise TypeError("each gains value must be a dict")
        if set(scenario_gain) != candidate_set:
            raise ValueError(
                "each gains dict keys must be exactly the candidate keys"
            )
        keys = scenario_keys[s]
        row: list[Decimal] = []
        for key in keys:
            gain_value = _portfolio_number(scenario_gain[key], "gain")
            if gain_value < 0:
                raise ValueError("each gain must be non-negative")
            row.append(gain_value)
        gains_by_scenario.append(row)

    return (
        scenario_names,
        scenario_keys[0],
        scenario_supports,
        weight_values,
        total_weight,
        gains_by_scenario,
        candidate_set,
    )


def portfolio_robustness(
    reports: dict,
    weights: dict,
    gains: dict,
    cost: dict,
    limit: float,
    rules: dict,
) -> str:
    """Rank portfolio selections across weighted consensus scenarios.

    ``reports`` must be a dict with at least two entries mapping
    non-empty string scenario names to
    :func:`decision_priority_consensus` JSON outputs whose candidate key
    sets are all identical. ``weights`` is a dict with exactly those
    scenario keys and strictly positive finite non-boolean int/float
    values. ``gains`` is a dict with exactly those scenario keys, each
    value a dict whose keys are exactly the shared candidate set and
    whose values are non-negative finite non-boolean int/float values.
    ``cost`` is a single candidate-keyed dict of strictly positive
    finite non-boolean int/float values shared by every scenario;
    ``limit`` and ``rules`` follow the :func:`portfolio` contract.

    For each scenario ``s`` and candidate ``i`` the per-candidate value
    is ``v_s_i = gain[s][i] * support[s][i]``. A feasible subset is one
    that is closed under every ``requires`` edge, never selects both
    ends of a mutex pair and has total cost at most ``limit``. For each
    feasible subset the scenario value is ``v_s = sum_i v_s_i`` and the
    summary statistics are ``expected = sum_s weight_s * v_s / sum_s
    weight_s``, ``worst = min_s v_s`` and
    ``sensitivity = max_s v_s - worst``; all arithmetic is
    ``Decimal(str(x))`` under a precision-1000, ROUND_HALF_EVEN context.
    Subsets rank by ``expected`` descending, then ``worst`` descending,
    then ``sensitivity`` ascending, then total cost ascending, then the
    selected-key tuple lexicographically ascending.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``total_weight,
    portfolios``. Each portfolio object uses the key order
    ``rank, pick, cost, expected, worst, sensitivity, scores``; ``rank``
    is an integer starting at 1, ``pick`` is the ascending selected-key
    array and ``scores`` lists ``{"key": ..., "score": ...}`` objects in
    ascending scenario-name order. Every numeric value except ``rank``
    renders with six decimals, negative zero normalized to
    ``0.000000``. A non-dict ``reports``, ``weights``, ``gains``,
    ``cost`` or ``rules``, or a non-dict per-scenario ``gains`` value,
    raises ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    if not isinstance(reports, dict):
        raise TypeError("reports must be a dict")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")
    if not isinstance(gains, dict):
        raise TypeError("gains must be a dict")
    if not isinstance(cost, dict):
        raise TypeError("cost must be a dict")

    (
        scenario_names,
        keys,
        scenario_supports,
        weight_values,
        total_weight,
        gains_by_scenario,
        candidate_set,
    ) = _portfolio_scenario_data(reports, weights, gains)

    if set(cost) != candidate_set:
        raise ValueError("cost keys must be exactly the candidate keys")
    costs: list[Decimal] = []
    for key in keys:
        cost_value = _portfolio_number(cost[key], "cost")
        if cost_value <= 0:
            raise ValueError("each cost must be positive")
        costs.append(cost_value)

    limit_value = _portfolio_number(limit, "limit")
    if limit_value < 0:
        raise ValueError("limit must be non-negative")

    index = {key: i for i, key in enumerate(keys)}
    mutex_edges, requires_edges = _portfolio_rule_pairs(
        rules, candidate_set, index
    )
    n = len(keys)
    requires_by_key: list[list[int]] = [[] for _ in range(n)]
    for left, right in requires_edges:
        requires_by_key[left].append(right)

    m = len(scenario_names)
    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        scenario_values = [
            [
                gains_by_scenario[s][i] * scenario_supports[s][i]
                for i in range(n)
            ]
            for s in range(m)
        ]

        def _closure(chosen: frozenset) -> frozenset | None:
            closure = set(chosen)
            while True:
                added = {
                    required
                    for member in closure
                    for required in requires_by_key[member]
                    if required not in closure
                }
                if not added:
                    break
                closure |= added
            if sum((costs[i] for i in closure), Decimal(0)) > limit_value:
                return None
            for edge in mutex_edges:
                left, right = tuple(edge)
                if left in closure and right in closure:
                    return None
            return frozenset(closure)

        records: list[
            tuple[
                Decimal,
                Decimal,
                Decimal,
                Decimal,
                tuple[str, ...],
                tuple[Decimal, ...],
            ]
        ] = []
        seen_closures: set[frozenset] = set()
        for size in range(n + 1):
            for idxs in combinations(range(n), size):
                closure = _closure(frozenset(idxs))
                if closure is None or closure in seen_closures:
                    continue
                seen_closures.add(closure)
                values = tuple(
                    sum(
                        (scenario_values[s][i] for i in closure),
                        Decimal(0),
                    )
                    for s in range(m)
                )
                total_cost = sum((costs[i] for i in closure), Decimal(0))
                weighted = Decimal(0)
                for s in range(m):
                    weighted += weight_values[s] * values[s]
                expected = weighted / total_weight
                worst = min(values)
                sensitivity = max(values) - worst
                chosen_keys = tuple(keys[i] for i in sorted(closure))
                records.append(
                    (
                        expected,
                        worst,
                        sensitivity,
                        total_cost,
                        chosen_keys,
                        values,
                    )
                )

        records.sort(
            key=lambda record: (
                -record[0],
                -record[1],
                record[2],
                record[3],
                record[4],
            )
        )

        portfolio_items: list[str] = []
        for rank, record in enumerate(records, start=1):
            expected, worst, sensitivity, total_cost, chosen_keys, values = (
                record
            )
            score_items = [
                '{"key":'
                + json.dumps(scenario_names[s], ensure_ascii=False)
                + ',"score":'
                + _format6(values[s])
                + "}"
                for s in range(m)
            ]
            portfolio_items.append(
                '{"rank":' + str(rank)
                + ',"pick":'
                + json.dumps(
                    list(chosen_keys),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + ',"cost":' + _format6(total_cost)
                + ',"expected":' + _format6(expected)
                + ',"worst":' + _format6(worst)
                + ',"sensitivity":' + _format6(sensitivity)
                + ',"scores":[' + ",".join(score_items) + "]}"
            )

    return (
        '{"total_weight":' + _format6(total_weight)
        + ',"portfolios":[' + ",".join(portfolio_items) + "]}\n"
    )


def portfolio_attribution(
    reports: dict,
    weights: dict,
    gains: dict,
    pick: list,
    *,
    alpha: float = 0.05,
) -> str:
    """Attribute picked candidates across weighted consensus scenarios.

    ``reports``, ``weights`` and ``gains`` follow the
    :func:`portfolio_robustness` contract; the scenario count ``n`` must
    additionally be at most 16. ``pick`` is a list of candidate keys
    without duplicates and may be empty. ``alpha`` must be a finite
    non-boolean number with ``0 < alpha <= 1``.

    For each picked candidate and scenario ``s`` the per-scenario value
    is ``v_s = gain[s][key] * support[s][key]``. With
    ``W = sum_s weight_s`` the summary statistics are
    ``expected = sum_s weight_s * v_s / W``, ``worst = min_s v_s`` and
    ``sensitivity = max_s v_s - worst``. The p-value enumerates all
    ``2**n`` sign vectors: ``p`` is the proportion of vectors with
    ``|sum_s sign_s * weight_s * v_s / W| >= |expected|``. All picked
    candidates are ranked ascending by ``(p, key)`` and each rank ``j``
    (1-based, of ``N``) gets the Benjamini-Hochberg q-value
    ``q_j = min(1, min(N * p_l / l for l in j..N))``; ``reject`` is
    ``q <= alpha``, compared on the unquantized values. All arithmetic
    is ``Decimal(str(x))`` under a precision-1000, ROUND_HALF_EVEN
    context.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``alpha, total_weight,
    items``. Each item uses the key order ``key, n, expected, worst,
    sensitivity, p, q, reject`` and items are sorted by ``key``
    ascending; ``n`` is the integer scenario count, ``reject`` a
    boolean and every other numeric value renders with six decimals,
    negative zero normalized to ``0.000000``. An empty ``pick`` yields
    ``"items":[]``. A non-dict ``reports``, ``weights`` or ``gains``, a
    non-dict per-scenario ``gains`` value or a non-list ``pick`` raises
    ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    (
        scenario_names,
        keys,
        scenario_supports,
        weight_values,
        total_weight,
        gains_by_scenario,
        candidate_set,
    ) = _portfolio_scenario_data(reports, weights, gains)

    if not isinstance(pick, list):
        raise TypeError("pick must be a list")

    n = len(scenario_names)
    if n > 16:
        raise ValueError("reports must contain at most sixteen scenarios")

    seen_pick: set[str] = set()
    for key in pick:
        if not isinstance(key, str) or key not in candidate_set:
            raise ValueError("each pick entry must be a candidate key")
        if key in seen_pick:
            raise ValueError("pick entries must not repeat")
        seen_pick.add(key)

    alpha_value = _validate_finite_number(alpha, "alpha")
    if alpha_value <= 0 or alpha_value > 1:
        raise ValueError("alpha must be greater than 0 and at most 1")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        index = {key: i for i, key in enumerate(keys)}
        sign_count = 1 << n
        # One record per picked candidate: ``[key, expected, worst,
        # sensitivity, p, q]`` with q filled in below.
        records: list[list] = []
        for key in pick:
            i = index[key]
            terms = [
                weight_values[s]
                * gains_by_scenario[s][i]
                * scenario_supports[s][i]
                for s in range(n)
            ]
            values = [
                gains_by_scenario[s][i] * scenario_supports[s][i]
                for s in range(n)
            ]
            expected = sum(terms, Decimal(0)) / total_weight
            worst = min(values)
            sensitivity = max(values) - worst
            threshold = abs(sum(terms, Decimal(0)))
            hits = 0
            for mask in range(sign_count):
                statistic = Decimal(0)
                for s in range(n):
                    if mask >> s & 1:
                        statistic += terms[s]
                    else:
                        statistic -= terms[s]
                if abs(statistic) >= threshold:
                    hits += 1
            p_value = Decimal(hits) / Decimal(sign_count)
            records.append([key, expected, worst, sensitivity, p_value, None])

        # Benjamini-Hochberg q-values across all picked candidates: rank
        # ascending by (p, key), then accumulate the running minimum of
        # N * p_l / l from the top rank down.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (records[idx][4], records[idx][0]),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate = Decimal(count) * records[idx][4] / rank
            if candidate < running:
                running = candidate
            records[idx][5] = running

        records.sort(key=lambda record: record[0])
        items: list[str] = []
        for key, expected, worst, sensitivity, p_value, q_value in records:
            items.append(
                '{"key":' + json.dumps(key, ensure_ascii=False)
                + ',"n":' + str(n)
                + ',"expected":' + _format6(expected)
                + ',"worst":' + _format6(worst)
                + ',"sensitivity":' + _format6(sensitivity)
                + ',"p":' + _format6(p_value)
                + ',"q":' + _format6(q_value)
                + ',"reject":'
                + ("true" if q_value <= alpha_value else "false")
                + "}"
            )

    return (
        '{"alpha":' + _format6(alpha_value)
        + ',"total_weight":' + _format6(total_weight)
        + ',"items":[' + ",".join(items) + "]}\n"
    )


_PANEL_ATTRIBUTION_ERROR = (
    "each reports value must be a portfolio_attribution JSON output"
)
_PANEL_ATTRIBUTION_INT_RE = re.compile(r"(?:0|[1-9][0-9]*)")
_PANEL_ATTRIBUTION_DECIMAL_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)\.[0-9]{6}"
)


class _PanelAttributionInt(Decimal):
    """Marker for a canonical unsigned JSON integer token."""


class _PanelAttributionDecimal(Decimal):
    """Marker for a canonical fixed-six-decimal JSON number token."""


def _panel_attribution_parse_constant(value: str) -> Decimal:
    raise ValueError(_PANEL_ATTRIBUTION_ERROR)


def _panel_attribution_signed_zero(value: Decimal) -> bool:
    return value == 0 and value.is_signed()


def _panel_attribution_items(raw: object) -> list[tuple[str, Decimal, bool]]:
    """Parse one :func:`portfolio_attribution` JSON output.

    Returns ``(key, expected, reject)`` triples in the report's item
    order. The input must be byte-for-byte identical to a canonical
    output (key order, escaping, spacing, the integer ``n`` token and
    the six-decimal numeric tokens included); any deviation raises
    ``ValueError``.
    """
    if not isinstance(raw, str):
        raise ValueError(_PANEL_ATTRIBUTION_ERROR)
    if not raw.endswith("\n") or raw.endswith("\n\n"):
        raise ValueError(_PANEL_ATTRIBUTION_ERROR)
    payload = raw[:-1]

    def _parse_integer(value: str) -> Decimal:
        if not _PANEL_ATTRIBUTION_INT_RE.fullmatch(value):
            raise ValueError(_PANEL_ATTRIBUTION_ERROR)
        return _PanelAttributionInt(value)

    def _parse_decimal(value: str) -> Decimal:
        if not _PANEL_ATTRIBUTION_DECIMAL_RE.fullmatch(value):
            raise ValueError(_PANEL_ATTRIBUTION_ERROR)
        return _PanelAttributionDecimal(value)

    try:
        data = json.loads(
            payload,
            parse_float=_parse_decimal,
            parse_int=_parse_integer,
            parse_constant=_panel_attribution_parse_constant,
        )
    except ValueError:
        raise ValueError(_PANEL_ATTRIBUTION_ERROR) from None
    if not isinstance(data, dict) or set(data) != {
        "alpha", "total_weight", "items",
    }:
        raise ValueError(_PANEL_ATTRIBUTION_ERROR)
    alpha_raw = data["alpha"]
    total_weight_raw = data["total_weight"]
    items_raw = data["items"]
    if (
        not isinstance(alpha_raw, _PanelAttributionDecimal)
        or not isinstance(total_weight_raw, _PanelAttributionDecimal)
        or not isinstance(items_raw, list)
    ):
        raise ValueError(_PANEL_ATTRIBUTION_ERROR)
    # Canonical values are always non-negative (weights/gains are), the
    # fixed-six serializer never emits negative zero and alpha is in
    # (0, 1] while total_weight is a sum of positive weights.
    if (
        alpha_raw <= 0
        or alpha_raw > 1
        or total_weight_raw <= 0
        or _panel_attribution_signed_zero(alpha_raw)
        or _panel_attribution_signed_zero(total_weight_raw)
    ):
        raise ValueError(_PANEL_ATTRIBUTION_ERROR)

    triples: list[tuple[str, Decimal, bool]] = []
    item_tokens: list[str] = []
    seen: set[str] = set()
    last_key: str | None = None
    n_value: _PanelAttributionInt | None = None
    for item in items_raw:
        if not isinstance(item, dict) or set(item) != {
            "key", "n", "expected", "worst", "sensitivity", "p", "q",
            "reject",
        }:
            raise ValueError(_PANEL_ATTRIBUTION_ERROR)
        key = item["key"]
        n_item = item["n"]
        expected = item["expected"]
        worst = item["worst"]
        sensitivity = item["sensitivity"]
        p_value = item["p"]
        q_value = item["q"]
        reject = item["reject"]
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(n_item, _PanelAttributionInt)
            or not isinstance(expected, _PanelAttributionDecimal)
            or not isinstance(worst, _PanelAttributionDecimal)
            or not isinstance(sensitivity, _PanelAttributionDecimal)
            or not isinstance(p_value, _PanelAttributionDecimal)
            or not isinstance(q_value, _PanelAttributionDecimal)
            or not isinstance(reject, bool)
        ):
            raise ValueError(_PANEL_ATTRIBUTION_ERROR)
        # A canonical report attributes across 2..16 scenarios.
        if n_item < 2 or n_item > 16:
            raise ValueError(_PANEL_ATTRIBUTION_ERROR)
        if (
            expected < 0
            or worst < 0
            or sensitivity < 0
            or p_value < 0
            or p_value > 1
            or q_value < 0
            or q_value > 1
            or _panel_attribution_signed_zero(expected)
            or _panel_attribution_signed_zero(worst)
            or _panel_attribution_signed_zero(sensitivity)
            or _panel_attribution_signed_zero(p_value)
            or _panel_attribution_signed_zero(q_value)
        ):
            raise ValueError(_PANEL_ATTRIBUTION_ERROR)
        if key in seen or (last_key is not None and key <= last_key):
            raise ValueError(_PANEL_ATTRIBUTION_ERROR)
        seen.add(key)
        last_key = key
        if n_value is None:
            n_value = n_item
        elif n_item != n_value:
            raise ValueError(_PANEL_ATTRIBUTION_ERROR)
        triples.append((key, expected, reject))
        item_tokens.append(
            '{"key":' + json.dumps(key, ensure_ascii=False)
            + ',"n":' + str(n_item)
            + ',"expected":' + str(expected)
            + ',"worst":' + str(worst)
            + ',"sensitivity":' + str(sensitivity)
            + ',"p":' + str(p_value)
            + ',"q":' + str(q_value)
            + ',"reject":' + ("true" if reject else "false")
            + "}"
        )

    # Structural validation alone accepts equivalent re-serializations
    # (whitespace, escaping, key order); the payload must reproduce the
    # canonical output byte-for-byte.
    canonical = (
        '{"alpha":' + str(alpha_raw)
        + ',"total_weight":' + str(total_weight_raw)
        + ',"items":[' + ",".join(item_tokens) + "]}"
    )
    if payload != canonical:
        raise ValueError(_PANEL_ATTRIBUTION_ERROR)
    return triples


_PANEL_GROUP_ORDER = {"region": 0, "window": 1, "overall": 2}


def _panel_report_data(
    reports: dict, weights: dict, alpha: float
) -> tuple[Decimal, list[tuple[str, str, list[int]]], list[list]]:
    """Validate the :func:`panel_report` contract and compute group records.

    Returns ``(alpha_value, groups_spec, records)``: ``groups_spec``
    lists ``(by, group_key, member panel indices)`` in canonical group
    emission order and each record is
    ``[by, group_key, candidate, n, mean, range, stable, p, q]`` with
    unquantized ``Decimal`` statistics and pooled Benjamini-Hochberg
    ``q`` values. See :func:`panel_report` for the full contract.
    """
    if not isinstance(reports, dict):
        raise TypeError("reports must be a dict")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")

    if len(reports) < 2:
        raise ValueError("reports must contain at least two panels")
    panel_keys: list[tuple[str, str]] = []
    for panel in reports:
        if (
            not isinstance(panel, tuple)
            or len(panel) != 2
            or not isinstance(panel[0], str)
            or not panel[0]
            or not isinstance(panel[1], str)
            or not panel[1]
        ):
            raise ValueError(
                "each reports key must be a non-empty (region, window) "
                "string pair"
            )
        panel_keys.append(panel)
    panel_keys.sort()

    if set(weights) != set(panel_keys):
        raise ValueError("weights keys must be exactly the reports keys")
    weight_values: list[Decimal] = []
    total_weight = Decimal(0)
    for panel in panel_keys:
        weight_value = _portfolio_number(weights[panel], "weight")
        if weight_value <= 0:
            raise ValueError("each weight must be positive")
        weight_values.append(weight_value)
        total_weight += weight_value

    alpha_value = _validate_finite_number(alpha, "alpha")
    if alpha_value <= 0 or alpha_value > 1:
        raise ValueError("alpha must be greater than 0 and at most 1")

    panel_items: list[list[tuple[str, Decimal, bool]]] = []
    candidate_set: set[str] | None = None
    for panel in panel_keys:
        triples = _panel_attribution_items(reports[panel])
        keys = {key for key, _, _ in triples}
        if candidate_set is None:
            candidate_set = keys
        elif keys != candidate_set:
            raise ValueError("every report must share the same candidate keys")
        panel_items.append(triples)
    assert candidate_set is not None

    n = len(panel_keys)
    if n > 16:
        raise ValueError("reports must contain at most sixteen panels")

    candidates = sorted(candidate_set)
    candidate_index = {candidate: i for i, candidate in enumerate(candidates)}
    m = len(candidates)
    # Panel-aligned source expected values and reject flags per candidate.
    expected_rows = [[Decimal(0)] * n for _ in range(m)]
    reject_rows = [[False] * n for _ in range(m)]
    for panel_index, triples in enumerate(panel_items):
        for key, expected, source_reject in triples:
            i = candidate_index[key]
            expected_rows[i][panel_index] = expected
            reject_rows[i][panel_index] = source_reject

    # (by, key, member panel indices) in canonical group emission order.
    groups_spec: list[tuple[str, str, list[int]]] = []
    regions = sorted({panel[0] for panel in panel_keys})
    windows = sorted({panel[1] for panel in panel_keys})
    for region in regions:
        members = [
            j for j, panel in enumerate(panel_keys) if panel[0] == region
        ]
        groups_spec.append(("region", region, members))
    for window in windows:
        members = [
            j for j, panel in enumerate(panel_keys) if panel[1] == window
        ]
        groups_spec.append(("window", window, members))
    groups_spec.append(("overall", "all", list(range(n))))

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        # One record per (group, candidate):
        # ``[by, group_key, candidate, n, mean, range, stable, p, q]``
        # with q filled in by the pooled BH pass below.
        records: list[list] = []
        for by, group_key, members in groups_spec:
            group_n = len(members)
            group_weights = [weight_values[j] for j in members]
            group_weight_total = sum(group_weights, Decimal(0))
            sign_count = 1 << group_n
            for i, candidate in enumerate(candidates):
                values = [expected_rows[i][j] for j in members]
                terms = [
                    group_weights[k] * values[k] for k in range(group_n)
                ]
                mean = sum(terms, Decimal(0)) / group_weight_total
                value_range = max(values) - min(values)
                stable_weight = sum(
                    (
                        group_weights[k]
                        for k in range(group_n)
                        if reject_rows[i][members[k]]
                    ),
                    Decimal(0),
                )
                stable = stable_weight / group_weight_total
                threshold = abs(sum(terms, Decimal(0)))
                hits = 0
                for mask in range(sign_count):
                    statistic = Decimal(0)
                    for k in range(group_n):
                        if mask >> k & 1:
                            statistic += terms[k]
                        else:
                            statistic -= terms[k]
                    if abs(statistic) >= threshold:
                        hits += 1
                p_value = Decimal(hits) / Decimal(sign_count)
                records.append(
                    [
                        by,
                        group_key,
                        candidate,
                        group_n,
                        mean,
                        value_range,
                        stable,
                        p_value,
                        None,
                    ]
                )

        # Pooled Benjamini-Hochberg across every group and candidate:
        # rank by (p, dimension order, group key, candidate), then
        # accumulate the running minimum of N * p_l / l top-down.
        count = len(records)
        ranked = sorted(
            range(count),
            key=lambda idx: (
                records[idx][7],
                _PANEL_GROUP_ORDER[records[idx][0]],
                records[idx][1],
                records[idx][2],
            ),
        )
        running = Decimal(1)
        for rank in range(count, 0, -1):
            idx = ranked[rank - 1]
            candidate_q = Decimal(count) * records[idx][7] / rank
            if candidate_q < running:
                running = candidate_q
            records[idx][8] = running

    return alpha_value, groups_spec, records


def panel_report(
    reports: dict,
    weights: dict,
    *,
    alpha: float = 0.05,
) -> str:
    """Aggregate portfolio attribution reports across a region/window panel.

    ``reports`` must be a dict with 2 to 16 entries mapping non-empty
    ``(region, window)`` string pairs to canonical
    :func:`portfolio_attribution` JSON outputs whose candidate key sets
    are all identical. ``weights`` is a dict with exactly those pair
    keys and strictly positive finite non-boolean int/float values.
    ``alpha`` must be a finite non-boolean number with
    ``0 < alpha <= 1``.

    Candidates are tested within three group dimensions, sorted by
    region, by window and overall across every panel (the overall group
    key is ``"all"``). For a group containing panels with weights
    ``w`` and source per-candidate values ``v`` (each report's
    ``expected``) and ``n`` panels, ``mean = sum(w*v)/sum(w)``,
    ``range = max(v) - min(v)`` and ``stable = sum(w * source reject) /
    sum(w)`` with the source ``reject`` flags taken straight from the
    reports. The p-value enumerates all ``2**n`` sign vectors: ``p`` is
    the proportion of vectors with
    ``|sum_i sign_i * w_i * v_i / sum(w)| >= |mean|``. All tests across
    every group and candidate are pooled for one Benjamini-Hochberg
    pass: tests rank ascending by ``(p, dimension order, group key,
    candidate)`` with dimensions ordered region, window, overall, and
    rank ``j`` (1-based, of ``N``) gets
    ``q_j = min(1, min(N * p_l / l for l in j..N))``; ``reject`` is
    ``q <= alpha``, compared on the unquantized values. All arithmetic
    is ``Decimal(str(x))`` under a precision-1000, ROUND_HALF_EVEN
    context.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``alpha, groups``.
    Each group object uses the key order ``by, key, items``, groups
    appear as all region groups (key ascending), then all window groups
    (key ascending), then the single overall group, and each item uses
    the key order ``key, n, mean, range, stable, p, q, reject`` with
    items in ascending candidate order. ``n`` is an integer and
    ``reject`` a boolean; every other numeric value renders with six
    decimals, negative zero normalized to ``0.000000``, and Unicode is
    preserved. A non-dict ``reports`` or ``weights`` raises
    ``TypeError``; every other contract violation raises
    ``ValueError``.
    """
    alpha_value, groups_spec, records = _panel_report_data(
        reports, weights, alpha
    )

    group_strings: list[str] = []
    for by, group_key, _members in groups_spec:
        group_records = [
            record
            for record in records
            if record[0] == by and record[1] == group_key
        ]
        group_records.sort(key=lambda record: record[2])
        item_strings: list[str] = []
        for (
            _by,
            _group_key,
            candidate,
            group_n,
            mean,
            value_range,
            stable,
            p_value,
            q_value,
        ) in group_records:
            item_strings.append(
                '{"key":' + json.dumps(candidate, ensure_ascii=False)
                + ',"n":' + str(group_n)
                + ',"mean":' + _format6(mean)
                + ',"range":' + _format6(value_range)
                + ',"stable":' + _format6(stable)
                + ',"p":' + _format6(p_value)
                + ',"q":' + _format6(q_value)
                + ',"reject":'
                + ("true" if q_value <= alpha_value else "false")
                + "}"
            )
        group_strings.append(
            '{"by":' + json.dumps(by, ensure_ascii=False)
            + ',"key":' + json.dumps(group_key, ensure_ascii=False)
            + ',"items":[' + ",".join(item_strings) + "]}"
        )

    return (
        '{"alpha":' + _format6(alpha_value)
        + ',"groups":[' + ",".join(group_strings) + "]}\n"
    )


def panel_priority(
    reports: dict,
    weights: dict,
    cost: dict,
    limit: float,
    *,
    alpha: float = 0.05,
) -> str:
    """Select the best affordable significant-candidate subset of a panel.

    ``reports``, ``weights`` and ``alpha`` follow the
    :func:`panel_report` contract. ``cost`` is a dict whose keys are
    exactly the candidate keys with strictly positive finite non-boolean
    int/float values and ``limit`` is a non-negative finite non-boolean
    int/float. A non-dict ``reports``, ``weights`` or ``cost`` raises
    ``TypeError``; every other contract violation raises ``ValueError``.

    The grouped items are computed as in :func:`panel_report`. Each
    candidate's ``stable`` is the minimum of its per-group ``stable``
    values, its ``score`` is the overall group's ``mean`` times that
    ``stable`` and it is ``significant`` exactly when every group's
    ``reject`` is true. Candidates are ranked by descending ``score``,
    then ascending ``cost``, then ascending key, with integer ranks
    starting at 1. Every subset of the significant candidates whose
    total cost is at most ``limit`` is enumerated exhaustively — the
    empty subset is always feasible — and the winner maximizes total
    score, then minimizes total cost, then minimizes the
    lexicographically ascending selected-key list. All arithmetic is
    ``Decimal(str(x))`` under a precision-1000, ROUND_HALF_EVEN
    context.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``alpha, limit, cost,
    score, pick, items`` with ``cost`` and ``score`` the selected
    totals. ``pick`` is the ascending selected-key array. Each item
    uses the key order ``key, cost, score, stable, significant, rank,
    pick`` and items are sorted by ``rank``; ``significant`` and
    ``pick`` are booleans, ``rank`` is an integer and every other
    numeric value renders with six decimals, negative zero normalized
    to ``0.000000``. With no candidates ``cost`` and ``score`` are
    ``0.000000`` and both arrays are empty.
    """
    if not isinstance(reports, dict):
        raise TypeError("reports must be a dict")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")
    if not isinstance(cost, dict):
        raise TypeError("cost must be a dict")

    alpha_value, _groups_spec, records = _panel_report_data(
        reports, weights, alpha
    )

    overall_mean: dict[str, Decimal] = {}
    min_stable: dict[str, Decimal] = {}
    significant: dict[str, bool] = {}
    for by, _group_key, candidate, _n, mean, _range, stable, _p, q in records:
        if candidate not in min_stable or stable < min_stable[candidate]:
            min_stable[candidate] = stable
        if by == "overall":
            overall_mean[candidate] = mean
        reject = q <= alpha_value
        if candidate in significant:
            significant[candidate] = significant[candidate] and reject
        else:
            significant[candidate] = reject

    candidates = sorted(overall_mean)
    if set(cost) != set(candidates):
        raise ValueError("cost keys must be exactly the candidate keys")
    costs: dict[str, Decimal] = {}
    for key in candidates:
        cost_value = _portfolio_number(cost[key], "cost")
        if cost_value <= 0:
            raise ValueError("each cost must be positive")
        costs[key] = cost_value
    limit_value = _portfolio_number(limit, "limit")
    if limit_value < 0:
        raise ValueError("limit must be non-negative")

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        scores = {
            key: overall_mean[key] * min_stable[key] for key in candidates
        }
        ranked = sorted(
            candidates, key=lambda key: (-scores[key], costs[key], key)
        )
        ranks = {key: rank for rank, key in enumerate(ranked, 1)}

        sig_keys = [key for key in candidates if significant[key]]
        sig_count = len(sig_keys)
        best: tuple[Decimal, Decimal, tuple[str, ...]] | None = None
        for size in range(sig_count + 1):
            for idxs in combinations(range(sig_count), size):
                total_cost = sum(
                    (costs[sig_keys[i]] for i in idxs), Decimal(0)
                )
                if total_cost > limit_value:
                    continue
                total_score = sum(
                    (scores[sig_keys[i]] for i in idxs), Decimal(0)
                )
                chosen_keys = tuple(sig_keys[i] for i in idxs)
                if (
                    best is None
                    or total_score > best[0]
                    or (
                        total_score == best[0]
                        and (
                            total_cost < best[1]
                            or (
                                total_cost == best[1]
                                and chosen_keys < best[2]
                            )
                        )
                    )
                ):
                    best = (total_score, total_cost, chosen_keys)

        assert best is not None  # the empty subset is always feasible
        total_score, total_cost, chosen_keys = best
        chosen = frozenset(chosen_keys)

        item_strings: list[str] = []
        for key in ranked:
            item_strings.append(
                '{"key":' + json.dumps(key, ensure_ascii=False)
                + ',"cost":' + _format6(costs[key])
                + ',"score":' + _format6(scores[key])
                + ',"stable":' + _format6(min_stable[key])
                + ',"significant":'
                + ("true" if significant[key] else "false")
                + ',"rank":' + str(ranks[key])
                + ',"pick":' + ("true" if key in chosen else "false")
                + "}"
            )

    pick_json = json.dumps(
        list(chosen_keys), ensure_ascii=False, separators=(",", ":")
    )
    return (
        '{"alpha":' + _format6(alpha_value)
        + ',"limit":' + _format6(limit_value)
        + ',"cost":' + _format6(total_cost)
        + ',"score":' + _format6(total_score)
        + ',"pick":' + pick_json
        + ',"items":[' + ",".join(item_strings) + "]}\n"
    )


def panel_priority_frontier(
    reports: dict,
    weights: dict,
    cost: dict,
    limits: list,
    *,
    alpha: float = 0.05,
) -> str:
    """Trace the best affordable significant-candidate subset over limits.

    ``reports``, ``weights``, ``cost`` and ``alpha`` follow the
    :func:`panel_priority` contract and each candidate's ``score`` and
    ``significant`` are computed exactly as there. ``limits`` must be a
    non-empty list of non-negative finite non-boolean int/float values
    that are pairwise distinct by their ``Decimal(str(x))`` value. A
    non-dict ``reports``, ``weights`` or ``cost`` or a non-list
    ``limits`` raises ``TypeError``; every other contract violation
    raises ``ValueError``.

    The limits are processed in ascending numeric order. For each limit
    ``L`` every subset of the significant candidates whose total cost is
    at most ``L`` is enumerated exhaustively — the empty subset is
    always feasible — and the winner maximizes total score, then
    minimizes total cost, then minimizes the lexicographically
    ascending selected-key list. The first frontier entry is compared
    against a baseline of ``score`` 0 and an empty pick; each later
    entry is compared against the previous entry's result, with
    ``marginal`` the score difference and ``added``/``removed`` the
    ascending set differences between the current and previous picks.
    All arithmetic is ``Decimal(str(x))`` under a precision-1000,
    ROUND_HALF_EVEN context and comparisons use the unquantized values.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``alpha, frontier``.
    Each frontier entry uses the key order ``limit, cost, score,
    marginal, pick, added, removed`` and entries are sorted by ascending
    ``limit``; ``pick``, ``added`` and ``removed`` are string arrays and
    every numeric value renders with six decimals, negative zero
    normalized to ``0.000000``.
    """
    if not isinstance(reports, dict):
        raise TypeError("reports must be a dict")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")
    if not isinstance(cost, dict):
        raise TypeError("cost must be a dict")
    if not isinstance(limits, list):
        raise TypeError("limits must be a list")
    if not limits:
        raise ValueError("limits must be a non-empty list")

    alpha_value, _groups_spec, records = _panel_report_data(
        reports, weights, alpha
    )

    overall_mean: dict[str, Decimal] = {}
    min_stable: dict[str, Decimal] = {}
    significant: dict[str, bool] = {}
    for by, _group_key, candidate, _n, mean, _range, stable, _p, q in records:
        if candidate not in min_stable or stable < min_stable[candidate]:
            min_stable[candidate] = stable
        if by == "overall":
            overall_mean[candidate] = mean
        reject = q <= alpha_value
        if candidate in significant:
            significant[candidate] = significant[candidate] and reject
        else:
            significant[candidate] = reject

    candidates = sorted(overall_mean)
    if set(cost) != set(candidates):
        raise ValueError("cost keys must be exactly the candidate keys")
    costs: dict[str, Decimal] = {}
    for key in candidates:
        cost_value = _portfolio_number(cost[key], "cost")
        if cost_value <= 0:
            raise ValueError("each cost must be positive")
        costs[key] = cost_value

    limit_values: list[Decimal] = []
    seen_limits: set[Decimal] = set()
    for limit in limits:
        limit_value = _portfolio_number(limit, "limits")
        if limit_value < 0:
            raise ValueError("each limit must be non-negative")
        if limit_value in seen_limits:
            raise ValueError("limits must be distinct")
        seen_limits.add(limit_value)
        limit_values.append(limit_value)
    limit_values.sort()

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        scores = {
            key: overall_mean[key] * min_stable[key] for key in candidates
        }
        sig_keys = [key for key in candidates if significant[key]]
        sig_count = len(sig_keys)
        subsets: list[tuple[Decimal, Decimal, tuple[str, ...]]] = []
        for size in range(sig_count + 1):
            for idxs in combinations(range(sig_count), size):
                total_cost = sum(
                    (costs[sig_keys[i]] for i in idxs), Decimal(0)
                )
                total_score = sum(
                    (scores[sig_keys[i]] for i in idxs), Decimal(0)
                )
                subsets.append(
                    (total_score, total_cost, tuple(sig_keys[i] for i in idxs))
                )

        frontier_strings: list[str] = []
        prev_score = Decimal(0)
        prev_pick: tuple[str, ...] = ()
        for limit_value in limit_values:
            best: tuple[Decimal, Decimal, tuple[str, ...]] | None = None
            for total_score, total_cost, chosen_keys in subsets:
                if total_cost > limit_value:
                    continue
                if (
                    best is None
                    or total_score > best[0]
                    or (
                        total_score == best[0]
                        and (
                            total_cost < best[1]
                            or (
                                total_cost == best[1]
                                and chosen_keys < best[2]
                            )
                        )
                    )
                ):
                    best = (total_score, total_cost, chosen_keys)

            assert best is not None  # the empty subset is always feasible
            total_score, total_cost, chosen_keys = best
            chosen = set(chosen_keys)
            previous = set(prev_pick)
            added = sorted(chosen - previous)
            removed = sorted(previous - chosen)
            marginal = total_score - prev_score
            frontier_strings.append(
                '{"limit":' + _format6(limit_value)
                + ',"cost":' + _format6(total_cost)
                + ',"score":' + _format6(total_score)
                + ',"marginal":' + _format6(marginal)
                + ',"pick":'
                + json.dumps(
                    list(chosen_keys), ensure_ascii=False, separators=(",", ":")
                )
                + ',"added":'
                + json.dumps(added, ensure_ascii=False, separators=(",", ":"))
                + ',"removed":'
                + json.dumps(removed, ensure_ascii=False, separators=(",", ":"))
                + "}"
            )
            prev_score = total_score
            prev_pick = chosen_keys

    return (
        '{"alpha":' + _format6(alpha_value)
        + ',"frontier":[' + ",".join(frontier_strings) + "]}\n"
    )


def frontier_sensitivity(
    reports: dict,
    weights: dict,
    cost: dict,
    limits: list,
    cases: dict,
    *,
    alpha: float = 0.05,
) -> str:
    """Trace the best affordable subset over limits under cost-multiplier cases.

    ``reports``, ``weights``, ``cost``, ``limits`` and ``alpha`` follow the
    :func:`panel_priority_frontier` contract and each candidate's ``score``
    and ``significant`` are computed exactly as there. ``cases`` must be a
    non-empty dict mapping a non-empty string case key to a dict whose keys
    are exactly the candidate keys and whose values are positive finite
    non-boolean int/float cost multipliers. A non-dict ``cases`` raises
    ``TypeError``; every other ``cases`` contract violation raises
    ``ValueError``.

    For each case the effective cost of a candidate is its ``cost`` times
    the case multiplier and, exactly as in :func:`panel_priority_frontier`,
    every subset of the significant candidates whose total effective cost
    is at most the limit is enumerated exhaustively — the empty subset is
    always feasible — and the winner maximizes total score, then minimizes
    total cost, then minimizes the lexicographically ascending selected-key
    list. Cases are processed in ascending key order and limits in
    ascending numeric order; within each case the first limit entry is
    compared against a baseline of ``score`` 0 and an empty pick and each
    later entry against the previous limit's result, with ``marginal`` the
    score difference, ``added``/``removed`` the ascending set differences
    between the current and previous picks and ``switch`` whether the pick
    changed. ``stability`` is the share of cases whose pick at that limit
    equals the most common pick. All arithmetic is ``Decimal(str(x))``
    under a precision-1000, ROUND_HALF_EVEN context and comparisons use the
    unquantized values.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``alpha, frontier``. Each
    frontier entry uses the key order ``limit, stability, cases`` and each
    case entry the key order ``key, cost, score, marginal, pick, added,
    removed, switch``; frontier entries are sorted by ascending ``limit``
    and case entries by ascending ``key``. ``key`` is the case key,
    ``pick``, ``added`` and ``removed`` are string arrays, ``switch`` is a
    boolean and every numeric value renders with six decimals, negative
    zero normalized to ``0.000000``.
    """
    if not isinstance(reports, dict):
        raise TypeError("reports must be a dict")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")
    if not isinstance(cost, dict):
        raise TypeError("cost must be a dict")
    if not isinstance(limits, list):
        raise TypeError("limits must be a list")
    if not isinstance(cases, dict):
        raise TypeError("cases must be a dict")
    if not limits:
        raise ValueError("limits must be a non-empty list")
    if not cases:
        raise ValueError("cases must be a non-empty dict")

    alpha_value, _groups_spec, records = _panel_report_data(
        reports, weights, alpha
    )

    overall_mean: dict[str, Decimal] = {}
    min_stable: dict[str, Decimal] = {}
    significant: dict[str, bool] = {}
    for by, _group_key, candidate, _n, mean, _range, stable, _p, q in records:
        if candidate not in min_stable or stable < min_stable[candidate]:
            min_stable[candidate] = stable
        if by == "overall":
            overall_mean[candidate] = mean
        reject = q <= alpha_value
        if candidate in significant:
            significant[candidate] = significant[candidate] and reject
        else:
            significant[candidate] = reject

    candidates = sorted(overall_mean)
    if set(cost) != set(candidates):
        raise ValueError("cost keys must be exactly the candidate keys")
    costs: dict[str, Decimal] = {}
    for key in candidates:
        cost_value = _portfolio_number(cost[key], "cost")
        if cost_value <= 0:
            raise ValueError("each cost must be positive")
        costs[key] = cost_value

    limit_values: list[Decimal] = []
    seen_limits: set[Decimal] = set()
    for limit in limits:
        limit_value = _portfolio_number(limit, "limits")
        if limit_value < 0:
            raise ValueError("each limit must be non-negative")
        if limit_value in seen_limits:
            raise ValueError("limits must be distinct")
        seen_limits.add(limit_value)
        limit_values.append(limit_value)
    limit_values.sort()

    case_keys: list[str] = []
    multipliers: dict[str, dict[str, Decimal]] = {}
    for case_key, mapping in cases.items():
        if not isinstance(case_key, str) or not case_key:
            raise ValueError("each cases key must be a non-empty string")
        if not isinstance(mapping, dict) or set(mapping) != set(candidates):
            raise ValueError(
                "each cases value must be a dict keyed by exactly the "
                "candidate keys"
            )
        case_multipliers: dict[str, Decimal] = {}
        for key in candidates:
            multiplier = _portfolio_number(mapping[key], "multiplier")
            if multiplier <= 0:
                raise ValueError("each multiplier must be positive")
            case_multipliers[key] = multiplier
        case_keys.append(case_key)
        multipliers[case_key] = case_multipliers
    case_keys.sort()

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        scores = {
            key: overall_mean[key] * min_stable[key] for key in candidates
        }
        sig_keys = [key for key in candidates if significant[key]]
        sig_count = len(sig_keys)

        # case key -> per-limit winning (score, cost, keys) triples
        case_winners: dict[str, list[tuple[Decimal, Decimal, tuple[str, ...]]]] = {}
        for case_key in case_keys:
            case_costs = {
                key: costs[key] * multipliers[case_key][key]
                for key in sig_keys
            }
            subsets: list[tuple[Decimal, Decimal, tuple[str, ...]]] = []
            for size in range(sig_count + 1):
                for idxs in combinations(range(sig_count), size):
                    total_cost = sum(
                        (case_costs[sig_keys[i]] for i in idxs), Decimal(0)
                    )
                    total_score = sum(
                        (scores[sig_keys[i]] for i in idxs), Decimal(0)
                    )
                    subsets.append(
                        (total_score, total_cost, tuple(sig_keys[i] for i in idxs))
                    )
            winners: list[tuple[Decimal, Decimal, tuple[str, ...]]] = []
            for limit_value in limit_values:
                best: tuple[Decimal, Decimal, tuple[str, ...]] | None = None
                for total_score, total_cost, chosen_keys in subsets:
                    if total_cost > limit_value:
                        continue
                    if (
                        best is None
                        or total_score > best[0]
                        or (
                            total_score == best[0]
                            and (
                                total_cost < best[1]
                                or (
                                    total_cost == best[1]
                                    and chosen_keys < best[2]
                                )
                            )
                        )
                    ):
                        best = (total_score, total_cost, chosen_keys)
                assert best is not None  # the empty subset is always feasible
                winners.append(best)
            case_winners[case_key] = winners

        case_total = len(case_keys)
        frontier_strings: list[str] = []
        for limit_index, limit_value in enumerate(limit_values):
            pick_counts: dict[tuple[str, ...], int] = {}
            for case_key in case_keys:
                pick = case_winners[case_key][limit_index][2]
                pick_counts[pick] = pick_counts.get(pick, 0) + 1
            stability = Decimal(max(pick_counts.values())) / Decimal(case_total)

            case_strings: list[str] = []
            for case_key in case_keys:
                total_score, total_cost, chosen_keys = case_winners[case_key][
                    limit_index
                ]
                if limit_index == 0:
                    prev_score = Decimal(0)
                    prev_pick: tuple[str, ...] = ()
                else:
                    previous = case_winners[case_key][limit_index - 1]
                    prev_score = previous[0]
                    prev_pick = previous[2]
                chosen = set(chosen_keys)
                previous_set = set(prev_pick)
                added = sorted(chosen - previous_set)
                removed = sorted(previous_set - chosen)
                marginal = total_score - prev_score
                switch = chosen_keys != prev_pick
                case_strings.append(
                    '{"key":' + json.dumps(case_key, ensure_ascii=False)
                    + ',"cost":' + _format6(total_cost)
                    + ',"score":' + _format6(total_score)
                    + ',"marginal":' + _format6(marginal)
                    + ',"pick":'
                    + json.dumps(
                        list(chosen_keys), ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + ',"added":'
                    + json.dumps(added, ensure_ascii=False, separators=(",", ":"))
                    + ',"removed":'
                    + json.dumps(removed, ensure_ascii=False, separators=(",", ":"))
                    + ',"switch":' + json.dumps(switch)
                    + "}"
                )
            frontier_strings.append(
                '{"limit":' + _format6(limit_value)
                + ',"stability":' + _format6(stability)
                + ',"cases":[' + ",".join(case_strings) + "]}"
            )

    return (
        '{"alpha":' + _format6(alpha_value)
        + ',"frontier":[' + ",".join(frontier_strings) + "]}\n"
    )


def frontier_joint(
    reports: dict,
    weights: dict,
    cost: dict,
    limits: list,
    cases: dict,
    *,
    alpha: float = 0.05,
) -> str:
    """Aggregate per-case best affordable subsets into a joint frontier.

    ``reports``, ``weights``, ``cost``, ``limits`` and ``alpha`` follow
    the :func:`panel_priority_frontier` contract and each candidate's
    ``score`` and ``significant`` are computed exactly as there.
    ``cases`` must be a non-empty dict mapping a non-empty string case
    key to a dict with exactly the keys ``"cost"`` and ``"gain"``, each
    a dict keyed by exactly the candidate keys whose values are positive
    finite non-boolean int/float multipliers. A non-dict ``reports``,
    ``weights``, ``cost`` or ``cases`` or a non-list ``limits`` raises
    ``TypeError``; every other contract violation raises ``ValueError``.

    For each case and each limit every subset of the significant
    candidates is enumerated exhaustively — the empty subset is always
    feasible — with total cost the sum of ``cost`` times the case's
    ``"cost"`` multiplier and total gain the sum of ``score`` times the
    case's ``"gain"`` multiplier; within budget the winner maximizes
    total gain, then minimizes total cost, then minimizes the
    lexicographically ascending selected-key list. For each limit the
    joint ``pick`` is the modal winning pick across cases (ties broken
    by the lexicographically smaller ascending key list), ``stability``
    is its frequency divided by the number of cases, ``worst`` is the
    smallest winning total gain across cases, ``switch`` is whether the
    joint pick differs from the previous limit's pick (``false`` for
    the first limit) and ``items`` lists every candidate's selection
    frequency across cases. All arithmetic is ``Decimal(str(x))`` under
    a precision-1000, ROUND_HALF_EVEN context and comparisons use the
    unquantized values.

    Returns a compact UTF-8 JSON string with no spaces and exactly one
    trailing newline; the top-level key order is ``alpha, frontier``.
    Each frontier entry uses the key order ``limit, pick, stability,
    worst, switch, items`` and each item the key order ``key,
    frequency``; entries sort by ascending ``limit`` and items by
    ascending ``key``. ``pick`` is an ascending string array, ``switch``
    a boolean and every other numeric value renders with six decimals,
    negative zero normalized to ``0.000000``.
    """
    if not isinstance(reports, dict):
        raise TypeError("reports must be a dict")
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict")
    if not isinstance(cost, dict):
        raise TypeError("cost must be a dict")
    if not isinstance(limits, list):
        raise TypeError("limits must be a list")
    if not isinstance(cases, dict):
        raise TypeError("cases must be a dict")
    if not limits:
        raise ValueError("limits must be a non-empty list")
    if not cases:
        raise ValueError("cases must be a non-empty dict")

    alpha_value, _groups_spec, records = _panel_report_data(
        reports, weights, alpha
    )

    overall_mean: dict[str, Decimal] = {}
    min_stable: dict[str, Decimal] = {}
    significant: dict[str, bool] = {}
    for by, _group_key, candidate, _n, mean, _range, stable, _p, q in records:
        if candidate not in min_stable or stable < min_stable[candidate]:
            min_stable[candidate] = stable
        if by == "overall":
            overall_mean[candidate] = mean
        reject = q <= alpha_value
        if candidate in significant:
            significant[candidate] = significant[candidate] and reject
        else:
            significant[candidate] = reject

    candidates = sorted(overall_mean)
    if set(cost) != set(candidates):
        raise ValueError("cost keys must be exactly the candidate keys")
    costs: dict[str, Decimal] = {}
    for key in candidates:
        cost_value = _portfolio_number(cost[key], "cost")
        if cost_value <= 0:
            raise ValueError("each cost must be positive")
        costs[key] = cost_value

    limit_values: list[Decimal] = []
    seen_limits: set[Decimal] = set()
    for limit in limits:
        limit_value = _portfolio_number(limit, "limits")
        if limit_value < 0:
            raise ValueError("each limit must be non-negative")
        if limit_value in seen_limits:
            raise ValueError("limits must be distinct")
        seen_limits.add(limit_value)
        limit_values.append(limit_value)
    limit_values.sort()

    case_keys: list[str] = []
    cost_multipliers: dict[str, dict[str, Decimal]] = {}
    gain_multipliers: dict[str, dict[str, Decimal]] = {}
    for case_key, mapping in cases.items():
        if not isinstance(case_key, str) or not case_key:
            raise ValueError("each cases key must be a non-empty string")
        if not isinstance(mapping, dict) or set(mapping) != {"cost", "gain"}:
            raise ValueError(
                'each cases value must be a dict with exactly the keys '
                '"cost" and "gain"'
            )
        case_cost: dict[str, Decimal] = {}
        case_gain: dict[str, Decimal] = {}
        for name, target in (("cost", case_cost), ("gain", case_gain)):
            multipliers = mapping[name]
            if not isinstance(multipliers, dict) or set(multipliers) != set(
                candidates
            ):
                raise ValueError(
                    f'each cases "{name}" must be a dict keyed by exactly '
                    "the candidate keys"
                )
            for key in candidates:
                multiplier = _portfolio_number(multipliers[key], "multiplier")
                if multiplier <= 0:
                    raise ValueError("each multiplier must be positive")
                target[key] = multiplier
        case_keys.append(case_key)
        cost_multipliers[case_key] = case_cost
        gain_multipliers[case_key] = case_gain
    case_keys.sort()

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        scores = {
            key: overall_mean[key] * min_stable[key] for key in candidates
        }
        sig_keys = [key for key in candidates if significant[key]]
        sig_count = len(sig_keys)

        # case key -> per-limit winning (gain, cost, keys) triples
        case_winners: dict[
            str, list[tuple[Decimal, Decimal, tuple[str, ...]]]
        ] = {}
        for case_key in case_keys:
            multipliers_cost = cost_multipliers[case_key]
            multipliers_gain = gain_multipliers[case_key]
            subsets: list[tuple[Decimal, Decimal, tuple[str, ...]]] = []
            for size in range(sig_count + 1):
                for idxs in combinations(range(sig_count), size):
                    total_cost = sum(
                        (
                            costs[sig_keys[i]] * multipliers_cost[sig_keys[i]]
                            for i in idxs
                        ),
                        Decimal(0),
                    )
                    total_gain = sum(
                        (
                            scores[sig_keys[i]] * multipliers_gain[sig_keys[i]]
                            for i in idxs
                        ),
                        Decimal(0),
                    )
                    subsets.append(
                        (
                            total_gain,
                            total_cost,
                            tuple(sig_keys[i] for i in idxs),
                        )
                    )
            winners: list[tuple[Decimal, Decimal, tuple[str, ...]]] = []
            for limit_value in limit_values:
                best: tuple[Decimal, Decimal, tuple[str, ...]] | None = None
                for total_gain, total_cost, chosen_keys in subsets:
                    if total_cost > limit_value:
                        continue
                    if (
                        best is None
                        or total_gain > best[0]
                        or (
                            total_gain == best[0]
                            and (
                                total_cost < best[1]
                                or (
                                    total_cost == best[1]
                                    and chosen_keys < best[2]
                                )
                            )
                        )
                    ):
                        best = (total_gain, total_cost, chosen_keys)
                assert best is not None  # the empty subset is always feasible
                winners.append(best)
            case_winners[case_key] = winners

        case_total = len(case_keys)
        frontier_strings: list[str] = []
        prev_pick: tuple[str, ...] | None = None
        for limit_index, limit_value in enumerate(limit_values):
            pick_counts: dict[tuple[str, ...], int] = {}
            member_counts = {key: 0 for key in candidates}
            worst: Decimal | None = None
            for case_key in case_keys:
                total_gain, _total_cost, chosen_keys = case_winners[case_key][
                    limit_index
                ]
                pick_counts[chosen_keys] = pick_counts.get(chosen_keys, 0) + 1
                for key in chosen_keys:
                    member_counts[key] += 1
                if worst is None or total_gain < worst:
                    worst = total_gain
            assert worst is not None  # cases is non-empty
            pick, pick_count = min(
                pick_counts.items(), key=lambda item: (-item[1], item[0])
            )
            stability = Decimal(pick_count) / Decimal(case_total)
            switch = prev_pick is not None and pick != prev_pick
            item_strings: list[str] = []
            for key in candidates:
                frequency = Decimal(member_counts[key]) / Decimal(case_total)
                item_strings.append(
                    '{"key":' + json.dumps(key, ensure_ascii=False)
                    + ',"frequency":' + _format6(frequency)
                    + "}"
                )
            frontier_strings.append(
                '{"limit":' + _format6(limit_value)
                + ',"pick":'
                + json.dumps(
                    list(pick), ensure_ascii=False, separators=(",", ":")
                )
                + ',"stability":' + _format6(stability)
                + ',"worst":' + _format6(worst)
                + ',"switch":' + ("true" if switch else "false")
                + ',"items":[' + ",".join(item_strings) + "]}"
            )
            prev_pick = pick

    return (
        '{"alpha":' + _format6(alpha_value)
        + ',"frontier":[' + ",".join(frontier_strings) + "]}\n"
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
