"""Urban heat island (UHI) intensity computation."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
import math

__all__ = [
    "compute_uhi",
    "align_temp",
    "grid_features",
    "fit_uhi_model",
    "scenario",
    "attribute_effects",
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


def _validate_effect_row(
    row: object,
) -> tuple[int, str, Decimal, Decimal, Decimal, Decimal, Decimal, Decimal]:
    """Validate one scenario ``(t, c, base, post, delta, cg, cr, cm)`` eight-tuple."""
    if not isinstance(row, tuple) or len(row) != 8:
        raise ValueError(
            "each detail must be a "
            "(t, c, base, post, delta, cg, cr, cm) eight-tuple"
        )
    timestamp, cell_id, base, post, delta, cg, cr, cm = row
    _validate_timestamp(timestamp)
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("c must be a non-empty string")
    return (
        timestamp,
        cell_id,
        _validate_finite_number(base, "base"),
        _validate_finite_number(post, "post"),
        _validate_finite_number(delta, "delta"),
        _validate_finite_number(cg, "cg"),
        _validate_finite_number(cr, "cr"),
        _validate_finite_number(cm, "cm"),
    )


def attribute_effects(
    details: list,
    *,
    by: str,
    minutes: int = 60,
) -> list[tuple[object, int, float, float, float, float, float, float, float]]:
    """Aggregate scenario details by time bucket or grid cell.

    ``details`` is a list of scenario eight-tuples
    ``(t, c, base, post, delta, cg, cr, cm)``: ``t`` is a non-boolean
    non-negative integer, ``c`` is a non-empty string and the other six
    fields are non-boolean finite ints or floats; ``(t, c)`` pairs must be
    unique. An empty list returns ``[]``.

    ``by`` is ``"time"`` (bucket ``t`` by Unix epoch, floored to
    ``minutes``-sized buckets) or ``"cell"`` (group by ``c``); ``minutes``
    must be a non-boolean integer in ``1..1440`` that divides 1440.

    Returns ``(key, n, base, post, delta, cg, cr, cm, p)`` tuples sorted by
    key, with the six numeric fields the arithmetic means over the group.
    ``p`` is the (capped at 1) two-sided exact sign-test p-value from the
    ``delta`` values: ``r``/``s`` count positive/negative samples (zeros
    ignored), ``m = r + s``, ``q = min(r, s)``; ``p = 1`` when ``m == 0``,
    otherwise ``min(1, 2 * sum(C(m, k), k=0..q) / 2**m)``. Every number
    enters as ``Decimal(str(x))`` under a precision-1000, ROUND_HALF_EVEN
    local context; the six means and ``p`` are quantized to 6 decimals and
    returned as floats, negative zero normalized to ``0.0``.

    ``details`` not being a list raises ``TypeError``; every other contract
    violation (field types, tuple shape, duplicate ``(t, c)``, ``by`` or
    ``minutes``) raises ``ValueError``.
    """
    if not isinstance(details, list):
        raise TypeError("details must be a list")
    if by not in ("time", "cell"):
        raise ValueError("by must be 'time' or 'cell'")
    minutes = _validate_minutes(minutes)

    if not details:
        return []

    parsed = [_validate_effect_row(row) for row in details]

    seen: set[tuple[int, str]] = set()
    groups: dict[object, list] = {}
    bucket_seconds = minutes * 60
    for timestamp, cell_id, base, post, delta, cg, cr, cm in parsed:
        pair = (timestamp, cell_id)
        if pair in seen:
            raise ValueError(f"duplicate (t, c) pair: {pair!r}")
        seen.add(pair)
        key = (timestamp // bucket_seconds) * bucket_seconds if by == "time" else cell_id
        groups.setdefault(key, []).append((base, post, delta, cg, cr, cm))

    with localcontext() as ctx:
        ctx.prec = _MODEL_PRECISION
        ctx.rounding = ROUND_HALF_EVEN

        zero = Decimal(0)
        one = Decimal(1)
        two = Decimal(2)
        results = []
        for key in sorted(groups):
            rows = groups[key]
            n = len(rows)
            sums = [zero] * 6
            positives = 0
            negatives = 0
            for base, post, delta, cg, cr, cm in rows:
                values = (base, post, delta, cg, cr, cm)
                for j in range(6):
                    sums[j] += values[j]
                if delta > 0:
                    positives += 1
                elif delta < 0:
                    negatives += 1

            m = positives + negatives
            if m == 0:
                p_value = one
            else:
                q = min(positives, negatives)
                # Exact binomial tail: sum C(m, k), k = 0..q, over 2**m.
                tail = zero
                combin = 1
                for k in range(q + 1):
                    tail += Decimal(combin)
                    combin = combin * (m - k) // (k + 1)
                p_value = two * tail / (two ** m)
                if p_value > one:
                    p_value = one

            results.append(
                (
                    key,
                    n,
                    _quantize6(sums[0] / n),
                    _quantize6(sums[1] / n),
                    _quantize6(sums[2] / n),
                    _quantize6(sums[3] / n),
                    _quantize6(sums[4] / n),
                    _quantize6(sums[5] / n),
                    _quantize6(p_value),
                )
            )
    return results
