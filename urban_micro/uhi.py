"""Urban heat island (UHI) intensity computation."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
import math

__all__ = ["compute_uhi", "align_temp", "grid_features", "fit_uhi_model"]

_RECORD_KEYS = frozenset({"station_id", "timestamp", "temp_c"})
_SATELLITE_KEYS = frozenset({"cell_id", "timestamp", "lst_c"})
_ZONES = frozenset({"urban", "rural"})
_ITEM_KINDS = frozenset({"building", "impervious", "green", "other"})
_COVERAGE_KINDS = frozenset({"impervious", "green", "other"})
_QUANT = Decimal("0.001")
_QUANT4 = Decimal("0.0001")
_QUANT6 = Decimal("0.000001")
_RIDGE = Decimal("0.000001")


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


def _validate_aligned_fields(
    timestamp: object, cell_id: object, station_c: object, satellite_c: object,
    diff_c: object,
) -> None:
    _validate_timestamp(timestamp)
    if not isinstance(cell_id, str) or not cell_id:
        raise ValueError("cell_id must be a non-empty string")
    _validate_temperature(station_c, "station_c")
    _validate_temperature(satellite_c, "satellite_c")
    _validate_temperature(diff_c, "diff_c")


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
    ``(timestamp, cell_id, station_c, satellite_c, diff_c)``; each timestamp
    must be a non-bool non-negative int, each cell_id a non-empty string and
    each temperature a non-bool finite int/float (the original values are
    passed through unchanged). ``items`` is a
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
    for row in aligned:
        if not isinstance(row, tuple) or len(row) != 5:
            raise ValueError(
                "each aligned row must be an align_temp "
                "(timestamp, cell_id, station_c, satellite_c, diff_c) five-tuple"
            )
        _validate_aligned_fields(*row)

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
    for row in aligned:
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


def _validate_model_number(
    value: object, name: str, *, minimum: Decimal, maximum: Decimal | None
) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite int or float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    decimal_value = Decimal(str(value))
    if decimal_value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and decimal_value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return decimal_value


def _validate_model_row(
    row: object,
) -> tuple[int, str, Decimal, Decimal, Decimal, Decimal, Decimal]:
    if not isinstance(row, tuple) or len(row) != 9:
        raise ValueError(
            "each row must be a (timestamp, cell_id, station_c, satellite_c, "
            "diff_c, height, building, impervious, green) nine-tuple"
        )
    t, c, s, l, d, h, b, i, g = row
    _validate_aligned_fields(t, c, s, l, d)
    target = Decimal(str(d))
    height = _validate_model_number(h, "height", minimum=Decimal(0), maximum=None)
    building = _validate_model_number(
        b, "building", minimum=Decimal(0), maximum=Decimal(1)
    )
    impervious = _validate_model_number(
        i, "impervious", minimum=Decimal(0), maximum=Decimal(1)
    )
    green = _validate_model_number(
        g, "green", minimum=Decimal(0), maximum=Decimal(1)
    )
    return t, c, target, height, building, impervious, green


def _solve_linear_system(matrix: list[list[Decimal]], rhs: list[Decimal]) -> list[Decimal]:
    """Solve ``matrix @ x = rhs`` by Gauss-Jordan elimination.

    For each column the pivot is the remaining row with the largest absolute
    value (earliest row on ties); a zero pivot raises ``ValueError``.
    """
    size = len(matrix)
    for col in range(size):
        pivot_row = max(range(col, size), key=lambda r: abs(matrix[r][col]))
        pivot = matrix[pivot_row][col]
        if pivot == 0:
            raise ValueError("normal equations are singular (zero pivot)")
        if pivot_row != col:
            matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
            rhs[col], rhs[pivot_row] = rhs[pivot_row], rhs[col]
        matrix[col] = [value / pivot for value in matrix[col]]
        rhs[col] = rhs[col] / pivot
        for row in range(size):
            if row == col:
                continue
            factor = matrix[row][col]
            if factor == 0:
                continue
            matrix[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(matrix[row], matrix[col])
            ]
            rhs[row] = rhs[row] - factor * rhs[col]
    return rhs


def _quantize6(value: Decimal) -> float:
    result = float(value.quantize(_QUANT6, rounding=ROUND_HALF_EVEN))
    return 0.0 if result == 0.0 else result


def fit_uhi_model(
    rows: list,
) -> tuple[int, float, float, float, float, float, float]:
    """Fit a ridge regression of ``diff_c`` on the grid-geometry features.

    ``rows`` is a non-empty list of ``grid_features`` nine-tuples
    ``(timestamp, cell_id, station_c, satellite_c, diff_c, height, building,
    impervious, green)``. The first five fields follow the ``grid_features``
    aligned-row contract; ``height`` must be a non-bool finite number >= 0
    and the three coverage fractions non-bool finite numbers in [0, 1].
    ``(timestamp, cell_id)`` pairs must be unique. Rows are ordered by
    ``(timestamp, cell_id)`` ascending before fitting.

    The model is ``diff_c ~ b0 + bh*height + bb*building + bi*impervious +
    bg*green`` with an unpenalized intercept. All arithmetic uses
    ``Decimal(str(x))`` under a local context with precision 1000 and
    ROUND_HALF_EVEN rounding. The normal equations add ``0.000001`` to the
    diagonal of the four feature columns (the intercept diagonal is left
    unchanged) and are solved by Gauss-Jordan elimination; for each column
    the pivot is the remaining row with the largest absolute value (earliest
    on ties), and a zero pivot raises ``ValueError``.

    Returns ``(n, b0, bh, bb, bi, bg, r2)`` where ``n`` is the sample count
    and ``r2`` is ``1 - SSE/SST`` with ``SST = sum((d - mean(d))**2)`` (0
    when ``SST`` is 0). The six Decimal results are quantized to 6 decimals
    (ROUND_HALF_EVEN), converted to floats and negative zero is normalized
    to ``0.0``.
    """
    if not isinstance(rows, list):
        raise TypeError("rows must be a list")
    if not rows:
        raise ValueError("rows must not be empty")

    parsed = [_validate_model_row(row) for row in rows]

    seen: set[tuple[int, str]] = set()
    for t, c, *_ in parsed:
        key = (t, c)
        if key in seen:
            raise ValueError(f"duplicate (timestamp, cell_id) pair: {key!r}")
        seen.add(key)

    parsed.sort(key=lambda row: (row[0], row[1]))
    n = len(parsed)

    with localcontext() as ctx:
        ctx.prec = 1000
        ctx.rounding = ROUND_HALF_EVEN

        # Normal equations for the design matrix [1, h, b, i, g].
        size = 5
        matrix = [[Decimal(0)] * size for _ in range(size)]
        rhs = [Decimal(0)] * size
        targets: list[Decimal] = []
        for _, _, target, height, building, impervious, green in parsed:
            design = (Decimal(1), height, building, impervious, green)
            targets.append(target)
            for j in range(size):
                rhs[j] += design[j] * target
                row_j = matrix[j]
                x_j = design[j]
                for k in range(size):
                    row_j[k] += x_j * design[k]
        for k in range(1, size):
            matrix[k][k] += _RIDGE

        beta = _solve_linear_system(matrix, rhs)
        b0, bh, bb, bi, bg = beta

        mean = sum(targets) / n
        sst = sum((target - mean) ** 2 for target in targets)
        sse = Decimal(0)
        for _, _, target, height, building, impervious, green in parsed:
            predicted = b0 + bh * height + bb * building + bi * impervious + bg * green
            sse += (target - predicted) ** 2
        r2 = Decimal(0) if sst == 0 else 1 - sse / sst

    return (
        n,
        _quantize6(b0),
        _quantize6(bh),
        _quantize6(bb),
        _quantize6(bi),
        _quantize6(bg),
        _quantize6(r2),
    )
