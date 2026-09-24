"""Tests for urban_micro.ventilation_report."""

from decimal import ROUND_HALF_EVEN, Decimal, localcontext
import json

import pytest

from urban_micro import ventilation_report


def _row(t, c, u, v, h, d):
    """Build one ventilation record mapping."""
    return {
        "timestamp": t,
        "cell_id": c,
        "wind_u": u,
        "wind_v": v,
        "height": h,
        "density": d,
    }


def _format6(value):
    quantized = value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    if quantized == 0:
        quantized = abs(quantized)
    return f"{quantized:.6f}"


def _reference(records, minutes):
    """Reference implementation straight from the spec."""
    buckets = {}
    for row in records:
        bucket = (row["timestamp"] // (minutes * 60)) * (minutes * 60)
        buckets.setdefault(bucket, {}).setdefault(row["cell_id"], []).append(row)
    with localcontext() as ctx:
        ctx.prec = 1000
        ctx.rounding = ROUND_HALF_EVEN
        groups = []
        for bucket in sorted(buckets):
            cells = []
            for cell_id in sorted(buckets[bucket]):
                rows = buckets[bucket][cell_id]
                n = len(rows)
                count = Decimal(n)
                u = sum((Decimal(str(r["wind_u"])) for r in rows), Decimal(0)) / count
                v = sum((Decimal(str(r["wind_v"])) for r in rows), Decimal(0)) / count
                h = sum((Decimal(str(r["height"])) for r in rows), Decimal(0)) / count
                d = sum((Decimal(str(r["density"])) for r in rows), Decimal(0)) / count
                speed = (u * u + v * v).sqrt()
                ventilation = speed * (1 - d) / (1 + h / 10)
                cells.append(
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
                + ',"cells":[' + ",".join(cells) + ']}'
            )
        return (
            '{"minutes":' + str(minutes)
            + ',"groups":[' + ",".join(groups) + ']}'
        )


def test_empty_records():
    assert ventilation_report([]) == '{"minutes":60,"groups":[]}'
    assert ventilation_report([], minutes=30) == '{"minutes":30,"groups":[]}'


def test_records_must_be_list():
    for bad in (None, "x", {}, (), 1, 1.5):
        with pytest.raises(TypeError):
            ventilation_report(bad)


def test_minutes_validation():
    row = _row(0, "a", 1.0, 0.0, 0.0, 0.0)
    for bad in (0, -1, 7, 100, 1441, True, 1.5, "60", None):
        with pytest.raises(ValueError):
            ventilation_report([row], minutes=bad)
    for good in (1, 2, 30, 60, 720, 1440):
        ventilation_report([row], minutes=good)


def test_row_must_be_mapping_with_exact_keys():
    with pytest.raises(ValueError):
        ventilation_report([("t",)])
    with pytest.raises(ValueError):
        ventilation_report([{"timestamp": 0}])
    extra = _row(0, "a", 1.0, 0.0, 0.0, 0.0)
    extra["extra"] = 1
    with pytest.raises(ValueError):
        ventilation_report([extra])
    missing = _row(0, "a", 1.0, 0.0, 0.0, 0.0)
    del missing["wind_v"]
    with pytest.raises(ValueError):
        ventilation_report([missing])


def test_timestamp_validation():
    for bad in (True, -1, 1.5, "0", None):
        with pytest.raises(ValueError):
            ventilation_report([_row(bad, "a", 1.0, 0.0, 0.0, 0.0)])


def test_cell_id_validation():
    for bad in ("", 1, None, b"a"):
        with pytest.raises(ValueError):
            ventilation_report([_row(0, bad, 1.0, 0.0, 0.0, 0.0)])


def test_numeric_field_validation():
    for field in ("wind_u", "wind_v", "height", "density"):
        for bad in (True, float("nan"), float("inf"), "1", None):
            row = _row(0, "a", 1.0, 0.0, 0.0, 0.0)
            row[field] = bad
            with pytest.raises(ValueError):
                ventilation_report([row])


def test_height_and_density_ranges():
    with pytest.raises(ValueError):
        ventilation_report([_row(0, "a", 1.0, 0.0, -0.5, 0.0)])
    with pytest.raises(ValueError):
        ventilation_report([_row(0, "a", 1.0, 0.0, 0.0, -0.1)])
    with pytest.raises(ValueError):
        ventilation_report([_row(0, "a", 1.0, 0.0, 0.0, 1.1)])
    ventilation_report([_row(0, "a", 1.0, 0.0, 0.0, 0.0)])
    ventilation_report([_row(0, "a", 1.0, 0.0, 0.0, 1.0)])


def test_single_record_exact_output():
    report = ventilation_report([_row(0, "a", 3.0, 4.0, 10.0, 0.5)])
    assert report == (
        '{"minutes":60,"groups":[{"key":0,"cells":[{"key":"a","n":1,'
        '"wind_u":3.000000,"wind_v":4.000000,"speed":5.000000,'
        '"height":10.000000,"density":0.500000,"ventilation":1.250000}]}]}'
    )


def test_matches_reference():
    records = [
        _row(0, "b", 1.0, 2.0, 5.0, 0.25),
        _row(100, "b", 3.0, -4.0, 15.0, 0.75),
        _row(3599, "a", -1.5, 0.5, 0.0, 1.0),
        _row(3600, "a", 2.5, 2.5, 3.0, 0.1),
        _row(7200, "细胞", 0.1, 0.2, 0.3, 0.4),
    ]
    assert ventilation_report(records) == _reference(records, 60)
    assert ventilation_report(records, minutes=30) == _reference(records, 30)
    assert ventilation_report(records, minutes=1) == _reference(records, 1)


def test_bucketing_and_ordering():
    records = [
        _row(3700, "z", 1.0, 0.0, 0.0, 0.0),
        _row(100, "b", 1.0, 0.0, 0.0, 0.0),
        _row(100, "a", 1.0, 0.0, 0.0, 0.0),
    ]
    report = json.loads(ventilation_report(records))
    assert [group["key"] for group in report["groups"]] == [0, 3600]
    assert [cell["key"] for cell in report["groups"][0]["cells"]] == ["a", "b"]


def test_negative_zero_normalized():
    report = ventilation_report([_row(0, "a", -0.0, 0.0, 0.0, 0.0)])
    assert "-0.000000" not in report
    assert '"wind_u":0.000000' in report
    assert '"speed":0.000000' in report
    assert '"ventilation":0.000000' in report


def test_unicode_cell_id_preserved():
    report = ventilation_report([_row(0, "细胞α", 1.0, 0.0, 0.0, 0.0)])
    assert '"key":"细胞α"' in report
    assert "\\u" not in report


def test_compact_output():
    report = ventilation_report([_row(0, "a", 1.0, 2.0, 3.0, 0.5)])
    assert " " not in report
    assert not report.endswith("\n")
    json.loads(report)


def test_mean_uses_unquantized_values():
    records = [
        _row(0, "a", 1.0, 0.0, 0.0, 0.0),
        _row(0, "a", 2.0, 0.0, 0.0, 0.0),
    ]
    report = ventilation_report(records)
    assert '"n":2' in report
    assert '"wind_u":1.500000' in report
    assert '"speed":1.500000' in report
