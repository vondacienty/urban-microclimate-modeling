"""Tests for urban_micro.ventilation_report."""

import json
from decimal import ROUND_HALF_EVEN, Decimal, localcontext

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


def _reference(records, minutes=60):
    """Reference compact report computed independently with Decimal."""
    with localcontext() as ctx:
        ctx.prec = 1000
        ctx.rounding = ROUND_HALF_EVEN
        buckets = {}
        for record in records:
            bucket = (record["timestamp"] // (minutes * 60)) * (minutes * 60)
            buckets.setdefault(bucket, {}).setdefault(
                record["cell_id"], []
            ).append(
                tuple(Decimal(str(record[k])) for k in
                      ("wind_u", "wind_v", "height", "density"))
            )

        def fmt(value):
            quantized = value.quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_EVEN
            )
            if quantized == 0:
                quantized = abs(quantized)
            return f"{quantized:.6f}"

        groups = []
        for bucket in sorted(buckets):
            cells = []
            for cell_id in sorted(buckets[bucket]):
                rows = buckets[bucket][cell_id]
                n = len(rows)
                u = sum(r[0] for r in rows) / n
                v = sum(r[1] for r in rows) / n
                h = sum(r[2] for r in rows) / n
                d = sum(r[3] for r in rows) / n
                speed = (u * u + v * v).sqrt()
                ventilation = speed * (1 - d) / (1 + h / 10)
                cells.append(
                    '{"key":' + json.dumps(cell_id, ensure_ascii=False)
                    + ',"n":' + str(n)
                    + ',"wind_u":' + fmt(u)
                    + ',"wind_v":' + fmt(v)
                    + ',"speed":' + fmt(speed)
                    + ',"height":' + fmt(h)
                    + ',"density":' + fmt(d)
                    + ',"ventilation":' + fmt(ventilation)
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


def test_single_record():
    records = [_row(0, "a", 3.0, 4.0, 10.0, 0.5)]
    report = ventilation_report(records)
    assert report == _reference(records)
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["n"] == 1
    # ventilation = 5 * (1 - 0.5) / (1 + 10 / 10) = 1.25
    assert (
        '"wind_u":3.000000,"wind_v":4.000000,"speed":5.000000,'
        '"height":10.000000,"density":0.500000,"ventilation":1.250000'
    ) in report


def test_grouping_means_and_order():
    records = [
        _row(3599, "b", 1.0, 0.0, 0.0, 0.0),
        _row(0, "b", 3.0, 0.0, 2.0, 0.2),
        _row(3600, "a", 0.0, 2.0, 4.0, 0.4),
        _row(7200, "a", 2.0, 2.0, 0.0, 1.0),
        _row(1800, "b", 5.0, 0.0, 4.0, 0.6),
    ]
    report = ventilation_report(records)
    assert report == _reference(records)
    parsed = json.loads(report)
    assert [group["key"] for group in parsed["groups"]] == [0, 3600, 7200]
    first = parsed["groups"][0]
    assert [cell["key"] for cell in first["cells"]] == ["b"]
    cell = first["cells"][0]
    assert cell["n"] == 3
    assert '"wind_u":3.000000' in report
    assert '"height":2.000000' in report


def test_minutes_bucketing():
    records = [
        _row(0, "a", 1.0, 1.0, 0.0, 0.0),
        _row(899, "a", 3.0, 3.0, 0.0, 0.0),
        _row(900, "a", 5.0, 5.0, 0.0, 0.0),
    ]
    report = ventilation_report(records, minutes=15)
    assert report == _reference(records, 15)
    parsed = json.loads(report)
    assert [group["key"] for group in parsed["groups"]] == [0, 900]
    assert parsed["groups"][0]["cells"][0]["n"] == 2
    assert parsed["groups"][1]["cells"][0]["n"] == 1


def test_cell_sorting_and_unicode():
    records = [
        _row(0, "β", 1.0, 0.0, 0.0, 0.0),
        _row(0, "alpha", 2.0, 0.0, 0.0, 0.0),
        _row(0, "细胞", 3.0, 0.0, 0.0, 0.0),
    ]
    report = ventilation_report(records)
    assert report == _reference(records)
    assert "细胞" in report
    assert "\\u" not in report
    keys = [cell["key"] for cell in json.loads(report)["groups"][0]["cells"]]
    assert keys == sorted(keys)


def test_negative_zero_normalized():
    records = [_row(0, "a", -0.0, 0.0, 0.0, 1.0)]
    report = ventilation_report(records)
    assert "-0.000000" not in report
    assert '"wind_u":0.000000' in report
    assert '"ventilation":0.000000' in report


def test_compact_no_trailing_newline():
    report = ventilation_report([_row(0, "a", 1.0, 2.0, 3.0, 0.1)])
    assert not report.endswith("\n")
    assert " " not in report
    assert json.loads(report) is not None


def test_records_not_list():
    with pytest.raises(TypeError):
        ventilation_report({"timestamp": 0})
    with pytest.raises(TypeError):
        ventilation_report(None)
    with pytest.raises(TypeError):
        ventilation_report("records")


def test_record_not_mapping():
    with pytest.raises(ValueError):
        ventilation_report([("a", 0)])
    with pytest.raises(ValueError):
        ventilation_report([None])


def test_record_key_set():
    with pytest.raises(ValueError):
        ventilation_report([{"timestamp": 0, "cell_id": "a", "wind_u": 1.0,
                             "wind_v": 1.0, "height": 1.0}])
    extra = _row(0, "a", 1.0, 1.0, 1.0, 0.5)
    extra["temp_c"] = 20.0
    with pytest.raises(ValueError):
        ventilation_report([extra])


def test_bad_timestamp():
    for bad in (True, -1, 1.5, "0", None):
        with pytest.raises(ValueError):
            ventilation_report([_row(bad, "a", 1.0, 1.0, 1.0, 0.5)])


def test_bad_cell_id():
    for bad in ("", 1, None, True):
        with pytest.raises(ValueError):
            ventilation_report([_row(0, bad, 1.0, 1.0, 1.0, 0.5)])


def test_bad_numbers():
    for key in ("wind_u", "wind_v", "height", "density"):
        for bad in (True, float("nan"), float("inf"), "1", None):
            row = _row(0, "a", 1.0, 1.0, 1.0, 0.5)
            row[key] = bad
            with pytest.raises(ValueError):
                ventilation_report([row])


def test_range_constraints():
    with pytest.raises(ValueError):
        ventilation_report([_row(0, "a", 1.0, 1.0, -0.1, 0.5)])
    with pytest.raises(ValueError):
        ventilation_report([_row(0, "a", 1.0, 1.0, 1.0, -0.1)])
    with pytest.raises(ValueError):
        ventilation_report([_row(0, "a", 1.0, 1.0, 1.0, 1.1)])
    # boundary values are accepted
    ventilation_report([_row(0, "a", 1.0, 1.0, 0, 0)])
    ventilation_report([_row(0, "a", 1.0, 1.0, 0, 1)])


def test_bad_minutes():
    for bad in (True, 0, -60, 1.5, "60", 1441, 7, 100):
        with pytest.raises(ValueError):
            ventilation_report([], minutes=bad)
    for good in (1, 2, 3, 5, 6, 10, 12, 15, 20, 30, 60, 120, 720, 1440):
        assert json.loads(ventilation_report([], minutes=good))["minutes"] == good


def test_int_and_float_mix():
    records = [
        _row(0, "a", 1, 2, 3, 0),
        _row(1, "a", 0.5, 0.25, 1.5, 0.75),
    ]
    assert ventilation_report(records) == _reference(records)
