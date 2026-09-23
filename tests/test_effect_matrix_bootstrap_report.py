"""Tests for urban_micro.effect_matrix_bootstrap_report."""

import itertools
import json
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, localcontext

import pytest

from urban_micro import effect_matrix_bootstrap_report

_QUANT6 = Decimal("0.000001")


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _format6(value):
    quantized = value.quantize(_QUANT6, rounding=ROUND_HALF_EVEN)
    if quantized == 0:
        quantized = abs(quantized)
    return f"{quantized:.6f}"


def _brute_force(deltas, confidence):
    """Full n**n enumeration reference for delta/lower/upper."""
    with localcontext() as ctx:
        ctx.prec = 1000
        ctx.rounding = ROUND_HALF_EVEN
        values = [Decimal(str(d)) for d in deltas]
        n = len(values)
        means = sorted(
            sum(draw) / n for draw in itertools.product(values, repeat=n)
        )
        total = sum(values) / n
        count = n**n
        q_value = (Decimal(1) - Decimal(str(confidence))) / 2
        r_value = (Decimal(count) - 1) * q_value
        floor_index = int(r_value.to_integral_value(rounding=ROUND_FLOOR))
        weight = r_value - floor_index

        def at(position):
            return means[position]

        lower = at(floor_index)
        upper = at(count - 1 - floor_index)
        if weight != 0:
            lower = lower + weight * (at(floor_index + 1) - lower)
            upper = upper + weight * (at(count - 2 - floor_index) - upper)
        return _format6(total), _format6(lower), _format6(upper)


def test_empty_details_gives_empty_groups():
    report = effect_matrix_bootstrap_report([])
    assert report == '{"minutes":60,"confidence":0.950000,"groups":[]}'
    assert json.loads(report) == {
        "minutes": 60,
        "confidence": 0.95,
        "groups": [],
    }


def test_single_row_cell():
    report = effect_matrix_bootstrap_report([_row(0, "a", 1.0)])
    assert report == (
        '{"minutes":60,"confidence":0.950000,"groups":['
        '{"key":0,"cells":['
        '{"key":"a","n":1,"delta":1.000000,"lower":1.000000,"upper":1.000000}'
        ']}]}'
    )


def test_buckets_floor_and_cells_sorted_within_bucket():
    details = [
        _row(0, "b", 1.0),
        _row(1799, "a", 1.0),
        _row(1800, "a", 1.0),
        _row(1800, "b", 1.0),
    ]
    report = effect_matrix_bootstrap_report(details, minutes=30)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [0, 1800]
    assert [cell["key"] for cell in groups[0]["cells"]] == ["a", "b"]
    assert [cell["n"] for cell in groups[0]["cells"]] == [1, 1]
    assert [cell["key"] for cell in groups[1]["cells"]] == ["a", "b"]


def test_same_cell_rows_aggregate_within_bucket_but_split_across_buckets():
    details = [_row(0, "a", 0.0), _row(5, "a", 2.0), _row(3600, "a", 4.0)]
    groups = json.loads(effect_matrix_bootstrap_report(details))["groups"]
    assert [g["key"] for g in groups] == [0, 3600]
    assert groups[0]["cells"][0]["n"] == 2
    assert groups[0]["cells"][0]["delta"] == 1.0
    assert groups[1]["cells"][0]["n"] == 1
    assert groups[1]["cells"][0]["delta"] == 4.0


def test_interpolated_bounds_n2_confidence_95():
    # d = (0, 2): sorted resample means are 0, 1, 1, 2; r = 3 * 0.025 = 0.075
    # -> lower = 0.075, upper = 2 - 0.075 = 1.925.
    report = effect_matrix_bootstrap_report([_row(0, "a", 0.0), _row(1, "a", 2.0)])
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell == {"key": "a", "n": 2, "delta": 1.0,
                    "lower": 0.075, "upper": 1.925}
    assert '"lower":0.075000' in report
    assert '"upper":1.925000' in report


def test_integer_r_takes_position_directly():
    # confidence 0.5 -> q = 0.25 -> r = (25 - 1) * 0.25 = 6 exactly for n = 5.
    deltas = [-1.0, 1.0, 3.0, -3.0, 5.0]
    details = [_row(t, "a", d) for t, d in enumerate(deltas)]
    delta, lower, upper = _brute_force(deltas, 0.5)
    report = effect_matrix_bootstrap_report(details, confidence=0.5)
    cell = json.loads(report)["groups"][0]["cells"][0]
    assert cell["n"] == 5
    assert f'"delta":{delta}' in report
    assert f'"lower":{lower}' in report
    assert f'"upper":{upper}' in report


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("confidence", [0.5, 0.9, 0.95])
def test_matches_full_enumeration(n, confidence):
    deltas = [0.1 * index - 0.7 for index in range(n)]
    details = [_row(t, "a", d) for t, d in enumerate(deltas)]
    delta, lower, upper = _brute_force(deltas, confidence)
    report = effect_matrix_bootstrap_report(details, confidence=confidence)
    expected_cell = (
        '{"key":"a","n":' + str(n)
        + ',"delta":' + delta
        + ',"lower":' + lower
        + ',"upper":' + upper + '}'
    )
    assert expected_cell in report


def test_n8_allowed_n9_rejected():
    details = [_row(t, "a", 1.0) for t in range(8)]
    report = effect_matrix_bootstrap_report(details, minutes=1440)
    assert json.loads(report)["groups"][0]["cells"][0]["n"] == 8

    too_many = [_row(t, "a", 1.0) for t in range(9)]
    with pytest.raises(ValueError):
        effect_matrix_bootstrap_report(too_many, minutes=1440)


def test_n9_in_other_bucket_still_rejected():
    details = [_row(t, "a", 1.0) for t in range(8)]
    details.append(_row(86400, "a", 1.0))  # next day, second bucket
    report = effect_matrix_bootstrap_report(details, minutes=1440)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == [0, 86400]
    assert [g["cells"][0]["n"] for g in groups] == [8, 1]

    details.append(_row(86401, "a", 1.0))
    # bucket 86400 now has n = 2 still fine; push it to 9:
    details += [_row(86400 + t, "b", 0.0) for t in range(9)]
    with pytest.raises(ValueError):
        effect_matrix_bootstrap_report(details, minutes=1440)


def test_negative_zero_normalized():
    report = effect_matrix_bootstrap_report([_row(0, "a", -0.0)])
    assert "-0.000000" not in report
    assert '"delta":0.000000' in report
    assert '"lower":0.000000' in report
    assert '"upper":0.000000' in report


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_matrix_bootstrap_report([_row(0, "céll", 1.0)])
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_item_key_order():
    report = effect_matrix_bootstrap_report(
        [_row(0, "a", 1.0)], minutes=30, confidence=0.9
    )
    assert list(json.loads(report)) == ["minutes", "confidence", "groups"]
    assert list(json.loads(report)["groups"][0]) == ["key", "cells"]
    assert list(json.loads(report)["groups"][0]["cells"][0]) == [
        "key", "n", "delta", "lower", "upper",
    ]
    assert '"minutes":30' in report
    assert '"confidence":0.900000' in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_bootstrap_report("not a list")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes": 0},
        {"minutes": 7},
        {"minutes": 1441},
        {"minutes": True},
        {"minutes": 60.0},
        {"minutes": "60"},
        {"confidence": 0.0},
        {"confidence": 1.0},
        {"confidence": -0.5},
        {"confidence": True},
        {"confidence": float("nan")},
        {"confidence": "0.95"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_matrix_bootstrap_report([], **kwargs)


@pytest.mark.parametrize(
    "row",
    [
        [0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # not a tuple
        (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0),  # wrong length
        (True, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # bool timestamp
        (-1, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # negative timestamp
        (0, "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # empty cell id
        (0, "a", 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0),  # non-finite
        (0, "a", 0.0, 0.0, True, 0.0, 0.0, 0.0),  # bool value
        (0, 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # non-string cell id
    ],
)
def test_invalid_rows_raise_value_error(row):
    with pytest.raises(ValueError):
        effect_matrix_bootstrap_report([row])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 0.0)
    with pytest.raises(ValueError):
        effect_matrix_bootstrap_report([row, row])
