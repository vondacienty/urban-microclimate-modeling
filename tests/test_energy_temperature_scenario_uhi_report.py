"""Tests for urban_micro.energy_temperature_scenario_uhi_report."""

import json

import pytest

from urban_micro import energy_temperature_scenario_uhi_report

ZONES = {"A": "urban", "B": "rural"}


def _e(t, c, n, s=0, l=0, st=0):
    return (t, c, n, s, l, st)


def _v(t, c, value):
    return (t, c, value)


def _a(c, dn=0, ds=0, dl=0, dst=0):
    return (c, dn, ds, dl, dst)


def test_empty_inputs_yield_empty_groups():
    assert (
        energy_temperature_scenario_uhi_report([], [], {}, [])
        == '{"minutes":60,"groups":[]}'
    )
    assert (
        energy_temperature_scenario_uhi_report([], [], {}, [], minutes=30)
        == '{"minutes":30,"groups":[]}'
    )


def test_basic_single_bucket_without_actions():
    energy = [_e(0, "A", 10, 2, 1, 1), _e(0, "B", 8, 1, 1, 0)]
    temp = [_v(0, "A", 30.0), _v(0, "B", 20.0)]
    result = energy_temperature_scenario_uhi_report(energy, temp, ZONES, [])
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"base":"0.000000",'
        '"post":"0.000000","uhi":"10.000000"}]}'
    )
    json.loads(result)


def test_actions_shift_only_post_uhi():
    energy = [_e(0, "A", 10, 2, 1, 1), _e(0, "B", 8, 1, 1, 0)]
    temp = [_v(0, "A", 30.0), _v(0, "B", 20.0)]
    actions = [_a("A", dn=4, ds=1)]  # a = 3 -> post residual A 9, B stays 6
    result = energy_temperature_scenario_uhi_report(
        energy, temp, ZONES, actions
    )
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"base":"0.000000",'
        '"post":"3.000000","uhi":"10.000000"}]}'
    )


def test_zone_averages_cells_not_rows():
    zones = {"A1": "urban", "A2": "urban", "B": "rural"}
    energy = [
        _e(0, "A1", 10),  # r 10
        _e(0, "A2", 20),  # r 20
        _e(0, "B", 5),    # r 5
    ]
    temp = [_v(0, "A1", 30.0), _v(0, "A2", 32.0), _v(0, "B", 20.0)]
    # action on A2: a = -5 -> post 15; urban post mean (10 + 15) / 2 = 12.5
    actions = [_a("A2", ds=5)]
    result = energy_temperature_scenario_uhi_report(
        energy, temp, zones, actions
    )
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"base":"10.000000",'
        '"post":"7.500000","uhi":"11.000000"}]}'
    )


def test_rows_averaged_before_zone_average():
    energy = [
        _e(0, "A", 10),
        _e(100, "A", 20),  # cell A base residual mean 15
        _e(0, "B", 4),
    ]
    temp = [
        _v(0, "A", 30.0),
        _v(100, "A", 40.0),  # cell A temp mean 35
        _v(0, "B", 20.0),
    ]
    result = energy_temperature_scenario_uhi_report(energy, temp, ZONES, [])
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"base":"11.000000",'
        '"post":"11.000000","uhi":"15.000000"}]}'
    )


def test_intersection_keeps_only_paired_cells_per_bucket():
    # energy-only pair A2 and temp-only pair A3 never contribute
    zones = {"A1": "urban", "A2": "urban", "A3": "urban", "B": "rural"}
    energy = [_e(0, "A1", 10), _e(0, "A2", 20), _e(0, "B", 4)]
    temp = [_v(0, "A1", 30.0), _v(0, "A3", 50.0), _v(0, "B", 20.0)]
    result = energy_temperature_scenario_uhi_report(energy, temp, zones, [])
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"base":"6.000000",'
        '"post":"6.000000","uhi":"10.000000"}]}'
    )


def test_bucket_missing_zone_is_dropped():
    energy = [
        _e(0, "A", 10),
        _e(0, "B", 4),
        _e(7200, "A", 12),  # second bucket has urban only
    ]
    temp = [_v(0, "A", 30.0), _v(0, "B", 20.0), _v(7200, "A", 31.0)]
    result = energy_temperature_scenario_uhi_report(energy, temp, ZONES, [])
    assert result == (
        '{"minutes":60,"groups":[{"key":0,"base":"6.000000",'
        '"post":"6.000000","uhi":"10.000000"}]}'
    )


def test_bucketing_and_minutes_30():
    energy = [
        _e(1799, "A", 10),
        _e(1800, "A", 20),
        _e(1799, "B", 4),
        _e(1800, "B", 8),
    ]
    temp = [
        _v(1799, "A", 30.0),
        _v(1800, "A", 40.0),
        _v(1799, "B", 20.0),
        _v(1800, "B", 22.0),
    ]
    result = energy_temperature_scenario_uhi_report(
        energy, temp, ZONES, [], minutes=30
    )
    assert result == (
        '{"minutes":30,"groups":['
        '{"key":0,"base":"6.000000","post":"6.000000","uhi":"10.000000"},'
        '{"key":1800,"base":"12.000000","post":"12.000000",'
        '"uhi":"18.000000"}]}'
    )


def test_action_applies_in_every_bucket_cell_appears():
    energy = [
        _e(0, "A", 10),
        _e(3600, "A", 10),
        _e(0, "B", 4),
        _e(3600, "B", 4),
    ]
    temp = [
        _v(0, "A", 30.0),
        _v(3600, "A", 30.0),
        _v(0, "B", 20.0),
        _v(3600, "B", 20.0),
    ]
    actions = [_a("A", dn=2)]  # post A 12 in both buckets
    result = energy_temperature_scenario_uhi_report(
        energy, temp, ZONES, actions
    )
    assert result == (
        '{"minutes":60,"groups":['
        '{"key":0,"base":"6.000000","post":"8.000000","uhi":"10.000000"},'
        '{"key":3600,"base":"6.000000","post":"8.000000",'
        '"uhi":"10.000000"}]}'
    )


def test_decimal_str_semantics_and_negative_zero():
    energy = [
        _e(0, "A", 0.1),
        _e(1, "A", 0.2),  # mean residual 0.15
        _e(0, "B", -0.0),
    ]
    temp = [_v(0, "A", 0.0), _v(1, "A", 0.0), _v(0, "B", 0.0)]
    result = energy_temperature_scenario_uhi_report(energy, temp, ZONES, [])
    assert '"base":"0.150000"' in result
    assert '"post":"0.150000"' in result
    assert '"uhi":"0.000000"' in result
    assert "-0.000000" not in result


def test_no_trailing_newline_and_no_spaces():
    energy = [_e(0, "A", 10), _e(0, "B", 4)]
    temp = [_v(0, "A", 30.0), _v(0, "B", 20.0)]
    result = energy_temperature_scenario_uhi_report(energy, temp, ZONES, [])
    assert not result.endswith("\n")
    assert " " not in result


def test_extra_zones_entries_allowed():
    energy = [_e(0, "A", 10), _e(0, "B", 4)]
    temp = [_v(0, "A", 30.0), _v(0, "B", 20.0)]
    zones = {"A": "urban", "B": "rural", "C": "urban"}
    result = energy_temperature_scenario_uhi_report(
        energy, temp, zones, []
    )
    assert '"key":0' in result


def test_type_errors():
    energy = [_e(0, "A", 10), _e(0, "B", 4)]
    temp = [_v(0, "A", 30.0), _v(0, "B", 20.0)]
    with pytest.raises(TypeError):
        energy_temperature_scenario_uhi_report("x", temp, ZONES, [])
    with pytest.raises(TypeError):
        energy_temperature_scenario_uhi_report(energy, "x", ZONES, [])
    with pytest.raises(TypeError):
        energy_temperature_scenario_uhi_report(energy, temp, "x", [])
    with pytest.raises(TypeError):
        energy_temperature_scenario_uhi_report(energy, temp, ZONES, "x")


def test_value_errors():
    energy = [_e(0, "A", 10), _e(0, "B", 4)]
    temp = [_v(0, "A", 30.0), _v(0, "B", 20.0)]

    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, {"A": "city", "B": "rural"}, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, {"": "urban", "B": "rural"}, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, {"A": "urban"}, []
        )
    with pytest.raises(ValueError):
        # temp cell uncovered by zones
        energy_temperature_scenario_uhi_report(
            energy, [_v(0, "A", 1.0), _v(0, "Z", 2.0)], ZONES, []
        )
    with pytest.raises(ValueError):
        # duplicate energy pair
        energy_temperature_scenario_uhi_report(
            energy + [_e(0, "A", 1)], temp, ZONES, []
        )
    with pytest.raises(ValueError):
        # duplicate temp pair
        energy_temperature_scenario_uhi_report(
            energy, temp + [_v(0, "A", 1.0)], ZONES, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [_e(-1, "A", 10), _e(0, "B", 4)], temp, ZONES, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [_e(True, "A", 10), _e(0, "B", 4)], temp, ZONES, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [(0, "A", 10), _e(0, "B", 4)], temp, ZONES, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [_e(0, "", 10), _e(0, "B", 4)], temp, ZONES, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [_e(0, "A", float("nan")), _e(0, "B", 4)], temp, ZONES, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [_e(0, "A", True), _e(0, "B", 4)], temp, ZONES, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, [_v(0, "A", True), _v(0, "B", 20.0)], ZONES, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, ZONES, [], minutes=7
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, ZONES, [], minutes=True
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, ZONES, [], minutes=0
        )


def test_action_validation():
    energy = [_e(0, "A", 10), _e(0, "B", 4)]
    temp = [_v(0, "A", 30.0), _v(0, "B", 20.0)]

    with pytest.raises(ValueError):
        # not a five-tuple
        energy_temperature_scenario_uhi_report(
            energy, temp, ZONES, [("A", 1, 0, 0)]
        )
    with pytest.raises(ValueError):
        # action cell absent from energy
        energy_temperature_scenario_uhi_report(
            energy, temp, ZONES, [_a("Z", dn=1)]
        )
    with pytest.raises(ValueError):
        # duplicate action cell
        energy_temperature_scenario_uhi_report(
            energy, temp, ZONES, [_a("A", dn=1), _a("A", ds=1)]
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, ZONES, [_a("", dn=1)]
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, ZONES, [_a("A", dn=True)]
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, ZONES, [_a("A", dn=float("inf"))]
        )
