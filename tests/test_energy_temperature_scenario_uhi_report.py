"""Tests for urban_micro.energy_temperature_scenario_uhi_report."""

import json

import pytest

from urban_micro import energy_temperature_scenario_uhi_report

ZONES = {"A": "urban", "B": "urban", "C": "rural"}


def _energy(t, c, n, s, l, st):
    return (t, c, n, s, l, st)


def _temp(t, c, v):
    return (t, c, v)


def _action(c, dn=0.0, ds=0.0, dl=0.0, dst=0.0):
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


def test_basic_single_bucket():
    # urban cells A (r=6) and B (r=6); rural C (r=4)
    energy = [
        _energy(0, "A", 10, 2, 1, 1),
        _energy(0, "B", 8, 1, 1, 0),
        _energy(0, "C", 5, 1, 0, 0),
    ]
    # urban mean v = (30 + 20) / 2 = 25; rural v = 10
    temp = [_temp(0, "A", 30), _temp(0, "B", 20), _temp(0, "C", 10)]
    # action on A: a = 2 -> post A = 8; B unchanged post = 6; urban post = 7
    actions = [_action("A", dn=2)]
    result = energy_temperature_scenario_uhi_report(
        energy, temp, ZONES, actions
    )
    assert result == (
        '{"minutes":60,"groups":[{"key":0,'
        '"base":"2.000000","post":"3.000000","uhi":"15.000000"}]}'
    )
    json.loads(result)


def test_no_actions_means_post_equals_base():
    energy = [
        _energy(0, "A", 10, 2, 1, 1),
        _energy(0, "C", 5, 1, 0, 0),
    ]
    temp = [_temp(0, "A", 30), _temp(0, "C", 10)]
    zones = {"A": "urban", "C": "rural"}
    result = energy_temperature_scenario_uhi_report(energy, temp, zones, [])
    assert result == (
        '{"minutes":60,"groups":[{"key":0,'
        '"base":"2.000000","post":"2.000000","uhi":"20.000000"}]}'
    )


def test_rows_averaged_within_pair_before_action():
    # A has two rows in bucket 0: r = 10 and 12 -> mean r = 11; post = 12
    # B urban: r = 5; rural C: r = 4
    energy = [
        _energy(0, "A", 10, 0, 0, 0),
        _energy(10, "A", 12, 0, 0, 0),
        _energy(0, "B", 5, 0, 0, 0),
        _energy(0, "C", 4, 0, 0, 0),
    ]
    temp = [_temp(0, "A", 0), _temp(0, "B", 0), _temp(0, "C", 0)]
    actions = [_action("A", dn=1)]
    result = energy_temperature_scenario_uhi_report(
        energy, temp, ZONES, actions
    )
    # urban base = (11 + 5) / 2 = 8; urban post = (12 + 5) / 2 = 8.5
    assert '"base":"4.000000"' in result
    assert '"post":"4.500000"' in result
    assert '"uhi":"0.000000"' in result


def test_bucketing_and_minutes_30():
    # t=1799 -> bucket 0; t=1800 -> bucket 1800
    energy = [
        _energy(1799, "A", 10, 0, 0, 0),
        _energy(1799, "C", 4, 0, 0, 0),
        _energy(1800, "A", 8, 0, 0, 0),
        _energy(1800, "C", 4, 0, 0, 0),
    ]
    temp = [
        _temp(1799, "A", 30),
        _temp(1799, "C", 10),
        _temp(1800, "A", 26),
        _temp(1800, "C", 10),
    ]
    zones = {"A": "urban", "C": "rural"}
    result = energy_temperature_scenario_uhi_report(
        energy, temp, zones, [], minutes=30
    )
    assert result == (
        '{"minutes":30,"groups":['
        '{"key":0,"base":"6.000000","post":"6.000000","uhi":"20.000000"},'
        '{"key":1800,"base":"4.000000","post":"4.000000","uhi":"16.000000"}]}'
    )


def test_bucket_missing_zone_dropped():
    # bucket 0 has both zones; bucket 3600 only urban -> dropped
    energy = [
        _energy(0, "A", 10, 0, 0, 0),
        _energy(0, "C", 4, 0, 0, 0),
        _energy(3600, "A", 10, 0, 0, 0),
    ]
    temp = [
        _temp(0, "A", 30),
        _temp(0, "C", 10),
        _temp(3600, "A", 30),
    ]
    zones = {"A": "urban", "C": "rural"}
    result = energy_temperature_scenario_uhi_report(energy, temp, zones, [])
    assert result == (
        '{"minutes":60,"groups":[{"key":0,'
        '"base":"6.000000","post":"6.000000","uhi":"20.000000"}]}'
    )


def test_pair_missing_from_temp_dropped():
    # urban cell A has no temp rows -> urban zone empty -> bucket dropped
    energy = [
        _energy(0, "A", 10, 0, 0, 0),
        _energy(0, "C", 4, 0, 0, 0),
    ]
    temp = [_temp(0, "C", 10)]
    zones = {"A": "urban", "C": "rural"}
    assert (
        energy_temperature_scenario_uhi_report(energy, temp, zones, [])
        == '{"minutes":60,"groups":[]}'
    )


def test_decimal_str_semantics_and_negative_zero():
    energy = [
        _energy(0, "A", 0.1, 0, 0, 0),
        _energy(0, "C", 0, 0, 0, 0),
    ]
    temp = [_temp(0, "A", 0.2), _temp(0, "C", -0.0)]
    zones = {"A": "urban", "C": "rural"}
    result = energy_temperature_scenario_uhi_report(energy, temp, zones, [])
    assert '"base":"0.100000"' in result
    assert '"post":"0.100000"' in result
    assert '"uhi":"0.200000"' in result
    assert "-0.000000" not in result


def test_negative_action_increment_and_difference_signs():
    # urban r = 3, post = 1; rural r = 5, post = 5 -> base -2, post -4
    energy = [
        _energy(0, "A", 3, 0, 0, 0),
        _energy(0, "C", 5, 0, 0, 0),
    ]
    temp = [_temp(0, "A", 10), _temp(0, "C", 12)]
    zones = {"A": "urban", "C": "rural"}
    actions = [_action("A", ds=2)]  # a = -2
    result = energy_temperature_scenario_uhi_report(
        energy, temp, zones, actions
    )
    assert '"base":"-2.000000"' in result
    assert '"post":"-4.000000"' in result
    assert '"uhi":"-2.000000"' in result


def test_no_trailing_newline_and_no_spaces():
    energy = [
        _energy(0, "A", 10, 0, 0, 0),
        _energy(0, "C", 4, 0, 0, 0),
    ]
    temp = [_temp(0, "A", 30), _temp(0, "C", 10)]
    zones = {"A": "urban", "C": "rural"}
    result = energy_temperature_scenario_uhi_report(energy, temp, zones, [])
    assert not result.endswith("\n")
    assert " " not in result


def test_type_errors():
    with pytest.raises(TypeError):
        energy_temperature_scenario_uhi_report("x", [], {}, [])
    with pytest.raises(TypeError):
        energy_temperature_scenario_uhi_report([], "x", {}, [])
    with pytest.raises(TypeError):
        energy_temperature_scenario_uhi_report([], [], "x", [])
    with pytest.raises(TypeError):
        energy_temperature_scenario_uhi_report([], [], {}, "x")


def test_value_errors():
    energy = [
        _energy(0, "A", 10, 0, 0, 0),
        _energy(0, "C", 4, 0, 0, 0),
    ]
    temp = [_temp(0, "A", 30), _temp(0, "C", 10)]
    zones = {"A": "urban", "C": "rural"}

    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [_energy(True, "A", 1, 0, 0, 0)], temp, zones, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [_energy(-1, "A", 1, 0, 0, 0)], temp, zones, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [_energy(0, "", 1, 0, 0, 0)], temp, zones, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [_energy(0, "A", True, 0, 0, 0)], temp, zones, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [_energy(0, "A", float("nan"), 0, 0, 0)], temp, zones, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            [(0, "A", 1, 0, 0)], temp, zones, []
        )
    with pytest.raises(ValueError):  # bool temp timestamp
        energy_temperature_scenario_uhi_report(
            energy, [_temp(True, "A", 1)], zones, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, [_temp(-1, "A", 1)], zones, []
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, [_temp(0, "A", True)], zones, []
        )
    with pytest.raises(ValueError):  # duplicate energy (t, c)
        energy_temperature_scenario_uhi_report(
            energy + [_energy(0, "A", 1, 0, 0, 0)], temp, zones, []
        )
    with pytest.raises(ValueError):  # duplicate temp (t, c)
        energy_temperature_scenario_uhi_report(
            energy, temp + [_temp(0, "A", 1)], zones, []
        )
    with pytest.raises(ValueError):  # zones missing energy cell
        energy_temperature_scenario_uhi_report(
            energy, temp, {"A": "urban"}, []
        )
    with pytest.raises(ValueError):  # zones missing temp-only cell
        energy_temperature_scenario_uhi_report(
            [_energy(0, "A", 10, 0, 0, 0)],
            [_temp(0, "A", 30), _temp(0, "D", 20)],
            {"A": "urban"},
            [],
        )
    with pytest.raises(ValueError):  # bad zone value
        energy_temperature_scenario_uhi_report(
            energy, temp, {"A": "city", "C": "rural"}, []
        )
    with pytest.raises(ValueError):  # empty zone key
        energy_temperature_scenario_uhi_report(
            energy, temp, {"A": "urban", "": "rural", "C": "rural"}, []
        )
    with pytest.raises(ValueError):  # action cell not in energy
        energy_temperature_scenario_uhi_report(
            energy, temp, zones, [_action("Z")]
        )
    with pytest.raises(ValueError):  # duplicate action cell
        energy_temperature_scenario_uhi_report(
            energy, temp, zones, [_action("A"), _action("A", dn=1)]
        )
    with pytest.raises(ValueError):  # non-finite action increment
        energy_temperature_scenario_uhi_report(
            energy, temp, zones, [_action("A", dn=float("inf"))]
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(energy, temp, zones, [], minutes=7)
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(energy, temp, zones, [], minutes=0)
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, zones, [], minutes=True
        )
    with pytest.raises(ValueError):
        energy_temperature_scenario_uhi_report(
            energy, temp, zones, [], minutes=1441
        )
