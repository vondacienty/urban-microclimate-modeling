"""Tests for urban_micro.driver_plan_frontier: driver_plan optimum picks
traced across ascending limits, per region/window group."""

import json

import pytest

from urban_micro import driver_plan, driver_plan_frontier
from urban_micro import uhi as _uhi

from test_driver_plan import (
    _actions,
    _group,
    _item,
    _one_region,
    _report,
)

_POINT_KEYS = ["limit", "groups"]
_GROUP_KEYS = [
    "by", "cost", "priority", "marginal", "pick", "added", "removed",
    "effects", "range", "dominated",
]
_EMPTY_REPORT = '{"alpha":0.050000,"groups":[]}\n'


def _payload(out):
    return json.loads(out)


def _groups(payload, by="region"):
    return [
        next(group for group in point["groups"] if group["by"] == by)
        for point in payload["points"]
    ]


def _all_ineligible_group(by):
    return _group(by, [
        _item(1, "station", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        _item(2, "lst", "energy", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        _item(3, "morph", "vent", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        _item(4, "cover", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
    ])


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    out = driver_plan_frontier(_one_region(), _actions(), [0, 10.0])
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = _payload(out)
    assert list(payload) == ["points"]
    assert isinstance(payload["points"], list)
    for point in payload["points"]:
        assert list(point) == _POINT_KEYS
        assert isinstance(point["groups"], list)
        for group in point["groups"]:
            assert list(group) == _GROUP_KEYS
            assert list(group["effects"]) == ["uhi", "energy", "vent"]
            assert list(group["range"]) == ["low", "high"]
            assert isinstance(group["pick"], list)
            assert isinstance(group["added"], list)
            assert isinstance(group["removed"], list)
            assert isinstance(group["dominated"], bool)


def test_points_sorted_ascending_regardless_of_input_order():
    out = driver_plan_frontier(_one_region(), _actions(), [10, 0, 1])
    limits = [point["limit"] for point in _payload(out)["points"]]
    assert limits == [0.0, 1.0, 10.0]
    assert out.count('"limit":') == 3


def test_region_before_window_groups():
    report = _report(
        _all_ineligible_group("region"),
        _all_ineligible_group("window"),
    )
    out = driver_plan_frontier(report, _actions(), [1.0])
    groups = _payload(out)["points"][0]["groups"]
    assert [group["by"] for group in groups] == ["region", "window"]


def test_empty_report_emits_empty_groups_per_point():
    out = driver_plan_frontier(_EMPTY_REPORT, _actions(), [2, 1])
    assert out == (
        '{"points":[{"limit":1.000000,"groups":[]},'
        '{"limit":2.000000,"groups":[]}]}\n'
    )


# --- parity with driver_plan ------------------------------------------------

@pytest.mark.parametrize("limits", [[0], [0, 1, 2, 10], [10, 1, 0, 2]])
def test_each_point_matches_driver_plan(limits):
    report = _one_region()
    actions = _actions()
    payload = _payload(driver_plan_frontier(report, actions, limits))
    for point in payload["points"]:
        single = _payload(driver_plan(report, actions, point["limit"]))
        assert len(point["groups"]) == len(single["groups"]) == 1
        group, single_group = point["groups"][0], single["groups"][0]
        assert group["by"] == single_group["by"]
        for key in ("cost", "priority", "pick", "effects", "range"):
            assert group[key] == single_group[key]


def test_each_point_matches_driver_plan_region_and_window():
    report = _report(
        _all_ineligible_group("region"),
        _all_ineligible_group("window"),
    )
    actions = _actions()
    payload = _payload(driver_plan_frontier(report, actions, [0, 5]))
    for point in payload["points"]:
        single = _payload(driver_plan(report, actions, point["limit"]))
        assert [g["by"] for g in point["groups"]] == [
            g["by"] for g in single["groups"]
        ]
        for group, single_group in zip(point["groups"], single["groups"]):
            for key in ("cost", "priority", "pick", "effects", "range"):
                assert group[key] == single_group[key]


# --- marginal / added / removed ---------------------------------------------

def test_first_point_baseline_state():
    # From the implicit priority=0, pick=[] state.
    group = _groups(_payload(driver_plan_frontier(
        _one_region(), _actions(), [10.0]
    )))[0]
    assert group["pick"] == ["green", "roof"]
    assert group["priority"] == pytest.approx(5.0)
    assert group["marginal"] == pytest.approx(5.0)
    assert group["added"] == ["green", "roof"]
    assert group["removed"] == []


def test_marginal_added_and_removed_track_previous_point():
    # green->station score 30 at cost 10 (ratio 3); roof->lst score 2 at
    # cost 1 (ratio 2); both together cost 11. Limit 1 picks roof alone;
    # limit 10 replaces it by green (strictly higher priority).
    report = _one_region(
        station=(30.0, -2.0, -0.5, -0.2, True),
        lst=(2.0, -5.0, -0.4, -0.1, True),
    )
    actions = _actions(green=("station", 10.0), roof=("lst", 1.0))
    groups = _groups(_payload(driver_plan_frontier(
        report, actions, [1, 10]
    )))
    first, second = groups
    assert first["pick"] == ["roof"]
    assert first["marginal"] == pytest.approx(2.0)
    assert first["added"] == ["roof"]
    assert first["removed"] == []
    assert second["pick"] == ["green"]
    assert second["priority"] == pytest.approx(3.0)
    assert second["marginal"] == pytest.approx(1.0)
    assert second["added"] == ["green"]
    assert second["removed"] == ["roof"]
    assert second["effects"] == {"uhi": -2.0, "energy": 0.0, "vent": 0.0}
    assert second["range"] == {"low": -0.5, "high": -0.2}


def test_unchanged_pick_has_zero_marginal_and_empty_differences():
    # roof wins at limit 1; limit 2 cannot add green (cost 2 on its own,
    # roof+green costs 3) so the optimum holds.
    groups = _groups(_payload(driver_plan_frontier(
        _one_region(), _actions(), [1, 2]
    )))
    first, second = groups
    assert first["pick"] == second["pick"] == ["roof"]
    assert second["marginal"] == 0.0
    assert second["added"] == []
    assert second["removed"] == []


def test_differences_are_per_group_not_global():
    # Window group stays empty throughout; its added/removed never leak
    # the region picks.
    report = _report(
        _group("region", [
            _item(1, "station", "uhi", 4.0, -0.5, -0.2, 0.01, 4.0, True),
            _item(2, "lst", "energy", 3.0, -0.4, -0.1, 0.01, 3.0, True),
            _item(3, "morph", "vent", 0.0, 0.0, 0.0, 1.0, 0.0, False),
            _item(4, "cover", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        ]),
        _all_ineligible_group("window"),
    )
    payload = _payload(driver_plan_frontier(report, _actions(), [1, 10]))
    for point in payload["points"]:
        window = next(g for g in point["groups"] if g["by"] == "window")
        assert window["pick"] == []
        assert window["added"] == []
        assert window["removed"] == []
        assert window["marginal"] == 0.0


# --- dominated --------------------------------------------------------------

def test_dominated_flags():
    # limit 1: roof (priority 3); limit 2: identical optimum -> dominated;
    # limit 3: green+roof (priority 5) -> not dominated.
    groups = _groups(_payload(driver_plan_frontier(
        _one_region(), _actions(), [1, 2, 3]
    )))
    assert [group["dominated"] for group in groups] == [False, True, False]


def test_first_point_never_dominated():
    groups = _groups(_payload(driver_plan_frontier(
        _one_region(), _actions(), [0]
    )))
    assert groups[0]["dominated"] is False


def test_empty_pick_hold_is_dominated():
    # With no eligible items the empty state repeats from the second
    # point on; the equal earlier point dominates it.
    report = _one_region(
        station=(0.0, -2.0, 0.0, 0.0, False),
        lst=(0.0, -5.0, 0.0, 0.0, False),
    )
    groups = _groups(_payload(driver_plan_frontier(
        report, _actions(), [0, 1, 2]
    )))
    assert [group["dominated"] for group in groups] == [
        False, True, True
    ]


def test_strictly_improving_points_never_dominated():
    report = _one_region(
        station=(30.0, -2.0, -0.5, -0.2, True),
        lst=(2.0, -5.0, -0.4, -0.1, True),
    )
    actions = _actions(green=("station", 10.0), roof=("lst", 1.0))
    groups = _groups(_payload(driver_plan_frontier(
        report, actions, [1, 10]
    )))
    assert [group["dominated"] for group in groups] == [False, False]


# --- number formatting ------------------------------------------------------

def test_six_decimals_and_no_negative_zero():
    out = driver_plan_frontier(_one_region(), _actions(), [0, 1])
    assert "-0.000000" not in out
    assert '"limit":0.000000' in out
    assert '"marginal":0.000000' in out
    assert '"effects":{"uhi":0.000000,"energy":0.000000,"vent":0.000000}' in out


# --- validation -------------------------------------------------------------

def test_type_errors():
    report = _one_region()
    actions = _actions()
    for bad in (1, None, [], True, {}):
        with pytest.raises(TypeError):
            driver_plan_frontier(bad, actions, [1.0])
    for bad in ([], None, 42, "x"):
        with pytest.raises(TypeError):
            driver_plan_frontier(report, bad, [1.0])
    for bad in ((), None, 42, "x", {}, (1.0,)):
        with pytest.raises(TypeError):
            driver_plan_frontier(report, actions, bad)


def test_type_errors_take_precedence():
    with pytest.raises(TypeError):
        driver_plan_frontier(1, [], 0)


@pytest.mark.parametrize(
    "actions",
    [
        {},
        {"green": ("station", 1.0), "roof": ("lst", 1.0)},
        dict(_actions(), extra=("morph", 1.0)),
        {"Green": ("station", 1.0), "roof": ("lst", 1.0),
         "material": ("morph", 1.0)},
    ],
)
def test_actions_key_set_must_be_exact(actions):
    with pytest.raises(ValueError):
        driver_plan_frontier(_one_region(), actions, [1.0])


@pytest.mark.parametrize(
    "value",
    [
        ["station", 1.0],
        ("station",),
        ("station", 1.0, 2.0),
        ("station", "1.0"),
        ("station", 0),
        ("station", -1.0),
        ("station", True),
        ("station", float("nan")),
        (1, 1.0),
        ("nope", 1.0),
    ],
)
def test_bad_action_values_rejected(value):
    with pytest.raises(ValueError):
        driver_plan_frontier(
            _one_region(), dict(_actions(), green=value), [1.0]
        )


def test_duplicate_factors_rejected():
    actions = _actions(
        green=("station", 1.0), roof=("station", 2.0),
        material=("morph", 3.0),
    )
    with pytest.raises(ValueError):
        driver_plan_frontier(_one_region(), actions, [1.0])


def test_empty_limits_rejected():
    with pytest.raises(ValueError):
        driver_plan_frontier(_one_region(), _actions(), [])


@pytest.mark.parametrize(
    "limits",
    [
        [-1],
        [-0.1, 1.0],
        [True],
        [False],
        ["1"],
        [None],
        [float("nan")],
        [float("inf")],
        [float("-inf")],
        [[1.0]],
    ],
)
def test_bad_limit_elements_rejected(limits):
    with pytest.raises(ValueError):
        driver_plan_frontier(_one_region(), _actions(), limits)


@pytest.mark.parametrize("limits", [[1, 1], [1.0, 1], [2.0, 2.00], [1, 1.0]])
def test_duplicate_limits_rejected(limits):
    with pytest.raises(ValueError):
        driver_plan_frontier(_one_region(), _actions(), limits)


def test_bool_limit_and_non_bool_distinction():
    # bool is not accepted even though it is an int subclass.
    with pytest.raises(ValueError):
        driver_plan_frontier(_one_region(), _actions(), [0, False])


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json\n",
        '{"alpha":0.050000,"groups":[]}',
        '{"alpha":0.050000,"groups":[]}\n\n',
        '{"groups":[],"alpha":0.050000}\n',
        '{"alpha":0.05,"groups":[]}\n',
        '{"groups":[]}\n',
    ],
)
def test_malformed_report_value_error(raw):
    with pytest.raises(ValueError):
        driver_plan_frontier(raw, _actions(), [1.0])


def test_validation_runs_before_report_parse_for_limits():
    # A canonical report but invalid limits still surfaces ValueError.
    with pytest.raises(ValueError):
        driver_plan_frontier(_one_region(), _actions(), [1, 1])


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.driver_plan_frontier is urban_micro.driver_plan_frontier
    assert "driver_plan_frontier" in urban_micro.__all__
