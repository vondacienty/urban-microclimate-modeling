"""Tests for urban_micro.driver_plan_frontier: trace the driver_plan
optimum over several budget limits with marginal changes and dominance."""

import json

import pytest

from urban_micro import driver_plan, driver_plan_frontier
from urban_micro import uhi as _uhi

_GROUP_KEYS = [
    "by", "cost", "priority", "marginal", "pick", "added", "removed",
    "effects", "range", "dominated",
]
_POINT_KEYS = ["limit", "groups"]


# --- direct canonical driver_link report builders ---------------------------

def _item(rank, factor, metric, effect, low, high, q, score, eligible):
    return (
        '{"factor":' + json.dumps(factor)
        + ',"metric":' + json.dumps(metric)
        + f',"effect":{effect:.6f},"low":{low:.6f},"high":{high:.6f},'
        + f'"q":{q:.6f},"score":{score:.6f},'
        + f'"eligible":{str(eligible).lower()},"rank":{rank}}}'
    )


def _group(by, items):
    return '{"by":' + json.dumps(by) + ',"items":[' + ",".join(items) + "]}"


def _report(*groups, alpha=0.05):
    return ('{"alpha":' + f"{alpha:.6f}" + ',"groups":['
            + ",".join(groups) + "]}\n")


def _one_region(station=(4.0, -2.0, -0.5, -0.2, True),
                lst=(3.0, -5.0, -0.4, -0.1, True),
                morph=(0.0, 3.0, 0.0, 0.0, False),
                cover=(0.0, 1.0, 0.0, 0.0, False)):
    rows = (
        ("station", "uhi", station),
        ("lst", "energy", lst),
        ("morph", "vent", morph),
        ("cover", "uhi", cover),
    )
    tokens = [
        _item(rank, factor, metric, effect, low, high, 0.01, score, eligible)
        for rank, (factor, metric, (score, effect, low, high, eligible))
        in enumerate(rows, start=1)
    ]
    return _report(_group("region", tokens))


def _one_window(station=(1.0, -1.0, -0.1, -0.05, True),
                lst=(6.0, -2.0, -0.2, -0.1, True),
                morph=(0.0, 3.0, 0.0, 0.0, False),
                cover=(0.0, 1.0, 0.0, 0.0, False)):
    rows = (
        ("station", "uhi", station),
        ("lst", "energy", lst),
        ("morph", "vent", morph),
        ("cover", "uhi", cover),
    )
    tokens = [
        _item(rank, factor, metric, effect, low, high, 0.01, score, eligible)
        for rank, (factor, metric, (score, effect, low, high, eligible))
        in enumerate(rows, start=1)
    ]
    return _group("window", tokens)


def _actions(green=("station", 2.0), roof=("lst", 1.0),
             material=("morph", 1.0)):
    return {"green": green, "roof": roof, "material": material}


def _points(out):
    return json.loads(out)["points"]


def _groups(out):
    return [
        {g["by"]: g for g in point["groups"]}
        for point in _points(out)
    ]


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    out = driver_plan_frontier(_one_region(), _actions(), [10.0])
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["points"]
    point = payload["points"][0]
    assert list(point) == _POINT_KEYS
    group = point["groups"][0]
    assert list(group) == _GROUP_KEYS
    assert list(group["effects"]) == ["uhi", "energy", "vent"]
    assert list(group["range"]) == ["low", "high"]
    assert isinstance(group["dominated"], bool)


def test_points_sorted_ascending_and_groups_region_before_window():
    report = _report(
        _group("region", [
            _item(r, f, m, 0.0, 0.0, 0.0, 1.0, 0.0, False)
            for r, (f, m) in enumerate(
                [("station", "uhi"), ("lst", "energy"),
                 ("morph", "vent"), ("cover", "uhi")], start=1)
        ]),
        _one_window(),
    )
    out = driver_plan_frontier(report, _actions(), [10, 0, 2.5, 1])
    points = _points(out)
    assert [p["limit"] for p in points] == [0.0, 1.0, 2.5, 10.0]
    for point in points:
        assert [g["by"] for g in point["groups"]] == ["region", "window"]


def test_empty_report_yields_empty_groups_per_point():
    out = driver_plan_frontier(
        '{"alpha":0.050000,"groups":[]}\n', _actions(), [1, 0]
    )
    assert out == (
        '{"points":['
        '{"limit":0.000000,"groups":[]},'
        '{"limit":1.000000,"groups":[]}'
        "]}\n"
    )


def test_single_limit_matches_driver_plan():
    report = _report(
        _group("region", [
            _item(r, *args)
            for r, args in enumerate([
                ("station", "uhi", -2.0, -0.5, -0.2, 0.01, 4.0, True),
                ("lst", "energy", -5.0, -0.4, -0.1, 0.01, 3.0, True),
                ("morph", "vent", 0.0, 0.0, 0.0, 1.0, 0.0, False),
                ("cover", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
            ], start=1)
        ])
    )
    actions = _actions()
    for limit in (0, 1, 2, 10):
        frontier = _groups(driver_plan_frontier(report, actions, [limit]))[0]
        single = {
            g["by"]: g
            for g in json.loads(driver_plan(report, actions, float(limit)))[
                "groups"
            ]
        }
        for by, group in frontier.items():
            ref = single[by]
            for key in ("cost", "priority", "pick", "effects", "range"):
                assert group[key] == ref[key]


def test_first_uses_baseline_and_later_use_previous_point():
    # region ratios: green->station 4/2 = 2, roof->lst 3/1 = 3.
    out = driver_plan_frontier(_one_region(), _actions(), [0, 1, 2, 10])
    region = [g["region"] for g in _groups(out)]
    assert region[0]["pick"] == []
    assert region[0]["marginal"] == 0.0
    assert region[0]["added"] == [] and region[0]["removed"] == []
    assert region[1]["pick"] == ["roof"]
    assert region[1]["marginal"] == 3.0
    assert region[1]["added"] == ["roof"] and region[1]["removed"] == []
    # limit 2 still only buys roof: identical pick, zero marginal.
    assert region[2]["pick"] == ["roof"]
    assert region[2]["marginal"] == 0.0
    assert region[2]["added"] == [] and region[2]["removed"] == []
    # limit 10 takes both: marginal 2.0 and green added.
    assert region[3]["pick"] == ["green", "roof"]
    assert region[3]["marginal"] == 2.0
    assert region[3]["added"] == ["green"] and region[3]["removed"] == []


def test_removed_diff_when_pick_shrinks():
    # green->station: score 15 cost 3 (ratio 5); roof->lst and
    # material->morph: score 6 cost 2 each (ratio 3). At limit 3 green
    # alone wins (priority 5); at limit 4 green is still feasible but
    # the roof+material pair scores 6, so green is removed.
    report = _one_region(
        station=(15.0, -2.0, -0.5, -0.2, True),
        lst=(6.0, -5.0, -0.4, -0.1, True),
        morph=(6.0, 3.0, 0.1, 0.2, True),
    )
    actions = {
        "green": ("station", 3.0),
        "roof": ("lst", 2.0),
        "material": ("morph", 2.0),
    }
    out = driver_plan_frontier(report, actions, [3, 4])
    region = [g["region"] for g in _groups(out)]
    assert region[0]["pick"] == ["green"]
    assert region[1]["pick"] == ["material", "roof"]
    assert region[1]["added"] == ["material", "roof"]
    assert region[1]["removed"] == ["green"]
    assert region[1]["marginal"] == pytest.approx(1.0)


def test_dominated_when_smaller_limit_matches_priority_and_cost():
    out = driver_plan_frontier(_one_region(), _actions(), [1, 2])
    region = [g["region"] for g in _groups(out)]
    assert region[0]["dominated"] is False
    assert region[1]["dominated"] is True


def test_not_dominated_when_priority_improves():
    # At limit 1 roof alone (priority 3); at limit 10 both (priority 5).
    out = driver_plan_frontier(_one_region(), _actions(), [1, 10])
    region = [g["region"] for g in _groups(out)]
    assert region[0]["dominated"] is False
    assert region[1]["dominated"] is False


def test_dominance_is_per_group_and_per_point():
    report = _report(
        _group("region", [
            _item(r, *args)
            for r, args in enumerate([
                ("station", "uhi", -2.0, -0.5, -0.2, 0.01, 4.0, True),
                ("lst", "energy", -5.0, -0.4, -0.1, 0.01, 3.0, True),
                ("morph", "vent", 0.0, 0.0, 0.0, 1.0, 0.0, False),
                ("cover", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
            ], start=1)
        ]),
        _one_window(),
    )
    # region plateaus after limit 1 (dominated at 2); window jumps at 10.
    out = driver_plan_frontier(report, _actions(), [1, 2, 10])
    groups = _groups(out)
    assert [g["region"]["dominated"] for g in groups] == [
        False, True, False
    ]
    assert [g["window"]["dominated"] for g in groups] == [
        False, True, False
    ]


def test_dominated_only_uses_strictly_smaller_limits():
    # Two distinct Decimal limits that sort apart never compare each other
    # at equality; a single point is never dominated.
    out = driver_plan_frontier(_one_region(), _actions(), [10])
    assert _groups(out)[0]["region"]["dominated"] is False


def test_six_decimals_and_no_negative_zero():
    out = driver_plan_frontier(_one_region(), _actions(), [0, 10])
    assert "-0.000000" not in out
    assert '"limit":0.000000' in out
    assert '"effects":{"uhi":-2.000000,"energy":-5.000000,"vent":0.000000}' in out


# --- validation -------------------------------------------------------------

def test_type_errors():
    report = _one_region()
    actions = _actions()
    for bad in (1, None, [], True, {}):
        with pytest.raises(TypeError):
            driver_plan_frontier(bad, actions, [10.0])
    for bad in ([], None, 42, "x"):
        with pytest.raises(TypeError):
            driver_plan_frontier(report, bad, [10.0])
    for bad in ({}, None, 1, "x", (1,)):
        with pytest.raises(TypeError):
            driver_plan_frontier(report, actions, bad)


def test_type_errors_take_precedence():
    with pytest.raises(TypeError):
        driver_plan_frontier(1, [], {})


@pytest.mark.parametrize(
    "bad",
    [
        [],               # empty
        [1, 1.0],         # equal by Decimal(str())
        [True],           # boolean
        [False],
        ["1"],            # string
        [None],
        [float("nan")],
        [float("inf")],
        [float("-inf")],
        [-1],
        [-0.1],
        [1, 1],           # duplicate
        [1, [2]],
    ],
)
def test_bad_limits_rejected(bad):
    with pytest.raises(ValueError):
        driver_plan_frontier(_one_region(), _actions(), bad)


def test_actions_violations_still_value_error():
    with pytest.raises(ValueError):
        driver_plan_frontier(_one_region(), {}, [1.0])


def test_malformed_report_value_error():
    with pytest.raises(ValueError):
        driver_plan_frontier("not json\n", _actions(), [1.0])


def test_integer_limits_accepted():
    out = driver_plan_frontier(_one_region(), _actions(), [0, 1, 2])
    assert [p["limit"] for p in _points(out)] == [0.0, 1.0, 2.0]


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.driver_plan_frontier is urban_micro.driver_plan_frontier
    assert "driver_plan_frontier" in urban_micro.__all__
