"""Tests for urban_micro.driver_plan_pareto: non-dominated plan
combinations selected from a canonical driver_plan_frontier output."""

import json

import pytest

from urban_micro import driver_plan_frontier, driver_plan_pareto
from urban_micro import uhi as _uhi

from test_driver_plan import _actions, _one_region, _report, _group, _item

_POINT_KEYS = ["limit", "groups"]
_GROUP_KEYS = ["by", "recommend", "items"]
_ITEM_KEYS = ["pick", "cost", "uhi", "energy", "vent", "score"]
_WEIGHTS = {"uhi": 1.0, "energy": 1.0, "vent": 1.0}


def _payload(out):
    return json.loads(out)


def _strarr(values):
    return "[" + ",".join(json.dumps(value) for value in values) + "]"


def _fgroup(by, cost, priority, marginal, pick, added, removed,
            uhi, energy, vent, dominated, low=0.0, high=0.0):
    return (
        '{"by":' + json.dumps(by)
        + f',"cost":{cost:.6f},"priority":{priority:.6f}'
        + f',"marginal":{marginal:.6f}'
        + ',"pick":' + _strarr(pick)
        + ',"added":' + _strarr(added)
        + ',"removed":' + _strarr(removed)
        + f',"effects":{{"uhi":{uhi:.6f},"energy":{energy:.6f},'
        + f'"vent":{vent:.6f}}}'
        + f',"range":{{"low":{low:.6f},"high":{high:.6f}}}'
        + ',"dominated":' + ("true" if dominated else "false")
        + "}"
    )


def _fpoint(limit, *groups):
    return (
        f'{{"limit":{limit:.6f},"groups":[' + ",".join(groups) + "]}"
    )


def _frontier(*points):
    return '{"points":[' + ",".join(points) + "]}\n"


def _state(by, pick, cost, uhi=0.0, energy=0.0, vent=0.0, dominated=False):
    """One frontier group with values chosen so the text is canonical."""
    return _fgroup(
        by, cost, cost, cost, pick, pick, [], uhi, energy, vent, dominated
    )


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3, 10])
    out = driver_plan_pareto(frontier, _WEIGHTS, 10.0)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = _payload(out)
    assert list(payload) == _POINT_KEYS
    assert payload["limit"] == 10.0
    assert isinstance(payload["groups"], list)
    group = payload["groups"][0]
    assert list(group) == _GROUP_KEYS
    assert isinstance(group["items"], list)
    for item in group["items"]:
        assert list(item) == _ITEM_KEYS
        assert isinstance(item["pick"], list)


def test_region_before_window_groups():
    all_ineligible = [
        _item(rank, factor, metric, 0.0, 0.0, 0.0, 1.0, 0.0, False)
        for rank, (factor, metric) in enumerate(
            (("station", "uhi"), ("lst", "energy"),
             ("morph", "vent"), ("cover", "uhi")), start=1
        )
    ]
    report = _report(
        _group("region", all_ineligible),
        _group("window", all_ineligible),
    )
    frontier = driver_plan_frontier(report, _actions(), [1.0])
    out = driver_plan_pareto(frontier, _WEIGHTS, 5.0)
    payload = _payload(out)
    assert [group["by"] for group in payload["groups"]] == [
        "region", "window"
    ]
    for group in payload["groups"]:
        assert group["recommend"] == []
        assert [item["pick"] for item in group["items"]] == [[]]


def test_empty_report_frontier_emits_no_groups():
    frontier = driver_plan_frontier(
        '{"alpha":0.050000,"groups":[]}\n', _actions(), [1, 2]
    )
    out = driver_plan_pareto(frontier, _WEIGHTS, 5.0)
    assert out == '{"limit":5.000000,"groups":[]}\n'


# --- gathering, dedup and filtering -----------------------------------------

def test_gathers_distinct_picks_across_points():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3, 10])
    payload = _payload(driver_plan_pareto(frontier, _WEIGHTS, 10.0))
    items = payload["groups"][0]["items"]
    assert [item["pick"] for item in items] == [
        ["green", "roof"], ["roof"], []
    ]
    assert payload["groups"][0]["recommend"] == ["green", "roof"]


def test_repeated_pick_is_deduplicated():
    # Limits 1 and 2 hold the same roof optimum in the real frontier.
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 2])
    payload = _payload(driver_plan_pareto(frontier, _WEIGHTS, 10.0))
    picks = [item["pick"] for item in payload["groups"][0]["items"]]
    assert picks.count(["roof"]) == 1


def test_cost_limit_is_inclusive_boundary():
    frontier = _frontier(
        _fpoint(
            0.0,
            _state("region", [], 0.0),
        ),
        _fpoint(
            2.0,
            _state("region", ["roof"], 2.0, energy=-4.0),
        ),
    )
    out = driver_plan_pareto(frontier, _WEIGHTS, 2)
    picks = [item["pick"] for item in _payload(out)["groups"][0]["items"]]
    assert picks == [["roof"], []]


def test_cost_limit_filters_expensive_combinations():
    frontier = _frontier(
        _fpoint(
            0.0,
            _state("region", [], 0.0),
        ),
        _fpoint(
            1.0,
            _state("region", ["roof"], 1.0, energy=-5.0),
        ),
        _fpoint(
            3.0,
            _state("region", ["green", "roof"], 3.0, uhi=-2.0,
                   energy=-5.0),
        ),
    )
    out = driver_plan_pareto(frontier, _WEIGHTS, 1)
    items = _payload(out)["groups"][0]["items"]
    assert [item["pick"] for item in items] == [["roof"], []]
    assert [item["cost"] for item in items] == [1.0, 0.0]


def test_group_present_with_null_recommend_when_nothing_admissible():
    frontier = _frontier(
        _fpoint(
            2.0,
            _state("region", ["roof"], 2.0, energy=-4.0),
        ),
    )
    out = driver_plan_pareto(frontier, _WEIGHTS, 1)
    assert out == (
        '{"limit":1.000000,"groups":[{"by":"region","recommend":null,'
        '"items":[]}]}\n'
    )


# --- Pareto dominance -------------------------------------------------------

def test_positive_energy_combo_dominated_by_empty():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=5.0)),
        _fpoint(2.0, _state("region", ["green"], 2.0, uhi=-1.0)),
    )
    items = _payload(driver_plan_pareto(frontier, _WEIGHTS, 10))[
        "groups"
    ][0]["items"]
    assert [item["pick"] for item in items] == [["green"], []]


def test_negative_vent_combo_dominated_via_negated_vent_component():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["material"], 1.0, vent=-2.0)),
        _fpoint(2.0, _state("region", ["roof"], 2.0, vent=3.0)),
    )
    items = _payload(driver_plan_pareto(frontier, _WEIGHTS, 10))[
        "groups"
    ][0]["items"]
    # material: objective (1,0,0,2) is dominated by [] (0,0,0,0);
    # roof with vent 3 -> component -3 is incomparable with [].
    assert [item["pick"] for item in items] == [["roof"], []]


def test_non_empty_combo_can_dominate_another():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(
            2.0,
            _state("region", ["green"], 2.0, uhi=-1.0, energy=-1.0),
        ),
        _fpoint(
            3.0,
            _state("region", ["roof"], 3.0, uhi=-1.0, energy=0.0),
        ),
    )
    items = _payload(driver_plan_pareto(frontier, _WEIGHTS, 10))[
        "groups"
    ][0]["items"]
    # green (2,-1,-1,0) dominates roof (3,-1,0,0).
    assert [item["pick"] for item in items] == [["green"], []]


# --- scoring and ordering ---------------------------------------------------

def test_score_uses_weights_and_vent_negation():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["material"], 1.0, vent=3.0)),
    )
    out = driver_plan_pareto(
        frontier, {"uhi": 2.0, "energy": 3.0, "vent": 5.0}, 10
    )
    material = _payload(out)["groups"][0]["items"][0]
    assert material["score"] == -15.0
    assert material["vent"] == 3.0


def test_ordering_follows_weighted_scores():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=-2.0)),
        _fpoint(1.0, _state("region", ["green"], 1.0, uhi=-1.0)),
    )
    # Distinct points may carry the same frontier limit here because the
    # parser only requires strictly ascending limits; emit them at 1 and 2.
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=-2.0)),
        _fpoint(2.0, _state("region", ["green"], 1.0, uhi=-1.0)),
    )
    weights_uhi = {"uhi": 10.0, "energy": 1.0, "vent": 1.0}
    items = _payload(driver_plan_pareto(frontier, weights_uhi, 10))[
        "groups"
    ][0]["items"]
    assert [item["pick"] for item in items] == [
        ["green"], ["roof"], []
    ]
    weights_energy = {"uhi": 1.0, "energy": 10.0, "vent": 1.0}
    items = _payload(driver_plan_pareto(frontier, weights_energy, 10))[
        "groups"
    ][0]["items"]
    assert [item["pick"] for item in items] == [
        ["roof"], ["green"], []
    ]


def test_tied_score_falls_back_to_cost_then_pick():
    # Both plans cost 1: roof uhi -2 scores -4 under w_uhi=2; green
    # energy -4 scores -4 under w_energy=1; vectors are incomparable.
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, uhi=-2.0)),
        _fpoint(2.0, _state("region", ["green"], 1.0, energy=-4.0)),
    )
    out = driver_plan_pareto(
        frontier, {"uhi": 2.0, "energy": 1.0, "vent": 1.0}, 10
    )
    group = _payload(out)["groups"][0]
    assert group["recommend"] == ["green"]
    assert [item["pick"] for item in group["items"]] == [
        ["green"], ["roof"], []
    ]
    assert group["items"][0]["score"] == group["items"][1]["score"] == -4.0


def test_pick_lexicographic_order_uses_full_tuple():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, uhi=-1.0)),
        _fpoint(2.0, _state("region", ["material"], 1.0, energy=-1.0)),
    )
    weights = {"uhi": 1.0, "energy": 1.0, "vent": 1.0}
    items = _payload(driver_plan_pareto(frontier, weights, 10))[
        "groups"
    ][0]["items"]
    # Equal score (-1) and cost (1): ("material",) < ("roof",).
    assert [item["pick"] for item in items[:2]] == [
        ["material"], ["roof"]
    ]


def test_dominance_compares_unquantized_decimal_values():
    # 0.1 + 0.2 floats would mis-compare if re-parsed as float; the
    # effects differ by exactly Decimal('0.000001') after six-decimal
    # quantization, so dominance must still be decided exactly.
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(
            1.0,
            _state("region", ["green"], 1.0, uhi=-1.000001),
        ),
        _fpoint(
            2.0,
            _state("region", ["roof"], 1.0, uhi=-1.000000),
        ),
    )
    items = _payload(driver_plan_pareto(frontier, _WEIGHTS, 10))[
        "groups"
    ][0]["items"]
    # Equal cost; green's uhi is strictly smaller -> green dominates roof.
    assert [item["pick"] for item in items] == [["green"], []]
    assert items[0]["uhi"] == -1.000001


# --- number formatting ------------------------------------------------------

def test_six_decimals_and_no_negative_zero():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0)),
    )
    out = driver_plan_pareto(frontier, _WEIGHTS, 10)
    assert "-0.000000" not in out
    assert '"limit":10.000000' in out
    assert (
        '{"pick":[],"cost":0.000000,"uhi":0.000000,"energy":0.000000,'
        '"vent":0.000000,"score":0.000000}' in out
    )


# --- validation -------------------------------------------------------------

def test_type_errors():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    for bad in (1, None, [], True, {}):
        with pytest.raises(TypeError):
            driver_plan_pareto(bad, _WEIGHTS, 1.0)
    for bad in ([], None, 42, "x", True):
        with pytest.raises(TypeError):
            driver_plan_pareto(frontier, bad, 1.0)


def test_type_errors_take_precedence():
    with pytest.raises(TypeError):
        driver_plan_pareto(1, [], 0)


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"uhi": 1.0, "energy": 1.0},
        {"uhi": 1.0, "energy": 1.0, "vent": 1.0, "extra": 1.0},
        {"UHI": 1.0, "energy": 1.0, "vent": 1.0},
    ],
)
def test_weights_key_set_must_be_exact(weights):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        driver_plan_pareto(frontier, weights, 1.0)


@pytest.mark.parametrize(
    "value",
    [0, -1.0, True, False, None, "1", float("nan"), float("inf"),
     float("-inf"), [1.0], (1.0,)],
)
def test_bad_weight_values_rejected(value):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    weights = dict(_WEIGHTS, uhi=value)
    with pytest.raises(ValueError):
        driver_plan_pareto(frontier, weights, 1.0)


@pytest.mark.parametrize(
    "limit",
    [-1, -0.1, True, False, "1", None, float("nan"), float("inf"),
     float("-inf"), [1], (1.0,)],
)
def test_bad_limit_rejected(limit):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        driver_plan_pareto(frontier, _WEIGHTS, limit)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json\n",
        '{"points":[]}',
        '{"points":[]}\n\n',
        '{"points":[]} ',
        '{"points":[]}\n ',
        '[]\n',
        '{}',
        '{"points":[]}\n',
        '{"points":[1]}\n',
        '{"points":[{"limit":0.000000}]}\n',
        '{"points":[{"groups":[]}]}\n',
        '{"points":[{"limit":0,"groups":[]}]}\n',
        '{"points":[{"limit":0.0,"groups":[]}]}\n',
        '{"points":[{"limit":-0.000000,"groups":[]}]}\n',
        '{"points":[{"limit":0.000000,"groups":{}}]}\n',
        # limits must be strictly ascending.
        '{"points":[{"limit":1.000000,"groups":[]},'
        '{"limit":1.000000,"groups":[]}]}\n',
        # the by set must be consistent across points.
        '{"points":[{"limit":1.000000,"groups":[]},'
        '{"limit":2.000000,"groups":['
        '{"by":"region","cost":0.000000,"priority":0.000000,'
        '"marginal":0.000000,"pick":[],"added":[],"removed":[],'
        '"effects":{"uhi":0.000000,"energy":0.000000,"vent":0.000000},'
        '"range":{"low":0.000000,"high":0.000000},"dominated":false}]}\n',
        # group key order is fixed.
        '{"points":[{"limit":0.000000,"groups":['
        '{"cost":0.000000,"by":"region","priority":0.000000,'
        '"marginal":0.000000,"pick":[],"added":[],"removed":[],'
        '"effects":{"uhi":0.000000,"energy":0.000000,"vent":0.000000},'
        '"range":{"low":0.000000,"high":0.000000},"dominated":false}]}]}\n',
        # effects key order is fixed.
        '{"points":[{"limit":0.000000,"groups":['
        '{"by":"region","cost":0.000000,"priority":0.000000,'
        '"marginal":0.000000,"pick":[],"added":[],"removed":[],'
        '"effects":{"energy":0.000000,"uhi":0.000000,"vent":0.000000},'
        '"range":{"low":0.000000,"high":0.000000},"dominated":false}]}]}\n',
        # pick must hold known kinds in ascending order.
        '{"points":[{"limit":0.000000,"groups":['
        '{"by":"region","cost":0.000000,"priority":0.000000,'
        '"marginal":0.000000,"pick":["roof","green"],"added":[],'
        '"removed":[],'
        '"effects":{"uhi":0.000000,"energy":0.000000,"vent":0.000000},'
        '"range":{"low":0.000000,"high":0.000000},"dominated":false}]}]}\n',
        '{"points":[{"limit":0.000000,"groups":['
        '{"by":"region","cost":0.000000,"priority":0.000000,'
        '"marginal":0.000000,"pick":["nope"],"added":[],"removed":[],'
        '"effects":{"uhi":0.000000,"energy":0.000000,"vent":0.000000},'
        '"range":{"low":0.000000,"high":0.000000},"dominated":false}]}]}\n',
        # by must be region or window.
        '{"points":[{"limit":0.000000,"groups":['
        '{"by":"city","cost":0.000000,"priority":0.000000,'
        '"marginal":0.000000,"pick":[],"added":[],"removed":[],'
        '"effects":{"uhi":0.000000,"energy":0.000000,"vent":0.000000},'
        '"range":{"low":0.000000,"high":0.000000},"dominated":false}]}]}\n',
        # window must not precede region.
        '{"points":[{"limit":0.000000,"groups":['
        '{"by":"window","cost":0.000000,"priority":0.000000,'
        '"marginal":0.000000,"pick":[],"added":[],"removed":[],'
        '"effects":{"uhi":0.000000,"energy":0.000000,"vent":0.000000},'
        '"range":{"low":0.000000,"high":0.000000},"dominated":false},'
        '{"by":"region","cost":0.000000,"priority":0.000000,'
        '"marginal":0.000000,"pick":[],"added":[],"removed":[],'
        '"effects":{"uhi":0.000000,"energy":0.000000,"vent":0.000000},'
        '"range":{"low":0.000000,"high":0.000000},"dominated":false}]}]}\n',
        # cost must be non-negative.
        '{"points":[{"limit":0.000000,"groups":['
        '{"by":"region","cost":-1.000000,"priority":0.000000,'
        '"marginal":0.000000,"pick":[],"added":[],"removed":[],'
        '"effects":{"uhi":0.000000,"energy":0.000000,"vent":0.000000},'
        '"range":{"low":0.000000,"high":0.000000},"dominated":false}]}]}\n',
    ],
)
def test_malformed_frontier_value_error(raw):
    with pytest.raises(ValueError):
        driver_plan_pareto(raw, _WEIGHTS, 1.0)


def test_real_frontier_roundtrip_is_accepted():
    report = _one_region()
    actions = _actions()
    frontier = driver_plan_frontier(report, actions, [0, 1, 3])
    out = driver_plan_pareto(frontier, _WEIGHTS, 3)
    payload = _payload(out)
    assert payload["limit"] == 3.0
    assert payload["groups"][0]["recommend"] == ["green", "roof"]


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.driver_plan_pareto is urban_micro.driver_plan_pareto
    assert "driver_plan_pareto" in urban_micro.__all__
