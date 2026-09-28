"""Tests for urban_micro.driver_plan_pareto: non-dominated budgeted
combinations selected over a canonical driver_plan_frontier report."""

import json

import pytest

from urban_micro import driver_plan_frontier, driver_plan_pareto
from urban_micro import uhi as _uhi

from test_driver_plan import (
    _actions,
    _group,
    _item,
    _one_region,
    _report,
)

_WEIGHTS = {"uhi": 1.0, "energy": 1.0, "vent": 1.0}
_TOP_KEYS = ["limit", "groups"]
_GROUP_KEYS = ["by", "recommend", "items"]
_ITEM_KEYS = ["pick", "cost", "uhi", "energy", "vent", "score"]
_EMPTY_REPORT = '{"alpha":0.050000,"groups":[]}\n'


def _payload(out):
    return json.loads(out)


def _only_group(out):
    groups = _payload(out)["groups"]
    assert len(groups) == 1
    return groups[0]


# --- hand-built canonical frontier strings ----------------------------------

def _js_array(values):
    return "[" + ",".join(json.dumps(value) for value in values) + "]"


def _fgroup(by, cost, priority, marginal, pick, added, removed,
            uhi, energy, vent, low, high, dominated):
    return (
        '{"by":' + json.dumps(by)
        + f',"cost":{cost:.6f},"priority":{priority:.6f},'
        + f'"marginal":{marginal:.6f}'
        + ',"pick":' + _js_array(pick)
        + ',"added":' + _js_array(added)
        + ',"removed":' + _js_array(removed)
        + f',"effects":{{"uhi":{uhi:.6f},"energy":{energy:.6f},'
        + f'"vent":{vent:.6f}}},'
        + f'"range":{{"low":{low:.6f},"high":{high:.6f}}},'
        + f'"dominated":{str(dominated).lower()}}}'
    )


def _fpoint(limit, groups):
    return (
        f'{{"limit":{limit:.6f},"groups":[' + ",".join(groups) + "]}"
    )


def _frontier(*points):
    return '{"points":[' + ",".join(points) + "]}\n"


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    front = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3])
    out = driver_plan_pareto(front, _WEIGHTS, 10.0)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = _payload(out)
    assert list(payload) == _TOP_KEYS
    assert isinstance(payload["groups"], list)
    for group in payload["groups"]:
        assert list(group) == _GROUP_KEYS
        assert isinstance(group["items"], list)
        for item in group["items"]:
            assert list(item) == _ITEM_KEYS
            assert isinstance(item["pick"], list)


def test_region_before_window_groups():
    def all_ineligible(by):
        return _group(by, [
            _item(1, "station", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
            _item(2, "lst", "energy", 0.0, 0.0, 0.0, 1.0, 0.0, False),
            _item(3, "morph", "vent", 0.0, 0.0, 0.0, 1.0, 0.0, False),
            _item(4, "cover", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        ])

    report = _report(all_ineligible("region"), all_ineligible("window"))
    front = driver_plan_frontier(report, _actions(), [1.0])
    out = driver_plan_pareto(front, _WEIGHTS, 5.0)
    assert [g["by"] for g in _payload(out)["groups"]] == ["region", "window"]


def test_empty_report_emits_empty_groups():
    front = driver_plan_frontier(_EMPTY_REPORT, _actions(), [2, 1])
    assert driver_plan_pareto(front, _WEIGHTS, 5) == (
        '{"limit":5.000000,"groups":[]}\n'
    )


def test_limit_renders_six_decimals_and_integer_accepted():
    front = driver_plan_frontier(_one_region(), _actions(), [0])
    out = driver_plan_pareto(front, {"uhi": 1, "energy": 1, "vent": 1}, 0)
    assert out.startswith('{"limit":0.000000,')


def test_six_decimals_and_no_negative_zero():
    front = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3])
    out = driver_plan_pareto(front, _WEIGHTS, 10.0)
    assert "-0.000000" not in out
    assert '"cost":3.000000' in out
    assert '"score":0.000000' in out


# --- de-duplication by pick -------------------------------------------------

def test_repeated_pick_deduplicated_keeping_first_point():
    # The producer repeats the optimum pick across held limits; those
    # collapse to one candidate per by.
    front = driver_plan_frontier(
        _one_region(), _actions(), [0, 1, 2, 3, 10]
    )
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 10.0))
    picks = [tuple(item["pick"]) for item in group["items"]]
    assert picks == [("green", "roof"), ("roof",), ()]
    assert len(picks) == len(set(picks))


def test_dedup_keeps_lowest_limit_values_for_pick():
    # Same pick at two limits with different effects: the first (lowest
    # limit) point is the retained combination.
    front = _frontier(
        _fpoint(1, [_fgroup("region", 2.0, 1.0, 1.0, ["green"], ["green"],
                            [], -2.0, 0.0, 0.0, -0.5, -0.2, False)]),
        _fpoint(2, [_fgroup("region", 2.0, 1.0, 0.0, ["green"], [], [],
                            -1.0, 0.0, 0.0, -0.5, -0.2, False)]),
    )
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 10.0))
    assert len(group["items"]) == 1
    assert group["items"][0]["pick"] == ["green"]
    assert group["items"][0]["uhi"] == -2.0


# --- dominance --------------------------------------------------------------

def test_positive_effect_combination_dominated_by_empty():
    # green raises uhi by +2 at a positive cost: the empty combination
    # dominates it on cost and uhi.
    report = _one_region(
        station=(4.0, 2.0, 0.1, 0.5, True),
        lst=(0.0, 0.0, 0.0, 0.0, False),
    )
    front = driver_plan_frontier(report, _actions(), [0, 10])
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 10.0))
    assert group["items"] == [{
        "pick": [], "cost": 0.0, "uhi": 0.0, "energy": 0.0,
        "vent": 0.0, "score": 0.0,
    }]
    assert group["recommend"] == []


def test_negative_vent_combination_dominated_by_empty():
    # material lowers vent (-1): under objective -vent that is worse, and
    # it additionally costs more, so empty dominates it.
    report = _one_region(
        station=(0.0, 0.0, 0.0, 0.0, False),
        lst=(0.0, 0.0, 0.0, 0.0, False),
        morph=(4.0, -1.0, -0.2, -0.1, True),
    )
    front = driver_plan_frontier(
        report, _actions(material=("morph", 1.0)), [0, 5]
    )
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 5.0))
    assert [tuple(item["pick"]) for item in group["items"]] == [()]


def test_positive_vent_survives_and_is_recommended():
    # material raises vent (+3); higher vent is preferred (objective
    # component -vent), so it is non-dominated and, with vent weighted
    # heavily, scores best.
    report = _one_region(
        station=(0.0, 0.0, 0.0, 0.0, False),
        lst=(0.0, 0.0, 0.0, 0.0, False),
        morph=(6.0, 3.0, 0.1, 0.5, True),
    )
    front = driver_plan_frontier(
        report, _actions(material=("morph", 1.0)), [0, 5]
    )
    group = _only_group(
        driver_plan_pareto(front, {"uhi": 1, "energy": 1, "vent": 10}, 5.0)
    )
    assert group["recommend"] == ["material"]
    assert group["items"][0]["pick"] == ["material"]
    assert group["items"][0]["score"] == -30.0


def test_tradeoff_combinations_both_survive():
    # green lowers uhi (-2), roof lowers energy (-2): neither dominates
    # the other.
    report = _one_region(
        station=(2.0, -2.0, -0.5, -0.2, True),
        lst=(6.0, -2.0, -0.4, -0.1, True),
    )
    front = driver_plan_frontier(
        report,
        _actions(green=("station", 1.0), roof=("lst", 2.0)),
        [0, 1, 2],
    )
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 2.0))
    picks = {tuple(item["pick"]) for item in group["items"]}
    assert picks == {(), ("green",), ("roof",)}


# --- scoring, sorting and recommend ----------------------------------------

def test_items_sort_by_score_and_recommend_matches_first():
    front = driver_plan_frontier(
        _one_region(), _actions(), [0, 1, 2, 3, 10]
    )
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 10.0))
    items = group["items"]
    assert [item["score"] for item in items] == [-7.0, -5.0, 0.0]
    assert [item["pick"] for item in items] == [
        ["green", "roof"], ["roof"], []
    ]
    assert group["recommend"] == items[0]["pick"] == ["green", "roof"]
    # Spot-check the objective inputs and score.
    assert items[0]["cost"] == 3.0
    assert items[0]["uhi"] == -2.0
    assert items[0]["energy"] == -5.0
    assert items[0]["vent"] == 0.0


def test_score_tie_breaks_by_cost():
    # green (cost 1, uhi -2) and roof (cost 2, energy -2) share score -2;
    # the cheaper one sorts first.
    front = _frontier(
        _fpoint(1, [_fgroup("region", 1.0, 2.0, 2.0, ["green"], ["green"],
                            [], -2.0, 0.0, 0.0, -0.5, -0.2, False)]),
        _fpoint(2, [_fgroup("region", 2.0, 3.0, 3.0, ["roof"], ["roof"],
                            ["green"], 0.0, -2.0, 0.0, -0.4, -0.1,
                            False)]),
    )
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 10.0))
    items = group["items"]
    assert [item["score"] for item in items] == [-2.0, -2.0]
    assert [item["pick"] for item in items] == [["green"], ["roof"]]
    assert group["recommend"] == ["green"]


def test_score_and_cost_tie_breaks_by_pick_lexicographic():
    front = _frontier(
        _fpoint(1, [_fgroup("region", 1.0, 2.0, 2.0, ["green"], ["green"],
                            [], -2.0, 0.0, 0.0, -0.5, -0.2, False)]),
        _fpoint(2, [_fgroup("region", 1.0, 2.0, 0.0, ["roof"], ["roof"],
                            ["green"], 0.0, -2.0, 0.0, -0.4, -0.1,
                            False)]),
    )
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 10.0))
    assert [item["pick"] for item in group["items"]] == [
        ["green"], ["roof"]
    ]
    assert group["recommend"] == ["green"]


def test_no_candidate_when_all_costs_exceed_limit():
    # A frontier whose single point costs 3 (no zero-limit point): a
    # limit of 0 admits nothing -> recommend null and empty items.
    front = driver_plan_frontier(_one_region(), _actions(), [5])
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 0))
    assert group["recommend"] is None
    assert group["items"] == []


def test_limit_is_inclusive():
    # roof point costs 1: at limit 1 it survives alongside empty.
    front = driver_plan_frontier(_one_region(), _actions(), [0, 1])
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 1))
    assert {tuple(i["pick"]) for i in group["items"]} == {(), ("roof",)}
    # Just below (limit 0) only the empty combination remains.
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 0))
    assert [tuple(i["pick"]) for i in group["items"]] == [()]


def test_groups_are_solved_independently():
    def all_ineligible(by):
        return _group(by, [
            _item(1, "station", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
            _item(2, "lst", "energy", 0.0, 0.0, 0.0, 1.0, 0.0, False),
            _item(3, "morph", "vent", 0.0, 0.0, 0.0, 1.0, 0.0, False),
            _item(4, "cover", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        ])

    report = _report(all_ineligible("region"), all_ineligible("window"))
    front = driver_plan_frontier(report, _actions(), [0, 5])
    payload = _payload(driver_plan_pareto(front, _WEIGHTS, 5.0))
    for group in payload["groups"]:
        assert group["recommend"] == []
        assert len(group["items"]) == 1
        assert group["items"][0]["pick"] == []


# --- genuine producer chain -------------------------------------------------

def test_genuine_chain_canonical_roundtrip():
    # The output parses and reproduces the frontier's effects verbatim
    # for every surviving item.
    front = driver_plan_frontier(
        _one_region(), _actions(), [0, 1, 2, 3, 10]
    )
    points = _payload(front)["points"]
    effects_by_pick = {}
    for point in points:
        g = point["groups"][0]
        effects_by_pick.setdefault(tuple(g["pick"]), g["effects"])
    group = _only_group(driver_plan_pareto(front, _WEIGHTS, 10.0))
    for item in group["items"]:
        effects = effects_by_pick[tuple(item["pick"])]
        assert item["uhi"] == effects["uhi"]
        assert item["energy"] == effects["energy"]
        assert item["vent"] == effects["vent"]


# --- validation -------------------------------------------------------------

def test_type_errors():
    front = driver_plan_frontier(_one_region(), _actions(), [1.0])
    for bad in (1, None, [], True, {}):
        with pytest.raises(TypeError):
            driver_plan_pareto(bad, _WEIGHTS, 1.0)
    for bad in ([], None, 42, "x"):
        with pytest.raises(TypeError):
            driver_plan_pareto(front, bad, 1.0)


def test_type_errors_take_precedence():
    with pytest.raises(TypeError):
        driver_plan_pareto(1, [], 0)


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"uhi": 1.0, "energy": 1.0},
        {"uhi": 1.0, "energy": 1.0, "vent": 1.0, "extra": 1.0},
        {"Uhi": 1.0, "energy": 1.0, "vent": 1.0},
        {"uhi": 0, "energy": 1.0, "vent": 1.0},
        {"uhi": 0.0, "energy": 1.0, "vent": 1.0},
        {"uhi": -1.0, "energy": 1.0, "vent": 1.0},
        {"uhi": True, "energy": 1.0, "vent": 1.0},
        {"uhi": False, "energy": 1.0, "vent": 1.0},
        {"uhi": "1", "energy": 1.0, "vent": 1.0},
        {"uhi": None, "energy": 1.0, "vent": 1.0},
        {"uhi": float("nan"), "energy": 1.0, "vent": 1.0},
        {"uhi": float("inf"), "energy": 1.0, "vent": 1.0},
    ],
)
def test_bad_weights_rejected(weights):
    front = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        driver_plan_pareto(front, weights, 1.0)


@pytest.mark.parametrize(
    "limit",
    [-1, -0.1, True, False, "1", None, float("nan"), float("inf"),
     float("-inf"), [1.0]],
)
def test_bad_limit_rejected(limit):
    front = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        driver_plan_pareto(front, _WEIGHTS, limit)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json\n",
        '{"points":[{"limit":1.000000,"groups":[]}]}',
        '{"points":[{"limit":1.000000,"groups":[]}]}\n\n',
        '{"Points":[]}\n',
        '{"points":[]}\n',
        '{"points":[{"groups":[],"limit":1.000000}]}\n',
        '{"points":[{"limit":1,"groups":[]}]}\n',
        '{"points":[{"limit":-1.000000,"groups":[]}]}\n',
        '{"points":[{"limit":-0.000000,"groups":[]}]}\n',
        '{"points":[{"limit":2.000000,"groups":[]},'
        '{"limit":1.000000,"groups":[]}]}\n',
        '{"points":[{"limit":1.000000,"groups":[]},'
        '{"limit":1.000000,"groups":[]}]}\n',
    ],
)
def test_malformed_frontier_value_error(raw):
    with pytest.raises(ValueError):
        driver_plan_pareto(raw, _WEIGHTS, 5.0)


def test_frontier_with_unknown_by_rejected():
    raw = _frontier(_fpoint(
        1.0,
        [_fgroup("cell", 0.0, 0.0, 0.0, [], [], [], 0.0, 0.0, 0.0,
                 0.0, 0.0, False)],
    ))
    with pytest.raises(ValueError):
        driver_plan_pareto(raw, _WEIGHTS, 5.0)


def test_frontier_with_bad_pick_kind_rejected():
    raw = _frontier(_fpoint(
        1.0,
        [_fgroup("region", 1.0, 1.0, 1.0, ["albedo"], ["albedo"], [],
                 0.0, 0.0, 0.0, 0.0, 0.0, False)],
    ))
    with pytest.raises(ValueError):
        driver_plan_pareto(raw, _WEIGHTS, 5.0)


def test_frontier_wrong_group_key_order_rejected():
    good = driver_plan_frontier(_one_region(), _actions(), [0, 1])
    bad = good.replace('"by":"region","cost"', '"cost":0.000000,"by":"region"')
    # The replacement breaks both the key order and shape; either way it
    # must be rejected as non-canonical.
    with pytest.raises(ValueError):
        driver_plan_pareto(bad, _WEIGHTS, 5.0)


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.driver_plan_pareto is urban_micro.driver_plan_pareto
    assert "driver_plan_pareto" in urban_micro.__all__
