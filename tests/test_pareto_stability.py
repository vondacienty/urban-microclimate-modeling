"""Tests for urban_micro.pareto_stability: Pareto recommendation
stability aggregated across weight sets and budget limits."""

import json

import pytest

from urban_micro import driver_plan_frontier, pareto_stability
from urban_micro import uhi as _uhi

from test_driver_plan import _actions, _one_region
from test_driver_plan_pareto import _frontier, _fpoint, _state

_TOP_KEYS = ["points"]
_POINT_KEYS = ["limit", "groups"]
_GROUP_KEYS = ["by", "recommend", "stability", "switch", "items"]
_ITEM_KEYS = ["pick", "frequency", "uhi", "energy", "vent"]

_UHI_HEAVY = {"uhi": 10.0, "energy": 1.0, "vent": 1.0}
_ENERGY_HEAVY = {"uhi": 1.0, "energy": 10.0, "vent": 1.0}
_BALANCED = {"uhi": 1.0, "energy": 1.0, "vent": 1.0}


def _payload(out):
    return json.loads(out)


def _frequencies(group):
    return [item["frequency"] for item in group["items"]]


def _two_weights():
    return {"a": _UHI_HEAVY, "b": _ENERGY_HEAVY}


def _split_frontier():
    # roof rewards the energy weight, green the uhi weight; the empty
    # plan stays a non-dominated survivor but never wins either vote.
    return _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=-2.0)),
        _fpoint(2.0, _state("region", ["green"], 2.0, uhi=-3.0)),
    )


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3])
    out = pareto_stability(frontier, _two_weights(), [3, 1])
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = _payload(out)
    assert list(payload) == _TOP_KEYS
    points = payload["points"]
    assert [point["limit"] for point in points] == [1.0, 3.0]
    for point in points:
        assert list(point) == _POINT_KEYS
        for group in point["groups"]:
            assert list(group) == _GROUP_KEYS
            assert isinstance(group["switch"], bool)
            for item in group["items"]:
                assert list(item) == _ITEM_KEYS
                assert isinstance(item["pick"], list)


def test_region_before_window_groups():
    frontier = _frontier(
        _fpoint(
            0.0,
            _state("region", [], 0.0),
            _state("window", ["roof"], 0.0, energy=-1.0),
        ),
    )
    payload = _payload(pareto_stability(frontier, _two_weights(), [1]))
    groups = payload["points"][0]["groups"]
    assert [group["by"] for group in groups] == ["region", "window"]


def test_empty_frontier_groups_stay_empty():
    frontier = _frontier(
        _fpoint(1.0),
        _fpoint(2.0),
    )
    out = pareto_stability(frontier, _two_weights(), [2, 1])
    assert out == (
        '{"points":[{"limit":1.000000,"groups":[]},'
        '{"limit":2.000000,"groups":[]}]}\n'
    )


# --- voting, ties and frequencies -------------------------------------------

def test_split_vote_takes_lexicographic_first_pick():
    out = pareto_stability(_split_frontier(), _two_weights(), [2])
    assert out == (
        '{"points":[{"limit":2.000000,"groups":[{"by":"region",'
        '"recommend":["green"],"stability":0.500000,"switch":false,'
        '"items":['
        '{"pick":["green"],"frequency":0.500000,"uhi":-3.000000,'
        '"energy":0.000000,"vent":0.000000},'
        '{"pick":["roof"],"frequency":0.500000,"uhi":0.000000,'
        '"energy":-2.000000,"vent":0.000000}'
        "]}]}]}\n"
    )


def test_majority_vote_sets_recommend_and_stability():
    weights = {
        "a": _UHI_HEAVY,
        "b": _UHI_HEAVY,
        "c": _ENERGY_HEAVY,
    }
    group = _payload(pareto_stability(_split_frontier(), weights, [2]))[
        "points"
    ][0]["groups"][0]
    assert group["recommend"] == ["green"]
    assert group["stability"] == pytest.approx(2 / 3, abs=1e-6)
    assert [item["pick"] for item in group["items"]] == [
        ["green"], ["roof"]
    ]
    assert _frequencies(group) == pytest.approx([2 / 3, 1 / 3], abs=1e-6)


def test_items_sort_frequency_desc_then_pick_asc():
    weights = {
        "a": _BALANCED,
        "b": _UHI_HEAVY,
        "c": _ENERGY_HEAVY,
        "d": _BALANCED,
    }
    # balanced scores green -3 and roof -2, so the two balanced votes go
    # to green; the heavy votes split one each: green 3, roof 1.
    group = _payload(pareto_stability(_split_frontier(), weights, [2]))[
        "points"
    ][0]["groups"][0]
    picks = [item["pick"] for item in group["items"]]
    assert picks == [["green"], ["roof"]]
    assert group["items"][0]["frequency"] == 0.75
    assert group["items"][1]["frequency"] == 0.25


def test_item_effects_come_from_the_deduped_combination():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(
            2.0,
            _state(
                "region", ["green", "roof"], 2.0,
                uhi=-1.5, energy=-4.25, vent=3.0,
            ),
        ),
    )
    group = _payload(pareto_stability(frontier, _two_weights(), [5]))[
        "points"
    ][0]["groups"][0]
    item = group["items"][0]
    assert item["pick"] == ["green", "roof"]
    assert item["uhi"] == -1.5
    assert item["energy"] == -4.25
    assert item["vent"] == 3.0


# --- all-null points --------------------------------------------------------

def test_all_null_recommend_has_stability_one_and_empty_items():
    # The only group costs 2, so at limit 1 nothing is admissible.
    frontier = _frontier(
        _fpoint(2.0, _state("region", ["roof"], 2.0, energy=-4.0)),
    )
    out = pareto_stability(frontier, _two_weights(), [1])
    assert out == (
        '{"points":[{"limit":1.000000,"groups":[{"by":"region",'
        '"recommend":null,"stability":1.000000,"switch":false,'
        '"items":[]}]}]}\n'
    )


# --- switch -----------------------------------------------------------------

def test_switch_fires_when_winner_changes_across_limits():
    frontier = _frontier(
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=-2.0)),
        _fpoint(2.0, _state("region", ["green"], 2.0, uhi=-3.0)),
    )
    weights = {
        "a": _UHI_HEAVY,
        "b": _UHI_HEAVY,
        "c": _ENERGY_HEAVY,
    }
    # limit 1: green inadmissible, every vote goes to roof; limit 2: the
    # two uhi-heavy votes move to green, flipping the winner.
    groups = [
        point["groups"][0]
        for point in _payload(pareto_stability(frontier, weights, [1, 2]))[
            "points"
        ]
    ]
    assert groups[0]["recommend"] == ["roof"]
    assert groups[0]["switch"] is False
    assert groups[1]["recommend"] == ["green"]
    assert groups[1]["switch"] is True


def test_switch_fires_from_null_to_pick():
    frontier = _frontier(
        _fpoint(2.0, _state("region", ["roof"], 2.0, energy=-4.0)),
        _fpoint(3.0, _state("region", ["material"], 3.0, vent=3.0)),
    )
    weights = {"a": _UHI_HEAVY, "b": _UHI_HEAVY, "c": _ENERGY_HEAVY}
    # material rewards vent, which no weight set emphasizes, so roof keeps
    # every vote once it becomes admissible at limit 2.
    groups = [
        point["groups"][0]
        for point in _payload(pareto_stability(frontier, weights, [1, 2, 3]))[
            "points"
        ]
    ]
    assert groups[0]["recommend"] is None
    assert groups[0]["switch"] is False
    assert groups[1]["recommend"] == ["roof"]
    assert groups[1]["switch"] is True
    assert groups[2]["recommend"] == ["roof"]
    assert groups[2]["switch"] is False


def test_identical_winners_do_not_switch():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3])
    groups = [
        point["groups"][0]
        for point in _payload(
            pareto_stability(frontier, _two_weights(), [3, 10])
        )["points"]
    ]
    assert [group["recommend"] for group in groups] == [
        ["green", "roof"], ["green", "roof"]
    ]
    assert all(group["switch"] is False for group in groups)


# --- number formatting ------------------------------------------------------

def test_six_decimals_and_no_negative_zero():
    out = pareto_stability(_split_frontier(), _two_weights(), [0])
    assert "-0.000000" not in out
    assert '"limit":0.000000' in out
    assert '"stability":1.000000' in out
    assert '"frequency":1.000000' in out
    assert '"switch":false' in out


# --- validation -------------------------------------------------------------

def test_type_errors():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    good_weights = _two_weights()
    for bad in (1, None, [], True, {}):
        with pytest.raises(TypeError):
            pareto_stability(bad, good_weights, [1.0])
    for bad in ([], None, 42, "x", True):
        with pytest.raises(TypeError):
            pareto_stability(frontier, bad, [1.0])
    for bad in (None, 42, "x", True, {}):
        with pytest.raises(TypeError):
            pareto_stability(frontier, good_weights, bad)


def test_non_dict_weight_entry_is_type_error():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(TypeError):
        pareto_stability(
            frontier, {"a": None, "b": _BALANCED}, [1.0]
        )
    with pytest.raises(TypeError):
        pareto_stability(
            frontier, {"a": [1, 1, 1], "b": _BALANCED}, [1.0]
        )


def test_type_errors_take_precedence():
    with pytest.raises(TypeError):
        pareto_stability(1, [], None)


def test_weights_must_hold_at_least_two_entries():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_stability(frontier, {"a": _BALANCED}, [1.0])


@pytest.mark.parametrize("key", [1, None, "", True])
def test_weights_keys_must_be_non_empty_strings(key):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    weights = {key: _BALANCED, "b": _BALANCED}
    with pytest.raises(ValueError):
        pareto_stability(frontier, weights, [1.0])


@pytest.mark.parametrize(
    "entry",
    [
        {},
        {"uhi": 1.0, "energy": 1.0},
        {"uhi": 1.0, "energy": 1.0, "vent": 1.0, "extra": 1.0},
        {"UHI": 1.0, "energy": 1.0, "vent": 1.0},
    ],
)
def test_weight_entry_key_set_must_be_exact(entry):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_stability(
            frontier, {"a": entry, "b": _BALANCED}, [1.0]
        )


@pytest.mark.parametrize(
    "value",
    [0, -1.0, True, False, None, "1", float("nan"), float("inf"),
     float("-inf"), [1.0], (1.0,)],
)
def test_bad_weight_values_rejected(value):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    weights = {"a": dict(_BALANCED, uhi=value), "b": _BALANCED}
    with pytest.raises(ValueError):
        pareto_stability(frontier, weights, [1.0])


@pytest.mark.parametrize(
    "limits",
    [
        [],
        [-1],
        [-0.1],
        [True],
        [False],
        [None],
        ["1"],
        [float("nan")],
        [float("inf")],
        [[1]],
        [(1.0,)],
    ],
)
def test_bad_limits_rejected(limits):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_stability(frontier, _two_weights(), limits)


@pytest.mark.parametrize(
    "limits",
    [[1, 1], [1.0, 1], [2, 2.0, 3], [0.5, 0.50]],
)
def test_limits_must_be_pairwise_distinct_as_decimal(limits):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_stability(frontier, _two_weights(), limits)


def test_malformed_frontier_value_error():
    with pytest.raises(ValueError):
        pareto_stability('{"points":[]}\n', _two_weights(), [1.0])


def test_real_frontier_roundtrip_is_accepted():
    report = _one_region()
    actions = _actions()
    frontier = driver_plan_frontier(report, actions, [0, 1, 3])
    out = pareto_stability(
        frontier,
        {"a": _BALANCED, "b": _UHI_HEAVY},
        [3, 0],
    )
    payload = _payload(out)
    assert [point["limit"] for point in payload["points"]] == [0.0, 3.0]
    assert payload["points"][-1]["groups"][0]["recommend"] == [
        "green", "roof"
    ]


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.pareto_stability is urban_micro.pareto_stability
    assert "pareto_stability" in urban_micro.__all__
