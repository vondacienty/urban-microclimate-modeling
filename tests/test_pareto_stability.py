"""Tests for urban_micro.pareto_stability: driver_plan_pareto
recommendations aggregated across weight scenarios and limits."""

import json

import pytest

from urban_micro import driver_plan_frontier, pareto_stability
from urban_micro import uhi as _uhi

from test_driver_plan import _actions, _one_region
from test_driver_plan_pareto import _frontier, _fpoint, _state

_POINT_KEYS = ["limit", "groups"]
_GROUP_KEYS = ["by", "recommend", "stability", "switch", "items"]
_ITEM_KEYS = ["pick", "frequency", "uhi", "energy", "vent"]
_WA = {"uhi": 10.0, "energy": 1.0, "vent": 1.0}
_WB = {"uhi": 1.0, "energy": 10.0, "vent": 1.0}
_WC = {"uhi": 10.0, "energy": 1.0, "vent": 1.0}


def _payload(out):
    return json.loads(out)


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3])
    out = pareto_stability(
        frontier, {"a": _WA, "b": _WB, "c": _WC}, [0, 1, 3]
    )
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = _payload(out)
    assert list(payload) == ["points"]
    assert [point["limit"] for point in payload["points"]] == [0.0, 1.0, 3.0]
    for point in payload["points"]:
        assert list(point) == _POINT_KEYS
        group = point["groups"][0]
        assert list(group) == _GROUP_KEYS
        for item in group["items"]:
            assert list(item) == _ITEM_KEYS
            assert isinstance(item["pick"], list)
        assert isinstance(group["switch"], bool)


def test_region_before_window_groups():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0), _state("window", [], 0.0)),
        _fpoint(
            1.0,
            _state("region", ["roof"], 1.0, energy=-4.0),
            _state("window", ["green"], 1.0, uhi=-2.0),
        ),
    )
    payload = _payload(
        pareto_stability(frontier, {"a": _WA, "b": _WB}, [1])
    )
    assert [group["by"] for group in payload["points"][0]["groups"]] == [
        "region", "window"
    ]


def test_empty_report_frontier_emits_empty_groups_per_point():
    frontier = driver_plan_frontier(
        '{"alpha":0.050000,"groups":[]}\n', _actions(), [1, 2]
    )
    out = pareto_stability(frontier, {"a": _WA, "b": _WB}, [2, 1])
    assert out == (
        '{"points":[{"limit":1.000000,"groups":[]},'
        '{"limit":2.000000,"groups":[]}]}\n'
    )


# --- tallying, ties and items -----------------------------------------------

def test_majority_pick_wins_with_count_ratio_and_items():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=-5.0)),
        _fpoint(2.0, _state("region", ["green"], 2.0, uhi=-3.0)),
    )
    group = _payload(
        pareto_stability(
            frontier, {"a": _WA, "b": _WB, "c": _WC}, [0, 1, 2]
        )
    )["points"][2]["groups"][0]
    assert group["recommend"] == ["green"]
    assert group["stability"] == pytest.approx(2 / 3)
    assert group["switch"] is True
    assert [[item["pick"], item["frequency"]] for item in group["items"]] == [
        [["green"], pytest.approx(2 / 3)],
        [["roof"], pytest.approx(1 / 3)],
    ]
    green = group["items"][0]
    assert (green["uhi"], green["energy"], green["vent"]) == (-3.0, 0.0, 0.0)
    roof = group["items"][1]
    assert (roof["uhi"], roof["energy"], roof["vent"]) == (0.0, -5.0, 0.0)


def test_tied_counts_take_lexicographically_first_pick():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=-5.0)),
        _fpoint(2.0, _state("region", ["green"], 2.0, uhi=-3.0)),
    )
    group = _payload(
        pareto_stability(frontier, {"a": _WA, "b": _WB}, [2])
    )["points"][0]["groups"][0]
    assert group["recommend"] == ["green"]
    assert group["stability"] == 0.5
    assert [item["pick"] for item in group["items"]] == [
        ["green"], ["roof"]
    ]


def test_empty_pick_is_a_real_recommendation():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1])
    group = _payload(
        pareto_stability(frontier, {"a": _WA, "b": _WB}, [0])
    )["points"][0]["groups"][0]
    assert group["recommend"] == []
    assert group["stability"] == 1.0
    assert group["switch"] is False
    item = group["items"][0]
    assert item["pick"] == []
    assert item["frequency"] == 1.0
    assert (item["uhi"], item["energy"], item["vent"]) == (0.0, 0.0, 0.0)


def test_all_null_recommendations():
    frontier = _frontier(
        _fpoint(5.0, _state("region", ["roof"], 5.0, energy=-4.0)),
    )
    point = _payload(pareto_stability(frontier, {"a": _WA, "b": _WB}, [0]))[
        "points"
    ][0]
    assert point == {
        "limit": 0.0,
        "groups": [
            {
                "by": "region",
                "recommend": None,
                "stability": 1.0,
                "switch": False,
                "items": [],
            }
        ],
    }


# --- switches across ascending limits ----------------------------------------

def test_switch_tracks_changes_including_null():
    frontier = _frontier(
        _fpoint(5.0, _state("region", ["roof"], 5.0, energy=-4.0)),
    )
    # Limits given out of order must be processed ascending.
    points = _payload(
        pareto_stability(frontier, {"a": _WA, "b": _WB}, [5, 0])
    )["points"]
    assert [point["limit"] for point in points] == [0.0, 5.0]
    assert points[0]["groups"][0]["recommend"] is None
    assert points[0]["groups"][0]["switch"] is False
    assert points[1]["groups"][0]["recommend"] == ["roof"]
    assert points[1]["groups"][0]["switch"] is True


def test_switch_flags_follow_the_trajectory():
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=-5.0)),
        _fpoint(2.0, _state("region", ["green"], 2.0, uhi=-3.0)),
    )
    groups = [
        point["groups"][0]
        for point in _payload(
            pareto_stability(frontier, {"a": _WA, "c": _WC}, [0, 1, 2])
        )["points"]
    ]
    assert [group["recommend"] for group in groups] == [
        [], ["roof"], ["green"]
    ]
    assert [group["switch"] for group in groups] == [
        False, True, True
    ]


def test_real_frontier_unanimous_trajectory():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3, 10])
    points = _payload(
        pareto_stability(
            frontier, {"a": _WA, "b": _WB, "c": _WC}, [10, 3, 1, 0]
        )
    )["points"]
    assert [point["limit"] for point in points] == [0.0, 1.0, 3.0, 10.0]
    expects = [
        ([], False),
        (["roof"], True),
        (["green", "roof"], True),
        (["green", "roof"], False),
    ]
    for point, (pick, switched) in zip(points, expects):
        group = point["groups"][0]
        assert group["recommend"] == pick
        assert group["stability"] == 1.0
        assert group["switch"] is switched
        assert [item["pick"] for item in group["items"]] == [pick]
        assert group["items"][0]["frequency"] == 1.0


# --- number formatting ------------------------------------------------------

def test_six_decimals_and_no_negative_zero():
    frontier = _frontier(
        _fpoint(5.0, _state("region", ["roof"], 5.0, energy=-4.0)),
    )
    out = pareto_stability(frontier, {"a": _WA, "b": _WB}, [0, 5])
    assert "-0.000000" not in out
    assert '"limit":0.000000' in out
    assert '"stability":1.000000' in out
    assert '"frequency":1.000000' in out


# --- validation -------------------------------------------------------------

def test_type_errors():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    for bad in (1, None, [], True, {}):
        with pytest.raises(TypeError):
            pareto_stability(bad, {"a": _WA, "b": _WB}, [1.0])
    for bad in ([], None, 42, "x", True):
        with pytest.raises(TypeError):
            pareto_stability(frontier, bad, [1.0])
    for bad in (None, {}, 1, "x", True, (1.0, 2.0)):
        with pytest.raises(TypeError):
            pareto_stability(frontier, {"a": _WA, "b": _WB}, bad)


def test_nested_weights_value_must_be_dict():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(TypeError):
        pareto_stability(frontier, {"a": _WA, "b": 5}, [1.0])


def test_type_errors_take_precedence():
    with pytest.raises(TypeError):
        pareto_stability(1, [], 0)


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"a": _WA},
        {"": _WA, "b": _WB},
        {"a": _WA, "a": _WB},
        {"a": {"uhi": 1.0, "energy": 1.0}, "b": _WB},
        {"a": {"uhi": 1.0, "energy": 1.0, "vent": 1.0, "x": 1.0},
         "b": _WB},
        {"a": {"UHI": 1.0, "energy": 1.0, "vent": 1.0}, "b": _WB},
    ],
)
def test_bad_weights_structure_value_error(weights):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_stability(frontier, weights, [1.0])


@pytest.mark.parametrize(
    "value",
    [0, -1.0, True, False, None, "1", float("nan"), float("inf"),
     float("-inf"), [1.0], (1.0,)],
)
def test_bad_weight_values_rejected(value):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    weights = {"a": dict(_WA, uhi=value), "b": _WB}
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
        ["1"],
        [None],
        [float("nan")],
        [float("inf")],
        [float("-inf")],
        [[1]],
        [(1.0,)],
        [1, 1],
        [1.0, 1.0],
    ],
)
def test_bad_limits_rejected(limits):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_stability(frontier, {"a": _WA, "b": _WB}, limits)


def test_malformed_frontier_value_error():
    with pytest.raises(ValueError):
        pareto_stability("not json\n", {"a": _WA, "b": _WB}, [1.0])


def test_real_frontier_roundtrip_is_accepted():
    report = _one_region()
    actions = _actions()
    frontier = driver_plan_frontier(report, actions, [0, 1, 3])
    out = pareto_stability(frontier, {"a": _WA, "b": _WB}, [3])
    payload = _payload(out)
    assert payload["points"][0]["groups"][0]["recommend"] == [
        "green", "roof"
    ]


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.pareto_stability is urban_micro.pareto_stability
    assert "pareto_stability" in urban_micro.__all__
