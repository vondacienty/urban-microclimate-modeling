"""Tests for urban_micro.pareto_bootstrap: exact bootstrap confidence
intervals around the modal Pareto recommendation."""

import json

import pytest

from urban_micro import driver_plan_frontier, pareto_bootstrap
from urban_micro import uhi as _uhi

from test_driver_plan import _actions, _one_region
from test_driver_plan_pareto import _frontier, _fpoint, _state

_TOP_KEYS = ["confidence", "points"]
_POINT_KEYS = ["limit", "groups"]
_GROUP_KEYS = ["by", "pick", "n", "lower", "upper", "p"]

_UHI_HEAVY = {"uhi": 10.0, "energy": 1.0, "vent": 1.0}
_ENERGY_HEAVY = {"uhi": 1.0, "energy": 10.0, "vent": 1.0}
_BALANCED = {"uhi": 1.0, "energy": 1.0, "vent": 1.0}


def _payload(out):
    return json.loads(out)


def _split_frontier():
    return _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=-2.0)),
        _fpoint(2.0, _state("region", ["green"], 2.0, uhi=-3.0)),
    )


def _weights(*specs):
    return {
        name: dict(weight)
        for name, weight in zip("abcdefgh", specs)
    }


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3])
    weights = _weights(_UHI_HEAVY, _ENERGY_HEAVY)
    out = pareto_bootstrap(frontier, weights, [3, 1])
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = _payload(out)
    assert list(payload) == _TOP_KEYS
    assert payload["confidence"] == 0.95
    points = payload["points"]
    assert [point["limit"] for point in points] == [1.0, 3.0]
    for point in points:
        assert list(point) == _POINT_KEYS
        for group in point["groups"]:
            assert list(group) == _GROUP_KEYS
            assert isinstance(group["n"], int)


def test_points_ascending_groups_region_before_window():
    frontier = _frontier(
        _fpoint(
            0.0,
            _state("region", [], 0.0),
            _state("window", ["roof"], 0.0, energy=-1.0),
        ),
    )
    weights = _weights(_UHI_HEAVY, _ENERGY_HEAVY)
    payload = _payload(pareto_bootstrap(frontier, weights, [1]))
    groups = payload["points"][0]["groups"]
    assert [group["by"] for group in groups] == ["region", "window"]


def test_empty_frontier_groups_stay_empty():
    frontier = _frontier(_fpoint(1.0), _fpoint(2.0))
    weights = _weights(_UHI_HEAVY, _ENERGY_HEAVY)
    out = pareto_bootstrap(frontier, weights, [2, 1])
    assert out == (
        '{"confidence":0.950000,"points":['
        '{"limit":1.000000,"groups":[]},'
        '{"limit":2.000000,"groups":[]}]}\n'
    )


# --- modal pick -------------------------------------------------------------

def test_split_vote_takes_lexicographic_first_pick():
    out = pareto_bootstrap(
        _split_frontier(), _weights(_UHI_HEAVY, _ENERGY_HEAVY), [2]
    )
    assert '"pick":["green"]' in out


def test_majority_pick_wins():
    weights = _weights(_UHI_HEAVY, _UHI_HEAVY, _ENERGY_HEAVY)
    group = _payload(pareto_bootstrap(_split_frontier(), weights, [2]))[
        "points"
    ][0]["groups"][0]
    assert group["pick"] == ["green"]
    assert group["n"] == 3


# --- all-null ----------------------------------------------------------------

def test_all_null_pick_has_unit_bounds_and_p():
    frontier = _frontier(
        _fpoint(2.0, _state("region", ["roof"], 2.0, energy=-4.0)),
    )
    weights = _weights(_UHI_HEAVY, _ENERGY_HEAVY)
    out = pareto_bootstrap(frontier, weights, [1])
    assert out == (
        '{"confidence":0.950000,"points":[{"limit":1.000000,"groups":['
        '{"by":"region","pick":null,"n":2,"lower":1.000000,'
        '"upper":1.000000,"p":1.000000}]}]}\n'
    )


# --- intervals and p-values --------------------------------------------------

def test_n2_split_vote_interval_and_p():
    # N=4; a=.025 -> indices floor(.075)=0 and ceil(2.925)=3 of the sorted
    # distribution [0, .5, .5, 1]; p = 2*(C(2,0)+C(2,1))/4 = 1.
    out = pareto_bootstrap(
        _split_frontier(), _weights(_UHI_HEAVY, _ENERGY_HEAVY), [2]
    )
    group = _payload(out)["points"][0]["groups"][0]
    assert group["n"] == 2
    assert group["lower"] == 0.0
    assert group["upper"] == 1.0
    assert group["p"] == 1.0


def test_n3_unanimous_pick_is_certain_with_sign_test_p():
    weights = _weights(_UHI_HEAVY, _UHI_HEAVY, _UHI_HEAVY)
    group = _payload(pareto_bootstrap(_split_frontier(), weights, [2]))[
        "points"
    ][0]["groups"][0]
    assert group["pick"] == ["green"]
    assert group["lower"] == 1.0
    assert group["upper"] == 1.0
    assert group["p"] == pytest.approx(0.25, abs=1e-6)


def test_n4_three_votes_interval_and_p():
    # N=256; blocks 1, 12, 54, 108, 81 for k=0..4; index 6 -> k=1 (.25),
    # index 249 -> k=4 (1); p = 2*(C(4,0)+C(4,1))/16 = .625.
    weights = _weights(
        _UHI_HEAVY, _UHI_HEAVY, _UHI_HEAVY, _ENERGY_HEAVY
    )
    group = _payload(pareto_bootstrap(_split_frontier(), weights, [2]))[
        "points"
    ][0]["groups"][0]
    assert group["pick"] == ["green"]
    assert group["lower"] == 0.25
    assert group["upper"] == 1.0
    assert group["p"] == pytest.approx(0.625, abs=1e-6)


def test_custom_confidence_renders_and_changes_indices():
    weights = _weights(_UHI_HEAVY, _UHI_HEAVY, _ENERGY_HEAVY)
    out = pareto_bootstrap(
        _split_frontier(), weights, [2], confidence=0.5
    )
    assert '"confidence":0.500000' in out
    group = _payload(out)["points"][0]["groups"][0]
    # N=27, a=.25: lower index floor(26*.25)=6 falls in the k=1 block
    # (multiplicity 6, indices 1..6) -> 1/3; upper index ceil(26*.75)=20
    # falls in the k=3 block (indices 19..26) -> 1.
    assert group["lower"] == pytest.approx(1 / 3, abs=1e-6)
    assert group["upper"] == 1.0


def test_n8_runs_instantly_via_block_enumeration():
    weights = _weights(
        _UHI_HEAVY, _ENERGY_HEAVY, _UHI_HEAVY, _ENERGY_HEAVY,
        _UHI_HEAVY, _ENERGY_HEAVY, _UHI_HEAVY, _ENERGY_HEAVY,
    )
    group = _payload(pareto_bootstrap(_split_frontier(), weights, [2]))[
        "points"
    ][0]["groups"][0]
    assert group["n"] == 8
    # r=4: index floor((8**8-1)*.025) lands in the k=1 block (.125) and
    # the symmetric upper index in the k=7 block (.875).
    assert group["lower"] == 0.125
    assert group["upper"] == 0.875
    assert group["p"] == 1.0


# --- number formatting -------------------------------------------------------

def test_no_negative_zero_and_six_decimals():
    out = pareto_bootstrap(
        _split_frontier(), _weights(_UHI_HEAVY, _ENERGY_HEAVY), [0]
    )
    assert "-0.000000" not in out
    assert '"confidence":0.950000' in out
    assert '"limit":0.000000' in out


def test_huge_int_weights_and_limits_stay_exact():
    huge = 10**1000
    # Two picks identical in uhi, differing only in energy; a huge uhi
    # weight must not bury the energy ordering under precision rounding.
    frontier = _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(
            1.0,
            _state("region", ["roof"], 1.0, uhi=-2.123456, energy=-1.0),
        ),
        _fpoint(
            2.0,
            _state("region", ["green"], 2.0, uhi=-2.123456, energy=-2.0),
        ),
    )
    weights = _weights(
        {"uhi": huge, "energy": 1, "vent": 1},
        {"uhi": huge, "energy": 1, "vent": 1},
    )
    group = _payload(pareto_bootstrap(frontier, weights, [10**500]))[
        "points"
    ][0]["groups"][0]
    assert group["pick"] == ["green"]
    assert group["lower"] == 1.0 and group["upper"] == 1.0


# --- validation --------------------------------------------------------------

def test_type_errors_follow_pareto_stability():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    good = _weights(_BALANCED, _BALANCED)
    for bad in (1, None, [], True, {}):
        with pytest.raises(TypeError):
            pareto_bootstrap(bad, good, [1.0])
    for bad in ([], None, 42, "x", True):
        with pytest.raises(TypeError):
            pareto_bootstrap(frontier, bad, [1.0])
    for bad in (None, 42, "x", True, {}):
        with pytest.raises(TypeError):
            pareto_bootstrap(frontier, good, bad)


def test_type_errors_take_precedence():
    with pytest.raises(TypeError):
        pareto_bootstrap(1, [], None)


def test_weight_sets_between_two_and_eight():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_bootstrap(frontier, _weights(_BALANCED), [1.0])
    with pytest.raises(ValueError):
        pareto_bootstrap(
            frontier,
            {c: _BALANCED for c in "abcdefghi"},
            [1.0],
        )


@pytest.mark.parametrize(
    "value",
    [True, False, 0, 1, -0.5, 1.0, 0.0, None, "0.95",
     float("nan"), float("inf"), float("-inf"), [0.95]],
)
def test_bad_confidence_rejected(value):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_bootstrap(
            frontier, _weights(_BALANCED, _BALANCED), [1.0],
            confidence=value,
        )


def test_weight_entry_contract_is_inherited():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_bootstrap(
            frontier,
            {"a": {"uhi": 1.0, "energy": 1.0}, "b": _BALANCED},
            [1.0],
        )
    with pytest.raises(TypeError):
        pareto_bootstrap(
            frontier, {"a": None, "b": _BALANCED}, [1.0]
        )


def test_bad_limits_contract_is_inherited():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_bootstrap(
            frontier, _weights(_BALANCED, _BALANCED), [1, 1]
        )


def test_malformed_frontier_value_error():
    with pytest.raises(ValueError):
        pareto_bootstrap(
            '{"points":[]}\n', _weights(_BALANCED, _BALANCED), [1.0]
        )


# --- export ------------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.pareto_bootstrap is urban_micro.pareto_bootstrap
    assert "pareto_bootstrap" in urban_micro.__all__
