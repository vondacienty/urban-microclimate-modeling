"""Tests for urban_micro.pareto_bootstrap: exact bootstrap intervals and
two-sided p-values for the pareto_stability winner across weight sets and
budget limits."""

import itertools
import json
import math

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


def _two_weights():
    return {"a": _UHI_HEAVY, "b": _ENERGY_HEAVY}


def _split_frontier():
    return _frontier(
        _fpoint(0.0, _state("region", [], 0.0)),
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=-2.0)),
        _fpoint(2.0, _state("region", ["green"], 2.0, uhi=-3.0)),
    )


def _enum_interval(n, r, confidence):
    """Reference interval: explicitly enumerate the n**n index sequences."""
    level = float(confidence)
    alpha = (1 - level) / 2
    proportions = sorted(
        sum(index < r for index in sequence) / n
        for sequence in itertools.product(range(n), repeat=n)
    )
    total = n ** n
    lower = proportions[math.floor((total - 1) * alpha)]
    upper = proportions[math.ceil((total - 1) * (1 - alpha))]
    return lower, upper


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3])
    out = pareto_bootstrap(frontier, _two_weights(), [3, 1])
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
            assert isinstance(group["pick"], (list, type(None)))
            assert isinstance(group["n"], int)


def test_region_before_window_groups():
    frontier = _frontier(
        _fpoint(
            0.0,
            _state("region", [], 0.0),
            _state("window", ["roof"], 0.0, energy=-1.0),
        ),
    )
    payload = _payload(pareto_bootstrap(frontier, _two_weights(), [1]))
    groups = payload["points"][0]["groups"]
    assert [group["by"] for group in groups] == ["region", "window"]


def test_empty_frontier_groups_stay_empty():
    frontier = _frontier(_fpoint(1.0), _fpoint(2.0))
    out = pareto_bootstrap(frontier, _two_weights(), [2, 1])
    assert out == (
        '{"confidence":0.950000,"points":['
        '{"limit":1.000000,"groups":[]},'
        '{"limit":2.000000,"groups":[]}]}\n'
    )


def test_confidence_echoes_six_decimals():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1])
    out = pareto_bootstrap(frontier, _two_weights(), [1], confidence=0.1234567)
    assert out.startswith('{"confidence":0.123457,')


# --- pick voting ------------------------------------------------------------

def test_pick_uses_mode_with_lexicographic_tiebreak():
    out = pareto_bootstrap(_split_frontier(), _two_weights(), [2])
    group = _payload(out)["points"][0]["groups"][0]
    assert group["pick"] == ["green"]
    assert group["n"] == 2


def test_majority_pick():
    weights = {"a": _UHI_HEAVY, "b": _UHI_HEAVY, "c": _ENERGY_HEAVY}
    group = _payload(pareto_bootstrap(_split_frontier(), weights, [2]))[
        "points"
    ][0]["groups"][0]
    assert group["pick"] == ["green"]
    assert group["n"] == 3


# --- all-null ---------------------------------------------------------------

def test_all_null_pick_has_unit_interval_and_p():
    frontier = _frontier(
        _fpoint(2.0, _state("region", ["roof"], 2.0, energy=-4.0)),
    )
    out = pareto_bootstrap(frontier, _two_weights(), [1])
    assert out == (
        '{"confidence":0.950000,"points":[{"limit":1.000000,"groups":['
        '{"by":"region","pick":null,"n":2,'
        '"lower":1.000000,"upper":1.000000,"p":1.000000}]}]}\n'
    )


# --- intervals and p-value --------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("confidence", [0.95, 0.9, 0.8, 0.5])
def test_interval_matches_explicit_nn_enumeration(n, confidence):
    # r uhi-heavy (green) and n-r energy-heavy (roof) weight sets split the
    # two non-empty picks; the winner holds the larger share (ties go to the
    # lexicographically smaller "green"). Bootstrap resamples with
    # replacement, so winner votes follow Binomial(n, winner_count/n).
    for r in range(1, n + 1):
        vote_weights = {
            **{f"u{i}": _UHI_HEAVY for i in range(r)},
            **{f"e{i}": _ENERGY_HEAVY for i in range(n - r)},
        }
        group = _payload(
            pareto_bootstrap(
                _split_frontier(), vote_weights, [2], confidence=confidence
            )
        )["points"][0]["groups"][0]
        winner_count = max(r, n - r)
        lower, upper = _enum_interval(n, winner_count, confidence)
        assert group["lower"] == pytest.approx(lower, abs=1e-6)
        assert group["upper"] == pytest.approx(upper, abs=1e-6)
        expected_p = min(
            1.0,
            2 * sum(math.comb(n, k) for k in range(min(r, n - r) + 1))
            / 2 ** n,
        )
        assert group["p"] == pytest.approx(expected_p, abs=1e-6)


def test_unanimous_n2_interval_collapses_and_p_is_a_half():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3])
    weights = {"a": _BALANCED, "b": _UHI_HEAVY}
    group = _payload(pareto_bootstrap(frontier, weights, [3]))["points"][0][
        "groups"
    ][0]
    assert group["pick"] == ["green", "roof"]
    assert group["n"] == 2
    assert group["lower"] == 1.0
    assert group["upper"] == 1.0
    assert group["p"] == 0.5


def test_split_n2_p_is_one_under_formula():
    # n=2, r=1: p = min(1, 2*(C(2,0)+C(2,1))/4) = min(1, 1.5) = 1.
    group = _payload(pareto_bootstrap(_split_frontier(), _two_weights(), [2]))[
        "points"
    ][0]["groups"][0]
    assert group["lower"] == 0.0
    assert group["upper"] == 1.0
    assert group["p"] == 1.0


def test_n3_unanimous_p_is_a_quarter():
    # n=3, r=3: p = 2*C(3,0)/8 = 0.25.
    frontier = _frontier(
        _fpoint(1.0, _state("region", ["roof"], 1.0, energy=-2.0)),
    )
    weights = {
        "a": _ENERGY_HEAVY, "b": _ENERGY_HEAVY, "c": _UHI_HEAVY
    }
    group = _payload(pareto_bootstrap(frontier, weights, [1]))["points"][0][
        "groups"
    ][0]
    assert group["n"] == 3
    assert group["p"] == 0.25


# --- number formatting ------------------------------------------------------

def test_six_decimals_and_no_negative_zero():
    out = pareto_bootstrap(_split_frontier(), _two_weights(), [0])
    assert "-0.000000" not in out
    assert '"limit":0.000000' in out
    assert '"n":2' in out


# --- validation -------------------------------------------------------------

def test_type_errors():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    good_weights = _two_weights()
    for bad in (1, None, [], True, {}):
        with pytest.raises(TypeError):
            pareto_bootstrap(bad, good_weights, [1.0])
    for bad in ([], None, 42, "x", True):
        with pytest.raises(TypeError):
            pareto_bootstrap(frontier, bad, [1.0])
    for bad in (None, 42, "x", True, {}):
        with pytest.raises(TypeError):
            pareto_bootstrap(frontier, good_weights, bad)


def test_type_errors_take_precedence():
    with pytest.raises(TypeError):
        pareto_bootstrap(1, [], None)


def test_weight_sets_must_number_two_to_eight():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_bootstrap(frontier, {"a": _BALANCED}, [1.0])
    with pytest.raises(ValueError):
        pareto_bootstrap(
            frontier, {str(i): _BALANCED for i in range(9)}, [1.0]
        )
    # exactly eight is admissible.
    pareto_bootstrap(
        frontier, {str(i): _UHI_HEAVY for i in range(8)}, [1.0]
    )


def test_weight_contract_is_inherited():
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_bootstrap(
            frontier, {"a": {"uhi": 0, "energy": 1.0, "vent": 1.0},
                       "b": _BALANCED},
            [1.0],
        )
    with pytest.raises(ValueError):
        pareto_bootstrap(frontier, {"a": _BALANCED, "b": {}}, [1.0])
    with pytest.raises(TypeError):
        pareto_bootstrap(frontier, {"a": None, "b": _BALANCED}, [1.0])
    with pytest.raises(ValueError):
        pareto_bootstrap(frontier, _two_weights(), [1, 1])


def test_malformed_frontier_value_error():
    with pytest.raises(ValueError):
        pareto_bootstrap(
            '{"points":[]}\n', _two_weights(), [1.0]
        )


@pytest.mark.parametrize(
    "confidence",
    [0, 1, -0.5, 1.0, 0.0, True, False, None, "0.9", 0j,
     float("nan"), float("inf"), float("-inf"), [0.9]],
)
def test_bad_confidence_rejected(confidence):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    with pytest.raises(ValueError):
        pareto_bootstrap(frontier, _two_weights(), [1.0], confidence=confidence)


@pytest.mark.parametrize("confidence", [0.0001, 0.5, 0.9999])
def test_good_confidence_accepted(confidence):
    frontier = driver_plan_frontier(_one_region(), _actions(), [1.0])
    payload = _payload(
        pareto_bootstrap(
            frontier, _two_weights(), [1.0], confidence=confidence
        )
    )
    assert payload["confidence"] == pytest.approx(confidence, abs=1e-6)


def test_confidence_check_runs_after_frontier_contract():
    # Bad first arguments raise their own errors even with bad confidence.
    with pytest.raises(TypeError):
        pareto_bootstrap(
            1, {"a": _BALANCED}, [1.0], confidence="nope"
        )


# --- huge integers stay exact Decimal ---------------------------------------

def test_huge_int_weights_and_limit_never_become_float():
    frontier = driver_plan_frontier(_one_region(), _actions(), [0, 1, 3])
    huge = 10 ** 10000
    weights = {
        "a": {"uhi": huge, "energy": 1, "vent": 1},
        "b": {"uhi": 1, "energy": huge, "vent": 1},
    }
    out = pareto_bootstrap(frontier, weights, [huge])
    payload = _payload(out)
    assert math.isinf(payload["points"][0]["limit"])
    token = out.split('"limit":', 1)[1].split(",", 1)[0]
    assert token == "1" + "0" * 10000 + ".000000"


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.pareto_bootstrap is urban_micro.pareto_bootstrap
    assert "pareto_bootstrap" in urban_micro.__all__
