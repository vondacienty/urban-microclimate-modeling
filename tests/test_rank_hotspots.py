"""Tests for frontier_rank_trajectory pick tightening and rank_hotspots."""

import json

import pytest

from urban_micro import (
    frontier_rank_trajectory,
    rank_hotspots,
)


def _panel(by, key, effect, rank, delta, direction, contribution):
    return (
        '{"by":' + json.dumps(by)
        + ',"key":' + json.dumps(key)
        + ',"effect":' + f"{effect:.6f}"
        + ',"rank":' + str(rank)
        + ',"delta":' + str(delta)
        + ',"direction":' + json.dumps(direction)
        + ',"contribution":' + f"{contribution:.6f}"
        + "}"
    )


def _rank(pick, n, support, effect, rank, driver, panels, stable=True):
    return (
        '{"pick":' + json.dumps(pick, separators=(",", ":"))
        + ',"n":' + str(n)
        + ',"support":' + f"{support:.6f}"
        + ',"effect":' + f"{effect:.6f}"
        + ',"rank":' + str(rank)
        + ',"stable":' + ("true" if stable else "false")
        + ',"driver":{"by":' + json.dumps(driver[0])
        + ',"key":' + json.dumps(driver[1])
        + '},"panels":[' + ",".join(panels) + "]}"
    )


def _test(reject):
    return (
        '{"a":["green"],"b":["roof"],"n_a":3,"n_b":2,"diff":0.100000,'
        '"p":0.500000,"q":' + ("0.040000" if reject else "0.100000")
        + ',"reject":' + ("true" if reject else "false") + "}"
    )


def _point(budget, ranks, reject):
    return (
        '{"budget":' + f"{budget:.6f}"
        + ',"ranks":[' + ",".join(ranks) + "]"
        + ',"tests":[' + _test(reject) + "]}"
    )


REGION = ("region", "north")
WINDOW = ("window", "w1")


def _attribution_report():
    # Two picks (green, roof) over four ascending budget points; ranks
    # swap at every step. Point 1 and 3 reject the pair-wise test,
    # point 2 does not.
    p0_green = _rank(
        ["green"], 3, 0.7, 1.0, 1, REGION,
        [
            _panel("region", "north", 1.2, 1, 0, "same", 0.2),
            _panel("window", "w1", 0.9, 1, 0, "same", -0.1),
        ],
    )
    p0_roof = _rank(
        ["roof"], 2, 0.3, 0.5, 2, REGION,
        [
            _panel("region", "north", 0.6, 2, 0, "same", 0.1),
            _panel("window", "w1", 0.4, 2, 0, "same", -0.1),
        ],
    )
    p1_roof = _rank(
        ["roof"], 2, 0.5, 1.2, 1, REGION,
        [
            _panel("region", "north", 0.3, 2, 1, "fall", -0.9),
            _panel("window", "w1", 1.4, 1, 0, "same", 0.2),
        ],
    )
    p1_green = _rank(
        ["green"], 3, 0.5, 0.8, 2, REGION,
        [
            _panel("region", "north", 1.5, 1, -1, "rise", 0.7),
            _panel("window", "w1", 0.2, 2, 0, "same", -0.6),
        ],
    )
    p2_green = _rank(
        ["green"], 3, 0.5, 1.1, 1, WINDOW,
        [
            _panel("region", "north", 1.8, 1, 0, "same", 0.7),
            _panel("window", "w1", 0.3, 2, 1, "fall", -0.8),
        ],
    )
    p2_roof = _rank(
        ["roof"], 2, 0.5, 0.4, 2, WINDOW,
        [
            _panel("region", "north", 0.1, 2, 0, "same", -0.3),
            _panel("window", "w1", 1.0, 1, -1, "rise", 0.6),
        ],
    )
    p3_roof = _rank(
        ["roof"], 2, 0.5, 1.3, 1, WINDOW,
        [
            _panel("region", "north", 1.6, 1, 0, "same", 0.3),
            _panel("window", "w1", 0.1, 2, 1, "fall", -1.2),
        ],
    )
    p3_green = _rank(
        ["green"], 3, 0.5, 0.7, 2, WINDOW,
        [
            _panel("region", "north", 0.2, 2, 0, "same", -0.5),
            _panel("window", "w1", 1.4, 1, -1, "rise", 0.7),
        ],
    )
    points = [
        _point(0, [p0_green, p0_roof], False),
        _point(1, [p1_roof, p1_green], True),
        _point(2, [p2_green, p2_roof], False),
        _point(3, [p3_roof, p3_green], True),
    ]
    return '{"alpha":0.050000,"points":[' + ",".join(points) + "]}\n"


TRAJECTORY_EXPECTED = (
    '{"alpha":0.050000,"trajectories":['
    '{"pick":["green"],"start":0.000000,"end":3.000000,"n":4,"max_abs":1,'
    '"driver":{"by":"window","key":"w1"},"persistence":0.666667,'
    '"steps":['
    '{"from":0.000000,"to":1.000000,"from_rank":1,"to_rank":2,"delta":1,'
    '"direction":"fall","driver":{"by":"region","key":"north"},'
    '"significant":true},'
    '{"from":1.000000,"to":2.000000,"from_rank":2,"to_rank":1,"delta":-1,'
    '"direction":"rise","driver":{"by":"window","key":"w1"},'
    '"significant":false},'
    '{"from":2.000000,"to":3.000000,"from_rank":1,"to_rank":2,"delta":1,'
    '"direction":"fall","driver":{"by":"window","key":"w1"},'
    '"significant":true}]},'
    '{"pick":["roof"],"start":0.000000,"end":3.000000,"n":4,"max_abs":1,'
    '"driver":{"by":"window","key":"w1"},"persistence":0.666667,'
    '"steps":['
    '{"from":0.000000,"to":1.000000,"from_rank":2,"to_rank":1,"delta":-1,'
    '"direction":"rise","driver":{"by":"region","key":"north"},'
    '"significant":true},'
    '{"from":1.000000,"to":2.000000,"from_rank":1,"to_rank":2,"delta":1,'
    '"direction":"fall","driver":{"by":"window","key":"w1"},'
    '"significant":false},'
    '{"from":2.000000,"to":3.000000,"from_rank":2,"to_rank":1,"delta":-1,'
    '"direction":"rise","driver":{"by":"window","key":"w1"},'
    '"significant":true}]}]}\n'
)

HOTSPOTS_EXPECTED = (
    '{"alpha":0.050000,"hotspots":['
    '{"pick":["green"],"by":"region","key":"north","steps":1,"hits":1,'
    '"max_abs":1,"persistence":0.333333},'
    '{"pick":["green"],"by":"window","key":"w1","steps":2,"hits":1,'
    '"max_abs":1,"persistence":0.666667},'
    '{"pick":["roof"],"by":"region","key":"north","steps":1,"hits":1,'
    '"max_abs":1,"persistence":0.333333},'
    '{"pick":["roof"],"by":"window","key":"w1","steps":2,"hits":1,'
    '"max_abs":1,"persistence":0.666667}]}\n'
)


def test_trajectory_canonical_fixture():
    report = _attribution_report()
    # The hand-built attribution report must reproduce the canonical
    # trajectory output byte-for-byte.
    assert frontier_rank_trajectory(report) == TRAJECTORY_EXPECTED


def test_rank_hotspots_aggregation():
    trajectory = frontier_rank_trajectory(_attribution_report())
    assert rank_hotspots(trajectory) == HOTSPOTS_EXPECTED


def test_rank_hotspots_no_significant_steps():
    report = _attribution_report().replace('"reject":true', '"reject":false')
    trajectory = frontier_rank_trajectory(report)
    assert '"significant":true' not in trajectory
    assert rank_hotspots(trajectory) == '{"alpha":0.050000,"hotspots":[]}\n'


def test_rank_hotspots_empty_trajectories():
    trajectory = '{"alpha":0.050000,"trajectories":[]}\n'
    assert rank_hotspots(trajectory) == '{"alpha":0.050000,"hotspots":[]}\n'


def test_rank_hotspots_max_abs_uses_significant_only():
    # A 3-step... 2-step run: the non-significant step moves three
    # ranks, the significant one only one; max_abs must be 1, not 3.
    trajectory = (
        '{"alpha":0.050000,"trajectories":['
        '{"pick":["material"],"start":0.000000,"end":2.000000,"n":3,'
        '"max_abs":3,"driver":{"by":"region","key":"north"},'
        '"persistence":1.000000,"steps":['
        '{"from":0.000000,"to":1.000000,"from_rank":4,"to_rank":1,'
        '"delta":-3,"direction":"rise",'
        '"driver":{"by":"region","key":"north"},"significant":false},'
        '{"from":1.000000,"to":2.000000,"from_rank":1,"to_rank":2,'
        '"delta":1,"direction":"fall",'
        '"driver":{"by":"region","key":"north"},"significant":true}]}]}\n'
    )
    assert rank_hotspots(trajectory) == (
        '{"alpha":0.050000,"hotspots":['
        '{"pick":["material"],"by":"region","key":"north","steps":2,'
        '"hits":1,"max_abs":1,"persistence":1.000000}]}\n'
    )


def test_rank_hotspots_type_error():
    with pytest.raises(TypeError):
        rank_hotspots(None)
    with pytest.raises(TypeError):
        rank_hotspots(b"{}")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "not json",
        "{}",
        '{"alpha":0.050000,"trajectories":[]}',  # missing newline
        '{"alpha":0.050000,"trajectories":[]}\n\n',
        '{"alpha":"0.050000","trajectories":[]}\n',
        '{"alpha":0.050000,"trajectories":{}}\n',
        '{"alpha":0.050000}\n',
    ],
)
def test_rank_hotspots_value_errors(bad):
    with pytest.raises(ValueError):
        rank_hotspots(bad)


def test_rank_hotspots_rejects_non_canonical_renderings():
    trajectory = frontier_rank_trajectory(_attribution_report())
    with pytest.raises(ValueError):
        rank_hotspots(" " + trajectory)
    # Whitespace before a separator.
    with pytest.raises(ValueError):
        rank_hotspots(trajectory.replace('",', '" ,', 1))
    # Key reordering/renaming.
    with pytest.raises(ValueError):
        rank_hotspots(trajectory.replace('"steps":', '"xx":', 1))
    # Tampered persistence.
    with pytest.raises(ValueError):
        rank_hotspots(trajectory.replace("0.666667", "0.500000", 1))
    # Tampered step delta breaks delta == to_rank - from_rank.
    with pytest.raises(ValueError):
        rank_hotspots(trajectory.replace('"delta":1', '"delta":2', 1))


def test_trajectory_rejects_empty_pick():
    report = _attribution_report().replace(
        '"pick":["green"]', '"pick":[]', 1
    )
    with pytest.raises(ValueError):
        frontier_rank_trajectory(report)


def test_trajectory_rejects_unknown_pick_member():
    report = _attribution_report().replace(
        '"pick":["green"]', '"pick":["building"]', 1
    )
    with pytest.raises(ValueError):
        frontier_rank_trajectory(report)


def test_trajectory_rejects_unsorted_pick():
    report = _attribution_report().replace(
        '"pick":["green"]', '"pick":["roof","green"]', 1
    )
    with pytest.raises(ValueError):
        frontier_rank_trajectory(report)


def test_trajectory_rejects_duplicate_pick_member():
    report = _attribution_report().replace(
        '"pick":["green"]', '"pick":["green","green"]', 1
    )
    with pytest.raises(ValueError):
        frontier_rank_trajectory(report)


def test_trajectory_non_string_raises_type_error():
    with pytest.raises(TypeError):
        frontier_rank_trajectory(None)
