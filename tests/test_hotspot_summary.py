"""Tests for urban_micro.hotspot_plan pick normalization and hotspot_summary."""

import json
from fractions import Fraction
from itertools import product

import pytest

from urban_micro import hotspot_plan, hotspot_plan_impact, hotspot_summary
from urban_micro import uhi as _uhi


def _priority(specs, alpha=0.05):
    """Build a canonical rank_hotspot_priority report.

    ``specs`` is a list of ``(pick, by, key, effect, q)`` rows; picks
    must already be in canonical rank_hotspots order (ascending pick,
    region before window, ascending key).
    """
    hotspots, effects = [], {}
    for index, (pick, by, key, effect, q) in enumerate(specs):
        hotspots.append(
            '{"pick":'
            + json.dumps(list(pick), separators=(",", ":"))
            + f',"by":"{by}","key":"{key}","steps":10,"hits":10,'
            + f'"max_abs":{index + 1},"persistence":1.000000}}'
        )
        effects[(tuple(pick), by, key)] = (effect, q)
    rank = f'{{"alpha":{alpha:.6f},"hotspots":[{",".join(hotspots)}]}}\n'
    return _uhi.rank_hotspot_priority(rank, effects)


def _impact_group(by, key, baseline, post, change, *, n=1, limit=3.0, cost=1.0):
    return (
        '{"by":' + json.dumps(by)
        + ',"key":' + json.dumps(key)
        + f',"limit":{limit:.6f},"cost":{cost:.6f},"n":{n},'
        + f'"baseline":{baseline:.6f},"post":{post:.6f},'
        + f'"change":{change:.6f}}}'
    )


def _impact(groups, alpha=0.05):
    return f'{{"alpha":{alpha:.6f},"groups":[{",".join(groups)}]}}\n'


# --- hotspot_plan accepts report picks lifted verbatim as JSON arrays ---

def test_hotspot_plan_accepts_list_picks_in_allow():
    priority = _priority(
        [
            (("green", "roof"), "region", "R1", -1.0, 0.01),
            (("material",), "region", "R1", -2.0, 0.02),
        ]
    )
    lifted = [
        item["pick"]
        for item in json.loads(priority)["groups"][0]["items"]
    ]
    out = hotspot_plan(
        priority,
        {("green", "roof"): 2.0, ("material",): 1.5},
        {("region", "R1"): 1.6},
        {("region", "R1"): lifted},
    )
    assert json.loads(out)["groups"][0]["pick"] == [["material"]]


def test_hotspot_plan_tuples_still_supported():
    priority = _priority(
        [
            (("green",), "region", "R1", -1.0, 0.01),
            (("roof",), "region", "R1", -2.0, 0.02),
        ]
    )
    out = hotspot_plan(
        priority,
        {("green",): 1.0, ("roof",): 1.5},
        {("region", "R1"): 3.0},
        {("region", "R1"): [("green",), ("roof",)]},
    )
    assert json.loads(out)["groups"][0]["pick"] == [["green"], ["roof"]]


@pytest.mark.parametrize(
    "bad",
    [
        ["water"],
        ["roof", "green"],
        ["green", "green"],
        [],
        [1],
        ["green", 2],
    ],
)
def test_hotspot_plan_rejects_malformed_list_picks(bad):
    priority = _priority(
        [
            (("green",), "region", "R1", -1.0, 0.01),
            (("roof",), "region", "R1", -2.0, 0.02),
        ]
    )
    with pytest.raises(ValueError):
        hotspot_plan(
            priority,
            {("green",): 1.0, ("roof",): 1.5},
            {("region", "R1"): 3.0},
            {("region", "R1"): [bad, ["roof"]]},
        )


# --- hotspot_summary contract ---

def test_empty_report_requires_empty_weights():
    empty = '{"alpha":0.050000,"groups":[]}\n'
    assert hotspot_summary(empty, {}) == (
        '{"alpha":0.050000,"groups":[]}\n'
    )
    with pytest.raises(ValueError):
        hotspot_summary(empty, {("region", "R1"): 1.0})


def test_type_errors():
    empty = '{"alpha":0.050000,"groups":[]}\n'
    with pytest.raises(TypeError):
        hotspot_summary(123, {})
    with pytest.raises(TypeError):
        hotspot_summary(None, {})
    with pytest.raises(TypeError):
        hotspot_summary(empty, [])


def test_weights_key_set_and_values():
    report = _impact([_impact_group("region", "R1", 10, 8, -2)])
    with pytest.raises(ValueError):
        hotspot_summary(report, {})
    with pytest.raises(ValueError):
        hotspot_summary(
            report,
            {("region", "R1"): 1.0, ("window", "W1"): 1.0},
        )
    for bad in (0, -1, float("nan"), float("inf"), True, "1"):
        with pytest.raises(ValueError):
            hotspot_summary(report, {("region", "R1"): bad})


def test_more_than_16_panels_in_one_dimension_raises():
    groups = [
        _impact_group("region", f"R{i:02d}", 10, 8, -2) for i in range(17)
    ]
    weights = {("region", f"R{i:02d}"): 1.0 for i in range(17)}
    with pytest.raises(ValueError):
        hotspot_summary(_impact(groups), weights)


def test_sixteen_panels_allowed():
    groups = [
        _impact_group("region", f"R{i:02d}", 10, 8, -2) for i in range(16)
    ]
    weights = {("region", f"R{i:02d}"): 1.0 for i in range(16)}
    out = json.loads(hotspot_summary(_impact(groups), weights))
    assert out["groups"][0]["n"] == 16


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        '{"alpha":0.05,"groups":[]}\n',
        '{"alpha":0.050000,"groups":[]}',
        '{"alpha":0.050000,"groups":[]}\n\n',
        '{"alpha":0.050000,"groups":[]} ',
    ],
)
def test_malformed_report_raises_value_error(raw):
    with pytest.raises(ValueError):
        hotspot_summary(raw, {})


def test_output_is_canonical_single_trailing_newline():
    report = _impact([_impact_group("region", "R1", 10, 8, -2)])
    out = hotspot_summary(report, {("region", "R1"): 1.0})
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    assert list(payload["groups"][0]) == [
        "by", "n", "base", "post", "change", "p", "q", "reject",
    ]
    assert isinstance(payload["groups"][0]["n"], int)
    assert isinstance(payload["groups"][0]["reject"], bool)


def test_matches_exact_fraction_reference():
    rows = [
        ("region", "R1", 10, 8.5, -1.5, 1.0),
        ("region", "R2", 20, 18.0, -2.0, 2.0),
        ("window", "W1", 5, 4.0, -1.0, 1.0),
        ("window", "W2", 7, 7.5, 0.5, 1.0),
    ]
    groups = [
        _impact_group(by, key, base, post, change)
        for by, key, base, post, change, _ in rows
    ]
    weights = {(by, key): w for by, key, _, _, _, w in rows}
    out = json.loads(hotspot_summary(_impact(groups), weights))

    alpha = Fraction("0.05")
    dims = {}
    for by, key, base, post, change, w in rows:
        dims.setdefault(by, []).append(
            (
                Fraction(str(base)),
                Fraction(str(post)),
                Fraction(str(change)),
                Fraction(str(w)),
            )
        )
    expected = []
    for by, panels in dims.items():
        total = sum((w for *_, w in panels), Fraction(0))
        base = sum(w * b for b, _, _, w in panels) / total
        post = sum(w * p for _, p, _, w in panels) / total
        change = sum(w * c for _, _, c, w in panels) / total
        contributions = [w * c / total for _, _, c, w in panels]
        n = len(contributions)
        tail = 0
        for signs in product((-1, 1), repeat=n):
            statistic = sum(s * v for s, v in zip(signs, contributions))
            if abs(statistic) >= abs(change):
                tail += 1
        expected.append(
            [by, n, base, post, change, Fraction(tail, 1 << n)]
        )
    expected.sort(
        key=lambda item: (item[5], 0 if item[0] == "region" else 1)
    )
    tested = len(expected)
    q_values, running = [], Fraction(1)
    for rank in range(tested, 0, -1):
        running = min(running, tested * expected[rank - 1][5] / rank)
        q_values.insert(0, min(Fraction(1), running))

    for group, row, q in zip(out["groups"], expected, q_values):
        by, n, base, post, change, p = row
        assert group["by"] == by
        assert group["n"] == n
        assert group["base"] == pytest.approx(float(base), abs=1e-6)
        assert group["post"] == pytest.approx(float(post), abs=1e-6)
        assert group["change"] == pytest.approx(float(change), abs=1e-6)
        assert group["p"] == pytest.approx(float(p), abs=1e-6)
        assert group["q"] == pytest.approx(float(q), abs=1e-6)
        assert group["reject"] is (q <= alpha)


def test_region_breaks_p_tie_before_window():
    report = _impact(
        [
            _impact_group("region", "R1", 10, 9, -1),
            _impact_group("window", "W1", 10, 9, -1),
        ]
    )
    weights = {("region", "R1"): 1.0, ("window", "W1"): 1.0}
    out = json.loads(hotspot_summary(report, weights))
    assert [g["by"] for g in out["groups"]] == ["region", "window"]


def test_groups_ordered_by_ascending_p():
    # Region: two coherent negative panels -> p = 0.5. Window: offsetting
    # panels where every sign vector reaches the small observed change ->
    # p = 1.0, so the region group sorts first.
    report = _impact(
        [
            _impact_group("region", "R1", 10, 8.5, -1.5),
            _impact_group("region", "R2", 20, 18.0, -2.0),
            _impact_group("window", "W1", 5, 4.0, -1.0),
            _impact_group("window", "W2", 7, 7.5, 0.5),
        ]
    )
    weights = {
        ("region", "R1"): 1.0,
        ("region", "R2"): 2.0,
        ("window", "W1"): 1.0,
        ("window", "W2"): 1.0,
    }
    groups = json.loads(hotspot_summary(report, weights))["groups"]
    assert [g["by"] for g in groups] == ["region", "window"]
    assert groups[0]["p"] == 0.5
    assert groups[1]["p"] == 1.0
    # BH adjusted p: region q = min(2*0.5/1, 2*1/2) = 1.0.
    assert groups[0]["q"] == 1.0
    assert groups[1]["q"] == 1.0
    assert all(g["reject"] is False for g in groups)


def test_coherent_panels_reject():
    groups = [
        _impact_group("region", f"R{i}", 10, 9, -1) for i in range(8)
    ]
    weights = {("region", f"R{i}"): 1.0 for i in range(8)}
    group = json.loads(hotspot_summary(_impact(groups), weights))["groups"][0]
    # 2 of 256 sign vectors (all signs equal) reach the threshold.
    assert group["p"] == 0.007812
    assert group["reject"] is True


def test_negative_zero_normalized():
    report = _impact([_impact_group("region", "R1", 0, 0, 0)])
    out = hotspot_summary(report, {("region", "R1"): 1.0})
    assert '"change":0.000000' in out
    assert '"base":0.000000' in out


def test_roundtrip_through_impact_report():
    """The summary accepts a byte-for-byte hotspot_plan_impact output."""
    priority = _priority(
        [
            (("green",), "region", "R1", -1.0, 0.01),
            (("roof",), "window", "W1", -2.0, 0.02),
        ]
    )
    plan = hotspot_plan(
        priority,
        {("green",): 1.0, ("roof",): 1.5},
        {("region", "R1"): 3.0, ("window", "W1"): 3.0},
        {("region", "R1"): [["green"]], ("window", "W1"): [["roof"]]},
    )
    effects = {
        ("region", "R1"): (10.0, {("green",): -1.0}),
        ("window", "W1"): (5.0, {("roof",): -2.0}),
    }
    impact = hotspot_plan_impact(plan, effects)
    weights = {("region", "R1"): 1.0, ("window", "W1"): 2.0}
    summary = hotspot_summary(impact, weights)
    assert summary.endswith("\n")
    # Each dimension holds one panel, so both p-values are 1.0 and the
    # region-before-window tie-break orders the groups.
    assert [g["by"] for g in json.loads(summary)["groups"]] == [
        "region", "window",
    ]
