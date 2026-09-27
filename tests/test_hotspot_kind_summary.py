"""Tests for urban_micro.hotspot_kind_summary."""

import json
from fractions import Fraction
from itertools import product

import pytest

from urban_micro import hotspot_kind_summary, hotspot_plan
from urban_micro import uhi as _uhi  # noqa: F401

from test_hotspot_summary import _priority


def _plan(specs, limits=None, allow=None, costs=None):
    """Build a hotspot_plan report from priority specs in any order."""
    specs = sorted(
        specs,
        key=lambda row: (
            row[0], 0 if row[1] == "region" else 1, row[2]
        ),
    )
    priority = _priority(specs)
    picks = sorted({pick for pick, *_ in specs})
    panels = sorted({(by, key) for _, by, key, *_ in specs})
    costs = costs or {pick: 1.0 for pick in picks}
    limits = limits or {panel: 100.0 for panel in panels}
    allow = allow or {
        panel: [pick for pick, by, key, *_ in specs if (by, key) == panel]
        for panel in panels
    }
    return hotspot_plan(priority, costs, limits, allow)


def test_empty_report_requires_empty_mappings():
    empty = '{"alpha":0.050000,"groups":[]}\n'
    assert hotspot_kind_summary(empty, {}, {}) == (
        '{"alpha":0.050000,"groups":[]}\n'
    )
    with pytest.raises(ValueError):
        hotspot_kind_summary(empty, {}, {("region", "R1"): 1.0})
    with pytest.raises(ValueError):
        hotspot_kind_summary(
            empty, {("region", "R1", ("green",)): 1.0}, {}
        )


def test_type_errors():
    empty = '{"alpha":0.050000,"groups":[]}\n'
    with pytest.raises(TypeError):
        hotspot_kind_summary(123, {}, {})
    with pytest.raises(TypeError):
        hotspot_kind_summary(None, {}, {})
    with pytest.raises(TypeError):
        hotspot_kind_summary(empty, [], {})
    with pytest.raises(TypeError):
        hotspot_kind_summary(empty, {}, [])


def test_malformed_report_is_value_error():
    with pytest.raises(ValueError):
        hotspot_kind_summary("not json\n", {}, {})
    with pytest.raises(ValueError):
        hotspot_kind_summary('{"alpha":0.05,"groups":[]}\n', {}, {})


def test_weights_key_set_and_values():
    plan = _plan(
        [
            (("green",), "region", "R1", -1.0, 0.01),
            (("roof",), "region", "R2", -2.0, 0.01),
        ]
    )
    effects = {
        ("region", "R1", ("green",)): -1.0,
        ("region", "R2", ("roof",)): -2.0,
    }
    with pytest.raises(ValueError):
        hotspot_kind_summary(plan, effects, {})
    with pytest.raises(ValueError):
        hotspot_kind_summary(
            plan, effects, {("region", "R1"): 1.0}
        )
    for bad in (0, -1, float("nan"), float("inf"), True, "1"):
        with pytest.raises(ValueError):
            hotspot_kind_summary(
                plan, effects,
                {("region", "R1"): bad, ("region", "R2"): 1.0},
            )


def test_effects_key_set_and_values():
    plan = _plan(
        [
            (("green",), "region", "R1", -1.0, 0.01),
            (("roof",), "region", "R2", -2.0, 0.01),
        ]
    )
    weights = {("region", "R1"): 1.0, ("region", "R2"): 1.0}
    # Skip-pick triples and missing triples are both errors.
    with pytest.raises(ValueError):
        hotspot_kind_summary(
            plan, {("region", "R1", ("green",)): -1.0}, weights
        )
    with pytest.raises(ValueError):
        hotspot_kind_summary(
            plan,
            {
                ("region", "R1", ("green",)): -1.0,
                ("region", "R2", ("roof",)): -2.0,
                ("region", "R1", ("roof",)): -3.0,
            },
            weights,
        )
    for bad in (float("nan"), float("inf"), True, False, "1"):
        with pytest.raises(ValueError):
            hotspot_kind_summary(
                plan,
                {
                    ("region", "R1", ("green",)): bad,
                    ("region", "R2", ("roof",)): -2.0,
                },
                weights,
            )


def test_effect_split_over_kinds_and_same_kind_sums():
    # R1 selects ("green",) effect -4 and ("green","roof") effect -2:
    # green per-panel v = -4 + (-2)/2 = -5, roof v = -1.
    plan = _plan(
        [
            (("green",), "region", "R1", -4.0, 0.01),
            (("green", "roof"), "region", "R1", -2.0, 0.01),
        ]
    )
    out = json.loads(
        hotspot_kind_summary(
            plan,
            {
                ("region", "R1", ("green",)): -4.0,
                ("region", "R1", ("green", "roof")): -2.0,
            },
            {("region", "R1"): 3.0},
        )
    )
    items = {
        item["kind"]: item for item in out["groups"][0]["items"]
    }
    assert items["green"]["change"] == -5.0
    assert items["roof"]["change"] == -1.0


def test_kind_summary_covers_only_panels_holding_the_kind():
    # Three region panels; only R3 holds material, so material has n == 1
    # while green spans R1 and R2 with n == 2.
    plan = _plan(
        [
            (("green",), "region", "R1", -1.0, 0.01),
            (("green",), "region", "R2", -3.0, 0.01),
            (("material",), "region", "R3", -2.0, 0.01),
        ]
    )
    out = json.loads(
        hotspot_kind_summary(
            plan,
            {
                ("region", "R1", ("green",)): -1.0,
                ("region", "R2", ("green",)): -3.0,
                ("region", "R3", ("material",)): -2.0,
            },
            {
                ("region", "R1"): 1.0,
                ("region", "R2"): 1.0,
                ("region", "R3"): 1.0,
            },
        )
    )
    items = {item["kind"]: item for item in out["groups"][0]["items"]}
    assert items["green"]["n"] == 2
    assert items["material"]["n"] == 1
    assert items["green"]["change"] == -2.0
    assert items["material"]["change"] == -2.0


def test_more_than_16_kind_panels_raises():
    specs = [
        (("green",), "region", f"R{i:02d}", -1.0, 0.01)
        for i in range(17)
    ]
    panels = [("region", f"R{i:02d}") for i in range(17)]
    plan = _plan(specs)
    effects = {
        (by, key, ("green",)): -1.0 for by, key in panels
    }
    weights = {panel: 1.0 for panel in panels}
    with pytest.raises(ValueError):
        hotspot_kind_summary(plan, effects, weights)


def test_17_panels_split_across_kinds_allowed():
    specs = [
        (("green",), "region", f"R{i:02d}", -1.0, 0.01)
        for i in range(16)
    ]
    specs.append((("roof",), "region", "R16", -1.0, 0.01))
    panels = [("region", f"R{i:02d}") for i in range(17)]
    plan = _plan(specs)
    effects = {
        (by, key, ("green",)): -1.0
        for by, key in panels
        if key != "R16"
    }
    effects[("region", "R16", ("roof",))] = -1.0
    out = json.loads(
        hotspot_kind_summary(plan, effects, {p: 1.0 for p in panels})
    )
    items = out["groups"][0]["items"]
    assert [(item["kind"], item["n"]) for item in items] == [
        ("green", 16),
        ("roof", 1),
    ]


def test_output_is_canonical_single_trailing_newline():
    plan = _plan(
        [
            (("green",), "region", "R1", -1.0, 0.01),
            (("roof",), "window", "W1", -2.0, 0.01),
        ]
    )
    out = hotspot_kind_summary(
        plan,
        {
            ("region", "R1", ("green",)): -1.0,
            ("window", "W1", ("roof",)): -2.0,
        },
        {("region", "R1"): 1.0, ("window", "W1"): 1.0},
    )
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    assert [g["by"] for g in payload["groups"]] == ["region", "window"]
    for group in payload["groups"]:
        assert list(group) == ["by", "items"]
        for item in group["items"]:
            assert list(item) == [
                "kind", "n", "change", "p", "q", "reject",
            ]
            assert isinstance(item["n"], int)
            assert isinstance(item["reject"], bool)


def test_items_sort_by_ascending_kind():
    plan = _plan(
        [
            (("roof",), "region", "R1", -1.0, 0.01),
            (("material",), "region", "R1", -2.0, 0.01),
            (("green",), "region", "R1", -3.0, 0.01),
        ]
    )
    out = json.loads(
        hotspot_kind_summary(
            plan,
            {
                ("region", "R1", ("roof",)): -1.0,
                ("region", "R1", ("material",)): -2.0,
                ("region", "R1", ("green",)): -3.0,
            },
            {("region", "R1"): 1.0},
        )
    )
    assert [item["kind"] for item in out["groups"][0]["items"]] == [
        "green", "material", "roof",
    ]


def test_matches_exact_fraction_reference():
    specs = [
        (("green",), "region", "R1", -1.0, 0.01),
        (("green", "roof"), "region", "R2", -2.0, 0.01),
        (("material",), "region", "R3", -6.0, 0.01),
        (("green", "material"), "window", "W1", -4.0, 0.01),
        (("roof",), "window", "W2", -1.0, 0.01),
    ]
    plan = _plan(specs)
    # kind -> list of (by, key, weight, per-kind v) expectations
    # R1 green: -1
    # R2 green: -1, roof: -1
    # R3 material: -6
    # W1 green: -2, material: -2 ; W2 roof: -1
    effects = {
        ("region", "R1", ("green",)): -1.0,
        ("region", "R2", ("green", "roof")): -2.0,
        ("region", "R3", ("material",)): -6.0,
        ("window", "W1", ("green", "material")): -4.0,
        ("window", "W2", ("roof",)): -1.0,
    }
    weights = {
        ("region", "R1"): 1.0,
        ("region", "R2"): 2.0,
        ("region", "R3"): 1.0,
        ("window", "W1"): 1.0,
        ("window", "W2"): 3.0,
    }
    out = json.loads(hotspot_kind_summary(plan, effects, weights))

    raw = {
        ("region", "green"): [(1, Fraction(-1)), (2, Fraction(-1))],
        ("region", "roof"): [(2, Fraction(-1))],
        ("region", "material"): [(1, Fraction(-6))],
        ("window", "green"): [(1, Fraction(-2))],
        ("window", "material"): [(1, Fraction(-2))],
        ("window", "roof"): [(3, Fraction(-1))],
    }
    expected = {}
    for key, panels in raw.items():
        n = len(panels)
        total = sum((w for w, _ in panels), Fraction(0))
        contributions = [w * v / total for w, v in panels]
        change = sum(contributions, Fraction(0))
        tail = 0
        for signs in product((-1, 1), repeat=n):
            statistic = sum(
                s * c for s, c in zip(signs, contributions)
            )
            if abs(statistic) >= abs(change):
                tail += 1
        expected[key] = (n, change, Fraction(tail, 1 << n))

    order = sorted(
        expected,
        key=lambda key: (
            expected[key][2],
            0 if key[0] == "region" else 1,
            key[1],
        ),
    )
    tested = len(order)
    q_values, running = {}, Fraction(1)
    for rank in range(tested, 0, -1):
        running = min(
            running, tested * expected[order[rank - 1]][2] / rank
        )
        q_values[order[rank - 1]] = min(Fraction(1), running)

    alpha = Fraction("0.05")
    for group in out["groups"]:
        for item in group["items"]:
            key = (group["by"], item["kind"])
            n, change, p = expected[key]
            assert item["n"] == n
            assert item["change"] == pytest.approx(
                float(change), abs=1e-6
            )
            assert item["p"] == pytest.approx(float(p), abs=1e-6)
            assert item["q"] == pytest.approx(
                float(q_values[key]), abs=1e-6
            )
            assert item["reject"] is (q_values[key] <= alpha)


def test_coherent_kind_panels_reject():
    specs = [
        (("green",), "region", f"R{i}", -1.0, 0.01)
        for i in range(8)
    ]
    panels = [("region", f"R{i}") for i in range(8)]
    plan = _plan(specs)
    out = json.loads(
        hotspot_kind_summary(
            plan,
            {(by, key, ("green",)): -1.0 for by, key in panels},
            {panel: 1.0 for panel in panels},
        )
    )
    item = out["groups"][0]["items"][0]
    # 2 of 256 sign vectors reach the all-coherent threshold.
    assert item["p"] == 0.007812
    assert item["reject"] is True


def test_negative_zero_normalized():
    # Both panels select the same two-kind pick; the summary effects
    # cancel with equal weights for both kinds even though the plan-time
    # priority effects were negative (and thus eligible).
    plan = _plan(
        [
            (("green", "roof"), "region", "R1", -1.0, 0.01),
            (("green", "roof"), "region", "R2", -1.0, 0.01),
        ]
    )
    out = hotspot_kind_summary(
        plan,
        {
            ("region", "R1", ("green", "roof")): -1.0,
            ("region", "R2", ("green", "roof")): 1.0,
        },
        {("region", "R1"): 1.0, ("region", "R2"): 1.0},
    )
    assert "-0.000000" not in out
    assert out.count('"change":0.000000') == 2


def test_plan_without_selected_picks_yields_empty_groups():
    # Every candidate exceeds the panel limit, so nothing is selected.
    plan = _plan(
        [
            (("green",), "region", "R1", -1.0, 0.01),
            (("roof",), "window", "W1", -2.0, 0.01),
        ],
        limits={("region", "R1"): 0.0, ("window", "W1"): 0.0},
        costs={("green",): 1.0, ("roof",): 1.0},
    )
    out = hotspot_kind_summary(
        plan,
        {},
        {("region", "R1"): 1.0, ("window", "W1"): 1.0},
    )
    assert out == '{"alpha":0.050000,"groups":[]}\n'
