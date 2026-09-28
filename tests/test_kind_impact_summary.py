"""Tests for urban_micro.kind_impact_summary and the kind_impact
portfolio-report consistency checks."""

import json
from fractions import Fraction

import pytest

from urban_micro import kind_impact, kind_impact_summary
from urban_micro import uhi as _uhi

_KINDS = ("green", "roof", "material")
_METRICS = ("uhi", "energy", "vent")


def _portfolio_group(by, pick, cost, remaining, budget=3.0, score=2.0):
    pick = tuple(sorted(pick))
    skip = [kind for kind in ("green", "material", "roof") if kind not in pick]
    skip_token = "[" + ",".join(
        '{"kind":' + json.dumps(kind) + ',"reason":"a"}' for kind in skip
    ) + "]"
    return (
        '{"by":' + json.dumps(by)
        + f',"budget":{budget:.6f},"cost":{cost:.6f},"remaining":{remaining:.6f},'
        + f'"score":{score:.6f},"pick":'
        + json.dumps(list(pick), separators=(",", ":"))
        + ',"skip":' + skip_token + "}"
    )


def _portfolio_report(*groups, alpha=0.05):
    return (
        f'{{"alpha":{alpha:.6f},"groups":[' + ",".join(groups) + "]}\n"
    )


def _panel(base, effects):
    return (
        tuple(base),
        {kind: tuple(effects.get(kind, (0, 0, 0))) for kind in _KINDS},
    )


def _summary(payload):
    return json.loads(payload)


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 1.0, 2.0)
    )
    data = {
        ("region", "r1"): _panel((10.0, 100.0, 5.0), {"green": (-2.0, -10.0, 1.0)}),
    }
    impact = kind_impact(report, data)
    out = kind_impact_summary(impact, {("region", "r1"): 1.0})
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = _summary(out)
    assert list(payload) == ["alpha", "items"]
    for item in payload["items"]:
        assert list(item) == [
            "by", "metric", "n", "base", "post", "change", "p", "q",
            "reject",
        ]


def test_items_order_region_before_window_and_metric_order():
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 1.0, 2.0),
        _portfolio_group("window", (), 0.0, 4.0, budget=4.0, score=0.0),
    )
    data = {
        ("region", "r1"): _panel((1.0, 2.0, 3.0), {"green": (1.0, -1.0, 2.0)}),
        ("window", "w1"): _panel((5.0, 6.0, 7.0), {}),
    }
    impact = kind_impact(report, data)
    out = _summary(
        kind_impact_summary(
            impact, {("region", "r1"): 1.0, ("window", "w1"): 2.0}
        )
    )
    bys = [item["by"] for item in out["items"]]
    assert bys == ["region"] * 3 + ["window"] * 3
    assert [item["metric"] for item in out["items"][:3]] == list(_METRICS)


def test_empty_report_only_with_empty_weights():
    empty = '{"alpha":0.050000,"groups":[]}\n'
    assert kind_impact_summary(empty, {}) == '{"alpha":0.050000,"items":[]}\n'
    with pytest.raises(ValueError):
        kind_impact_summary(empty, {("region", "r1"): 1.0})


def test_n_integer_reject_boolean_numbers_six_decimals():
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 1.0, 2.0)
    )
    data = {
        ("region", "r1"): _panel((1.0, 2.0, 3.0), {"green": (0.0, 0.0, 0.0)}),
    }
    out = kind_impact_summary(
        kind_impact(report, data), {("region", "r1"): 1.0}
    )
    assert '"n":1' in out
    assert '"reject":false' in out
    for token in (
        '"base":1.000000', '"post":1.000000', '"change":0.000000',
        '"p":1.000000', '"q":1.000000',
    ):
        assert token in out


def test_negative_zero_normalized():
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 0.0, 3.0)
    )
    data = {
        ("region", "r1"): _panel((0.0, 0.0, 0.0), {"green": (0.0, 0.0, 0.0)}),
    }
    out = kind_impact_summary(
        kind_impact(report, data), {("region", "r1"): 1.0}
    )
    assert "-0.000000" not in out


# --- weighted means ---------------------------------------------------------

def test_weighted_means():
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 1.0, 2.0)
    )
    data = {
        ("region", "r1"): _panel(
            (10.0, 100.0, 5.0), {"green": (-2.0, -10.0, 1.0)}
        ),
        ("region", "r2"): _panel(
            (12.0, 120.0, 7.0), {"green": (-1.0, -5.0, 2.0)}
        ),
    }
    impact = kind_impact(report, data)
    out = _summary(
        kind_impact_summary(
            impact, {("region", "r1"): 1.0, ("region", "r2"): 3.0}
        )
    )
    values = {item["metric"]: item for item in out["items"]}
    assert values["uhi"]["base"] == 11.5
    assert values["uhi"]["post"] == 10.25
    assert values["uhi"]["change"] == -1.25
    assert values["energy"]["base"] == 115.0
    assert values["energy"]["post"] == 108.75
    assert values["energy"]["change"] == -6.25
    assert values["vent"]["base"] == 6.5
    assert values["vent"]["post"] == 8.25
    assert values["vent"]["change"] == 1.75
    assert all(item["n"] == 2 for item in out["items"])


# --- sign-flip p-values -----------------------------------------------------

def test_single_panel_p_is_one_for_nonzero_change():
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 1.0, 2.0)
    )
    data = {
        ("region", "r1"): _panel((10.0, 0.0, 0.0), {"green": (-2.0, 1.0, 1.0)}),
    }
    out = _summary(
        kind_impact_summary(kind_impact(report, data), {("region", "r1"): 1.0})
    )
    assert all(item["p"] == 1.0 for item in out["items"])


def test_sign_flip_p_identical_changes_n_six():
    n = 6
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 0.0, 3.0)
    )
    data = {
        ("region", f"k{i}"): _panel(
            (10.0, 0.0, 0.0), {"green": (-2.0, 0.0, 0.0)}
        )
        for i in range(n)
    }
    impact = kind_impact(report, data)
    out = _summary(
        kind_impact_summary(
            impact, {("region", f"k{i}"): 1.0 for i in range(n)}
        )
    )
    uhi = next(item for item in out["items"] if item["metric"] == "uhi")
    # Only the two all-same-sign sign vectors reach |observed| with equal
    # weighted contributions: p = 2 / 2**n.
    assert uhi["p"] == 2 / 2**n
    assert uhi["reject"] is False


def test_sign_flip_p_and_reject_against_exact_rational():
    n = 7
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 0.0, 3.0)
    )
    data = {
        ("region", f"k{i}"): _panel(
            (10.0, 0.0, 0.0), {"green": (-2.0, 0.0, 0.0)}
        )
        for i in range(n)
    }
    impact = kind_impact(report, data)
    out = _summary(
        kind_impact_summary(
            impact, {("region", f"k{i}"): 1.0 for i in range(n)}
        )
    )
    p = Fraction(2, 1 << n)
    uhi = next(item for item in out["items"] if item["metric"] == "uhi")
    assert abs(uhi["p"] - float(p)) < 1e-9
    # BH over the three metric items (N=3); uhi is rank 1.
    q = Fraction(3) * p
    assert abs(uhi["q"] - float(q)) < 1e-9
    assert uhi["reject"] is True
    for metric in ("energy", "vent"):
        item = next(i for i in out["items"] if i["metric"] == metric)
        assert item["p"] == 1.0 and item["q"] == 1.0
        assert item["reject"] is False


def test_p_with_unequal_weights_and_changes_brute_force():
    panels = {"a": 1.0, "b": 2.0, "c": 4.0}
    changes = {"a": -3.0, "b": 1.0, "c": -2.0}
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 0.0, 3.0)
    )
    data = {
        ("region", key): _panel(
            (0.0, 0.0, 0.0), {"green": (changes[key], 0.0, 0.0)}
        )
        for key in panels
    }
    weights = {("region", key): weight for key, weight in panels.items()}
    impact = kind_impact(report, data)
    out = _summary(kind_impact_summary(impact, weights))
    uhi = next(item for item in out["items"] if item["metric"] == "uhi")
    contributions = [
        Fraction(str(panels[key])) * Fraction(str(changes[key]))
        for key in panels
    ]
    observed = abs(sum(contributions))
    tail = 0
    for mask in range(1 << 3):
        statistic = sum(
            (Fraction(1) if (mask >> index) & 1 else Fraction(-1))
            * contribution
            for index, contribution in enumerate(contributions)
        )
        if abs(statistic) >= observed:
            tail += 1
    assert abs(uhi["p"] - tail / 8) < 1e-9
    assert uhi["n"] == 3


# --- input validation -------------------------------------------------------

def test_type_errors():
    report = '{"alpha":0.050000,"groups":[]}\n'
    with pytest.raises(TypeError):
        kind_impact_summary(1, {})
    with pytest.raises(TypeError):
        kind_impact_summary(None, {})
    with pytest.raises(TypeError):
        kind_impact_summary(report, [])


def test_weights_key_set_must_match_panels():
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 1.0, 2.0)
    )
    data = {
        ("region", "r1"): _panel((1.0, 2.0, 3.0), {"green": (0.0, 0.0, 0.0)}),
    }
    impact = kind_impact(report, data)
    with pytest.raises(ValueError):
        kind_impact_summary(impact, {})
    with pytest.raises(ValueError):
        kind_impact_summary(
            impact,
            {("region", "r1"): 1.0, ("window", "w1"): 1.0},
        )


@pytest.mark.parametrize(
    "bad",
    [0, -1.0, True, False, float("inf"), -float("inf"), float("nan"), "1"],
)
def test_weights_values_rejected(bad):
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 1.0, 2.0)
    )
    data = {
        ("region", "r1"): _panel((1.0, 2.0, 3.0), {"green": (0.0, 0.0, 0.0)}),
    }
    impact = kind_impact(report, data)
    with pytest.raises(ValueError):
        kind_impact_summary(impact, {("region", "r1"): bad})


def test_too_many_panels_rejected():
    keys = [f"k{i}" for i in range(17)]
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 0.0, 3.0)
    )
    data = {
        ("region", key): _panel(
            (float(index), 0.0, 0.0), {"green": (1.0, 0.0, 0.0)}
        )
        for index, key in enumerate(keys)
    }
    impact = kind_impact(report, data)
    with pytest.raises(ValueError, match="16"):
        kind_impact_summary(impact, {("region", key): 1.0 for key in keys})


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        '{"alpha":0.050000,"groups":[]}',
        '{"alpha":0.050000,"groups":[]}\n\n',
        '{"alpha":0.05,"items":[]}\n',
    ],
)
def test_noncanonical_report_raises_value_error(raw):
    with pytest.raises(ValueError):
        kind_impact_summary(raw, {})


def test_portfolio_report_is_not_a_kind_impact_report():
    report = _portfolio_report(
        _portfolio_group("region", ("green",), 1.0, 2.0)
    )
    with pytest.raises(ValueError):
        kind_impact_summary(report, {})


# --- kind_impact portfolio parser consistency checks ------------------------

def _single_panel_data():
    return {
        ("region", "r1"): _panel(
            (1.0, 2.0, 3.0), {"green": (1.0, -1.0, 2.0)}
        ),
    }


def test_budget_identity_within_two_last_decimals():
    data = _single_panel_data()
    good = _portfolio_report(
        _portfolio_group("region", ("green",), 1.0, 2.000002)
    )
    impact = kind_impact(good, data)
    kind_impact_summary(impact, {("region", "r1"): 1.0})

    with pytest.raises(ValueError):
        kind_impact(
            _portfolio_report(
                _portfolio_group("region", ("green",), 1.0, 2.000003)
            ),
            data,
        )
    with pytest.raises(ValueError):
        kind_impact(
            _portfolio_report(
                _portfolio_group("region", ("green",), 1.0, 1.999997)
            ),
            data,
        )


@pytest.mark.parametrize(
    "cost,remaining,score",
    [
        (0.000001, 2.999999, 0.0),
        (0.0, 3.0, 0.000001),
        (0.0, 2.999999, 0.0),
        (0.0, 3.000001, 0.0),
    ],
)
def test_empty_pick_requires_zero_cost_score_and_full_remaining(
    cost, remaining, score
):
    with pytest.raises(ValueError):
        kind_impact(
            _portfolio_report(
                _portfolio_group(
                    "region", (), cost, remaining, score=score
                )
            ),
            _single_panel_data(),
        )


def test_empty_pick_consistent_report_accepted():
    report = _portfolio_report(
        _portfolio_group("region", (), 0.0, 3.0, score=0.0)
    )
    impact = kind_impact(report, _single_panel_data())
    out = _summary(kind_impact_summary(impact, {("region", "r1"): 1.0}))
    assert all(item["change"] == 0.0 for item in out["items"])


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    assert _uhi.kind_impact_summary is kind_impact_summary
    assert "kind_impact_summary" in __import__("urban_micro").__all__
