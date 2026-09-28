"""Tests for urban_micro.kind_impact_summary."""

import json
from decimal import Decimal

import pytest

from urban_micro import kind_impact, kind_impact_summary
from urban_micro import uhi as _uhi


_KINDS = ("green", "roof", "material")


def _f6(value) -> str:
    return f"{Decimal(str(value)):.6f}"


def _panel(base, green_effect):
    """A (base, effects) data entry whose only active kind is green."""
    return (
        tuple(base),
        {
            "green": tuple(green_effect),
            "roof": (0.0, 0.0, 0.0),
            "material": (0.0, 0.0, 0.0),
        },
    )


def _portfolio_group(by, *, budget=100.0, cost=1.0, score=1.0, pick=("green",)):
    # remaining = budget - cost; skip is empty, which the parser accepts as
    # long as the bytes re-serialize canonically.
    remaining = budget - cost
    pick_token = "[" + ",".join(json.dumps(k) for k in pick) + "]"
    return (
        '{"by":' + json.dumps(by)
        + f',"budget":{_f6(budget)},"cost":{_f6(cost)},'
        + f'"remaining":{_f6(remaining)},"score":{_f6(score)},'
        + f'"pick":{pick_token},"skip":[]}}'
    )


def _portfolio(groups, *, alpha=0.05):
    return (
        f'{{"alpha":{_f6(alpha)},"groups":[' + ",".join(groups) + "]}\n"
    )


def _impact(panels, *, bys=("region",), alpha=0.05, **group_kwargs):
    """Run panels (an ordered mapping of key -> panel) through kind_impact."""
    groups = [_portfolio_group(by, **group_kwargs) for by in bys]
    portfolio = _portfolio(groups, alpha=alpha)
    data = {
        (by, key): panel
        for by in bys
        for key, panel in panels.items()
    }
    return kind_impact(portfolio, data)


def _weights(panels, bys=("region",), value=1.0):
    return {(by, key): value for by in bys for key in panels}


# --- canonical output shape -------------------------------------------------

def test_basic_output_shape_and_key_order():
    panels = {"a": _panel((1.0, 10.0, 100.0), (-1.0, -2.0, -3.0))}
    report = _impact(panels)
    out = kind_impact_summary(report, _weights(panels))
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "items"]
    assert [list(item) for item in payload["items"]] == [
        ["by", "metric", "n", "base", "post", "change", "p", "q", "reject"],
        ["by", "metric", "n", "base", "post", "change", "p", "q", "reject"],
        ["by", "metric", "n", "base", "post", "change", "p", "q", "reject"],
    ]


def test_items_sort_by_then_metric():
    panels = {
        "a": _panel((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        "b": _panel((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
    }
    report = _impact(panels, bys=("region", "window"))
    data_weights = {
        **{("region", k): 1.0 for k in panels},
        **{("window", k): 1.0 for k in panels},
    }
    payload = json.loads(kind_impact_summary(report, data_weights))
    assert [(i["by"], i["metric"]) for i in payload["items"]] == [
        ("region", "uhi"),
        ("region", "energy"),
        ("region", "vent"),
        ("window", "uhi"),
        ("window", "energy"),
        ("window", "vent"),
    ]


def test_n_is_integer_and_reject_boolean_numbers_six_decimals():
    panels = {"a": _panel((0.0, 0.0, 0.0), (4.0, 0.0, 0.0))}
    report = _impact(panels)
    out = kind_impact_summary(report, _weights(panels))
    item = json.loads(out)["items"][0]
    assert isinstance(item["n"], int) and item["n"] == 1
    assert isinstance(item["reject"], bool)
    for token in (
        '"base":0.000000',
        '"post":4.000000',
        '"change":4.000000',
        '"p":1.000000',
        '"q":1.000000',
        '"n":1',
    ):
        assert token in out


def test_empty_report_yields_empty_items():
    empty = kind_impact(_portfolio([], alpha=0.05), {})
    assert empty == '{"alpha":0.050000,"groups":[]}\n'
    assert kind_impact_summary(empty, {}) == (
        '{"alpha":0.050000,"items":[]}\n'
    )


# --- weighted aggregates ----------------------------------------------------

def test_weighted_base_post_change():
    panels = {
        "a": _panel((10.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
        "b": _panel((20.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
    }
    report = _impact(panels)
    weights = {("region", "a"): 1.0, ("region", "b"): 3.0}
    item = json.loads(kind_impact_summary(report, weights))["items"][0]
    # W = 4; base = (10 + 60)/4 = 17.5; change = (2 + 12)/4 = 3.5;
    # post = (12 + 72)/4 = 21.
    assert item["base"] == 17.5
    assert item["change"] == 3.5
    assert item["post"] == 21.0
    assert item["n"] == 2


def test_weights_are_per_by_key_panel_tuples():
    panels_r = {"a": _panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
    panels_w = {"x": _panel((0.0, 0.0, 0.0), (9.0, 0.0, 0.0))}
    groups = [
        _portfolio_group("region", cost=1.0),
        _portfolio_group("window", cost=1.0),
    ]
    portfolio = _portfolio(groups)
    data = {
        ("region", "a"): panels_r["a"],
        ("window", "x"): panels_w["x"],
    }
    report = kind_impact(portfolio, data)
    weights = {("region", "a"): 2.0, ("window", "x"): 1.0}
    items = json.loads(kind_impact_summary(report, weights))["items"]
    region = next(i for i in items if i["by"] == "region")
    window = next(i for i in items if i["by"] == "window")
    assert region["n"] == 1 and region["change"] == 1.0
    assert window["n"] == 1 and window["change"] == 9.0


# --- sign-flip p-values -----------------------------------------------------

def test_n1_p_is_one():
    panels = {"a": _panel((0.0, 0.0, 0.0), (4.0, 5.0, -6.0))}
    report = _impact(panels)
    items = json.loads(kind_impact_summary(report, _weights(panels)))["items"]
    assert [i["p"] for i in items] == [1.0, 1.0, 1.0]


def test_n2_equal_changes_p_is_one_half():
    panels = {
        "a": _panel((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        "b": _panel((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
    }
    report = _impact(panels)
    item = json.loads(kind_impact_summary(report, _weights(panels)))["items"][0]
    # ++ and -- reach |observed|=8; +- and -+ give 0 -> 2/4.
    assert item["p"] == 0.5


def test_n3_equal_changes_p_is_one_quarter():
    panels = {
        key: _panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        for key in ("a", "b", "c")
    }
    report = _impact(panels)
    item = json.loads(kind_impact_summary(report, _weights(panels)))["items"][0]
    # only all-plus and all-minus reach |3| -> 2/8.
    assert item["p"] == 0.25


def test_zero_observed_change_p_is_one():
    panels = {
        "a": _panel((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        "b": _panel((0.0, 0.0, 0.0), (-4.0, 0.0, 0.0)),
    }
    report = _impact(panels)
    item = json.loads(kind_impact_summary(report, _weights(panels)))["items"][0]
    assert item["change"] == 0.0
    assert item["p"] == 1.0


def test_weighted_zero_contribution_panel_does_not_shrink_p():
    panels = {
        "a": _panel((0.0, 0.0, 0.0), (4.0, 0.0, 0.0)),
        "b": _panel((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    }
    report = _impact(panels)
    weights = {("region", "a"): 3.0, ("region", "b"): 1.0}
    item = json.loads(kind_impact_summary(report, weights))["items"][0]
    # panel b contributes nothing regardless of its sign -> all 4 vectors
    # reach |observed| = 3.
    assert item["change"] == 3.0
    assert item["p"] == 1.0


# --- Benjamini-Hochberg -----------------------------------------------------

def test_bh_across_all_six_items_rejects_at_alpha():
    panels = {
        f"k{i}": _panel((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        for i in range(6)
    }
    report = _impact(panels, alpha=0.05)
    items = json.loads(
        kind_impact_summary(report, _weights(panels))
    )["items"]
    # N = 3 metrics, each with p = 2/64 = 0.03125; the three-way tie keeps
    # q = p for every rank.
    assert items[0]["p"] == 0.03125
    assert all(i["q"] == 0.03125 for i in items)
    assert all(i["reject"] is True for i in items)


def test_bh_reject_threshold_uses_report_alpha():
    panels = {
        f"k{i}": _panel((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        for i in range(6)
    }
    report = _impact(panels, alpha=0.01)
    items = json.loads(
        kind_impact_summary(report, _weights(panels))
    )["items"]
    assert all(i["q"] == 0.03125 for i in items)
    assert all(i["reject"] is False for i in items)


def test_bh_stepup_q_is_running_minimum():
    # region: 6 strong panels -> p = 0.03125 on uhi only; energy/vent are
    # zero (p = 1). N = 3; rank order is uhi (0.03125), energy (1), vent (1).
    panels = {
        f"k{i}": _panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        for i in range(6)
    }
    report = _impact(panels, alpha=0.05)
    items = {
        i["metric"]: i
        for i in json.loads(kind_impact_summary(report, _weights(panels)))[
            "items"
        ]
    }
    # rank 1: q = min(3*0.03125, 3*1/2, 1) = 0.09375.
    assert items["uhi"]["q"] == 0.09375
    assert items["uhi"]["reject"] is False
    assert items["energy"]["q"] == 1.0
    assert items["vent"]["q"] == 1.0


def test_bh_unified_across_both_bys():
    # region gets 6 strong panels; window gets 1. N = 6 items.
    region_panels = {
        f"r{i}": _panel((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        for i in range(6)
    }
    window_panels = {"w": _panel((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))}
    groups = [
        _portfolio_group("region", cost=1.0),
        _portfolio_group("window", cost=1.0),
    ]
    portfolio = _portfolio(groups, alpha=0.05)
    data = {
        **{("region", k): p for k, p in region_panels.items()},
        **{("window", k): p for k, p in window_panels.items()},
    }
    report = kind_impact(portfolio, data)
    weights = {
        **{("region", k): 1.0 for k in region_panels},
        **{("window", k): 1.0 for k in window_panels},
    }
    items = json.loads(kind_impact_summary(report, weights))["items"]
    region = [i for i in items if i["by"] == "region"]
    window = [i for i in items if i["by"] == "window"]
    # The three region p's (0.03125) take ranks 1..3 and the window p's (1)
    # ranks 4..6; rank 3 already yields N*p/3 = 6*0.03125/3 = 0.0625.
    assert {i["q"] for i in region} == {0.0625}
    assert all(i["reject"] is False for i in region)
    assert all(i["q"] == 1.0 for i in window)


# --- input validation -------------------------------------------------------

def test_type_errors():
    panels = {"a": _panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
    report = _impact(panels)
    with pytest.raises(TypeError):
        kind_impact_summary(1, _weights(panels))
    with pytest.raises(TypeError):
        kind_impact_summary(None, _weights(panels))
    with pytest.raises(TypeError):
        kind_impact_summary(report, [])


def test_weights_key_set_must_match_panels():
    panels = {"a": _panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
    report = _impact(panels)
    with pytest.raises(ValueError):
        kind_impact_summary(report, {})
    with pytest.raises(ValueError):
        kind_impact_summary(report, {("region", "b"): 1.0})
    with pytest.raises(ValueError):
        kind_impact_summary(
            report,
            {("region", "a"): 1.0, ("region", "b"): 1.0},
        )
    with pytest.raises(ValueError):
        kind_impact_summary(report, {("window", "a"): 1.0})


def test_empty_report_requires_empty_weights():
    empty = kind_impact(_portfolio([], alpha=0.05), {})
    with pytest.raises(ValueError):
        kind_impact_summary(empty, {("region", "a"): 1.0})


@pytest.mark.parametrize(
    "bad",
    [0, -1, -0.5, float("nan"), float("inf"), float("-inf"), True, "1"],
)
def test_weights_values_rejected(bad):
    panels = {"a": _panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
    report = _impact(panels)
    with pytest.raises(ValueError):
        kind_impact_summary(report, {("region", "a"): bad})


def test_more_than_16_panels_raises():
    panels = {
        f"k{i:02d}": _panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        for i in range(17)
    }
    report = _impact(panels)
    with pytest.raises(ValueError):
        kind_impact_summary(report, _weights(panels))


def test_sixteen_panels_accepted():
    panels = {
        f"k{i:02d}": _panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        for i in range(16)
    }
    report = _impact(panels)
    items = json.loads(
        kind_impact_summary(report, _weights(panels))
    )["items"]
    assert items[0]["n"] == 16
    # all-plus / all-minus only -> 2 / 65536, rendered to six decimals.
    assert items[0]["p"] == round(2 / 65536, 6)


# --- report must be canonical kind_impact output ----------------------------

@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s[:-1],                          # no trailing newline
        lambda s: s + "\n",                        # two trailing newlines
        lambda s: s.replace('{"by"', '{ "by"', 1),
        lambda s: s.replace(",", ", ", 1),
        lambda s: s.replace("0.000000", "0", 1),   # non-six-decimal token
        lambda s: s.replace("1.000000", "1.0", 1),
        lambda s: s.replace("uhi", "uha", 1),
    ],
)
def test_noncanonical_report_raises_value_error(mutate):
    panels = {"a": _panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))}
    report = _impact(panels)
    with pytest.raises(ValueError):
        kind_impact_summary(mutate(report), _weights(panels))


def test_panels_out_of_key_order_raises():
    # Two panels canonically ordered a then b; exchanging the two item
    # objects breaks the ascending-key canonical serialization.
    panels = {
        "a": _panel((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        "b": _panel((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
    }
    report = _impact(panels)
    marker_a = '{"key":"a"'
    marker_b = '{"key":"b"'
    ia = report.index(marker_a)
    ib = report.index(marker_b)
    end_b = report.index("]}", ib)
    item_a = report[ia : ib - 1]
    item_b = report[ib:end_b]
    reordered = report[:ia] + item_b + "," + item_a + report[end_b:]
    assert reordered != report
    with pytest.raises(ValueError):
        kind_impact_summary(reordered, _weights(panels))


@pytest.mark.parametrize("raw", ["", "not json", '{"alpha":0.05}'])
def test_garbage_report_raises_value_error(raw):
    with pytest.raises(ValueError):
        kind_impact_summary(raw, {})


def test_negative_zero_token_rejected():
    panels = {"a": _panel((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))}
    report = _impact(panels)
    tampered = report.replace('"uhi":[0.000000', '"uhi":[-0.000000', 1)
    assert tampered != report
    with pytest.raises(ValueError):
        kind_impact_summary(tampered, _weights(panels))


def test_report_type_error_takes_precedence():
    with pytest.raises(TypeError):
        kind_impact_summary(123, [])


def test_exported_from_package_root():
    assert _uhi.kind_impact_summary is kind_impact_summary
    assert "kind_impact_summary" in __import__("urban_micro").__all__
