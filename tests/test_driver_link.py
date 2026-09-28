"""Tests for urban_micro.driver_link: join a canonical
driver_attr_stability_summary (s) with a canonical kind_impact_summary
(i) under a factor -> metric/direction links table."""

import json

import pytest

from urban_micro import (
    driver_attr_stability,
    driver_attr_stability_summary,
    driver_link,
    kind_impact,
    kind_impact_summary,
)
from urban_micro import uhi as _uhi

_FACTORS = ("station", "lst", "morph", "cover")
_METRICS = ("uhi", "energy", "vent")
_ITEM_KEYS = [
    "factor", "metric", "effect", "low", "high", "q", "score",
    "eligible", "rank",
]
_LINKS = {"station": "uhi-", "lst": "energy-", "morph": "vent+",
          "cover": "uhi+"}


# --- report builders --------------------------------------------------------

def _s_item(factor, direction, frequency, consistency, significant, low,
            high, stable, conflict, cross_zero):
    """One canonical driver_attr_stability_summary item token."""
    return (
        '{"factor":' + json.dumps(factor)
        + ',"direction":' + json.dumps(direction)
        + f',"frequency":{frequency:.6f},"consistency":{consistency:.6f},'
        + f'"significant":{significant:.6f},"low":{low:.6f},'
        + f'"high":{high:.6f},"stable":{str(stable).lower()},'
        + f'"conflict":{str(conflict).lower()},'
        + f'"cross_zero":{str(cross_zero).lower()}}}'
    )


def _stable_row(factor, low=-0.5, high=-0.2, direction="down"):
    return _s_item(factor, direction, 1.0, 1.0, 1.0, low, high,
                   True, False, False)


def _empty_row(factor):
    return _s_item(factor, "flat", 0.0, 0.0, 0.0, 0.0, 0.0,
                   False, False, False)


def _s_group(by, *items):
    return '{"by":' + json.dumps(by) + ',"items":[' + ",".join(items) + "]}"


def _s_report(*groups, alpha=0.05):
    return ('{"alpha":' + f"{alpha:.6f}" + ',"groups":['
            + ",".join(groups) + "]}\n")


def _s_region(*rows):
    """Region stability report; rows maps factor -> token, unstable
    empty rows fill the unspecified factors."""
    by_factor = {_FACTORS[i]: rows[i] if i < len(rows) else None
                 for i in range(4)}
    tokens = [by_factor[f] or _empty_row(f) for f in _FACTORS]
    return _s_report(_s_group("region", *tokens))


def _i_item(by, metric, change, q, reject, *, n=4, base=0.0, post=None,
            p=0.1):
    if post is None:
        post = base + change
    return (
        '{"by":' + json.dumps(by)
        + ',"metric":' + json.dumps(metric)
        + f',"n":{n},"base":{base:.6f},"post":{post:.6f},'
        + f'"change":{change:.6f},"p":{p:.6f},"q":{q:.6f},'
        + f'"reject":{str(reject).lower()}}}'
    )


def _i_report(*items, alpha=0.05):
    return ('{"alpha":' + f"{alpha:.6f}" + ',"items":['
            + ",".join(items) + "]}\n")


def _i_region(uhi=(-2.0, 0.01, True), energy=(-5.0, 0.2, False),
              vent=(3.0, 0.01, True), alpha=0.05):
    triples = (uhi, energy, vent)
    return _i_report(*[
        _i_item("region", metric, change, q, reject)
        for metric, (change, q, reject) in zip(_METRICS, triples)
    ], alpha=alpha)


def _rows(out):
    return json.loads(out)["groups"][0]["items"]


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    out = driver_link(_s_region(), _i_region(), _LINKS)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    assert list(payload["groups"][0]) == ["by", "items"]
    for item in _rows(out):
        assert list(item) == _ITEM_KEYS
        assert isinstance(item["rank"], int)
        assert isinstance(item["eligible"], bool)


def test_region_before_window_groups():
    s = _s_report(
        _s_group("region", *[_empty_row(f) for f in _FACTORS]),
        _s_group("window", *[_empty_row(f) for f in _FACTORS]),
    )
    i = _i_report(
        *[_i_item("region", m, 1.0, 0.5, False) for m in _METRICS],
        *[_i_item("window", m, 1.0, 0.5, False) for m in _METRICS],
    )
    out = json.loads(driver_link(s, i, _LINKS))
    assert [g["by"] for g in out["groups"]] == ["region", "window"]
    for group in out["groups"]:
        assert [item["rank"] for item in group["items"]] == [1, 2, 3, 4]


def test_empty_groups():
    s = '{"alpha":0.050000,"groups":[]}\n'
    i = '{"alpha":0.050000,"items":[]}\n'
    assert driver_link(s, i, _LINKS) == (
        '{"alpha":0.050000,"groups":[]}\n'
    )


def test_metric_and_effect_low_high_q_come_from_the_reports():
    s = _s_region(
        _stable_row("station", low=-0.5, high=-0.2),
        _stable_row("lst"),
        _stable_row("morph"),
        _empty_row("cover"),
    )
    out = _rows(driver_link(s, _i_region(), _LINKS))
    by_factor = {item["factor"]: item for item in out}
    station = by_factor["station"]
    assert station["metric"] == "uhi"
    assert station["effect"] == -2.0
    assert station["low"] == -0.5 and station["high"] == -0.2
    assert station["q"] == 0.01
    assert by_factor["lst"]["metric"] == "energy"
    assert by_factor["lst"]["effect"] == -5.0
    assert by_factor["lst"]["q"] == 0.2
    assert by_factor["morph"]["metric"] == "vent"
    assert by_factor["morph"]["effect"] == 3.0
    assert by_factor["cover"]["metric"] == "uhi"
    # Unstable factors echo zero stability low/high.
    assert by_factor["cover"]["low"] == 0.0
    assert by_factor["cover"]["high"] == 0.0


def test_eligible_needs_stable_reject_and_direction():
    s = _s_region(
        _stable_row("station"),
        _stable_row("lst"),
        _stable_row("morph"),
        _empty_row("cover"),
    )
    out = _rows(driver_link(s, _i_region(), _LINKS))
    by_factor = {item["factor"]: item for item in out}
    # station: stable, uhi rejects, change -2 matches "uhi-".
    assert by_factor["station"]["eligible"] is True
    # lst: stable and energy change -5 matches "energy-" but no reject.
    assert by_factor["lst"]["eligible"] is False
    # morph: stable, vent rejects, +3 matches "vent+".
    assert by_factor["morph"]["eligible"] is True
    # cover: uhi would otherwise qualify but the factor is not stable.
    assert by_factor["cover"]["eligible"] is False


def test_score_is_abs_change_times_one_minus_q():
    s = _s_region(
        _stable_row("station"),
        _stable_row("lst"),
        _stable_row("morph"),
        _empty_row("cover"),
    )
    out = _rows(driver_link(s, _i_region(), _LINKS))
    by_factor = {item["factor"]: item for item in out}
    assert by_factor["station"]["score"] == pytest.approx(2 * 0.99)
    assert by_factor["morph"]["score"] == pytest.approx(3 * 0.99)
    assert by_factor["lst"]["score"] == 0.0
    assert by_factor["cover"]["score"] == 0.0


def test_score_rounds_half_even_on_decimal_tokens():
    # 1.25 * (1 - 0.333333) = 1.25 * 0.666667 = 0.83333375 -> 0.833334.
    s = _s_region(_stable_row("station"))
    i = _i_report(
        _i_item("region", "uhi", -1.25, 0.333333, True),
        _i_item("region", "energy", 0.0, 1.0, False),
        _i_item("region", "vent", 0.0, 1.0, False),
    )
    out = driver_link(s, i, _LINKS)
    station = next(item for item in _rows(out) if item["factor"] == "station")
    assert station["score"] == 0.833334
    assert '"score":0.833334' in out


def test_ranks_eligible_first_score_desc_then_factor_order():
    s = _s_region(
        _stable_row("station"),
        _stable_row("lst"),
        _stable_row("morph"),
        _empty_row("cover"),
    )
    out = _rows(driver_link(s, _i_region(), _LINKS))
    # morph 2.97 > station 1.98; then non-eligible lst (factor index 1)
    # before cover (factor index 3).
    assert [(item["factor"], item["rank"]) for item in out] == [
        ("morph", 1), ("station", 2), ("lst", 3), ("cover", 4),
    ]


def test_rank_tie_resolves_by_factor_order():
    s = _s_region(
        _stable_row("station"),
        _empty_row("lst"),
        _stable_row("morph"),
        _empty_row("cover"),
    )
    i = _i_report(
        _i_item("region", "uhi", -2.0, 0.01, True),
        _i_item("region", "energy", 0.0, 1.0, False),
        _i_item("region", "vent", -2.0, 0.01, True),
    )
    links = dict(_LINKS, morph="vent-")
    out = _rows(driver_link(s, i, links))
    assert [(item["factor"], item["rank"], item["eligible"]) for item in out] == [
        ("station", 1, True),
        ("morph", 2, True),
        ("lst", 3, False),
        ("cover", 4, False),
    ]


@pytest.mark.parametrize(
    "change,suffix,eligible",
    [
        (1.0, "-", False),
        (-1.0, "-", True),
        (1.0, "+", True),
        (-1.0, "+", False),
        (0.0, "-", False),
        (0.0, "+", False),
    ],
)
def test_change_direction_matching(change, suffix, eligible):
    s = _s_region(_stable_row("station"))
    i = _i_report(
        _i_item("region", "uhi", change, 0.01, True),
        _i_item("region", "energy", 0.0, 1.0, False),
        _i_item("region", "vent", 0.0, 1.0, False),
    )
    links = dict(_LINKS, station="uhi" + suffix)
    out = _rows(driver_link(s, i, links))
    station = next(item for item in out if item["factor"] == "station")
    assert station["eligible"] is eligible
    assert station["score"] == (abs(change) * 0.99 if eligible else 0.0)


def test_six_decimals_and_no_negative_zero():
    s = _s_region(
        _stable_row("station"),
        _empty_row("lst"),
        _empty_row("morph"),
        _empty_row("cover"),
    )
    i = _i_report(
        _i_item("region", "uhi", 0.0, 1.0, False),
        _i_item("region", "energy", 0.0, 1.0, False),
        _i_item("region", "vent", 0.0, 1.0, False),
    )
    out = driver_link(s, i, _LINKS)
    assert "-0.000000" not in out
    assert '"rank":1' in out and '"eligible":false' in out
    assert '"score":0.000000' in out and '"effect":0.000000' in out


# --- validation -------------------------------------------------------------

def test_type_errors():
    s = _s_region()
    i = _i_region()
    for bad in (1, None, [], True, {}):
        with pytest.raises(TypeError):
            driver_link(bad, i, _LINKS)
        with pytest.raises(TypeError):
            driver_link(s, bad, _LINKS)
    for bad in ([], None, 42, "x"):
        with pytest.raises(TypeError):
            driver_link(s, i, bad)


def test_type_errors_take_precedence():
    with pytest.raises(TypeError):
        driver_link(1, "garbage\n", [])


@pytest.mark.parametrize(
    "links",
    [
        {},
        {"station": "uhi-", "lst": "energy-", "morph": "vent+"},
        dict(_LINKS, extra="uhi-"),
        {"Station": "uhi-", "lst": "energy-", "morph": "vent+",
         "cover": "uhi+"},
    ],
)
def test_links_key_set_must_be_exact(links):
    with pytest.raises(ValueError):
        driver_link(_s_region(), _i_region(), links)


@pytest.mark.parametrize(
    "token",
    ["uhi", "uhi--", "uhi+ ", " temp-", "temp-", "UHI-", "vent",
     "", 1, None, True, False, ("uhi-",)],
)
def test_links_values_rejected(token):
    links = dict(_LINKS, station=token)
    with pytest.raises(ValueError):
        driver_link(_s_region(), _i_region(), links)


def test_alpha_mismatch_rejected():
    with pytest.raises(ValueError):
        driver_link(_s_region(), _i_region(alpha=0.1), _LINKS)


@pytest.mark.parametrize(
    "s_bys,i_bys",
    [
        ((), ("region",)),
        (("region",), ()),
        (("region", "window"), ("region",)),
        (("window",), ("region",)),
    ],
)
def test_by_sets_must_match(s_bys, i_bys):
    s = _s_report(*[
        _s_group(by, *[_empty_row(f) for f in _FACTORS]) for by in s_bys
    ])
    i = _i_report(*[
        _i_item(by, metric, 0.0, 1.0, False)
        for by in i_bys for metric in _METRICS
    ])
    with pytest.raises(ValueError):
        driver_link(s, i, _LINKS)


def test_window_only_reports_accepted():
    s = _s_report(_s_group("window", *[_empty_row(f) for f in _FACTORS]))
    i = _i_report(*[
        _i_item("window", metric, 0.0, 1.0, False) for metric in _METRICS
    ])
    payload = json.loads(driver_link(s, i, _LINKS))
    assert [g["by"] for g in payload["groups"]] == ["window"]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json\n",
        '{"alpha":0.050000,"groups":[]}',
        '{"alpha":0.050000,"groups":[]}\n\n',
        '{"groups":[],"alpha":0.050000}\n',
        '{"alpha":0.05,"groups":[]}\n',
        '{"alpha":0.050000,"items":[]}\n',
    ],
)
def test_malformed_s_value_error(raw):
    with pytest.raises(ValueError):
        driver_link(raw, _i_region(), _LINKS)


def test_s_wrong_item_key_set_rejected():
    # Drop the trailing cross_zero flag from the station item.
    good = _s_region(_stable_row("station"))
    bad = good.replace(',"cross_zero":false', "")
    assert bad != good
    with pytest.raises(ValueError):
        driver_link(bad, _i_region(), _LINKS)


def test_s_factor_order_rejected():
    tokens = [
        _stable_row("lst"), _stable_row("station"),
        _empty_row("morph"), _empty_row("cover"),
    ]
    with pytest.raises(ValueError):
        driver_link(_s_report(_s_group("region", *tokens)), _i_region(),
                    _LINKS)


def test_s_negative_zero_token_rejected():
    token = _s_item("station", "down", 1.0, 1.0, 1.0, -0.0, -0.2,
                    True, False, False)
    bad = _s_report(_s_group("region", token, *[_empty_row(f)
                          for f in _FACTORS[1:]]))
    with pytest.raises(ValueError):
        driver_link(bad, _i_region(), _LINKS)


def test_i_kind_impact_shape_rejected():
    # A kind_impact report carries groups, not an items summary.
    impact = (
        '{"alpha":0.050000,"groups":[{"by":"region","items":['
        '{"key":"r1","pick":["green"],'
        '"uhi":[1.0,-1.0,-2.0],"energy":[0.0,0.0,0.0],'
        '"vent":[0.0,0.0,0.0]}]}]}\n'
    )
    with pytest.raises(ValueError):
        driver_link(_s_region(), impact, _LINKS)


def test_i_driver_attr_shape_rejected():
    da = (
        '{"alpha":0.050000,"items":[{"by":"region","factor":"station",'
        '"n":4,"effect":-1.000000,"p":0.100000,"q":0.200000,'
        '"reject":false,"direction":"down"}]}\n'
    )
    with pytest.raises(ValueError):
        driver_link(_s_region(), da, _LINKS)


def test_i_metrics_must_be_complete_and_ordered():
    missing = _i_report(_i_item("region", "uhi", -1.0, 0.01, True))
    with pytest.raises(ValueError):
        driver_link(_s_region(), missing, _LINKS)
    reordered = _i_report(
        _i_item("region", "energy", 0.0, 1.0, False),
        _i_item("region", "uhi", -1.0, 0.01, True),
        _i_item("region", "vent", 0.0, 1.0, False),
    )
    with pytest.raises(ValueError):
        driver_link(_s_region(), reordered, _LINKS)
    duplicate = _i_report(
        _i_item("region", "uhi", -1.0, 0.01, True),
        _i_item("region", "energy", 0.0, 1.0, False),
        _i_item("region", "energy", 0.0, 1.0, False),
    )
    with pytest.raises(ValueError):
        driver_link(_s_region(), duplicate, _LINKS)


def test_i_reordered_bys_rejected():
    s = _s_report(
        _s_group("region", *[_empty_row(f) for f in _FACTORS]),
        _s_group("window", *[_empty_row(f) for f in _FACTORS]),
    )
    i = _i_report(
        *[_i_item("window", m, 1.0, 0.5, False) for m in _METRICS],
        *[_i_item("region", m, 1.0, 0.5, False) for m in _METRICS],
    )
    with pytest.raises(ValueError):
        driver_link(s, i, _LINKS)


def test_i_negative_zero_token_rejected():
    i = _i_report(
        _i_item("region", "uhi", -0.0, 1.0, False),
        _i_item("region", "energy", 0.0, 1.0, False),
        _i_item("region", "vent", 0.0, 1.0, False),
    )
    with pytest.raises(ValueError):
        driver_link(_s_region(), i, _LINKS)


# --- genuine producer chain -------------------------------------------------

def _portfolio_group(by, pick, cost, remaining, budget=3.0, score=2.0):
    pick = tuple(sorted(pick))
    skip = [kind for kind in ("green", "material", "roof")
            if kind not in pick]
    skip_token = "[" + ",".join(
        '{"kind":' + json.dumps(kind) + ',"reason":"a"}' for kind in skip
    ) + "]"
    return (
        '{"by":' + json.dumps(by)
        + f',"budget":{budget:.6f},"cost":{cost:.6f},'
        + f'"remaining":{remaining:.6f},"score":{score:.6f},"pick":'
        + json.dumps(list(pick), separators=(",", ":"))
        + ',"skip":' + skip_token + "}"
    )


def _da_item(by, factor, effect, reject, direction, n=7,
             p="0.001000", q="0.008000", alpha=0.05):
    reject_token = "true" if reject else "false"
    return (
        f'{{"alpha":{alpha:.6f},"items":[{{"by":{json.dumps(by)},'
        f'"factor":{json.dumps(factor)},"n":{n},"effect":{effect:.6f},'
        f'"p":{p},"q":{q},"reject":{reject_token},'
        f'"direction":{json.dumps(direction)}}}]}}\n'
    )


def test_genuine_producer_chain():
    n = 7
    portfolio = '{"alpha":0.050000,"groups":[' + _portfolio_group(
        "region", ("green",), 1.0, 2.0
    ) + "]}\n"
    data = {
        ("region", f"k{index}"): (
            (10.0, 100.0, 5.0),
            {"green": (-2.0, 0.0, 0.0), "roof": (0.0, 0.0, 0.0),
             "material": (0.0, 0.0, 0.0)},
        )
        for index in range(n)
    }
    impact = kind_impact(portfolio, data)
    kis = kind_impact_summary(
        impact, {("region", f"k{index}"): 1.0 for index in range(n)}
    )
    kis_payload = {item["metric"]: item
                   for item in json.loads(kis)["items"]}
    # Seven identical negative moves make uhi reject with q = 3 * 2/128.
    assert kis_payload["uhi"]["reject"] is True
    assert kis_payload["uhi"]["q"] == pytest.approx(3 * 2 / 128)

    da = _da_item("region", "station", -1.0, True, "down")
    stability = driver_attr_stability({"r1": da, "r2": da})
    s = driver_attr_stability_summary(stability)

    out = json.loads(driver_link(s, kis, _LINKS))
    assert [g["by"] for g in out["groups"]] == ["region"]
    items = out["groups"][0]["items"]
    by_factor = {item["factor"]: item for item in items}
    station = by_factor["station"]
    assert station["metric"] == "uhi"
    assert station["eligible"] is True
    assert station["effect"] == -2.0
    assert station["low"] == -1.0 and station["high"] == -1.0
    assert station["q"] == pytest.approx(3 * 2 / 128)
    assert station["score"] == pytest.approx(2 * (1 - 3 * 2 / 128))
    assert station["rank"] == 1
    # Every other factor was never picked, so none is eligible; they
    # follow in factor order with ranks 2..4.
    assert [(item["factor"], item["rank"], item["eligible"])
            for item in items] == [
        ("station", 1, True),
        ("lst", 2, False),
        ("morph", 3, False),
        ("cover", 4, False),
    ]


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.driver_link is urban_micro.driver_link
    assert "driver_link" in urban_micro.__all__
