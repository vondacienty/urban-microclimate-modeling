"""Tests for urban_micro.driver_link: joining stable driver factors to
significant kind-impact metric items."""

import json

import pytest

from urban_micro import driver_link
from urban_micro import uhi as _uhi

_FACTORS = ("station", "lst", "morph", "cover")
_LINK_VALUES = ("uhi-", "uhi+", "energy-", "energy+", "vent-", "vent+")
_ITEM_KEYS = [
    "factor", "metric", "effect", "low", "high", "q", "score",
    "eligible", "rank",
]
_LINKS = {
    "station": "uhi-",
    "lst": "energy+",
    "morph": "vent+",
    "cover": "uhi+",
}


# --- report builders --------------------------------------------------------

def _kis_item(metric, change, q, reject, base=100.0, post=100.0, n=4,
              p=0.01):
    """One canonical kind_impact_summary item for the region by."""
    return (
        '{"by":"region","metric":' + json.dumps(metric)
        + f',"n":{n},"base":{base:.6f},"post":{post:.6f},'
        + f'"change":{change:.6f},"p":{p:.6f},"q":{q:.6f},'
        + '"reject":' + ("true" if reject else "false") + "}"
    )


def _impact_report(*item_tokens, alpha=0.05):
    return (
        f'{{"alpha":{alpha:.6f},"items":['
        + ",".join(item_tokens) + "]}\n"
    )


def _stab_item(factor, stable, *, direction="up", low=0.1, high=0.4,
               frequency=1.0, consistency=1.0, significant=1.0):
    """One canonical driver_attr_stability_summary region item."""
    return (
        '{"factor":' + json.dumps(factor)
        + ',"direction":' + json.dumps(direction)
        + f',"frequency":{frequency:.6f},"consistency":{consistency:.6f},'
        + f'"significant":{significant:.6f},"low":{low:.6f},'
        + f'"high":{high:.6f},"stable":' + ("true" if stable else "false")
        + ',"conflict":false,"cross_zero":false}'
    )


def _stability_report(*item_tokens, by="region", alpha=0.05):
    return (
        f'{{"alpha":{alpha:.6f},"groups":[{{"by":"{by}","items":['
        + ",".join(item_tokens) + "]}]}\n"
    )


def _four_stability(stables, directions=None, lows=None, highs=None):
    directions = directions or {}
    lows = lows or {}
    highs = highs or {}
    tokens = []
    for factor, stable in zip(_FACTORS, stables):
        tokens.append(
            _stab_item(
                factor,
                stable,
                direction=directions.get(factor, "up"),
                low=lows.get(factor, 0.1),
                high=highs.get(factor, 0.4),
            )
        )
    return tokens


def _region_items(out):
    return json.loads(out)["groups"][0]["items"]


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    i = _impact_report(
        _kis_item("uhi", -2.0, 0.04, True),
        _kis_item("energy", 1.0, 0.20, True),
        _kis_item("vent", 0.5, 0.20, False),
    )
    s = _stability_report(*_four_stability([True] * 4))
    out = driver_link(s, i, _LINKS)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    assert list(payload["groups"][0]) == ["by", "items"]
    assert payload["groups"][0]["by"] == "region"
    for item in payload["groups"][0]["items"]:
        assert list(item) == _ITEM_KEYS


def test_region_before_window_order():
    def group(by):
        return (
            '{"by":"' + by + '","items":['
            + ",".join(_four_stability([True] * 4)) + "]}"
        )

    s = (
        '{"alpha":0.050000,"groups":[' + group("region")
        + "," + group("window") + "]}\n"
    )
    items = [
        _kis_item("uhi", -2.0, 0.04, True),
        _kis_item("energy", 1.0, 0.20, True),
        _kis_item("vent", 0.5, 0.20, True),
    ]
    i_region = ",".join(items)
    i_window = ",".join(
        token.replace('"by":"region"', '"by":"window"') for token in items
    )
    i = (
        '{"alpha":0.050000,"items":[' + i_region + "," + i_window + "]}\n"
    )
    out = driver_link(s, i, _LINKS)
    assert [g["by"] for g in json.loads(out)["groups"]] == [
        "region", "window"
    ]


def test_empty_groups():
    s = '{"alpha":0.050000,"groups":[]}\n'
    i = '{"alpha":0.050000,"items":[]}\n'
    assert driver_link(s, i, _LINKS) == (
        '{"alpha":0.050000,"groups":[]}\n'
    )


def test_ranks_are_dense_ints_and_items_ranked():
    # lst/energy eligible and largest; station/uhi eligible but smaller;
    # morph/vent not rejected; cover/uhi stable but links to an up move
    # while change is down.
    i = _impact_report(
        _kis_item("uhi", -2.0, 0.05, True),
        _kis_item("energy", 10.0, 0.10, True),
        _kis_item("vent", 0.5, 0.20, False),
    )
    s = _stability_report(*_four_stability([True, True, True, True]))
    items = _region_items(driver_link(s, i, _LINKS))
    assert [it["factor"] for it in items] == [
        "lst", "station", "morph", "cover"
    ]
    assert [it["rank"] for it in items] == [1, 2, 3, 4]
    assert all(isinstance(it["rank"], int) for it in items)
    assert all(isinstance(it["eligible"], bool) for it in items)


def test_score_is_abs_change_times_one_minus_q():
    i = _impact_report(
        _kis_item("uhi", -2.0, 0.05, True),
        _kis_item("energy", 1.0, 0.9, False),
        _kis_item("vent", 0.5, 0.9, False),
    )
    s = _stability_report(
        *_four_stability([True, False, False, False])
    )
    items = _region_items(driver_link(s, i, _LINKS))
    station = next(it for it in items if it["factor"] == "station")
    assert station["eligible"] is True
    assert station["score"] == 1.9
    assert station["effect"] == -2.0
    assert station["q"] == 0.05
    assert station["metric"] == "uhi"
    # low/high echo the factor's stability-summary effect bounds.
    assert station["low"] == 0.1 and station["high"] == 0.4
    for factor in ("lst", "morph", "cover"):
        item = next(it for it in items if it["factor"] == factor)
        assert item["eligible"] is False
        assert item["score"] == 0.0


def test_ineligible_zero_scores_ordered_by_factor():
    i = _impact_report(
        _kis_item("uhi", -2.0, 0.05, True),
        _kis_item("energy", 10.0, 0.10, True),
        _kis_item("vent", 0.5, 0.20, True),
    )
    # Nothing is stable -> every item ineligible; factor order decides.
    s = _stability_report(*_four_stability([False] * 4))
    items = _region_items(driver_link(s, i, _LINKS))
    assert [it["factor"] for it in items] == list(_FACTORS)
    assert all(not it["eligible"] and it["score"] == 0.0 for it in items)


def test_direction_suffix_decides_eligibility():
    # station links uhi- (needs down); cover links uhi+ (needs up).
    i = _impact_report(
        _kis_item("uhi", -1.0, 0.02, True),
        _kis_item("energy", 1.0, 0.9, False),
        _kis_item("vent", 0.5, 0.9, False),
    )
    s = _stability_report(*_four_stability([True] * 4))
    items = _region_items(driver_link(s, i, _LINKS))
    flags = {it["factor"]: it["eligible"] for it in items}
    assert flags["station"] is True   # down matches uhi-
    assert flags["cover"] is False    # down does not match uhi+

    i_up = _impact_report(
        _kis_item("uhi", 1.0, 0.02, True),
        _kis_item("energy", 1.0, 0.9, False),
        _kis_item("vent", 0.5, 0.9, False),
    )
    items = _region_items(driver_link(s, i_up, _LINKS))
    flags = {it["factor"]: it["eligible"] for it in items}
    assert flags["station"] is False  # up does not match uhi-
    assert flags["cover"] is True     # up matches uhi+


def test_zero_change_matches_neither_direction():
    i = _impact_report(
        _kis_item("uhi", 0.0, 0.02, True),
        _kis_item("energy", 1.0, 0.9, False),
        _kis_item("vent", 0.5, 0.9, False),
    )
    s = _stability_report(*_four_stability([True] * 4))
    items = _region_items(driver_link(s, i, _LINKS))
    assert not any(it["eligible"] for it in items)


def test_reject_must_be_true():
    i = _impact_report(
        _kis_item("uhi", -2.0, 0.9, False),
        _kis_item("energy", 1.0, 0.9, False),
        _kis_item("vent", 0.5, 0.9, False),
    )
    s = _stability_report(*_four_stability([True] * 4))
    items = _region_items(driver_link(s, i, _LINKS))
    assert not any(it["eligible"] for it in items)


def test_six_decimals_and_no_negative_zero():
    i = _impact_report(
        _kis_item("uhi", 0.0, 0.5, False),
        _kis_item("energy", 0.0, 0.5, False),
        _kis_item("vent", 0.0, 0.5, False),
    )
    s = _stability_report(*_four_stability([False] * 4))
    out = driver_link(s, i, _LINKS)
    assert "-0.000000" not in out
    assert '"effect":0.000000' in out
    assert '"score":0.000000' in out


# --- validation -------------------------------------------------------------

def test_type_errors():
    s = '{"alpha":0.050000,"groups":[]}\n'
    i = '{"alpha":0.050000,"items":[]}\n'
    for bad in (1, None, [], {}, True):
        with pytest.raises(TypeError):
            driver_link(bad, i, _LINKS)
        with pytest.raises(TypeError):
            driver_link(s, bad, _LINKS)
    with pytest.raises(TypeError):
        driver_link(s, i, [])


def test_alpha_mismatch_value_error():
    s = '{"alpha":0.050000,"groups":[]}\n'
    i = '{"alpha":0.100000,"items":[]}\n'
    with pytest.raises(ValueError):
        driver_link(s, i, _LINKS)


def test_by_set_mismatch_value_error():
    s = '{"alpha":0.050000,"groups":[]}\n'
    window = (
        '{"alpha":0.050000,"items":['
        + _kis_item("uhi", -1.0, 0.05, True).replace(
            '"by":"region"', '"by":"window"'
        )
        + "]}\n"
    )
    with pytest.raises(ValueError):
        driver_link(s, window, _LINKS)


def test_links_key_set_must_be_exact():
    s = '{"alpha":0.050000,"groups":[]}\n'
    i = '{"alpha":0.050000,"items":[]}\n'
    with pytest.raises(ValueError):
        driver_link(s, i, {})
    with pytest.raises(ValueError):
        driver_link(
            s, i,
            {"station": "uhi-", "lst": "energy+", "morph": "vent+"},
        )
    with pytest.raises(ValueError):
        driver_link(
            s, i,
            {
                "station": "uhi-", "lst": "energy+",
                "morph": "vent+", "cover": "uhi+", "extra": "uhi-",
            },
        )


@pytest.mark.parametrize("value", ["uhi", "UHI-", "uhi- ", "uhi--", 1, None])
def test_bad_link_values_value_error(value):
    s = '{"alpha":0.050000,"groups":[]}\n'
    i = '{"alpha":0.050000,"items":[]}\n'
    links = dict(_LINKS)
    links["station"] = value
    with pytest.raises(ValueError):
        driver_link(s, i, links)


def test_all_six_link_values_accepted():
    for value in _LINK_VALUES:
        links = {factor: value for factor in _FACTORS}
        metric = value[:-1]
        change = -1.0 if value.endswith("-") else 1.0
        triples = []
        for name in ("uhi", "energy", "vent"):
            if name == metric:
                triples.append(_kis_item(name, change, 0.02, True))
            else:
                triples.append(_kis_item(name, 5.0, 0.9, False))
        i = _impact_report(*triples)
        s = _stability_report(*_four_stability([True] * 4))
        out = driver_link(s, i, links)
        items = _region_items(out)
        assert all(it["eligible"] for it in items)
        assert {it["metric"] for it in items} == {metric}


def test_noncanonical_s_value_error():
    i = '{"alpha":0.050000,"items":[]}\n'
    for bad in ("", "nope\n", '{"alpha":0.05,"groups":[]}\n'):
        with pytest.raises(ValueError):
            driver_link(bad, i, _LINKS)


def test_noncanonical_i_value_error():
    s = '{"alpha":0.050000,"groups":[]}\n'
    for bad in ("", "nope\n", '{"alpha":0.05,"items":[]}\n'):
        with pytest.raises(ValueError):
            driver_link(s, bad, _LINKS)


def test_wrong_report_shapes_rejected():
    s = '{"alpha":0.050000,"groups":[]}\n'
    i = '{"alpha":0.050000,"items":[]}\n'
    # An impact summary is not a stability summary and vice versa.
    with pytest.raises(ValueError):
        driver_link(i, i, _LINKS)
    with pytest.raises(ValueError):
        driver_link(s, s, _LINKS)


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.driver_link is urban_micro.driver_link
    assert "driver_link" in urban_micro.__all__
