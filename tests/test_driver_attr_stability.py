"""Tests for urban_micro.driver_attr_stability: selection stability of
repeated driver_attr reports aggregated per by and station/lst/morph/
cover factor."""

import json

import pytest

from urban_micro import driver_attr, driver_attr_stability, kind_impact
from urban_micro import uhi as _uhi

_FACTORS = ("station", "lst", "morph", "cover")


# --- report builders --------------------------------------------------------

def _portfolio_group(by, budget=3.0, cost=1.0, remaining=2.0, score=2.0):
    pick = ("green",)
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


def _panel(base, green=(0, 0, 0)):
    return (
        tuple(base),
        {
            "green": tuple(green),
            "roof": (0, 0, 0),
            "material": (0, 0, 0),
        },
    )


def _impact(by_to_changes, alpha=0.05):
    """Build a canonical kind_impact report: by -> {key: uhi change}."""
    groups = []
    panels = {}
    for by in ("region", "window"):
        if by not in by_to_changes:
            continue
        groups.append(_portfolio_group(by))
        for key, change in by_to_changes[by].items():
            panels[(by, key)] = _panel(
                (10.0, 0.0, 0.0), green=(change, 0.0, 0.0)
            )
    return kind_impact(_portfolio_report(*groups, alpha=alpha), panels)


def _driver_data(by_to_factors):
    return {
        (by, key): tuple(values)
        for by, per_key in by_to_factors.items()
        for key, values in per_key.items()
    }


def _da_item(by, factor, effect, reject, direction, n=3,
             p="0.333333", q="0.666667", alpha=0.05):
    """Hand-build one canonical driver_attr report string."""
    reject_token = "true" if reject else "false"
    return (
        f'{{"alpha":{alpha:.6f},"items":[{{"by":{json.dumps(by)},'
        f'"factor":{json.dumps(factor)},"n":{n},"effect":{effect:.6f},'
        f'"p":{p},"q":{q},"reject":{reject_token},'
        f'"direction":{json.dumps(direction)}}}]}}\n'
    )


def _da_item_token(by, factor, effect, reject, direction, n=3,
                   p="0.333333", q="0.666667"):
    """One item object token (no surrounding report), region-before-window."""
    reject_token = "true" if reject else "false"
    return (
        f'{{"by":{json.dumps(by)},"factor":{json.dumps(factor)},"n":{n},'
        f'"effect":{effect:.6f},"p":{p},"q":{q},'
        f'"reject":{reject_token},"direction":{json.dumps(direction)}}}'
    )


def _da_report(*tokens, alpha=0.05):
    return (
        f'{{"alpha":{alpha:.6f},"items":[' + ",".join(tokens) + "]}\n"
    )


# --- datasets producing genuine driver_attr reports -------------------------

def _region_station_down():
    changes = {"region": {"a": -1, "b": -2, "c": -3}}
    factors = {
        "region": {
            "a": (1.0, 10.0, 0.5, 0.2),
            "b": (2.0, 20.0, 0.5, 0.9),
            "c": (3.0, 30.0, 0.5, 0.1),
        }
    }
    return driver_attr(_impact(changes), _driver_data(factors))


def _region_lst_up():
    changes = {"region": {"a": -1, "b": 0, "c": 1}}
    factors = {
        "region": {
            # lst tracks y perfectly; station correlates only partially.
            "a": (1.0, 1.0, 0.2, 0.1),
            "b": (2.0, 2.0, 0.3, 0.8),
            "c": (2.0, 3.0, 0.9, 0.4),
        }
    }
    return driver_attr(_impact(changes), _driver_data(factors))


def _region_station_down_rejected():
    # n = 6 perfect ordering: p = 2/6!, reject true under BH at 0.05.
    changes = {"region": {f"k{i}": -float(i) for i in range(1, 7)}}
    factors = {
        "region": {
            f"k{i}": (float(i), 0.0, 0.0, 0.0) for i in range(1, 7)
        }
    }
    return driver_attr(_impact(changes), _driver_data(factors))


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    out = driver_attr_stability(
        {"r1": _region_station_down(), "r2": _region_lst_up()}
    )
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    assert list(payload["groups"][0]) == ["by", "items"]
    assert [item["factor"] for item in payload["groups"][0]["items"]] == list(
        _FACTORS
    )
    for item in payload["groups"][0]["items"]:
        assert list(item) == [
            "factor", "n", "frequency", "direction",
            "consistency", "low", "high", "significant",
        ]


def test_two_reports_same_pick_statistics():
    out = driver_attr_stability(
        {
            "r1": _da_item("region", "station", -1.0, False, "down"),
            "r2": _da_item("region", "station", -0.5, False, "down"),
        }
    )
    group = json.loads(out)["groups"][0]
    assert group["by"] == "region"
    station, lst, morph, cover = group["items"]
    assert station == {
        "factor": "station",
        "n": 2,
        "frequency": 1.0,
        "direction": "down",
        "consistency": 1.0,
        "low": -1.0,
        "high": -0.5,
        "significant": 0.0,
    }
    for item in (lst, morph, cover):
        assert item["n"] == 0
        assert item["frequency"] == 0.0
        assert item["direction"] == "flat"
        assert item["consistency"] == 0.0
        assert item["significant"] == 0.0
        assert item["low"] == 0.0
        assert item["high"] == 0.0


def test_frequency_and_significant_shares():
    out = driver_attr_stability(
        {
            "r1": _da_item("region", "station", -1.0, True, "down",
                           p="0.002778", q="0.005556"),
            "r2": _da_item("region", "lst", 1.0, False, "up"),
            "r3": _da_item("region", "station", -0.25, False, "down"),
        }
    )
    items = {item["factor"]: item for item in json.loads(out)["groups"][0]["items"]}
    assert items["station"]["n"] == 2
    assert abs(items["station"]["frequency"] - 2 / 3) < 5.1e-7
    assert items["station"]["significant"] == 0.5
    assert items["station"]["direction"] == "down"
    assert items["station"]["consistency"] == 1.0
    assert items["station"]["low"] == -1.0
    assert items["station"]["high"] == -0.25
    assert items["lst"]["n"] == 1
    assert abs(items["lst"]["frequency"] - 1 / 3) < 5.1e-7
    assert items["lst"]["significant"] == 0.0
    assert items["lst"]["direction"] == "up"


def test_direction_tie_resolves_down_up_flat():
    # down vs up tie -> down.
    out = driver_attr_stability(
        {
            "r1": _da_item("region", "morph", -0.4, False, "down"),
            "r2": _da_item("region", "morph", 0.4, False, "up"),
        }
    )
    item = json.loads(out)["groups"][0]["items"][2]
    assert item["direction"] == "down"
    assert item["consistency"] == 0.5
    assert item["low"] == -0.4
    assert item["high"] == 0.4

    # up vs flat tie -> up (down absent).
    out = driver_attr_stability(
        {
            "r1": _da_item("region", "morph", 0.4, False, "up"),
            "r2": _da_item("region", "morph", 0.0, False, "flat"),
        }
    )
    item = json.loads(out)["groups"][0]["items"][2]
    assert item["direction"] == "up"
    assert item["consistency"] == 0.5

    # down vs flat tie -> down.
    out = driver_attr_stability(
        {
            "r1": _da_item("region", "morph", -0.4, False, "down"),
            "r2": _da_item("region", "morph", 0.0, False, "flat"),
        }
    )
    item = json.loads(out)["groups"][0]["items"][2]
    assert item["direction"] == "down"


def test_consistency_is_modal_share_not_unanimous():
    out = driver_attr_stability(
        {
            "r1": _da_item("region", "cover", 0.1, True, "up"),
            "r2": _da_item("region", "cover", 0.2, False, "up"),
            "r3": _da_item("region", "cover", -0.3, False, "down"),
            "r4": _da_item("region", "cover", 0.4, True, "up"),
        }
    )
    item = json.loads(out)["groups"][0]["items"][3]
    assert item["n"] == 4
    assert item["frequency"] == 1.0
    assert item["direction"] == "up"
    assert item["consistency"] == 0.75
    assert item["significant"] == 0.5
    assert item["low"] == -0.3
    assert item["high"] == 0.4


def test_six_decimal_tokens_and_no_negative_zero():
    out = driver_attr_stability(
        {
            "r1": _da_item("region", "station", -1.0, False, "down"),
            "r2": _da_item("region", "lst", 1.0, False, "up"),
            "r3": _da_item("region", "lst", 1.0, False, "up"),
        }
    )
    assert '"frequency":0.333333' in out
    assert '"frequency":0.666667' in out
    assert '"consistency":1.000000' in out
    assert "-0.000000" not in out
    assert '"n":0' in out and '"n":1' in out and '"n":2' in out


def test_groups_region_before_window_and_window_only():
    out = driver_attr_stability(
        {
            "r1": _da_item("region", "station", -1.0, False, "down"),
            "r2": _da_item("region", "station", -1.0, False, "down"),
        }
    )
    groups = json.loads(out)["groups"]
    assert [group["by"] for group in groups] == ["region"]

    both_a = _da_report(
        _da_item_token("region", "station", -1.0, False, "down"),
        _da_item_token("window", "lst", 1.0, False, "up"),
    )
    out = driver_attr_stability({"r1": both_a, "r2": both_a})
    assert [group["by"] for group in json.loads(out)["groups"]] == [
        "region", "window"
    ]
    window = json.loads(out)["groups"][1]
    assert window["by"] == "window"
    assert window["items"][0]["n"] == 0
    assert window["items"][1]["n"] == 2


def test_end_to_end_with_genuine_reports():
    out = driver_attr_stability(
        {
            "r1": _region_station_down(),
            "r2": _region_lst_up(),
            "r3": _region_station_down_rejected(),
        }
    )
    items = {
        item["factor"]: item
        for item in json.loads(out)["groups"][0]["items"]
    }
    assert items["station"]["n"] == 2
    assert abs(items["station"]["frequency"] - 2 / 3) < 5.1e-7
    assert items["station"]["direction"] == "down"
    assert items["station"]["consistency"] == 1.0
    assert items["station"]["significant"] == 0.5
    assert items["station"]["low"] == -1.0
    assert items["station"]["high"] == -1.0
    assert items["lst"]["n"] == 1
    assert items["lst"]["direction"] == "up"
    assert abs(items["lst"]["frequency"] - 1 / 3) < 5.1e-7
    assert items["morph"]["n"] == 0
    assert items["cover"]["n"] == 0


def test_unicode_report_keys_preserved_elsewhere():
    # Non-empty keys may be arbitrary Unicode; they never appear in output.
    out = driver_attr_stability(
        {"报告一": _region_station_down(), "rapport 2": _region_lst_up()}
    )
    assert json.loads(out)["alpha"] == 0.05


# --- validation -------------------------------------------------------------

def test_non_dict_reports_type_error():
    for bad in ([], None, (), "x", 1):
        with pytest.raises(TypeError):
            driver_attr_stability(bad)


def test_fewer_than_two_entries_value_error():
    report = _da_item("region", "station", -1.0, False, "down")
    with pytest.raises(ValueError):
        driver_attr_stability({"only": report})
    with pytest.raises(ValueError):
        driver_attr_stability({})


def test_keys_must_be_non_empty_strings():
    report = _da_item("region", "station", -1.0, False, "down")
    with pytest.raises(ValueError):
        driver_attr_stability({"": report, "b": report})
    with pytest.raises(ValueError):
        driver_attr_stability({1: report, "b": report})
    with pytest.raises(ValueError):
        driver_attr_stability({None: report, "b": report})


def test_values_must_be_canonical_driver_attr_reports():
    report = _da_item("region", "station", -1.0, False, "down")
    good = {"r1": report, "r2": report}
    # driver_attr_stability itself does not accept non-string values as
    # anything but ValueError (they are report-shape violations).
    for bad_value in (1, None, [], {}):
        bad = dict(good)
        bad["r1"] = bad_value
        with pytest.raises(ValueError):
            driver_attr_stability(bad)
    for raw in (
        "",
        "not json\n",
        report.rstrip("\n"),
        report + "\n",
        '{"alpha":0.050000,"groups":[]}\n',
        '{"alpha":0.05,"items":[]}\n',
    ):
        with pytest.raises(ValueError):
            driver_attr_stability({"r1": raw, "r2": report})


def test_canonical_byte_for_byte_required():
    report = _da_item("region", "station", -1.0, False, "down")
    # Reordered item keys.
    reordered = report.replace(
        '"factor":"station","n":3', '"n":3,"factor":"station"'
    )
    assert reordered != report
    with pytest.raises(ValueError):
        driver_attr_stability({"r1": reordered, "r2": report})
    # Two region items (one by may hold only its single picked item).
    duplicate = _da_report(
        _da_item_token("region", "station", -1.0, False, "down"),
        _da_item_token("region", "lst", 1.0, False, "up"),
    )
    with pytest.raises(ValueError):
        driver_attr_stability({"r1": duplicate, "r2": report})


def test_alpha_must_match_across_reports():
    r1 = _da_item("region", "station", -1.0, False, "down", alpha=0.05)
    r2 = _da_item("region", "station", -1.0, False, "down", alpha=0.10)
    with pytest.raises(ValueError):
        driver_attr_stability({"r1": r1, "r2": r2})


def test_by_set_must_match_across_reports():
    region = _da_item("region", "station", -1.0, False, "down")
    window = _da_item("window", "station", -1.0, False, "down")
    with pytest.raises(ValueError):
        driver_attr_stability({"r1": region, "r2": window})


def test_kind_impact_report_shape_rejected():
    impact = _impact({"region": {"a": -1, "b": -2, "c": -3}})
    report = _da_item("region", "station", -1.0, False, "down")
    with pytest.raises(ValueError):
        driver_attr_stability({"r1": impact, "r2": report})


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.driver_attr_stability is urban_micro.driver_attr_stability
    assert "driver_attr_stability" in urban_micro.__all__
