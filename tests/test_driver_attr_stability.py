"""Tests for urban_micro.driver_attr_stability: aggregation of repeated
driver_attr picks into per-by, per-factor selection frequency, direction
consensus, significance share and effect extremes."""

import json

import pytest

from urban_micro import driver_attr, driver_attr_stability, kind_impact
from urban_micro import uhi as _uhi

_FACTORS = ("station", "lst", "morph", "cover")
_DIRECTIONS = ("down", "up", "flat")


# --- canonical driver_attr report builders ----------------------------------

def _item(by, factor, n, effect, p="0.333333", q="0.666667",
          reject=False, direction="flat"):
    return (
        '{"by":' + json.dumps(by)
        + ',"factor":' + json.dumps(factor)
        + f',"n":{n},"effect":{effect:.6f},"p":{p},"q":{q}'
        + ',"reject":' + ("true" if reject else "false")
        + ',"direction":' + json.dumps(direction)
        + "}"
    )


def _report(*items, alpha="0.050000"):
    return (
        f'{{"alpha":{alpha},"items":[' + ",".join(items) + "]}\n"
    )


def _pick(by, factor, *, direction="flat", effect=0.0, reject=False, n=3):
    return _item(
        by, factor, n, effect, reject=reject, direction=direction
    )


def _report_from_picks(by_to_pick, alpha="0.050000"):
    order = ("region", "window")
    items = [by_to_pick[by] for by in order if by in by_to_pick]
    return _report(*items, alpha=alpha)


# --- canonical shape ---------------------------------------------------------

def test_canonical_shape_and_key_order():
    reports = {
        "a": _report(_pick("region", "station", direction="down",
                           effect=-1.0, reject=True)),
        "b": _report(_pick("region", "lst", direction="up", effect=0.5)),
    }
    out = driver_attr_stability(reports)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    assert list(payload["groups"][0]) == ["by", "items"]
    assert list(payload["groups"][0]["items"][0]) == [
        "factor", "n", "frequency", "direction",
        "consistency", "low", "high", "significant",
    ]


def test_groups_region_before_window_and_factor_order():
    reports = {
        "a": _report(
            _pick("region", "cover", direction="up", effect=0.1),
            _pick("window", "station", direction="down", effect=-0.1),
        ),
        "b": _report(
            _pick("region", "morph", direction="flat"),
            _pick("window", "lst", direction="up", effect=0.2),
        ),
    }
    payload = json.loads(driver_attr_stability(reports))
    assert [group["by"] for group in payload["groups"]] == [
        "region", "window"
    ]
    for group in payload["groups"]:
        assert [item["factor"] for item in group["items"]] == list(
            _FACTORS
        )
        assert len(group["items"]) == 4


# --- frequency, n and the n == 0 rule ----------------------------------------

def test_frequency_and_n():
    reports = {
        "r1": _report(_pick("region", "station", direction="down",
                            effect=-1.0)),
        "r2": _report(_pick("region", "station", direction="down",
                            effect=-0.5)),
        "r3": _report(_pick("region", "lst", direction="up",
                            effect=0.25)),
    }
    items = json.loads(driver_attr_stability(reports))["groups"][0]["items"]
    by_factor = {item["factor"]: item for item in items}
    assert by_factor["station"]["n"] == 2
    assert abs(by_factor["station"]["frequency"] - 2 / 3) < 5.1e-7
    assert by_factor["lst"]["n"] == 1
    assert abs(by_factor["lst"]["frequency"] - 1 / 3) < 5.1e-7
    assert isinstance(by_factor["station"]["n"], int)
    assert '"frequency":0.666667' in driver_attr_stability(reports)
    assert '"frequency":0.333333' in driver_attr_stability(reports)


def test_never_selected_factor_is_flat_and_all_zero():
    reports = {
        "a": _report(_pick("region", "station", direction="down",
                           effect=-1.0, reject=True)),
        "b": _report(_pick("region", "station", direction="up",
                           effect=1.0)),
    }
    items = json.loads(driver_attr_stability(reports))["groups"][0]["items"]
    for item in items:
        if item["factor"] == "station":
            assert item["n"] == 2
        else:
            assert item == {
                "factor": item["factor"],
                "n": 0,
                "frequency": 0.0,
                "direction": "flat",
                "consistency": 0.0,
                "low": 0.0,
                "high": 0.0,
                "significant": 0.0,
            }


def test_all_factors_absent_yields_empty_groups():
    empty = '{"alpha":0.050000,"items":[]}\n'
    out = driver_attr_stability({"a": empty, "b": empty})
    assert out == '{"alpha":0.050000,"groups":[]}\n'


# --- direction mode with tie order -------------------------------------------

def test_direction_mode_down_up_tie_picks_down():
    reports = {
        "a": _report(_pick("region", "station", direction="down",
                           effect=-1.0)),
        "b": _report(_pick("region", "station", direction="up",
                           effect=1.0)),
    }
    item = json.loads(driver_attr_stability(reports))["groups"][0][
        "items"
    ][0]
    assert item["direction"] == "down"
    assert item["consistency"] == 0.5


def test_direction_mode_up_flat_tie_picks_up():
    reports = {
        "a": _report(_pick("region", "station", direction="up",
                           effect=1.0)),
        "b": _report(_pick("region", "station", direction="flat")),
    }
    item = json.loads(driver_attr_stability(reports))["groups"][0][
        "items"
    ][0]
    assert item["direction"] == "up"
    assert item["consistency"] == 0.5


def test_three_way_tie_picks_down():
    reports = {
        "a": _report(_pick("region", "station", direction="down",
                           effect=-1.0)),
        "b": _report(_pick("region", "station", direction="up",
                           effect=1.0)),
        "c": _report(_pick("region", "station", direction="flat")),
    }
    item = json.loads(driver_attr_stability(reports))["groups"][0][
        "items"
    ][0]
    assert item["direction"] == "down"
    assert abs(item["consistency"] - 1 / 3) < 5.1e-7


def test_mode_needs_no_majority():
    # down twice, up once, flat once: mode down, consistency 2/4.
    reports = {
        "a": _report(_pick("region", "station", direction="down",
                           effect=-1.0)),
        "b": _report(_pick("region", "station", direction="down",
                           effect=-0.2)),
        "c": _report(_pick("region", "station", direction="up",
                           effect=0.2)),
        "d": _report(_pick("region", "station", direction="flat")),
    }
    item = json.loads(driver_attr_stability(reports))["groups"][0][
        "items"
    ][0]
    assert item["direction"] == "down"
    assert item["consistency"] == 0.5


# --- significance share and effect extremes ----------------------------------

def test_significant_share():
    reports = {
        "a": _report(_pick("region", "station", direction="down",
                           effect=-1.0, reject=True)),
        "b": _report(_pick("region", "station", direction="up",
                           effect=0.5, reject=True)),
        "c": _report(_pick("region", "station", direction="up",
                           effect=0.25, reject=False)),
        "d": _report(_pick("region", "station", direction="up",
                           effect=0.75, reject=True)),
    }
    item = json.loads(driver_attr_stability(reports))["groups"][0][
        "items"
    ][0]
    assert item["n"] == 4
    assert item["direction"] == "up"
    assert item["consistency"] == 0.75
    assert item["significant"] == 0.75
    assert '"significant":0.750000' in driver_attr_stability(reports)


def test_low_and_high_are_effect_extremes():
    reports = {
        "a": _report(_pick("region", "station", direction="down",
                           effect=-1.0)),
        "b": _report(_pick("region", "station", direction="down",
                           effect=-0.25)),
        "c": _report(_pick("region", "station", direction="up",
                           effect=0.5)),
    }
    item = json.loads(driver_attr_stability(reports))["groups"][0][
        "items"
    ][0]
    assert item["low"] == -1.0
    assert item["high"] == 0.5
    out = driver_attr_stability(reports)
    assert '"low":-1.000000' in out
    assert '"high":0.500000' in out


def test_single_selection_low_equals_high():
    reports = {
        "a": _report(_pick("region", "morph", direction="up",
                           effect=0.125)),
        "b": _report(_pick("region", "station", direction="flat")),
    }
    item = [
        item for item in json.loads(driver_attr_stability(reports))[
            "groups"
        ][0]["items"] if item["factor"] == "morph"
    ][0]
    assert item["low"] == item["high"] == 0.125
    assert item["direction"] == "up"
    assert item["consistency"] == 1.0
    assert item["significant"] == 0.0


# --- independent by dimensions -----------------------------------------------

def test_bys_aggregated_independently():
    reports = {
        "a": _report(
            _pick("region", "station", direction="down", effect=-1.0,
                  reject=True),
            _pick("window", "cover", direction="up", effect=0.4),
        ),
        "b": _report(
            _pick("region", "station", direction="up", effect=0.5),
            _pick("window", "cover", direction="up", effect=0.1),
        ),
        "c": _report(
            _pick("region", "lst", direction="flat"),
            _pick("window", "station", direction="down", effect=-0.2),
        ),
    }
    groups = {
        group["by"]: {item["factor"]: item for item in group["items"]}
        for group in json.loads(driver_attr_stability(reports))["groups"]
    }
    assert set(groups) == {"region", "window"}
    assert groups["region"]["station"]["n"] == 2
    assert groups["region"]["station"]["direction"] == "down"
    assert groups["region"]["lst"]["n"] == 1
    assert groups["window"]["cover"]["n"] == 2
    assert groups["window"]["cover"]["direction"] == "up"
    assert groups["window"]["cover"]["low"] == 0.1
    assert groups["window"]["cover"]["high"] == 0.4
    assert groups["window"]["station"]["n"] == 1
    assert groups["window"]["station"]["direction"] == "down"


def test_result_independent_of_report_dict_order():
    first = _report(_pick("region", "station", direction="down",
                          effect=-1.0, reject=True))
    second = _report(_pick("region", "lst", direction="up", effect=0.5))
    third = _report(_pick("region", "station", direction="up", effect=0.25))
    out_one = driver_attr_stability({"a": first, "b": second, "c": third})
    out_two = driver_attr_stability({"c": third, "a": first, "b": second})
    assert out_one == out_two


# --- integration with real driver_attr reports -------------------------------

def _portfolio_group(by, pick=("green",), budget=3.0, cost=1.0,
                     remaining=2.0, score=2.0):
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


def _impact(region_changes):
    groups = [_portfolio_group("region")]
    panels = {}
    for key, change in region_changes.items():
        panels[("region", key)] = (
            (10.0, 0.0, 0.0),
            {
                "green": (float(change), 0.0, 0.0),
                "roof": (0.0, 0.0, 0.0),
                "material": (0.0, 0.0, 0.0),
            },
        )
    report = (
        '{"alpha":0.050000,"groups":[' + ",".join(groups) + "]}\n"
    )
    return kind_impact(report, panels)


def test_accepts_real_driver_attr_outputs():
    # Dataset A: station constant, lst tracks y perfectly -> lst/up.
    impact_a = _impact({"a": -1, "b": 0, "c": 1})
    data_a = {
        ("region", "a"): (5.0, 1.0, 0.2, 0.1),
        ("region", "b"): (5.0, 2.0, 0.3, 0.8),
        ("region", "c"): (5.0, 3.0, 0.9, 0.4),
    }
    # Dataset B: station anti-tracks y perfectly -> station/down.
    impact_b = _impact({"a": -1, "b": -2, "c": -3})
    data_b = {
        ("region", "a"): (1.0, 10.0, 0.5, 0.2),
        ("region", "b"): (2.0, 20.0, 0.5, 0.9),
        ("region", "c"): (3.0, 30.0, 0.5, 0.1),
    }
    report_a = driver_attr(impact_a, data_a)
    report_b = driver_attr(impact_b, data_b)
    assert json.loads(report_a)["items"][0]["factor"] == "lst"
    assert json.loads(report_b)["items"][0]["factor"] == "station"

    out = driver_attr_stability({"a": report_a, "b": report_b})
    items = json.loads(out)["groups"][0]["items"]
    by_factor = {item["factor"]: item for item in items}
    assert by_factor["lst"]["n"] == 1
    assert by_factor["lst"]["direction"] == "up"
    assert by_factor["lst"]["high"] == 1.0
    assert by_factor["station"]["n"] == 1
    assert by_factor["station"]["direction"] == "down"
    assert by_factor["station"]["low"] == -1.0
    assert by_factor["morph"]["n"] == 0


# --- formatting ---------------------------------------------------------------

def test_six_decimal_tokens_and_no_negative_zero():
    # A real driver_attr report may render a tiny negative r as
    # 0.000000 while keeping direction "down"; aggregating such picks must
    # never emit -0.000000 and the three-way tie resolves to down.
    reports = {
        "a": _report(_pick("region", "station", direction="down",
                           effect=0.0)),
        "b": _report(_pick("region", "station", direction="up",
                           effect=0.0)),
        "c": _report(_pick("region", "station", direction="flat")),
    }
    out = driver_attr_stability(reports)
    assert "-0.000000" not in out
    assert '"direction":"down","consistency":0.333333' in out
    assert '"low":0.000000,"high":0.000000' in out
    assert '"significant":0.000000' in out
    assert '"frequency":1.000000' in out
    assert '"n":3' in out


# --- validation ---------------------------------------------------------------

def test_non_dict_reports_type_error():
    good = _report(_pick("region", "station", direction="down"))
    for bad in ([good, good], (good, good), None, 1, "x"):
        with pytest.raises(TypeError):
            driver_attr_stability(bad)


def test_fewer_than_two_reports_value_error():
    good = _report(_pick("region", "station", direction="down"))
    with pytest.raises(ValueError):
        driver_attr_stability({})
    with pytest.raises(ValueError):
        driver_attr_stability({"only": good})


def test_keys_must_be_non_empty_strings():
    good = _report(_pick("region", "station", direction="down"))
    with pytest.raises(ValueError):
        driver_attr_stability({"": good, "b": good})
    with pytest.raises(ValueError):
        driver_attr_stability({1: good, "b": good})


def test_values_must_be_canonical_driver_attr_outputs():
    good = _report(_pick("region", "station", direction="down"))
    bad_values = [
        '{"alpha":0.050000,"items":[]}',            # missing newline
        '{"alpha":0.050000,"items":[]}\n\n',        # two newlines
        ' {"alpha":0.050000,"items":[]}\n',         # leading space
        '{"alpha":0.05,"items":[]}\n',              # uncanonical alpha
        '{"alpha":0.050000}\n',                     # missing items key
        '{"items":[],"alpha":0.050000}\n',          # swapped top keys
        42,
        None,
        # kind_impact-style report is not a driver_attr report.
        '{"alpha":0.050000,"groups":[]}\n',
    ]
    for bad in bad_values:
        with pytest.raises(ValueError):
            driver_attr_stability({"a": good, "b": bad})


def test_item_tokens_must_be_canonical():
    base = _pick("region", "station", direction="down", effect=-1.0)
    # Swapped item key order, wrong factor, bad direction, negative zero.
    bad_items = [
        '{"by":"region","n":3,"factor":"station","effect":-1.000000,'
        '"p":0.333333,"q":0.666667,"reject":false,"direction":"down"}',
        '{"by":"region","factor":"wind","n":3,"effect":-1.000000,'
        '"p":0.333333,"q":0.666667,"reject":false,"direction":"down"}',
        '{"by":"region","factor":"station","n":3,"effect":-1.000000,'
        '"p":0.333333,"q":0.666667,"reject":false,"direction":"sideways"}',
        '{"by":"region","factor":"station","n":3,"effect":-0.000000,'
        '"p":0.333333,"q":0.666667,"reject":false,"direction":"down"}',
        '{"by":"region","factor":"station","n":"3","effect":-1.000000,'
        '"p":0.333333,"q":0.666667,"reject":false,"direction":"down"}',
    ]
    for bad_item in bad_items:
        bad_report = (
            '{"alpha":0.050000,"items":[' + bad_item + "]}\n"
        )
        with pytest.raises(ValueError):
            driver_attr_stability({"a": bad_report, "b": _report(
                _pick("region", "station", direction="up"))})


def test_duplicate_by_in_one_report_rejected():
    duplicated = _report(
        _pick("region", "station", direction="down"),
        _pick("region", "lst", direction="up"),
    )
    with pytest.raises(ValueError):
        driver_attr_stability({
            "a": duplicated,
            "b": _report(_pick("region", "station", direction="up")),
        })


def test_alpha_must_match_across_reports():
    reports = {
        "a": _report_from_picks(
            {"region": _pick("region", "station", direction="down")},
            alpha="0.050000",
        ),
        "b": _report_from_picks(
            {"region": _pick("region", "station", direction="up")},
            alpha="0.010000",
        ),
    }
    with pytest.raises(ValueError):
        driver_attr_stability(reports)


def test_by_sets_must_match_across_reports():
    reports = {
        "a": _report(
            _pick("region", "station", direction="down"),
            _pick("window", "cover", direction="up"),
        ),
        "b": _report(_pick("region", "station", direction="up")),
    }
    with pytest.raises(ValueError):
        driver_attr_stability(reports)


def test_window_before_region_order_rejected():
    # Canonical reports put region before window.
    reordered = _report(
        _pick("window", "cover", direction="up"),
        _pick("region", "station", direction="down"),
    )
    with pytest.raises(ValueError):
        driver_attr_stability({
            "a": reordered,
            "b": _report(
                _pick("region", "station", direction="up"),
                _pick("window", "cover", direction="flat"),
            ),
        })


# --- exports and CLI ---------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.driver_attr_stability is urban_micro.driver_attr_stability
    assert "driver_attr_stability" in urban_micro.__all__
    # The sibling export stays intact.
    assert _uhi.driver_attr is urban_micro.driver_attr


def test_cli_version_and_help_unchanged(capsys):
    from urban_micro.cli import main

    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "urban-microclimate-modeling" in capsys.readouterr().out
