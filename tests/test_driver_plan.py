"""Tests for urban_micro.driver_plan: enumerate budgeted subsets of
interventions over a canonical driver_link report."""

import json

import pytest

from urban_micro import driver_link, driver_plan
from urban_micro import uhi as _uhi

_FACTORS = ("station", "lst", "morph", "cover")
_METRICS = ("uhi", "energy", "vent")
_KINDS = ("green", "roof", "material")
_GROUP_KEYS = ["by", "cost", "priority", "pick", "effects", "range"]
_LINKS = {"station": "uhi-", "lst": "energy-", "morph": "vent+",
          "cover": "uhi+"}


# --- direct canonical driver_link report builders ---------------------------

def _item(rank, factor, metric, effect, low, high, q, score, eligible):
    return (
        '{"factor":' + json.dumps(factor)
        + ',"metric":' + json.dumps(metric)
        + f',"effect":{effect:.6f},"low":{low:.6f},"high":{high:.6f},'
        + f'"q":{q:.6f},"score":{score:.6f},'
        + f'"eligible":{str(eligible).lower()},"rank":{rank}}}'
    )


def _group(by, items):
    return '{"by":' + json.dumps(by) + ',"items":[' + ",".join(items) + "]}"


def _report(*groups, alpha=0.05):
    return ('{"alpha":' + f"{alpha:.6f}" + ',"groups":['
            + ",".join(groups) + "]}\n")


def _one_region(station=(4.0, -2.0, -0.5, -0.2, True),
                lst=(3.0, -5.0, -0.4, -0.1, True),
                morph=(0.0, 3.0, 0.0, 0.0, False),
                cover=(0.0, 1.0, 0.0, 0.0, False)):
    """One region group; each tuple is (score, effect, low, high, eligible)."""
    rows = (
        ("station", "uhi", station),
        ("lst", "energy", lst),
        ("morph", "vent", morph),
        ("cover", "uhi", cover),
    )
    tokens = [
        _item(rank, factor, metric, effect, low, high, 0.01, score, eligible)
        for rank, (factor, metric, (score, effect, low, high, eligible))
        in enumerate(rows, start=1)
    ]
    return _report(_group("region", tokens))


def _actions(green=("station", 2.0), roof=("lst", 1.0),
             material=("morph", 1.0)):
    return {"green": green, "roof": roof, "material": material}


def _group_payload(out, by="region"):
    groups = json.loads(out)["groups"]
    return next(group for group in groups if group["by"] == by)


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    out = driver_plan(_one_region(), _actions(), 10.0)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["groups"]
    group = payload["groups"][0]
    assert list(group) == _GROUP_KEYS
    assert list(group["effects"]) == ["uhi", "energy", "vent"]
    assert list(group["range"]) == ["low", "high"]
    assert isinstance(group["pick"], list)


def test_region_before_window_groups():
    region = _group("region", [
        _item(1, "station", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        _item(2, "lst", "energy", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        _item(3, "morph", "vent", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        _item(4, "cover", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
    ])
    window = _group("window", [
        _item(1, "station", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        _item(2, "lst", "energy", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        _item(3, "morph", "vent", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        _item(4, "cover", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
    ])
    out = driver_plan(_report(region, window), _actions(), 10.0)
    assert [g["by"] for g in json.loads(out)["groups"]] == [
        "region", "window"
    ]


def test_empty_groups():
    assert driver_plan('{"alpha":0.050000,"groups":[]}\n', _actions(), 1.0) \
        == '{"groups":[]}\n'


def test_pick_combines_both_eligible_kinds():
    # green->station ratio 4/2 = 2; roof->lst ratio 3/1 = 3; together 5.
    group = _group_payload(driver_plan(_one_region(), _actions(), 10.0))
    assert group["pick"] == ["green", "roof"]
    assert group["cost"] == 3.0
    assert group["priority"] == pytest.approx(5.0)


def test_only_eligible_factor_kinds_are_candidates():
    # material maps to morph, whose item is not eligible; even a generous
    # budget never selects it.
    actions = _actions(material=("morph", 0.5))
    group = _group_payload(driver_plan(_one_region(), actions, 100.0))
    assert "material" not in group["pick"]
    assert group["pick"] == ["green", "roof"]


def test_limit_is_inclusive():
    # station scores 10 at cost 2 (ratio 5); lst scores 3 at cost 1 (ratio 3).
    report = _one_region(station=(10.0, -2.0, -0.5, -0.2, True))
    # At exactly 2 only green (cost 2) fits together with nothing else;
    # at 1 green is unaffordable and roof wins.
    group = _group_payload(driver_plan(report, _actions(), 2.0))
    assert group["pick"] == ["green"]
    assert group["cost"] == 2.0
    group = _group_payload(driver_plan(report, _actions(), 1.0))
    assert group["pick"] == ["roof"]


def test_priority_dominates_cost():
    # With limit 1 only roof (cost 1, ratio 3) fits; green costs 2.
    group = _group_payload(driver_plan(_one_region(), _actions(), 1.0))
    assert group["pick"] == ["roof"]
    assert group["cost"] == 1.0
    assert group["priority"] == pytest.approx(3.0)


def test_zero_limit_selects_empty_set():
    group = _group_payload(driver_plan(_one_region(), _actions(), 0))
    assert group["pick"] == []
    assert group["cost"] == 0.0 and group["priority"] == 0.0


def test_no_eligible_items_selects_empty_set():
    report = _one_region(
        station=(0.0, -2.0, 0.0, 0.0, False),
        lst=(0.0, -5.0, 0.0, 0.0, False),
    )
    group = _group_payload(driver_plan(report, _actions(), 10.0))
    assert group["pick"] == []
    assert group["cost"] == 0.0 and group["priority"] == 0.0


def test_priority_tie_breaks_by_cost_then_pick():
    # Both eligible items score 2.0; green costs 1, roof costs 1 so
    # priorities tie at 2.0 and costs tie at 1.0; lexicographic pick wins.
    report = _one_region(lst=(2.0, -5.0, -0.4, -0.1, True))
    actions = _actions(green=("station", 1.0), roof=("lst", 1.0))
    group = _group_payload(driver_plan(report, actions, 1.0))
    assert group["pick"] == ["green"]
    assert group["cost"] == 1.0
    # With equal priority but different cost, the cheaper single wins.
    actions = _actions(green=("station", 1.0), roof=("lst", 2.0))
    group = _group_payload(driver_plan(report, actions, 2.0))
    assert group["pick"] == ["green"]
    assert group["cost"] == 1.0


def test_effects_accumulate_by_metric_and_range_sums_intervals():
    # station (uhi -2, low -0.5 high -0.2) + lst (energy -5, low -0.4
    # high -0.1); vent untouched.
    group = _group_payload(driver_plan(_one_region(), _actions(), 10.0))
    assert group["effects"] == {"uhi": -2.0, "energy": -5.0, "vent": 0.0}
    assert group["range"] == {"low": -0.9, "high": -0.3}


def test_six_decimals_and_no_negative_zero():
    group = _group_payload(driver_plan(_one_region(), _actions(), 0))
    text = json.dumps(group, separators=(",", ":"))
    assert "-0.000000" not in text
    out = driver_plan(_one_region(), _actions(), 10.0)
    assert '"cost":3.000000' in out
    assert '"effects":{"uhi":-2.000000,"energy":-5.000000,"vent":0.000000}' in out


# --- validation -------------------------------------------------------------

def test_type_errors():
    report = _one_region()
    actions = _actions()
    for bad in (1, None, [], True, {}):
        with pytest.raises(TypeError):
            driver_plan(bad, actions, 10.0)
    for bad in ([], None, 42, "x"):
        with pytest.raises(TypeError):
            driver_plan(report, bad, 10.0)


def test_type_errors_take_precedence():
    with pytest.raises(TypeError):
        driver_plan(1, [], 0)


@pytest.mark.parametrize(
    "actions",
    [
        {},
        {"green": ("station", 1.0), "roof": ("lst", 1.0)},
        dict(_actions(), extra=("morph", 1.0)),
        {"Green": ("station", 1.0), "roof": ("lst", 1.0),
         "material": ("morph", 1.0)},
    ],
)
def test_actions_key_set_must_be_exact(actions):
    with pytest.raises(ValueError):
        driver_plan(_one_region(), actions, 10.0)


@pytest.mark.parametrize(
    "value",
    [
        ["station", 1.0],
        ("station",),
        ("station", 1.0, 2.0),
        "x",
        None,
        ("station", "1.0"),
        ("station", 0),
        ("station", 0.0),
        ("station", -1.0),
        ("station", True),
        ("station", float("nan")),
        ("station", float("inf")),
        (1, 1.0),
        ("nope", 1.0),
        ("STATION", 1.0),
    ],
)
def test_bad_action_values_rejected(value):
    with pytest.raises(ValueError):
        driver_plan(_one_region(), dict(_actions(), green=value), 10.0)


def test_duplicate_factors_rejected():
    actions = _actions(
        green=("station", 1.0), roof=("station", 2.0),
        material=("morph", 3.0),
    )
    with pytest.raises(ValueError):
        driver_plan(_one_region(), actions, 10.0)


@pytest.mark.parametrize("limit", [-1, -0.1, True, False, "1", None,
                                   float("nan"), float("inf"),
                                   float("-inf"), [1.0]])
def test_bad_limit_rejected(limit):
    with pytest.raises(ValueError):
        driver_plan(_one_region(), _actions(), limit)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json\n",
        '{"alpha":0.050000,"groups":[]}',
        '{"alpha":0.050000,"groups":[]}\n\n',
        '{"groups":[],"alpha":0.050000}\n',
        '{"alpha":0.05,"groups":[]}\n',
        '{"groups":[]}\n',
    ],
)
def test_malformed_report_value_error(raw):
    with pytest.raises(ValueError):
        driver_plan(raw, _actions(), 10.0)


def test_report_rank_gap_rejected():
    good = _one_region()
    bad = good.replace('"rank":4', '"rank":5')
    with pytest.raises(ValueError):
        driver_plan(bad, _actions(), 10.0)


def test_report_duplicate_factor_rejected():
    # Rewrite the cover item's factor to station; factors must be unique.
    tokens = [
        _item(1, "station", "uhi", 4.0, -0.5, -0.2, 0.01, 4.0, True),
        _item(2, "lst", "energy", 3.0, -0.4, -0.1, 0.01, 3.0, True),
        _item(3, "morph", "vent", 0.0, 0.0, 0.0, 1.0, 0.0, False),
        _item(4, "station", "uhi", 0.0, 0.0, 0.0, 1.0, 0.0, False),
    ]
    with pytest.raises(ValueError):
        driver_plan(_report(_group("region", tokens)), _actions(), 10.0)


def test_report_item_key_set_rejected():
    good = _one_region()
    bad = good.replace(',"eligible":true', "")
    assert bad != good
    with pytest.raises(ValueError):
        driver_plan(bad, _actions(), 10.0)


# --- genuine producer chain -------------------------------------------------

def test_genuine_driver_link_chain():
    def s_item(factor, low, high, stable, direction="down"):
        return (
            '{"factor":' + json.dumps(factor)
            + ',"direction":' + json.dumps(direction)
            + ',"frequency":1.000000,"consistency":1.000000,'
            + '"significant":1.000000,'
            + f'"low":{low:.6f},"high":{high:.6f},'
            + f'"stable":{str(stable).lower()},"conflict":false,'
            + '"cross_zero":false}'
        )

    def i_item(metric, change, q, reject):
        post = change
        return (
            '{"by":"region","metric":' + json.dumps(metric)
            + f',"n":4,"base":0.000000,"post":{post:.6f},'
            + f'"change":{change:.6f},"p":0.100000,"q":{q:.6f},'
            + f'"reject":{str(reject).lower()}}}'
        )

    s = (
        '{"alpha":0.050000,"groups":[{"by":"region","items":['
        + ",".join([
            s_item("station", -0.5, -0.2, True),
            s_item("lst", -0.4, -0.1, True),
            s_item("morph", 0.0, 0.0, False, "flat"),
            s_item("cover", 0.0, 0.0, False, "flat"),
        ])
        + "]}]}\n"
    )
    i = (
        '{"alpha":0.050000,"items":['
        + i_item("uhi", -2.0, 0.01, True) + ","
        + i_item("energy", -5.0, 0.20, False) + ","
        + i_item("vent", 3.0, 0.01, True)
        + "]}\n"
    )
    report = driver_link(s, i, _LINKS)
    actions = {
        "green": ("station", 2.0),
        "roof": ("lst", 1.0),
        "material": ("morph", 1.0),
    }
    # station eligible score 1.98; lst does not reject so is ineligible;
    # morph unstable. Only green survives, and it fits any non-zero budget.
    group = _group_payload(driver_plan(report, actions, 5.0))
    assert group["pick"] == ["green"]
    assert group["cost"] == 2.0
    assert group["priority"] == pytest.approx(1.98 / 2.0)
    assert group["effects"] == {"uhi": -2.0, "energy": 0.0, "vent": 0.0}
    assert group["range"] == {"low": -0.5, "high": -0.2}


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.driver_plan is urban_micro.driver_plan
    assert "driver_plan" in urban_micro.__all__
