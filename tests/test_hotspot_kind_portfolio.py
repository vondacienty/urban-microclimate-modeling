"""Tests for urban_micro.hotspot_kind_portfolio."""

import json
from decimal import Decimal

import pytest

from urban_micro import hotspot_kind_priority, hotspot_kind_portfolio
from urban_micro import uhi as _uhi


def _f6(value) -> str:
    return f"{Decimal(str(value)):.6f}"


def _item(kind, change, q, cost, score, eligible, rank, *, n=1):
    return (
        '{"kind":' + json.dumps(kind)
        + f',"n":{n},"change":{_f6(change)},"q":{_f6(q)},'
        + f'"cost":{_f6(cost)},"score":{_f6(score)},'
        + f'"eligible":{"true" if eligible else "false"},"rank":{rank}'
        + "}"
    )


def _group(by, *items):
    return '{"by":' + json.dumps(by) + ',"items":[' + ",".join(items) + "]}"


def _report(*groups, alpha=0.05):
    return (
        f'{{"alpha":{_f6(alpha)},"groups":[' + ",".join(groups) + "]}\n"
    )


# A region with three kinds: green and roof eligible (scores 2 and 1),
# material non-eligible.
def _three_kind_report():
    return _report(
        _group(
            "region",
            _item("green", -2.0, 0.01, 1.0, 2.0, True, 1),
            _item("roof", -1.0, 0.02, 2.0, 0.5, True, 2),
            _item("material", 0.5, 0.9, 3.0, 0.0, False, 3),
        )
    )


# --- canonical output shape -------------------------------------------------

def test_basic_selection_and_key_order():
    report = _three_kind_report()
    out = hotspot_kind_portfolio(
        report, {"region": 3}, {"region": ["green", "roof", "material"]}
    )
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    group = payload["groups"][0]
    assert list(group) == [
        "by", "budget", "cost", "remaining", "score", "pick", "skip",
    ]
    # green (cost 1, score 2) + roof (cost 2, score 0.5) fit budget 3.
    assert group["by"] == "region"
    assert group["pick"] == ["green", "roof"]
    assert group["budget"] == 3.0
    assert group["cost"] == 3.0
    assert group["remaining"] == 0.0
    assert group["score"] == 2.5
    assert group["skip"] == [{"kind": "material", "reason": "q"}]
    assert list(group["skip"][0]) == ["kind", "reason"]


def test_numbers_render_with_six_decimals():
    report = _three_kind_report()
    out = hotspot_kind_portfolio(
        report, {"region": 3}, {"region": ["green", "roof", "material"]}
    )
    for token in (
        '"budget":3.000000',
        '"cost":3.000000',
        '"remaining":0.000000',
        '"score":2.500000',
    ):
        assert token in out


def test_region_before_window_and_pick_skip_sorted_by_kind():
    # The report keeps region before window canonically; output must too.
    report = _report(
        _group(
            "region",
            _item("green", -2.0, 0.01, 1.0, 2.0, True, 1),
            _item("roof", -1.0, 0.02, 2.0, 0.5, True, 2),
            _item("material", 0.5, 0.9, 3.0, 0.0, False, 3),
        ),
        _group("window", _item("roof", -3.0, 0.01, 4.0, 0.6, True, 1)),
    )
    out = json.loads(
        hotspot_kind_portfolio(
            report,
            {"region": 3, "window": 4},
            {
                "region": ["green", "roof", "material"],
                "window": ["roof"],
            },
        )
    )
    assert [g["by"] for g in out["groups"]] == ["region", "window"]


def test_empty_report_requires_empty_dicts():
    empty = '{"alpha":0.050000,"groups":[]}\n'
    assert hotspot_kind_portfolio(empty, {}, {}) == (
        '{"alpha":0.050000,"groups":[]}\n'
    )
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(empty, {"region": 1}, {"region": []})
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(empty, {}, {"region": []})


# --- subset selection and tie-breaks ----------------------------------------

def test_chooses_descending_score_then_ascending_cost():
    # Equal scores; green costs 2, roof costs 1; budget 1 forces roof.
    report = _report(
        _group(
            "region",
            _item("green", -2.0, 0.01, 2.0, 1.0, True, 1),
            _item("roof", -1.0, 0.02, 1.0, 1.0, True, 2),
        )
    )
    out = json.loads(
        hotspot_kind_portfolio(
            report, {"region": 1}, {"region": ["green", "roof"]}
        )
    )
    assert out["groups"][0]["pick"] == ["roof"]


def test_lexicographic_kind_list_is_final_tie_break():
    # Equal score and equal unit cost; budget 1 admits one; green < roof.
    report = _report(
        _group(
            "region",
            _item("green", -1.0, 0.01, 1.0, 1.0, True, 1),
            _item("roof", -1.0, 0.02, 1.0, 1.0, True, 2),
        )
    )
    out = json.loads(
        hotspot_kind_portfolio(
            report, {"region": 1}, {"region": ["green", "roof"]}
        )
    )
    assert out["groups"][0]["pick"] == ["green"]


def test_empty_subset_always_feasible():
    report = _report(
        _group("region", _item("green", -2.0, 0.01, 5.0, 2.0, True, 1))
    )
    group = json.loads(
        hotspot_kind_portfolio(
            report, {"region": 3}, {"region": ["green"]}
        )
    )["groups"][0]
    assert group["pick"] == []
    assert group["cost"] == 0.0
    assert group["remaining"] == 3.0
    assert group["score"] == 0.0
    assert group["skip"] == [{"kind": "green", "reason": "b"}]


def test_zero_budget_and_exact_fit():
    report = _report(
        _group("region", _item("green", -2.0, 0.01, 2.0, 2.0, True, 1))
    )
    zero = json.loads(
        hotspot_kind_portfolio(
            report, {"region": 0}, {"region": ["green"]}
        )
    )["groups"][0]
    assert zero["pick"] == []
    assert zero["remaining"] == 0.0
    assert zero["skip"] == [{"kind": "green", "reason": "b"}]

    exact = json.loads(
        hotspot_kind_portfolio(
            report, {"region": 2}, {"region": ["green"]}
        )
    )["groups"][0]
    assert exact["pick"] == ["green"]
    assert exact["remaining"] == 0.0


def test_empty_allow_picks_nothing():
    report = _three_kind_report()
    group = json.loads(
        hotspot_kind_portfolio(report, {"region": 10}, {"region": []})
    )["groups"][0]
    assert group["pick"] == []
    assert group["cost"] == 0.0
    assert group["remaining"] == 10.0
    # Eligible excluded kinds are "a"; non-eligible kinds stay "q".
    assert group["skip"] == [
        {"kind": "green", "reason": "a"},
        {"kind": "material", "reason": "q"},
        {"kind": "roof", "reason": "a"},
    ]


# --- skip reasons -----------------------------------------------------------

def test_skip_reasons_q_a_b_d():
    # green: eligible but excluded from allow -> a
    # roof: eligible, allowed, affordable alongside green's stand-in only
    #       via budget pressure -> b
    # material: non-eligible -> q (even though allowed)
    report = _three_kind_report()
    group = json.loads(
        hotspot_kind_portfolio(
            report,
            {"region": 2},
            {"region": ["roof", "material"]},
        )
    )["groups"][0]
    # roof alone costs 2 and scores 0.5; green is excluded.
    assert group["pick"] == ["roof"]
    assert group["skip"] == [
        {"kind": "green", "reason": "a"},
        {"kind": "material", "reason": "q"},
    ]

    # budget 1: roof (cost 2) cannot be added to the empty pick -> b;
    # green excluded -> a; material non-eligible -> q.
    group = json.loads(
        hotspot_kind_portfolio(
            report,
            {"region": 1},
            {"region": ["roof", "material"]},
        )
    )["groups"][0]
    assert group["pick"] == []
    assert group["skip"] == [
        {"kind": "green", "reason": "a"},
        {"kind": "material", "reason": "q"},
        {"kind": "roof", "reason": "b"},
    ]


def test_skip_reason_d_for_affordable_dominated_kind():
    # roof is eligible, allowed and affordable, but adds no score; the
    # cheaper equal-score empty choice dominates it -> reason d.
    report = _report(
        _group(
            "region",
            _item("green", -1.0, 0.01, 1.0, 1.0, True, 1),
            _item("roof", 0.0, 0.02, 1.0, 0.0, True, 2),
        )
    )
    group = json.loads(
        hotspot_kind_portfolio(
            report, {"region": 2}, {"region": ["green", "roof"]}
        )
    )["groups"][0]
    assert group["pick"] == ["green"]
    assert group["skip"] == [{"kind": "roof", "reason": "d"}]


# --- input validation -------------------------------------------------------

def test_type_errors():
    report = _three_kind_report()
    with pytest.raises(TypeError):
        hotspot_kind_portfolio(1, {"region": 1}, {"region": []})
    with pytest.raises(TypeError):
        hotspot_kind_portfolio(None, {"region": 1}, {"region": []})
    with pytest.raises(TypeError):
        hotspot_kind_portfolio(report, [], {"region": []})
    with pytest.raises(TypeError):
        hotspot_kind_portfolio(report, {"region": 1}, [])


def test_budget_key_set_must_match_report_bys():
    report = _three_kind_report()
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(report, {}, {"region": []})
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(
            report,
            {"region": 1, "window": 1},
            {"region": [], "window": []},
        )


def test_allow_key_set_must_match_report_bys():
    report = _three_kind_report()
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(report, {"region": 1}, {})
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(
            report, {"region": 1}, {"region": [], "window": []}
        )


@pytest.mark.parametrize("bad", [-1, -0.5, float("nan"), float("inf"), True, "1"])
def test_budget_values_rejected(bad):
    report = _three_kind_report()
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(
            report, {"region": bad}, {"region": []}
        )


def test_allow_value_must_be_list():
    report = _three_kind_report()
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(
            report, {"region": 1}, {"region": ("green",)}
        )


@pytest.mark.parametrize(
    "bad",
    [
        ["water"],
        ["green", "green"],
        [1],
        ["green", 2],
    ],
)
def test_allow_values_rejected(bad):
    report = _three_kind_report()
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(
            report, {"region": 1}, {"region": bad}
        )


# --- report must be canonical hotspot_kind_priority output ------------------

@pytest.mark.parametrize(
    "raw,budgets,allow",
    [
        ('{"alpha":0.05,"groups":[]}\n', {}, {}),
        ('{"alpha":0.050000,"groups":[]}', {}, {}),
        ('{"alpha":0.050000,"groups":[]}\n\n', {}, {}),
        ('{"alpha":0.050000,"groups":[]} ', {}, {}),
        # window before region is non-canonical when both are present.
        (
            '{"alpha":0.050000,"groups":['
            '{"by":"window","items":['
            '{"kind":"roof","n":1,"change":-1.000000,"q":0.010000,'
            '"cost":1.000000,"score":1.000000,"eligible":true,"rank":1}]},'
            '{"by":"region","items":['
            '{"kind":"green","n":1,"change":-2.000000,"q":0.010000,'
            '"cost":1.000000,"score":2.000000,"eligible":true,"rank":1}]}]}\n',
            {"window": 1, "region": 1},
            {"window": [], "region": []},
        ),
        # rank mismatch.
        (
            '{"alpha":0.050000,"groups":[{"by":"region","items":['
            '{"kind":"green","n":1,"change":-1.000000,"q":0.010000,'
            '"cost":1.000000,"score":1.000000,"eligible":true,"rank":2}]}]}\n',
            {"region": 1},
            {"region": []},
        ),
        # duplicate kind.
        (
            '{"alpha":0.050000,"groups":[{"by":"region","items":['
            '{"kind":"green","n":1,"change":-1.000000,"q":0.010000,'
            '"cost":1.000000,"score":1.000000,"eligible":true,"rank":1},'
            '{"kind":"green","n":1,"change":-2.000000,"q":0.020000,'
            '"cost":1.000000,"score":2.000000,"eligible":true,"rank":2}]}]}\n',
            {"region": 1},
            {"region": []},
        ),
        # non-eligible item with a nonzero score is inconsistent.
        (
            '{"alpha":0.050000,"groups":[{"by":"region","items":['
            '{"kind":"green","n":1,"change":0.500000,"q":0.900000,'
            '"cost":1.000000,"score":0.500000,"eligible":false,"rank":1}]}]}\n',
            {"region": 1},
            {"region": []},
        ),
    ],
)
def test_noncanonical_report_raises_value_error(raw, budgets, allow):
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(raw, budgets, allow)


@pytest.mark.parametrize("raw", ["", "not json"])
def test_garbage_report_raises_value_error(raw):
    with pytest.raises(ValueError):
        hotspot_kind_portfolio(raw, {}, {})


def test_report_type_error_takes_precedence():
    # A non-string report is a TypeError even when the dicts are bad too.
    with pytest.raises(TypeError):
        hotspot_kind_portfolio(123, [], [])


# --- end-to-end through hotspot_kind_priority -------------------------------

def _summary_item(kind, n, change, p, q, reject):
    return (
        '{"kind":' + json.dumps(kind)
        + f',"n":{n},"change":{_f6(change)},"p":{_f6(p)},"q":{_f6(q)},'
        + f'"reject":{"true" if reject else "false"}}}'
    )


def test_roundtrip_through_hotspot_kind_priority():
    summary = (
        '{"alpha":0.050000,"groups":['
        '{"by":"region","items":['
        + _summary_item("green", 2, -2.0, 0.003906, 0.007812, True)
        + ","
        + _summary_item("material", 1, 0.5, 1.0, 1.0, False)
        + "]},"
        '{"by":"window","items":['
        + _summary_item("roof", 1, -3.0, 0.5, 0.5, True)
        + "]}]}\n"
    )
    priority = hotspot_kind_priority(
        summary, {"green": 2.0, "roof": 4.0, "material": 1.0}
    )
    out = json.loads(
        hotspot_kind_portfolio(
            priority,
            {"region": 3, "window": 2},
            {"region": ["green", "material"], "window": ["roof"]},
        )
    )
    region, window = out["groups"]
    assert region["pick"] == ["green"]
    assert region["cost"] == 2.0
    assert region["remaining"] == 1.0
    assert region["score"] == 1.0
    assert region["skip"] == [{"kind": "material", "reason": "q"}]
    # roof costs 4 against a window budget of 2 -> empty pick, reason b.
    assert window["pick"] == []
    assert window["remaining"] == 2.0
    assert window["skip"] == [{"kind": "roof", "reason": "b"}]


def test_exported_from_package_root():
    assert _uhi.hotspot_kind_portfolio is hotspot_kind_portfolio
    assert "hotspot_kind_portfolio" in __import__("urban_micro").__all__
