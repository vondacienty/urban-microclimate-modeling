"""Tests for urban_micro.region_attr_layer_report: categorical layer
stratification of repeated region_attr reports, pooling each layer's
factor effects with an exact 2**n sign-flip p-value, one global
Benjamini-Hochberg pass and q/reject per item. Also pins the
region_attr rank-order alignment fix (no (a, b) re-sort)."""

import json
from fractions import Fraction

import pytest

from urban_micro import (
    pareto_region_compare,
    pareto_region_stability,
    region_attr,
    region_attr_layer_report,
)
from urban_micro import uhi as _uhi

_GREEN = ("green",)
_ROOF = ("roof",)
_FACTORS = ("weather", "morph", "cover")
_LAYERS = ("weather", "morph", "cover")


def _row(r, w, p, u, e=0.0, v=0.0):
    return (r, w, 0, p, u, e, v)


def _stability(rows, *, alpha=0.05):
    return pareto_region_stability(
        {f"t{i}": pareto_region_compare(rows, alpha=alpha)
         for i in range(2)}
    )


def _n3_rows(u_c=9.0):
    return [
        _row("a", "a1", _GREEN, 0.0),
        _row("b", "b1", _GREEN, 3.0),
        _row("c", "c1", _GREEN, u_c),
    ]


_N3_DRIVERS = {
    "a": (0.0, 5.0, 9.0),
    "b": (1.0, 5.0, 2.0),
    "c": (3.0, 5.0, 1.0),
}


def _reports(spec, *, alpha=1.0):
    """spec: list of uhi values for the c region; name -> region_attr."""
    return {
        f"p{index}": region_attr(
            _stability(_n3_rows(u_c), alpha=alpha), _N3_DRIVERS
        )
        for index, u_c in enumerate(spec)
    }


def _layers(triples):
    return {f"p{index}": triple for index, triple in enumerate(triples)}


# --- exact Fraction reference ----------------------------------------------

def _reference(reports, layers):
    parsed = {name: json.loads(report) for name, report in reports.items()}
    alpha = Fraction(str(next(iter(parsed.values()))["alpha"]))
    identities = [
        (tuple(group["pick"]), item["factor"])
        for group in next(iter(parsed.values()))["groups"]
        for item in group["items"]
    ]
    effects = {}
    for name, payload in parsed.items():
        for group in payload["groups"]:
            pick = tuple(group["pick"])
            for item in group["items"]:
                effects[(name, pick, item["factor"])] = Fraction(
                    str(item["effect"])
                )

    names = list(reports)
    tests = {}
    for layer_index, layer in enumerate(_LAYERS):
        labels = sorted({layers[name][layer_index] for name in names})
        for label in labels:
            members = [
                name for name in names
                if layers[name][layer_index] == label
            ]
            n = len(members)
            for pick, factor in sorted(
                identities,
                key=lambda identity: (
                    identity[0], _FACTORS.index(identity[1])
                ),
            ):
                values = [
                    effects[(name, pick, factor)] for name in members
                ]
                total = sum(values, Fraction(0))
                mean = total / n
                hits = 0
                for mask in range(1 << n):
                    signed = sum(
                        (-value if (mask >> i) & 1 else value)
                        for i, value in enumerate(values)
                    )
                    if abs(signed) >= abs(total):
                        hits += 1
                p_value = Fraction(hits, 1 << n)
                tests[(layer_index, label, pick, factor)] = (
                    n, mean, p_value,
                )

    rows = [
        (p_value, layer_index, label, pick, _FACTORS.index(factor))
        for (layer_index, label, pick, factor), (
            _n, _mean, p_value
        ) in tests.items()
    ]
    rows.sort(key=lambda row: (
        row[0], row[1], row[2], row[3], row[4],
    ))
    tested = len(rows)
    q_values = {}
    running = Fraction(1)
    for rank in range(tested, 0, -1):
        p_value, layer_index, label, pick, factor_index = rows[rank - 1]
        running = min(running, Fraction(tested) * p_value / rank)
        q_values[(layer_index, label, pick, _FACTORS[factor_index])] = min(
            Fraction(1), running
        )

    groups = []
    for layer_index, layer in enumerate(_LAYERS):
        labels = sorted({layers[name][layer_index] for name in names})
        for label in labels:
            items = []
            for pick, factor in sorted(
                identities,
                key=lambda identity: (
                    identity[0], _FACTORS.index(identity[1])
                ),
            ):
                n, mean, p_value = tests[
                    (layer_index, label, pick, factor)
                ]
                q_value = q_values[(layer_index, label, pick, factor)]
                items.append(
                    {
                        "pick": list(pick),
                        "factor": factor,
                        "n": n,
                        "mean": float(mean),
                        "p": float(p_value),
                        "q": float(q_value),
                        "reject": q_value <= alpha,
                    }
                )
            groups.append({"by": layer, "key": label, "items": items})
    return {"alpha": float(alpha), "groups": groups}


def _assert_matches_reference(reports, layers):
    out = region_attr_layer_report(reports, layers)
    payload = json.loads(out)
    expected = _reference(reports, layers)
    assert payload["alpha"] == expected["alpha"]
    assert len(payload["groups"]) == len(expected["groups"])
    for group, exp_group in zip(payload["groups"], expected["groups"]):
        assert group["by"] == exp_group["by"]
        assert group["key"] == exp_group["key"]
        assert len(group["items"]) == len(exp_group["items"])
        for item, exp in zip(group["items"], exp_group["items"]):
            assert list(item) == [
                "pick", "factor", "n", "mean", "p", "q", "reject",
            ]
            assert item["pick"] == exp["pick"]
            assert item["factor"] == exp["factor"]
            assert item["n"] == exp["n"]
            assert isinstance(item["n"], int) and not isinstance(
                item["n"], bool
            )
            assert item["reject"] is exp["reject"]
            assert abs(item["mean"] - exp["mean"]) < 5.1e-7
            assert abs(item["p"] - exp["p"]) < 5.1e-7
            assert abs(item["q"] - exp["q"]) < 5.1e-7
    return out


# --- canonical output -------------------------------------------------------

def test_canonical_shape_key_order_and_group_sort():
    reports = _reports([9.0, 3.0, 9.0, 3.0])
    layers = _layers([
        ("hot", "dense", "low"),
        ("cold", "dense", "high"),
        ("hot", "sparse", "low"),
        ("cold", "sparse", "high"),
    ])
    out = _assert_matches_reference(reports, layers)
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    assert [(g["by"], g["key"]) for g in payload["groups"]] == [
        ("weather", "cold"), ("weather", "hot"),
        ("morph", "dense"), ("morph", "sparse"),
        ("cover", "high"), ("cover", "low"),
    ]
    for group in payload["groups"]:
        assert list(group) == ["by", "key", "items"]
        assert [tuple(item["pick"]) for item in group["items"]] == [
            _GREEN
        ] * 3
        assert [item["factor"] for item in group["items"]] == list(
            _FACTORS
        )
        for item in group["items"]:
            assert list(item) == [
                "pick", "factor", "n", "mean", "p", "q", "reject",
            ]
            assert isinstance(item["reject"], bool)


def test_n_is_pooled_report_count():
    reports = _reports([9.0, 9.0, 3.0, 3.0, 9.0])
    layers = _layers([
        ("hot", "a", "x"), ("hot", "a", "x"), ("hot", "a", "x"),
        ("cold", "a", "x"), ("cold", "a", "x"),
    ])
    payload = json.loads(region_attr_layer_report(reports, layers))
    by_key = {(g["by"], g["key"]): g for g in payload["groups"]}
    assert {item["n"] for item in by_key[("weather", "hot")]["items"]} == {3}
    assert {item["n"] for item in by_key[("weather", "cold")]["items"]} == {2}
    # morph/cover slots label every report "a"/"x", so they pool all five.
    assert {item["n"] for item in by_key[("morph", "a")]["items"]} == {5}
    assert {item["n"] for item in by_key[("cover", "x")]["items"]} == {5}


def test_items_pool_by_the_slot_label_not_report_identity():
    reports = _reports([9.0, 3.0, 6.0])
    layers = _layers([
        ("hot", "dense", "low"),
        ("cold", "dense", "low"),
        ("hot", "sparse", "high"),
    ])
    _assert_matches_reference(reports, layers)


def test_matches_reference_across_poolings():
    reports = _reports([9.0, 3.0, 6.0, 1.0, 9.0])
    layers = _layers([
        ("hot", "dense", "low"),
        ("cold", "dense", "low"),
        ("hot", "sparse", "high"),
        ("cold", "sparse", "high"),
        ("hot", "dense", "high"),
    ])
    _assert_matches_reference(reports, layers)


def test_identical_effects_give_half_flip_p():
    # Two identical reports pooled: equal non-zero effects give p = 2/4
    # (only the all-plus and all-minus flips clear the observed mean);
    # the constant morph factor has effect 0 and p = 1. Alpha 1 keeps the
    # source significant flags set (n=3 cannot reject at 0.05).
    reports = _reports([9.0, 9.0], alpha=1.0)
    layers = _layers([("hot", "a", "x"), ("hot", "a", "x")])
    items = {
        item["factor"]: item
        for group in json.loads(
            region_attr_layer_report(reports, layers)
        )["groups"]
        for item in group["items"]
    }
    assert items["weather"]["n"] == 2
    assert items["weather"]["p"] == 0.5
    assert items["cover"]["p"] == 0.5
    assert items["morph"]["p"] == 1.0
    assert items["morph"]["mean"] == 0.0
    assert items["morph"]["q"] == 1.0
    # At alpha 1 even q == 1 rejects.
    assert items["morph"]["reject"] is True


def test_alpha_one_rejects_everything():
    reports = _reports([9.0, 3.0], alpha=1.0)
    layers = _layers([("hot", "a", "x"), ("cold", "b", "y")])
    payload = json.loads(region_attr_layer_report(reports, layers))
    assert all(
        item["reject"]
        for group in payload["groups"]
        for item in group["items"]
    )


def test_two_picks_sort_ascending():
    rows = [
        _row("a", "a1", _GREEN, 1.0),
        _row("b", "b1", _GREEN, 9.0),
        _row("a", "x1", _ROOF, 2.0),
        _row("b", "x2", _ROOF, 8.0),
    ]
    drivers = {"a": (1.0, 2.0, 3.0), "b": (4.0, 1.0, 0.0)}
    report = region_attr(_stability(rows), drivers)
    reports = {"p0": report, "p1": report}
    layers = {"p0": ("k", "k", "k"), "p1": ("k", "k", "k")}
    for group in json.loads(
        region_attr_layer_report(reports, layers)
    )["groups"]:
        assert [item["pick"] for item in group["items"]] == [
            ["green"], ["green"], ["green"],
            ["roof"], ["roof"], ["roof"],
        ]


def test_sixteen_reports_accepted():
    one = _reports([9.0])["p0"]
    reports = {f"p{i}": one for i in range(16)}
    layers = {f"p{i}": ("a", "b", "c") for i in range(16)}
    payload = json.loads(region_attr_layer_report(reports, layers))
    assert len(payload["groups"]) == 3
    assert {
        item["n"]
        for group in payload["groups"]
        for item in group["items"]
    } == {16}


def test_no_negative_zero():
    reports = _reports([9.0, 3.0])
    layers = _layers([("hot", "a", "x"), ("cold", "b", "y")])
    assert "-0.000000" not in region_attr_layer_report(reports, layers)


def test_empty_identity_keeps_label_strata_with_empty_items():
    empty = region_attr(
        pareto_region_stability(
            {f"t{i}": pareto_region_compare([]) for i in range(2)}
        ),
        {},
    )
    out = region_attr_layer_report(
        {"p0": empty, "p1": empty},
        {"p0": ("hot", "dense", "low"), "p1": ("hot", "dense", "low")},
    )
    assert json.loads(out) == {
        "alpha": 0.05,
        "groups": [
            {"by": "weather", "key": "hot", "items": []},
            {"by": "morph", "key": "dense", "items": []},
            {"by": "cover", "key": "low", "items": []},
        ],
    }
    assert out.endswith("\n") and not out.endswith("\n\n")


# --- validation -------------------------------------------------------------

def test_type_errors():
    reports = _reports([9.0, 3.0])
    layers = _layers([("hot", "a", "x"), ("cold", "b", "y")])
    with pytest.raises(TypeError):
        region_attr_layer_report([], layers)
    with pytest.raises(TypeError):
        region_attr_layer_report(None, layers)
    with pytest.raises(TypeError):
        region_attr_layer_report(reports, [])
    with pytest.raises(TypeError):
        region_attr_layer_report(reports, None)


@pytest.mark.parametrize("count", [0, 1, 17])
def test_reports_count_bounds(count):
    one = _reports([9.0])["p0"]
    reports = {f"p{i}": one for i in range(count)}
    layers = {f"p{i}": ("a", "b", "c") for i in range(count)}
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, layers)


def test_report_keys_non_empty_strings():
    reports = _reports([9.0, 3.0])
    layers = _layers([("a", "b", "c"), ("a", "b", "c")])
    reports[""] = reports["p0"]
    layers[""] = ("a", "b", "c")
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, layers)


def test_layers_keys_must_match_reports_exactly():
    reports = _reports([9.0, 3.0])
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, {"p0": ("a", "b", "c")})
    with pytest.raises(ValueError):
        region_attr_layer_report(
            reports,
            {
                "p0": ("a", "b", "c"),
                "p1": ("a", "b", "c"),
                "p2": ("a", "b", "c"),
            },
        )


@pytest.mark.parametrize("bad", [
    ["hot", "a", "x"],
    ("hot", "a"),
    ("hot", "a", "x", "y"),
    ("hot", "", "x"),
    ("hot", 7, "x"),
    ("hot", None, "x"),
])
def test_layers_values_must_be_three_non_empty_strings(bad):
    reports = _reports([9.0, 3.0])
    layers = _layers([("hot", "a", "x"), ("cold", "b", "y")])
    layers["p0"] = bad
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, layers)


def test_mixed_alpha_rejected():
    loose = _reports([9.0], alpha=1.0)["p0"]
    strict = _reports([9.0], alpha=0.05)["p0"]
    with pytest.raises(ValueError):
        region_attr_layer_report(
            {"p0": loose, "p1": strict},
            {"p0": ("a", "b", "c"), "p1": ("a", "b", "c")},
        )


def test_different_identity_sets_rejected():
    green = _reports([9.0])["p0"]
    roof_rows = [
        _row("a", "a1", _ROOF, 1.0),
        _row("b", "b1", _ROOF, 9.0),
        _row("c", "c1", _ROOF, 2.0),
    ]
    roof = region_attr(_stability(roof_rows), _N3_DRIVERS)
    with pytest.raises(ValueError):
        region_attr_layer_report(
            {"p0": green, "p1": roof},
            {"p0": ("a", "b", "c"), "p1": ("a", "b", "c")},
        )


@pytest.mark.parametrize("mutate", [
    lambda s: s.rstrip("\n"),
    lambda s: s + "\n",
    lambda s: s.replace('"factor":', '"factorX":', 1),
    lambda s: '{"groups":[],"alpha":1.000000}\n',
    lambda s: "not json\n",
])
def test_non_canonical_report_value_error(mutate):
    reports = _reports([9.0, 3.0])
    reports["p1"] = mutate(reports["p1"])
    layers = _layers([("a", "b", "c"), ("a", "b", "c")])
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, layers)


def test_stability_report_rejected():
    stability = _stability(_n3_rows())
    reports = {"p0": stability, "p1": stability}
    layers = {"p0": ("a", "b", "c"), "p1": ("a", "b", "c")}
    with pytest.raises(ValueError):
        region_attr_layer_report(reports, layers)


# --- region_attr rank-order alignment --------------------------------------

def test_uhi_records_follow_report_rank_order_not_pair_order():
    # Identical periods with uhis a=0, b=3, c=9 give the stability uhi
    # block the rank order (a,c), (b,c), (a,b) -- descending abs(mean)
    # -- which is not the ascending (a, b) pair order (a,b), (a,c),
    # (b,c). The correlation must align along the former.
    _alpha, groups = _uhi._region_attr_stability_output_parse(
        _stability(_n3_rows(9.0), alpha=1.0)
    )
    (pick, items) = groups.popitem()
    assert pick == _GREEN
    records = _uhi._region_attr_uhi_records(items)
    assert [(r[0], r[1], r[2]) for r in records] == [
        ("a", "c", 1), ("b", "c", 2), ("a", "b", 3),
    ]


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.region_attr_layer_report is urban_micro.region_attr_layer_report
    assert "region_attr_layer_report" in urban_micro.__all__
    assert "region_attr_layer_report" in _uhi.__all__
