"""Tests for urban_micro.pareto_region_stability: cross-period
stability aggregation over canonical pareto_region_compare reports."""

import json

import pytest

from urban_micro import pareto_region_compare, pareto_region_stability
from urban_micro import uhi as _uhi

_GREEN = ("green",)
_ROOF = ("roof",)


def _row(r, w, p, u, e=0.0, v=0.0):
    return (r, w, 0, p, u, e, v)


def _report(rows, *, alpha=0.05):
    return pareto_region_compare(rows, alpha=alpha)


def _payload(out):
    return json.loads(out)


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    r1 = _report([
        _row("a", "w1", _GREEN, 1.0),
        _row("a", "w2", _GREEN, 3.0),
        _row("b", "w3", _GREEN, 9.0),
    ])
    r2 = _report([
        _row("a", "x1", _GREEN, 2.0),
        _row("b", "x2", _GREEN, 8.0),
    ])
    out = pareto_region_stability({"t1": r1, "t2": r2})
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    data = _payload(out)
    assert list(data) == ["alpha", "groups"]
    group = data["groups"][0]
    assert list(group) == ["pick", "items"]
    assert group["pick"] == ["green"]
    assert len(group["items"]) == 3
    for item in group["items"]:
        assert list(item) == [
            "a", "b", "mean", "direction", "consistency",
            "significant", "rank",
        ]
        assert isinstance(item["rank"], int) and not isinstance(item["rank"], bool)


def test_mean_direction_consistency_significant():
    r1 = _report([
        _row("a", "w1", _GREEN, 1.0),
        _row("a", "w2", _GREEN, 3.0),
        _row("b", "w3", _GREEN, 9.0),
    ])
    r2 = _report([
        _row("a", "x1", _GREEN, 2.0),
        _row("b", "x2", _GREEN, 8.0),
    ])
    uhi = _payload(pareto_region_stability({"t1": r1, "t2": r2}))[
        "groups"
    ][0]["items"][0]
    # Period diffs: 2 - 9 = -7 and 2 - 8 = -6; mean -6.5.
    assert uhi["a"] == "a"
    assert uhi["b"] == "b"
    assert uhi["mean"] == -6.5
    assert uhi["direction"] == "down"
    assert uhi["consistency"] == 1.0
    assert uhi["significant"] == 0.0
    assert uhi["rank"] == 1


def test_metric_blocks_in_uhi_energy_vent_order():
    # Period 1 is uniformly 3 below for a, period 2 uniformly 3 above;
    # every metric block therefore has one pair with mean 0 and rank 1.
    r1 = _report([
        _row("a", "w1", _GREEN, 1.0, 1.0, 1.0),
        _row("b", "w2", _GREEN, 4.0, 4.0, 4.0),
    ])
    r2 = _report([
        _row("a", "x1", _GREEN, 4.0, 4.0, 4.0),
        _row("b", "x2", _GREEN, 1.0, 1.0, 1.0),
    ])
    items = _payload(
        pareto_region_stability({"t1": r1, "t2": r2})
    )["groups"][0]["items"]
    assert len(items) == 3
    for item in items:
        assert item["mean"] == 0.0
        assert item["direction"] == "flat"
        assert item["consistency"] == 0.0
        assert item["rank"] == 1
    assert {(it["a"], it["b"]) for it in items} == {("a", "b")}


def test_mean_zero_counts_exact_zero_diffs_and_reject_share():
    # alpha = 1 forces rejection; the two periods reverse sign so the
    # mean diff is zero while neither period diff is zero.
    r1 = _report([
        _row("a", "w1", _GREEN, 0.0),
        _row("a", "w2", _GREEN, 0.0),
        _row("b", "w3", _GREEN, 100.0),
        _row("b", "w4", _GREEN, 100.0),
    ], alpha=1.0)
    r2 = _report([
        _row("a", "x1", _GREEN, 100.0),
        _row("a", "x2", _GREEN, 100.0),
        _row("b", "x3", _GREEN, 0.0),
        _row("b", "x4", _GREEN, 0.0),
    ], alpha=1.0)
    uhi = _payload(
        pareto_region_stability({"t1": r1, "t2": r2})
    )["groups"][0]["items"][0]
    assert uhi["mean"] == 0.0
    assert uhi["direction"] == "flat"
    assert uhi["consistency"] == 0.0
    assert uhi["significant"] == 1.0


def test_partial_rejection_share():
    # At alpha 0.5 a fully separated 3v2 pair rejects (exact p 0.2,
    # global BH over three separated metrics), while an equal-valued
    # period does not; two of each over four periods gives 0.5.
    def strong(prefix):
        return [
            _row("a", f"{prefix}a1", _GREEN, 0.0, 0.0, 0.0),
            _row("a", f"{prefix}a2", _GREEN, 0.0, 0.0, 0.0),
            _row("a", f"{prefix}a3", _GREEN, 0.0, 0.0, 0.0),
            _row("b", f"{prefix}b1", _GREEN, 100.0, 100.0, 100.0),
            _row("b", f"{prefix}b2", _GREEN, 100.0, 100.0, 100.0),
        ]

    def flat(prefix):
        return [
            _row("a", f"{prefix}a1", _GREEN, 5.0, 5.0, 5.0),
            _row("a", f"{prefix}a2", _GREEN, 5.0, 5.0, 5.0),
            _row("a", f"{prefix}a3", _GREEN, 5.0, 5.0, 5.0),
            _row("b", f"{prefix}b1", _GREEN, 5.0, 5.0, 5.0),
            _row("b", f"{prefix}b2", _GREEN, 5.0, 5.0, 5.0),
        ]

    reports = {
        "t1": _report(strong("1"), alpha=0.5),
        "t2": _report(strong("2"), alpha=0.5),
        "t3": _report(flat("3"), alpha=0.5),
        "t4": _report(flat("4"), alpha=0.5),
    }
    uhi = _payload(pareto_region_stability(reports))["groups"][0]["items"][0]
    assert uhi["significant"] == 0.5


def test_ranks_sort_by_significant_then_consistency_then_abs_mean():
    # Three region pairs per period; window counts of 1 per side give
    # p = 1.0 so nothing rejects at alpha 0.05: significant ties at 0.
    def period(values):
        rows = []
        for region, value in values.items():
            rows.append(_row(region, f"{region}-w", _GREEN, float(value)))
        return _report(rows)

    # a-b negative everywhere (consistency 1, mean -6); a-c only
    # negative in one of two periods (consistency 0.5); b-c mixed but
    # larger mean than a-c so it wins only after consistency.
    t1 = period({"a": 0, "b": 4, "c": 10})
    t2 = period({"a": 0, "b": 8, "c": 2})
    uhis = _payload(
        pareto_region_stability({"t1": t1, "t2": t2})
    )["groups"][0]["items"][:3]
    # a-b: diffs -4, -8 -> mean -6, consistency 1 -> rank 1.
    # a-c: diffs -10, -2 -> mean -6, consistency 1 -> ties a-b on
    # significant/consistency/|mean| and a < b resolves a-b first.
    # b-c: diffs -6, 6 -> mean 0, consistency 0 -> last.
    assert [(it["a"], it["b"]) for it in uhis] == [
        ("a", "b"), ("a", "c"), ("b", "c"),
    ]
    assert [it["rank"] for it in uhis] == [1, 2, 3]
    assert uhis[1]["mean"] == -6.0
    assert uhis[2]["direction"] == "flat"


def test_groups_sort_by_ascending_pick():
    rows = [
        _row("a", "w1", _ROOF, 1.0),
        _row("b", "w2", _ROOF, 4.0),
        _row("a", "x1", _GREEN, 1.0),
        _row("b", "x2", _GREEN, 4.0),
    ]
    out = pareto_region_stability({"t1": _report(rows), "t2": _report(rows)})
    assert [g["pick"] for g in _payload(out)["groups"]] == [
        ["green"], ["roof"],
    ]


def test_negative_zero_normalized():
    r1 = _report([
        _row("a", "w1", _GREEN, 2.0),
        _row("b", "w2", _GREEN, 2.0),
    ])
    out = pareto_region_stability({"t1": r1, "t2": r1})
    assert '"mean":0.000000' in out
    assert "-0.000000" not in out


def test_tiny_mean_rounds_to_positive_zero():
    r1 = _report([
        _row("a", "w1", _GREEN, 0.0000001),
        _row("b", "w2", _GREEN, -0.0000001),
    ])
    r2 = _report([
        _row("a", "x1", _GREEN, 0.0000001),
        _row("b", "x2", _GREEN, -0.0000001),
    ])
    out = pareto_region_stability({"t1": r1, "t2": r2})
    assert '"mean":0.000000' in out
    assert "-0.000000" not in out


def test_empty_reports_yield_empty_groups():
    empty = _report([])
    out = pareto_region_stability({"t1": empty, "t2": empty})
    assert out == '{"alpha":0.050000,"groups":[]}\n'


def test_report_count_bounds_sixteen_inclusive():
    empty = _report([])
    out = pareto_region_stability({f"t{i}": empty for i in range(16)})
    assert out == '{"alpha":0.050000,"groups":[]}\n'


# --- validation -------------------------------------------------------------

def test_non_dict_reports_type_error():
    for bad in ([], None, 1, "reports", True, ()):
        with pytest.raises(TypeError):
            pareto_region_stability(bad)


@pytest.mark.parametrize("count", [0, 1, 17])
def test_report_count_out_of_range_value_error(count):
    empty = _report([])
    with pytest.raises(ValueError):
        pareto_region_stability({f"t{i}": empty for i in range(count)})


def test_keys_must_be_non_empty_strings():
    empty = _report([])
    with pytest.raises(ValueError):
        pareto_region_stability({"t1": empty, "": empty})


def test_mismatched_alpha_value_error():
    r1 = _report([
        _row("a", "w1", _GREEN, 1.0),
        _row("b", "w2", _GREEN, 4.0),
    ])
    r2 = _report([
        _row("a", "w1", _GREEN, 1.0),
        _row("b", "w2", _GREEN, 4.0),
    ], alpha=0.5)
    with pytest.raises(ValueError):
        pareto_region_stability({"t1": r1, "t2": r2})


def test_mismatched_test_identities_value_error():
    full = _report([
        _row("a", "w1", _GREEN, 1.0),
        _row("b", "w2", _GREEN, 4.0),
        _row("c", "w3", _GREEN, 9.0),
    ])
    pair = _report([
        _row("a", "w1", _GREEN, 1.0),
        _row("b", "w2", _GREEN, 4.0),
    ])
    with pytest.raises(ValueError):
        pareto_region_stability({"t1": full, "t2": pair})
    empty = _report([])
    with pytest.raises(ValueError):
        pareto_region_stability({"t1": full, "t2": empty})


def test_non_string_report_value_error():
    empty = _report([])
    for bad in (1, None, [], {}, True):
        with pytest.raises(ValueError):
            pareto_region_stability({"t1": bad, "t2": empty})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.rstrip("\n"),
        lambda s: s + "\n",
        lambda s: s.replace(
            '"a":"a","b":"b"', '"b":"b","a":"a"'
        ),
        lambda s: s.replace('"diff":0.000000', '"diff":-0.000000', 1),
        lambda s: s.replace(
            '{"alpha":0.050000,"groups"', '{"alpha":0.05, "groups"'
        ),
        lambda s: '{"groups":[],"alpha":0.050000}\n',
        lambda s: "not json\n",
    ],
)
def test_non_canonical_report_value_error(mutate):
    base = _report([
        _row("a", "w1", _GREEN, 2.0),
        _row("b", "w2", _GREEN, 2.0),
    ])
    raw = mutate(base)
    with pytest.raises(ValueError):
        pareto_region_stability({"t1": raw, "t2": base})


def test_duplicate_pair_in_metric_block_rejected():
    # Hand-build a report whose uhi block repeats pair (a, b) while the
    # other metrics keep one pair: the parser must reject it.
    one = (
        '{"a":"a","b":"b","metric":METRIC,"n_a":1,"n_b":1,'
        '"diff":0.000000,"p":1.000000,"q":1.000000,"reject":false,"rank":1}'
    )
    tests = []
    tests.append(one.replace("METRIC", '"uhi"'))
    tests.append(one.replace("METRIC", '"uhi"'))
    tests.append(one.replace("METRIC", '"energy"'))
    tests.append(one.replace("METRIC", '"vent"'))
    raw = (
        '{"alpha":0.050000,"groups":[{"pick":["green"],'
        '"tests":[' + ",".join(tests) + "]}]}\n"
    )
    good = _report([
        _row("a", "w1", _GREEN, 2.0),
        _row("b", "w2", _GREEN, 2.0),
    ])
    with pytest.raises(ValueError):
        pareto_region_stability({"t1": raw, "t2": good})


def test_divergent_pair_sets_across_metrics_rejected():
    def token(metric, pair, rank=1):
        a, b = pair
        return (
            f'{{"a":"{a}","b":"{b}","metric":"{metric}","n_a":1,"n_b":1,'
            '"diff":0.000000,"p":1.000000,"q":1.000000,"reject":false,'
            f'"rank":{rank}}}'
        )
    tests = [
        token("uhi", ("a", "b")),
        token("energy", ("a", "c")),
        token("vent", ("a", "b")),
    ]
    raw = (
        '{"alpha":0.050000,"groups":[{"pick":["green"],'
        '"tests":[' + ",".join(tests) + "]}]}\n"
    )
    good = _report([
        _row("a", "w1", _GREEN, 2.0),
        _row("b", "w2", _GREEN, 2.0),
    ])
    with pytest.raises(ValueError):
        pareto_region_stability({"t1": raw, "t2": good})


def test_rank_must_be_ascending_permutation_in_block():
    def token(metric, rank, pair):
        a, b = pair
        return (
            f'{{"a":"{a}","b":"{b}","metric":"{metric}","n_a":1,"n_b":1,'
            '"diff":0.000000,"p":1.000000,"q":1.000000,"reject":false,'
            f'"rank":{rank}}}'
        )
    # uhi ranks 2,1 instead of 1,2.
    tests = [
        token("uhi", 2, ("a", "b")), token("uhi", 1, ("a", "c")),
        token("energy", 1, ("a", "b")), token("energy", 2, ("a", "c")),
        token("vent", 1, ("a", "b")), token("vent", 2, ("a", "c")),
    ]
    raw = (
        '{"alpha":0.050000,"groups":[{"pick":["green"],'
        '"tests":[' + ",".join(tests) + "]}]}\n"
    )
    good = _report([
        _row("a", "w1", _GREEN, 1.0),
        _row("b", "w2", _GREEN, 2.0),
        _row("c", "w3", _GREEN, 3.0),
    ])
    with pytest.raises(ValueError):
        pareto_region_stability({"t1": raw, "t2": good})


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert (
        _uhi.pareto_region_stability
        is urban_micro.pareto_region_stability
    )
    assert "pareto_region_stability" in urban_micro.__all__
