"""Tests for urban_micro.region_attr: Pearson attribution of
pareto_region_stability region pairs (uhi metric only) to
weather/morph/cover driver differences with an n! positional
permutation p-value, a global Benjamini-Hochberg pass and per-pick
factor ranks."""

import json
import math
from fractions import Fraction
from itertools import permutations

import pytest

from urban_micro import pareto_region_compare, pareto_region_stability
from urban_micro import region_attr
from urban_micro import uhi as _uhi

_GREEN = ("green",)
_ROOF = ("roof",)
_FACTORS = ("weather", "morph", "cover")


# --- report / driver builders ----------------------------------------------

def _row(r, w, p, u, e=0.0, v=0.0):
    return (r, w, 0, p, u, e, v)


def _report(periods, *, alpha=0.05):
    """Build a canonical pareto_region_stability report from periods,
    each a list of pareto_region_compare rows."""
    return pareto_region_stability(
        {
            f"t{i}": pareto_region_compare(rows, alpha=alpha)
            for i, rows in enumerate(periods)
        }
    )


def _identical(rows, *, alpha=0.05):
    return _report([rows, rows], alpha=alpha)


# --- exact Fraction reference ----------------------------------------------

def _reference(report, drivers):
    """Mirror region_attr with exact Fractions. Returns the expected
    parsed payload (alpha plus groups with ranked items)."""
    payload = json.loads(report)
    alpha = Fraction(str(payload["alpha"]))

    # pick -> uhi records in ascending (a, b) order, each
    # (a, b, rank, y).
    per_pick = {}
    for group in payload["groups"]:
        pick = tuple(group["pick"])
        # Items are three metric blocks (uhi, energy, vent), each with
        # ranks restarting at 1; the uhi block precedes the second rank-1.
        uhi_items = []
        seen_restart = False
        for item in group["items"]:
            if item["rank"] == 1:
                if seen_restart:
                    break
                seen_restart = True
            uhi_items.append(item)
        ordered = sorted(uhi_items, key=lambda it: (it["a"], it["b"]))
        per_pick[pick] = [
            (
                item["a"],
                item["b"],
                item["rank"],
                Fraction(str(item["mean"]))
                * Fraction(str(item["consistency"]))
                * Fraction(str(item["significant"]))
                / Fraction(item["rank"]),
            )
            for item in ordered
        ]

    tests = {}
    sort_rows = []
    for pick, records in per_pick.items():
        n = len(records)
        ys = [record[3] for record in records]
        y_bar = sum(ys, Fraction(0)) / n
        syy = sum((y - y_bar) ** 2 for y in ys)
        for factor_index, factor in enumerate(_FACTORS):
            xs = [
                Fraction(str(drivers[a][factor_index]))
                - Fraction(str(drivers[b][factor_index]))
                for a, b, _rank, _y in records
            ]
            x_bar = sum(xs, Fraction(0)) / n
            sxx = sum((x - x_bar) ** 2 for x in xs)
            if sxx == 0 or syy == 0:
                r_float = 0.0
                p = Fraction(1)
            else:
                x_centered = [x - x_bar for x in xs]
                y_centered = [y - y_bar for y in ys]
                sxy = sum(
                    dx * dy
                    for dx, dy in zip(x_centered, y_centered)
                )
                r_float = float(sxy) / math.sqrt(float(sxx * syy))
                total = math.factorial(n)
                tail = 0
                for permuted in permutations(y_centered):
                    permuted_sxy = sum(
                        dx * dy
                        for dx, dy in zip(x_centered, permuted)
                    )
                    permuted_syy = sum(dy * dy for dy in permuted)
                    if (
                        permuted_sxy * permuted_sxy * syy
                        >= sxy * sxy * permuted_syy
                    ):
                        tail += 1
                p = Fraction(tail, total)
            tests[(pick, factor)] = (n, r_float, p)
            sort_rows.append((p, pick, factor_index, factor))

    sort_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    tested = len(sort_rows)
    q_values = {}
    running = Fraction(1)
    for rank in range(tested, 0, -1):
        p, pick, factor_index, factor = sort_rows[rank - 1]
        running = min(running, Fraction(tested) * p / rank)
        q_values[(pick, factor)] = min(Fraction(1), running)

    groups = []
    for pick in sorted(per_pick):
        ranked = sorted(
            _FACTORS,
            key=lambda factor: (
                0 if q_values[(pick, factor)] <= alpha else 1,
                -abs(tests[(pick, factor)][1]),
                _FACTORS.index(factor),
            ),
        )
        items = []
        for rank, factor in enumerate(ranked, start=1):
            n, r_float, p = tests[(pick, factor)]
            q = q_values[(pick, factor)]
            items.append(
                {
                    "factor": factor,
                    "n": n,
                    "effect": r_float,
                    "p": float(p),
                    "q": float(q),
                    "reject": q <= alpha,
                    "rank": rank,
                }
            )
        groups.append({"pick": list(pick), "items": items})
    return {"alpha": float(alpha), "groups": groups}


def _assert_matches_reference(report, drivers):
    out = region_attr(report, drivers)
    payload = json.loads(out)
    expected = _reference(report, drivers)
    assert len(payload["groups"]) == len(expected["groups"])
    for group, exp_group in zip(payload["groups"], expected["groups"]):
        assert group["pick"] == exp_group["pick"]
        assert len(group["items"]) == len(exp_group["items"])
        for item, exp in zip(group["items"], exp_group["items"]):
            assert list(item) == [
                "factor", "n", "effect", "p", "q", "reject", "rank",
            ]
            assert item["factor"] == exp["factor"]
            assert item["n"] == exp["n"]
            assert isinstance(item["n"], int) and not isinstance(item["n"], bool)
            assert item["rank"] == exp["rank"]
            assert item["reject"] is exp["reject"]
            assert abs(item["effect"] - exp["effect"]) < 5.1e-7
            assert abs(item["p"] - exp["p"]) < 5.1e-7
            assert abs(item["q"] - exp["q"]) < 5.1e-7
    return out


# --- datasets ---------------------------------------------------------------

def _n3_rows():
    # UHIs 0/3/9 keep every mean a multiple of 3 so mean/rank (ranks
    # 1..3) terminates in decimal and the exact-Fraction reference
    # coincides bit-for-bit with the mandated precision-1000 arithmetic.
    return [
        _row("a", "a1", _GREEN, 0.0),
        _row("b", "b1", _GREEN, 3.0),
        _row("c", "c1", _GREEN, 9.0),
    ]


def _n6_rows():
    # Four regions -> six pairs with six distinct |mean| ranks; the 3/9/21
    # values are multiples of 3 so mean/rank (ranks 1..6) is exact.
    return [
        _row("a", "a1", _GREEN, 0.0),
        _row("b", "b1", _GREEN, 3.0),
        _row("c", "c1", _GREEN, 9.0),
        _row("d", "d1", _GREEN, 21.0),
    ]


def _two_pick_rows():
    return [
        _row("a", "a1", _GREEN, 1.0),
        _row("b", "b1", _GREEN, 9.0),
        _row("a", "x1", _ROOF, 2.0),
        _row("b", "x2", _ROOF, 8.0),
    ]


_N3_DRIVERS = {
    "a": (0.0, 5.0, 9.0),
    "b": (1.0, 5.0, 2.0),
    "c": (3.0, 5.0, 1.0),
}

_N6_DRIVERS = {
    "a": (0.0, 5.0, 9.0),
    "b": (1.0, 5.0, 3.0),
    "c": (4.0, 5.0, 2.0),
    "d": (8.0, 5.0, 0.0),
}

_TWO_PICK_DRIVERS = {
    "a": (1.0, 2.0, 3.0),
    "b": (4.0, 1.0, 0.0),
}


# --- canonical output -------------------------------------------------------

def test_canonical_shape_and_key_order():
    out = _assert_matches_reference(
        _identical(_n3_rows(), alpha=1.0), _N3_DRIVERS
    )
    assert out.endswith("\n") and not out.endswith("\n\n")
    assert " " not in out
    payload = json.loads(out)
    assert list(payload) == ["alpha", "groups"]
    group = payload["groups"][0]
    assert list(group) == ["pick", "items"]
    assert group["pick"] == ["green"]
    assert len(group["items"]) == 3
    for item in group["items"]:
        assert list(item) == [
            "factor", "n", "effect", "p", "q", "reject", "rank",
        ]
        assert isinstance(item["rank"], int) and not isinstance(item["rank"], bool)
    assert [item["rank"] for item in group["items"]] == [1, 2, 3]


def test_n_is_region_pair_count():
    out = _assert_matches_reference(
        _identical(_n3_rows(), alpha=1.0), _N3_DRIVERS
    )
    assert all(item["n"] == 3 for item in json.loads(out)["groups"][0]["items"])


def test_n6_matches_reference():
    _assert_matches_reference(_identical(_n6_rows(), alpha=1.0), _N6_DRIVERS)


def test_two_picks_share_global_bh_and_sort_by_pick():
    out = _assert_matches_reference(
        _identical(_two_pick_rows()), _TWO_PICK_DRIVERS
    )
    assert [g["pick"] for g in json.loads(out)["groups"]] == [
        ["green"], ["roof"],
    ]


def test_matches_reference_across_alpha_levels():
    for alpha in (0.05, 0.5, 1.0):
        _assert_matches_reference(
            _identical(_n3_rows(), alpha=alpha), _N3_DRIVERS
        )
        _assert_matches_reference(
            _identical(_n6_rows(), alpha=alpha), _N6_DRIVERS
        )


def test_uses_only_uhi_metric():
    # Energy/vent values vary across periods but the uhi block does not;
    # output must equal that of a report with zero energy/vent columns.
    rows_flat = [
        _row("a", "a1", _GREEN, 0.0, 0.0, 0.0),
        _row("b", "b1", _GREEN, 10.0, 0.0, 0.0),
        _row("c", "c1", _GREEN, 20.0, 0.0, 0.0),
    ]
    rows_mixed = [
        _row("a", "a1", _GREEN, 0.0, 100.0, -100.0),
        _row("b", "b1", _GREEN, 10.0, -5.0, 30.0),
        _row("c", "c1", _GREEN, 20.0, 7.0, 2.0),
    ]
    flat = _identical(rows_flat, alpha=1.0)
    mixed = _identical(rows_mixed, alpha=1.0)
    assert region_attr(flat, _N3_DRIVERS) == region_attr(
        mixed, _N3_DRIVERS
    )


def test_constant_driver_factor_is_zero_p_one():
    drivers = {
        "a": (0.0, 5.0, 9.0),
        "b": (1.0, 5.0, 2.0),
        "c": (3.0, 5.0, 1.0),
    }
    out = _assert_matches_reference(
        _identical(_n3_rows(), alpha=0.05), drivers
    )
    morph = next(
        item
        for item in json.loads(out)["groups"][0]["items"]
        if item["factor"] == "morph"
    )
    assert morph["effect"] == 0.0
    assert morph["p"] == 1.0
    assert morph["q"] == 1.0
    # p = 1 never rejects at alpha below 1.
    assert morph["reject"] is False


def test_alpha_one_ranks_rejected_by_abs_effect():
    out = _assert_matches_reference(
        _identical(_n3_rows(), alpha=1.0), _N3_DRIVERS
    )
    items = json.loads(out)["groups"][0]["items"]
    assert all(item["reject"] for item in items)
    effects = [abs(item["effect"]) for item in items]
    assert effects == sorted(effects, reverse=True)
    # weather (0.817) outranks cover (-0.679) outranks morph (0).
    assert [item["factor"] for item in items] == [
        "weather", "cover", "morph",
    ]


def test_rejection_flips_with_alpha():
    loose = region_attr(_identical(_n3_rows(), alpha=1.0), _N3_DRIVERS)
    strict = region_attr(_identical(_n3_rows(), alpha=0.05), _N3_DRIVERS)
    assert all(
        item["reject"] for item in json.loads(loose)["groups"][0]["items"]
    )
    assert not any(
        item["reject"] for item in json.loads(strict)["groups"][0]["items"]
    )


def test_integer_driver_values_accepted():
    drivers = {region: tuple(int(v) for v in vals)
               for region, vals in _N3_DRIVERS.items()}
    _assert_matches_reference(_identical(_n3_rows(), alpha=1.0), drivers)


def test_no_negative_zero_anywhere():
    out = region_attr(_identical(_two_pick_rows()), _TWO_PICK_DRIVERS)
    assert "-0.000000" not in out


# --- empty report -----------------------------------------------------------

def test_empty_report_only_with_empty_drivers():
    empty = '{"alpha":0.050000,"groups":[]}\n'
    assert region_attr(empty, {}) == '{"alpha":0.050000,"groups":[]}\n'
    with pytest.raises(ValueError):
        region_attr(empty, {"a": (1.0, 2.0, 3.0)})


# --- validation -------------------------------------------------------------

def test_type_errors():
    report = _identical(_two_pick_rows())
    with pytest.raises(TypeError):
        region_attr(1, _TWO_PICK_DRIVERS)
    with pytest.raises(TypeError):
        region_attr(None, _TWO_PICK_DRIVERS)
    with pytest.raises(TypeError):
        region_attr(report, [])
    with pytest.raises(TypeError):
        region_attr(report, None)


def test_driver_keys_must_match_regions_exactly():
    report = _identical(_two_pick_rows())
    missing = {"a": (1.0, 2.0, 3.0)}
    extra = dict(_TWO_PICK_DRIVERS)
    extra["zzz"] = (0.0, 0.0, 0.0)
    for bad in (missing, extra, {}):
        with pytest.raises(ValueError):
            region_attr(report, bad)


def test_driver_values_must_be_three_tuples():
    report = _identical(_two_pick_rows())
    for bad_value in (
        [1.0, 2.0, 3.0],
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
    ):
        bad = dict(_TWO_PICK_DRIVERS)
        bad["a"] = bad_value
        with pytest.raises(ValueError):
            region_attr(report, bad)


@pytest.mark.parametrize("bad", [
    True, False, float("inf"), -float("inf"), float("nan"), "1.0", None,
])
def test_driver_numbers_rejected(bad):
    report = _identical(_two_pick_rows())
    drivers = dict(_TWO_PICK_DRIVERS)
    drivers["a"] = (bad, 2.0, 3.0)
    with pytest.raises(ValueError):
        region_attr(report, drivers)


def test_more_than_eight_region_pairs_rejected():
    regions = [f"r{i}" for i in range(5)]
    rows = [
        _row(x, f"w{x}", _GREEN, float(i))
        for i, x in enumerate(regions)
    ]
    report = _identical(rows, alpha=1.0)
    drivers = {x: (float(i), 0.0, 0.0) for i, x in enumerate(regions)}
    with pytest.raises(ValueError, match="8"):
        region_attr(report, drivers)


def test_six_region_pairs_accepted():
    report = _identical(_n6_rows(), alpha=1.0)
    out = region_attr(report, _N6_DRIVERS)
    assert all(
        item["n"] == 6 for item in json.loads(out)["groups"][0]["items"]
    )


@pytest.mark.parametrize("mutate", [
    lambda s: s.rstrip("\n"),
    lambda s: s + "\n",
    lambda s: s.replace('"mean":', '"m":', 1),
    lambda s: s.replace(
        '{"alpha":0.050000,"groups"', '{"alpha":0.05, "groups"'
    ),
    lambda s: '{"groups":[],"alpha":0.050000}\n',
    lambda s: "not json\n",
])
def test_non_canonical_report_value_error(mutate):
    report = _identical(_two_pick_rows())
    with pytest.raises(ValueError):
        region_attr(mutate(report), _TWO_PICK_DRIVERS)


def test_region_compare_report_rejected():
    # A pareto_region_compare (tests) report is not a stability report.
    compare = pareto_region_compare(_two_pick_rows())
    with pytest.raises(ValueError):
        region_attr(compare, _TWO_PICK_DRIVERS)


# --- export -----------------------------------------------------------------

def test_exported_from_package_root():
    import urban_micro

    assert _uhi.region_attr is urban_micro.region_attr
    assert "region_attr" in urban_micro.__all__
