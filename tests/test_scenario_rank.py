"""Tests for urban_micro.scenario_rank."""

import json
from fractions import Fraction
from itertools import combinations

import pytest

from urban_micro import scenario_rank, scenario_shift, window_shift


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _two_cell_tables(slope_b, slope_a, timestamps):
    """Build (base, sets) tables for cells a,b both labelled ``"L"``.

    The base table is flat zero. Cell ``a`` keeps slope ``slope_a`` and
    cell ``b`` slope ``slope_b``, so each lag pairing on edge (a, b) gives
    ``x = slope_a - slope_b`` (the bucket means collapse to the slope).
    """
    base = []
    for t in timestamps:
        base.append(_row(t, "a", 0.0))
        base.append(_row(t, "b", 0.0))
    table = []
    for t in timestamps:
        table.append(_row(t, "a", slope_a * (t // 3600)))
        table.append(_row(t, "b", slope_b * (t // 3600)))
    return base, table


def _perm_p(sample_a, sample_b):
    """Reference two-sided equal-size permutation p-value in Fraction."""
    values = [Fraction(v) for v in sample_a + sample_b]
    n_a = len(sample_a)
    n_b = len(sample_b)
    sum_a = Fraction(0)
    for value in sample_a:
        sum_a += Fraction(value)
    sum_b = Fraction(0)
    for value in sample_b:
        sum_b += Fraction(value)
    threshold = abs(sum_a * n_b - sum_b * n_a)
    pooled_sum = sum_a + sum_b
    hits = 0
    for combo in combinations(range(n_a + n_b), n_a):
        perm_a = sum((values[i] for i in combo), Fraction(0))
        perm_b = pooled_sum - perm_a
        if abs(perm_a * n_b - perm_b * n_a) >= threshold:
            hits += 1
    assignments = _comb(n_a + n_b, n_a)
    return Fraction(hits, assignments)


def _comb(n, k):
    if k < 0 or k > n:
        return 0
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _groups(report):
    return json.loads(report)["groups"]


# ---------------------------------------------------------------------------
# rank content
# ---------------------------------------------------------------------------


def test_ranks_ordered_by_mean_then_key():
    timestamps = [k * 3600 for k in range(3)]
    base, t_plus = _two_cell_tables(0, 1, timestamps)   # x = +1
    _, t_minus = _two_cell_tables(0, -1, timestamps)    # x = -1
    report = scenario_rank(
        base,
        {"hi": t_plus, "lo": t_minus, "mid": base},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {t: "w" for t in timestamps},
    )
    ranks = _groups(report)[0]["ranks"]
    # m=2 pairings: means -1, 0, 1.
    assert [(r["key"], r["mean"], r["rank"]) for r in ranks] == [
        ("lo", -1.0, 1),
        ("mid", 0.0, 2),
        ("hi", 1.0, 3),
    ]
    for rank in ranks:
        assert isinstance(rank["rank"], int)


def test_equal_means_tie_break_on_key():
    timestamps = [k * 3600 for k in range(3)]
    base, t_plus = _two_cell_tables(0, 1, timestamps)
    _, t_minus = _two_cell_tables(0, -1, timestamps)
    report = scenario_rank(
        base,
        {"zz": t_minus, "aa": t_minus, "mm": t_plus},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {t: "w" for t in timestamps},
    )
    ranks = _groups(report)[0]["ranks"]
    assert [(r["key"], r["rank"]) for r in ranks] == [
        ("aa", 1),
        ("zz", 2),
        ("mm", 3),
    ]


def test_rank_means_are_of_set_minus_base_e_values():
    timestamps = [k * 3600 for k in range(3)]
    # Set cell values: a slope 3, b slope 1 -> x_set = 2; base x = 0; e = 2.
    base, table = _two_cell_tables(1, 3, timestamps)
    report = scenario_rank(
        base,
        {"s": table, "t": base},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {t: "w" for t in timestamps},
    )
    ranks = {r["key"]: r for r in _groups(report)[0]["ranks"]}
    assert ranks["s"]["mean"] == 2.0
    assert ranks["t"]["mean"] == 0.0


def test_window_concatenates_buckets_in_ascending_order():
    # Three buckets contribute pairings; the mean is over all three e values
    # regardless of how the buckets are split across windows.
    timestamps = [k * 3600 for k in range(4)]
    base, table = _two_cell_tables(0, 1, timestamps)
    one_window = scenario_rank(
        base,
        {"s": table, "t": base},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {t: "all" for t in timestamps},
    )
    ranks = {r["key"]: r for r in _groups(one_window)[0]["ranks"]}
    assert ranks["s"]["mean"] == 1.0  # three pairings, e == 1 each


# ---------------------------------------------------------------------------
# pairwise tests
# ---------------------------------------------------------------------------


def test_pairwise_tests_pair_keys_in_ascending_order():
    timestamps = [k * 3600 for k in range(3)]
    base, t_plus = _two_cell_tables(0, 1, timestamps)
    _, t_minus = _two_cell_tables(0, -1, timestamps)
    report = scenario_rank(
        base,
        {"hi": t_plus, "lo": t_minus, "mid": base},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {t: "w" for t in timestamps},
    )
    tests = _groups(report)[0]["tests"]
    assert [(t["a"], t["b"]) for t in tests] == [
        ("hi", "lo"),
        ("hi", "mid"),
        ("lo", "mid"),
    ]
    diffs = {("hi", "lo"): 2.0, ("hi", "mid"): 1.0, ("lo", "mid"): -1.0}
    for test in tests:
        assert test["diff"] == diffs[(test["a"], test["b"])]


def test_permutation_p_and_q_match_reference():
    timestamps = [k * 3600 for k in range(3)]
    base, t_plus = _two_cell_tables(0, 1, timestamps)
    _, t_minus = _two_cell_tables(0, -1, timestamps)
    report = scenario_rank(
        base,
        {"hi": t_plus, "lo": t_minus, "mid": base},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {t: "w" for t in timestamps},
    )
    tests = {(t["a"], t["b"]): t for t in _groups(report)[0]["tests"]}
    # m = 2 pairings per set: pooled (1, 1) vs (0, 0) etc.; two of the six
    # assignments reach the observed separation -> p = 2/6 = 1/3 for the
    # unit-diff comparisons, and (1,1) vs (-1,-1) likewise gives 2/6.
    for key, sample_a, sample_b in (
        (("hi", "lo"), [1, 1], [-1, -1]),
        (("hi", "mid"), [1, 1], [0, 0]),
        (("lo", "mid"), [-1, -1], [0, 0]),
    ):
        expected = _perm_p(sample_a, sample_b)
        assert tests[key]["p"] == pytest.approx(float(expected), abs=5e-7)
    # N = 3 equal p-values: BH leaves q = p = 1/3; all above alpha 0.05.
    for test in tests.values():
        assert test["q"] == pytest.approx(1.0 / 3.0, abs=5e-7)
        assert test["reject"] is False


def test_bh_rejects_significant_comparisons():
    # m = 4 pairings per set (buckets 1..4); unit shifts give p = 2/70.
    timestamps = [k * 3600 for k in range(5)]
    base, t_plus = _two_cell_tables(0, 1, timestamps)
    _, t_minus = _two_cell_tables(0, -1, timestamps)
    report = scenario_rank(
        base,
        {"hi": t_plus, "lo": t_minus, "mid": base},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {t: "w" for t in timestamps},
    )
    group = _groups(report)[0]
    expected = _perm_p([1] * 4, [0] * 4)
    assert expected == Fraction(2, 70)
    for test in group["tests"]:
        assert test["p"] == pytest.approx(float(expected), abs=5e-7)
        assert test["q"] == pytest.approx(float(expected), abs=5e-7)
        assert test["reject"] is True


def test_alpha_compared_on_unquantized_value():
    # m = 4 pairings per set -> p = 2/70 = 0.028571428... (renders 0.028571).
    timestamps = [k * 3600 for k in range(5)]
    base, t_plus = _two_cell_tables(0, 1, timestamps)
    kwargs = dict(
        edges=[("a", "b")],
        lags=[1],
        labels={"a": "L", "b": "L"},
        windows={t: "w" for t in timestamps},
    )
    # 0.0285713 is below the unquantized p but quantizes to the same
    # 0.028571 at six decimals; the unquantized comparison must NOT reject.
    report_below = scenario_rank(
        base, {"s": t_plus, "t": base}, alpha=0.0285713, **kwargs
    )
    assert _groups(report_below)[0]["tests"][0]["reject"] is False
    # 0.03 is above the unquantized p.
    report_above = scenario_rank(
        base, {"s": t_plus, "t": base}, alpha=0.03, **kwargs
    )
    assert _groups(report_above)[0]["tests"][0]["reject"] is True


# ---------------------------------------------------------------------------
# grouping / ordering
# ---------------------------------------------------------------------------


def test_groups_sorted_by_label_window_lag():
    timestamps = [k * 3600 for k in range(4)]
    cells = ["a", "b", "c", "d"]

    def table(value):
        rows = []
        for t in timestamps:
            rows.append(_row(t, "a", value * (t // 3600)))
            rows.append(_row(t, "b", 0.0))
            rows.append(_row(t, "c", value * (t // 3600)))
            rows.append(_row(t, "d", 0.0))
        return rows

    base = table(0)
    window_map = {
        0: "pm",
        3600: "am",
        7200: "pm",
        10800: "am",
    }
    report = scenario_rank(
        base,
        {"s": table(1), "t": table(2)},
        [("a", "b"), ("c", "d")],
        [1, 2],
        {"a": "X", "b": "X", "c": "Y", "d": "Y"},
        window_map,
    )
    triples = [(g["label"], g["window"], g["lag"]) for g in _groups(report)]
    assert triples == [
        ("X", "am", 1),
        ("X", "am", 2),
        ("X", "pm", 1),
        ("X", "pm", 2),
        ("Y", "am", 1),
        ("Y", "am", 2),
        ("Y", "pm", 1),
        ("Y", "pm", 2),
    ]


def test_windows_split_into_independent_groups():
    timestamps = [k * 3600 for k in range(4)]
    base, table = _two_cell_tables(0, 1, timestamps)
    report = scenario_rank(
        base,
        {"s": table, "t": base},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {0: "p", 3600: "p", 7200: "q", 10800: "q"},
    )
    groups = _groups(report)
    # lag-1 pairings: bucket 3600 (p), 7200 (q), 10800 (q).
    by_window = {g["window"]: g for g in groups}
    assert sorted(by_window) == ["p", "q"]
    # Every e value is 1, so set "s" has mean 1.0 in both windows even
    # though p holds one pairing and q two.
    for group in by_window.values():
        means = {r["key"]: r["mean"] for r in group["ranks"]}
        assert means == {"s": 1.0, "t": 0.0}


def test_bh_ties_resolved_by_label_window_lag_a_b():
    # Three groups share identical p-values; q must equal p for all and the
    # test ordering inside a group stays (a, b) ascending.
    timestamps = [k * 3600 for k in range(3)]
    cells = ["a", "b", "c", "d"]

    def table(value):
        rows = []
        for t in timestamps:
            rows.append(_row(t, "a", value * (t // 3600)))
            rows.append(_row(t, "b", 0.0))
            rows.append(_row(t, "c", value * (t // 3600)))
            rows.append(_row(t, "d", 0.0))
        return rows

    report = scenario_rank(
        table(0),
        {"s": table(1), "t": table(0)},
        [("a", "b"), ("c", "d")],
        [1],
        {"a": "X", "b": "X", "c": "Y", "d": "Y"},
        {t: "w" for t in timestamps},
    )
    groups = _groups(report)
    for group in groups:
        test = group["tests"][0]
        assert (test["a"], test["b"]) == ("s", "t")
        assert test["p"] == pytest.approx(1.0 / 3.0, abs=5e-7)
        # N = 2 comparisons here (one per label), both p = 1/3.
        assert test["q"] == pytest.approx(1.0 / 3.0, abs=5e-7)


def test_no_lag_pairings_yields_empty_groups():
    base = [_row(0, "a", 0.0), _row(0, "b", 0.0)]
    shifted = [_row(0, "a", 1.0), _row(0, "b", 0.0)]
    report = scenario_rank(
        base,
        {"s": shifted, "t": base},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {0: "w"},
    )
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}\n'


# ---------------------------------------------------------------------------
# size limit
# ---------------------------------------------------------------------------


def test_combined_sample_over_16_raises_value_error():
    # 10 buckets -> 9 pairings per set -> 9 + 9 = 18 > 16.
    timestamps = [k * 3600 for k in range(10)]
    base, table = _two_cell_tables(0, 1, timestamps)
    with pytest.raises(ValueError):
        scenario_rank(
            base,
            {"s": table, "t": base},
            [("a", "b")],
            [1],
            {"a": "L", "b": "L"},
            {t: "w" for t in timestamps},
        )


def test_sixteen_combined_values_allowed():
    # 9 buckets -> 8 pairings per set -> 8 + 8 = 16.
    timestamps = [k * 3600 for k in range(9)]
    base, table = _two_cell_tables(0, 1, timestamps)
    report = scenario_rank(
        base,
        {"s": table, "t": base},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {t: "w" for t in timestamps},
    )
    assert _groups(report)[0]["tests"]


# ---------------------------------------------------------------------------
# serialization
# ---------------------------------------------------------------------------


def _full_report():
    timestamps = [k * 3600 for k in range(3)]
    base, t_plus = _two_cell_tables(0, 1, timestamps)
    _, t_minus = _two_cell_tables(0, -1, timestamps)
    return scenario_rank(
        base,
        {"hi": t_plus, "lo": t_minus, "mid": base},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {t: "w" for t in timestamps},
    )


def test_compact_utf8_single_trailing_newline():
    report = _full_report()
    assert report.endswith("\n")
    assert not report.endswith("\n\n")
    assert " " not in report
    json.loads(report)  # valid JSON


def test_json_key_orders():
    report = _full_report()
    parsed = json.loads(report)
    assert list(parsed) == ["minutes", "alpha", "groups"]
    group = parsed["groups"][0]
    assert list(group) == ["label", "window", "lag", "ranks", "tests"]
    assert list(group["ranks"][0]) == ["key", "mean", "rank"]
    assert list(group["tests"][0]) == ["a", "b", "diff", "p", "q", "reject"]


def test_six_decimals_and_integer_and_bool_rendering():
    report = _full_report()
    assert '"alpha":0.050000' in report
    assert '"lag":1' in report
    assert '"rank":1' in report
    assert '"reject":false' in report
    for token in ('"mean":-1.000000', '"mean":0.000000', '"mean":1.000000'):
        assert token in report
    for token in ('"p":0.333333', '"q":0.333333', '"diff":2.000000'):
        assert token in report


def test_negative_zero_normalized():
    timestamps = [k * 3600 for k in range(3)]
    base, table = _two_cell_tables(0, 1, timestamps)
    # Two keys with identical e samples: diff between them is exactly zero.
    report = scenario_rank(
        base,
        {"s1": table, "s2": table},
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {t: "w" for t in timestamps},
    )
    assert "-0.000000" not in report
    assert '"diff":0.000000' in report


def test_unicode_keys_preserved():
    timestamps = [k * 3600 for k in range(3)]
    base, table = _two_cell_tables(0, 1, timestamps)
    report = scenario_rank(
        base,
        {"α": table, "β": base},
        [("a", "b")],
        [1],
        {"a": "λ", "b": "λ"},
        {t: "fenêtre" for t in timestamps},
    )
    assert '"α"' in report
    assert '"λ"' in report
    assert '"fenêtre"' in report


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _tiny_inputs():
    base = [_row(0, "a", 0.0), _row(0, "b", 0.0)]
    table = [_row(0, "a", 1.0), _row(0, "b", 0.0)]
    return (
        base,
        [("a", "b")],
        [1],
        {"a": "L", "b": "L"},
        {0: "w"},
        table,
    )


def test_sets_not_dict_raises_type_error():
    base, edges, lags, labels, windows, table = _tiny_inputs()
    for bad in ([("s", table), ("t", table)], (("s", table),), "nope"):
        with pytest.raises(TypeError):
            scenario_rank(base, bad, edges, lags, labels, windows)


def test_sets_needs_two_entries():
    base, edges, lags, labels, windows, table = _tiny_inputs()
    with pytest.raises(ValueError):
        scenario_rank(base, {}, edges, lags, labels, windows)
    with pytest.raises(ValueError):
        scenario_rank(base, {"s": table}, edges, lags, labels, windows)


def test_sets_keys_must_be_non_empty_strings():
    base, edges, lags, labels, windows, table = _tiny_inputs()
    for bad_key in ("", 1, ("s",), True):
        with pytest.raises(ValueError):
            scenario_rank(
                base,
                {bad_key: table, "t": table},
                edges, lags, labels, windows,
            )


def test_sets_values_must_be_lists():
    base, edges, lags, labels, windows, table = _tiny_inputs()
    for bad_value in (tuple(table), "rows", 42):
        with pytest.raises(ValueError):
            scenario_rank(
                base,
                {"s": bad_value, "t": table},
                edges, lags, labels, windows,
            )


def test_base_not_list_raises_type_error():
    _, edges, lags, labels, windows, table = _tiny_inputs()
    with pytest.raises(TypeError):
        scenario_rank((), {"s": table, "t": table}, edges, lags, labels, windows)


def test_set_table_contract_violation_raises_value_error():
    base, edges, lags, labels, windows, table = _tiny_inputs()
    bad_table = [table[0]] + [
        (0, "a", 1.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # duplicate (t, c) pair
    ]
    with pytest.raises(ValueError):
        scenario_rank(
            base, {"s": table, "t": bad_table}, edges, lags, labels, windows
        )


def test_set_table_key_set_mismatch_raises_value_error():
    base, edges, lags, labels, windows, _ = _tiny_inputs()
    other = [_row(0, "a", 1.0), _row(0, "z", 0.0)]
    with pytest.raises(ValueError):
        scenario_rank(
            base,
            {"s": [_row(0, "a", 1.0), _row(0, "b", 0.0)], "t": other},
            edges, lags, labels, windows,
        )


def test_windows_and_other_shared_contracts_validated():
    base, edges, lags, labels, _, table = _tiny_inputs()
    sets = {"s": table, "t": base}
    with pytest.raises(TypeError):
        scenario_rank(base, sets, edges, lags, labels, [(0, "w")])
    with pytest.raises(ValueError):
        scenario_rank(base, sets, edges, lags, labels, {3600: "w"})
    with pytest.raises(ValueError):
        scenario_rank(base, sets, edges, lags, labels, {0: ""})
    with pytest.raises(ValueError):
        scenario_rank(base, sets, edges, [0], labels, {0: "w"})
    with pytest.raises(ValueError):
        scenario_rank(base, sets, edges, lags, labels, {0: "w"}, minutes=7)


# ---------------------------------------------------------------------------
# old interface unchanged / exports
# ---------------------------------------------------------------------------


def test_scenario_rank_is_exported():
    import urban_micro

    assert "scenario_rank" in urban_micro.__all__
    assert urban_micro.scenario_rank is scenario_rank
    from urban_micro import uhi

    assert "scenario_rank" in uhi.__all__
    assert callable(scenario_shift)
    assert callable(window_shift)


def test_scenario_shift_still_works():
    # The legacy single-set interface keeps its own contract/output.
    timestamps = [k * 3600 for k in range(3)]
    base, table = _two_cell_tables(0, 1, timestamps)
    edges, lags, labels = [("a", "b")], [1], {"a": "L", "b": "L"}
    # Two window names so one comparison exists: pairings at 3600 land in
    # "p", the one at 7200 in "q"; both e values are 1, so diff = 0 and
    # the exact permutation p is 1.
    windows = {0: "p", 3600: "p", 7200: "q"}
    old = scenario_shift(base, table, edges, lags, labels, windows)
    parsed = json.loads(old)
    assert list(parsed) == ["minutes", "alpha", "groups"]
    group = parsed["groups"][0]
    assert group["key"] == "L"
    comparison = group["comparisons"][0]
    assert (comparison["a"], comparison["b"]) == ("p", "q")
    lag = comparison["lags"][0]
    assert lag == {
        "lag": 1,
        "n_a": 1,
        "n_b": 1,
        "diff": 0.0,
        "p": 1.0,
        "q": 1.0,
        "reject": False,
    }
    assert old.endswith("\n")
