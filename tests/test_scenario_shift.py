"""Tests for urban_micro.scenario_shift."""

import json

import pytest

from urban_micro import scenario_shift


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _table(cells, times, delta):
    """Build a details table with delta(t, c) taken from delta[(t, c)]."""
    return [_row(t, c, delta[(t, c)]) for t in times for c in cells]


def test_empty_tables_give_empty_groups():
    report = scenario_shift([], [], [], [1], {}, {})
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}\n'
    assert json.loads(report) == {"minutes": 60, "alpha": 0.05, "groups": []}


def test_basic_shift_exact_json():
    # One edge (a, b), lag 1; x = (dB,a - dB-lag,a) - (dB,b - dB-lag,b).
    # x_before = [-1, -3] and x_after = [-1, -4] at buckets 3600 and 7200,
    # so e = x_after - x_before = [0, -1]; w1 = [0], w2 = [-1].
    before = _table(
        ("a", "b"),
        (0, 3600, 7200),
        {(0, "a"): 1.0, (0, "b"): 2.0,
         (3600, "a"): 3.0, (3600, "b"): 5.0,
         (7200, "a"): 4.0, (7200, "b"): 9.0},
    )
    after = _table(
        ("a", "b"),
        (0, 3600, 7200),
        {(0, "a"): 1.0, (0, "b"): 2.0,
         (3600, "a"): 4.0, (3600, "b"): 6.0,
         (7200, "a"): 5.0, (7200, "b"): 11.0},
    )
    report = scenario_shift(
        before, after, [("a", "b")], [1], {"a": "u", "b": "u"},
        {0: "w1", 3600: "w1", 7200: "w2"},
    )
    assert report == (
        '{"minutes":60,"alpha":0.050000,"groups":['
        '{"key":"u","comparisons":['
        '{"a":"w1","b":"w2","lags":['
        '{"lag":1,"n_a":1,"n_b":1,"diff":1.000000,"p":1.000000,'
        '"q":1.000000,"reject":false}]}]}]}\n'
    )


def test_shift_permutation_p_exact():
    # e samples w1 = [0, 0] and w2 = [3, 3]: of the C(4, 2) = 6 equal-size
    # assignments only the two extreme splits reach |diff'| = |diff| = 3,
    # so p = 2/6 = 1/3.
    before = [_row(t, c, 1.0) for t in (0, 3600, 7200) for c in ("a", "b", "c")]
    after = _table(
        ("a", "b", "c"),
        (0, 3600, 7200),
        {(0, "a"): 0.0, (0, "b"): 0.0, (0, "c"): 0.0,
         (3600, "a"): 1.0, (3600, "b"): 1.0, (3600, "c"): 1.0,
         (7200, "a"): 7.0, (7200, "b"): 4.0, (7200, "c"): 1.0},
    )
    report = scenario_shift(
        before, after, [("a", "b"), ("b", "c")], [1],
        {"a": "u", "b": "u", "c": "u"},
        {0: "w1", 3600: "w1", 7200: "w2"},
    )
    lag = json.loads(report)["groups"][0]["comparisons"][0]["lags"][0]
    assert lag == {
        "lag": 1, "n_a": 2, "n_b": 2, "diff": -3.0,
        "p": 0.333333, "q": 0.333333,
        "reject": False,
    }
    assert '"diff":-3.000000' in report
    assert '"p":0.333333' in report
    assert '"q":0.333333' in report


def test_reject_true_when_q_below_alpha():
    # e samples w1 = [0, 0, 0, 0] and w2 = [2, 2, 2, 2]: only the two
    # extreme of the C(8, 4) = 70 assignments reach |diff| = 2, so
    # p = q = 2/70 = 1/35 <= 0.05 and the comparison is rejected.
    cells = ("a", "b", "c", "d", "e")
    before = [_row(t, c, 1.0) for t in (0, 3600, 7200) for c in cells]
    after = _table(
        cells,
        (0, 3600, 7200),
        {(0, c): 0.0 for c in cells}
        | {(3600, c): 1.0 for c in cells}
        | {(7200, c): d for c, d in zip(cells, (9.0, 7.0, 5.0, 3.0, 1.0))},
    )
    neighbors = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]
    report = scenario_shift(
        before, after, neighbors, [1], {c: "u" for c in cells},
        {0: "w1", 3600: "w1", 7200: "w2"},
    )
    lag = json.loads(report)["groups"][0]["comparisons"][0]["lags"][0]
    assert lag["n_a"] == 4
    assert lag["n_b"] == 4
    assert lag["diff"] == -2.0
    # Rendered with exactly six decimals: 2/70 = 1/35 -> 0.028571.
    assert lag["p"] == 0.028571
    assert lag["q"] == 0.028571
    assert lag["reject"] is True
    assert '"p":0.028571' in report
    assert '"reject":true' in report


def test_identical_tables_give_zero_shift():
    before = _table(
        ("a", "b"),
        (0, 3600, 7200),
        {(0, "a"): 1.0, (0, "b"): 2.0,
         (3600, "a"): 3.0, (3600, "b"): 5.0,
         (7200, "a"): 4.0, (7200, "b"): 9.0},
    )
    report = scenario_shift(
        before, list(before), [("a", "b")], [1], {"a": "u", "b": "u"},
        {0: "w1", 3600: "w1", 7200: "w2"},
    )
    lag = json.loads(report)["groups"][0]["comparisons"][0]["lags"][0]
    assert lag["diff"] == 0.0
    assert lag["p"] == 1.0
    assert lag["reject"] is False
    assert "-0.000000" not in report


def test_groups_sorted_by_label_and_bh_over_all_comparisons():
    # Two labels with one comparison each, both p = 1 -> both q = 1.
    cells = ("a", "b", "c", "d")
    before = [_row(t, c, 0.0) for t in (0, 3600, 7200) for c in cells]
    after = _table(
        cells,
        (0, 3600, 7200),
        {(0, c): 0.0 for c in cells}
        | {(3600, "a"): 1.0, (3600, "b"): 0.0, (3600, "c"): 0.0,
           (3600, "d"): 1.0}
        | {(7200, "a"): 2.0, (7200, "b"): 0.0, (7200, "c"): 0.0,
           (7200, "d"): 2.0},
    )
    labels = {"a": "u", "b": "u", "c": "v", "d": "v"}
    report = scenario_shift(
        before, after, [("a", "b"), ("c", "d")], [1], labels,
        {0: "w1", 3600: "w1", 7200: "w2"},
    )
    assert report == (
        '{"minutes":60,"alpha":0.050000,"groups":['
        '{"key":"u","comparisons":['
        '{"a":"w1","b":"w2","lags":['
        '{"lag":1,"n_a":1,"n_b":1,"diff":0.000000,"p":1.000000,'
        '"q":1.000000,"reject":false}]}]},'
        '{"key":"v","comparisons":['
        '{"a":"w1","b":"w2","lags":['
        '{"lag":1,"n_a":1,"n_b":1,"diff":0.000000,"p":1.000000,'
        '"q":1.000000,"reject":false}]}]}]}\n'
    )


def test_lags_sorted_in_output_and_bh_stepdown():
    # lags given as [2, 1] render ascending; lag 1 gets p = 1/3 and the
    # two-comparison BH step-down gives q = 2/3 while lag 2 keeps q = 1.
    before = [_row(t, c, 0.0) for t in (0, 3600, 7200, 10800) for c in ("a", "b")]
    after = _table(
        ("a", "b"),
        (0, 3600, 7200, 10800),
        {(0, "a"): 0.0, (0, "b"): 0.0,
         (3600, "a"): 1.0, (3600, "b"): 0.0,
         (7200, "a"): 2.0, (7200, "b"): 0.0,
         (10800, "a"): 4.0, (10800, "b"): 0.0},
    )
    report = scenario_shift(
        before, after, [("a", "b")], [2, 1], {"a": "u", "b": "u"},
        {0: "w1", 3600: "w1", 7200: "w1", 10800: "w2"},
    )
    lags = json.loads(report)["groups"][0]["comparisons"][0]["lags"]
    assert [lag["lag"] for lag in lags] == [1, 2]
    assert lags[0]["n_a"] == 2
    assert lags[0]["n_b"] == 1
    assert lags[0]["diff"] == -1.0
    assert lags[0]["p"] == 0.333333
    assert lags[0]["q"] == 0.666667
    assert lags[1]["n_a"] == 1
    assert lags[1]["n_b"] == 1
    assert lags[1]["p"] == 1.0
    assert lags[1]["q"] == 1.0
    assert '"q":0.666667' in report


def test_negative_zero_shift_normalized():
    before = [_row(t, c, 0.0) for t in (0, 3600, 7200) for c in ("a", "b")]
    after = _table(
        ("a", "b"),
        (0, 3600, 7200),
        {(0, "a"): 0.0, (0, "b"): 0.0,
         (3600, "a"): -0.0, (3600, "b"): 0.0,
         (7200, "a"): 0.0, (7200, "b"): 0.0},
    )
    report = scenario_shift(
        before, after, [("a", "b")], [1], {"a": "u", "b": "u"},
        {0: "w1", 3600: "w1", 7200: "w2"},
    )
    assert "-0.000000" not in report
    assert '"diff":0.000000' in report


def test_compact_utf8_no_spaces_one_trailing_newline():
    before = _table(
        ("a", "b"),
        (0, 3600, 7200),
        {(t, c): 1.0 for t in (0, 3600, 7200) for c in ("a", "b")},
    )
    after = _table(
        ("a", "b"),
        (0, 3600, 7200),
        {(0, "a"): 0.0, (0, "b"): 0.0,
         (3600, "a"): 1.0, (3600, "b"): 0.0,
         (7200, "a"): 2.0, (7200, "b"): 0.0},
    )
    report = scenario_shift(
        before, after, [("a", "b")], [1], {"a": "céll", "b": "céll"},
        {0: "wü", 3600: "wü", 7200: "wø"},
    )
    assert " " not in report
    assert report.endswith("\n") and not report.endswith("\n\n")
    assert '"key":"céll"' in report
    assert '"a":"wø"' in report  # "wø" sorts before "wü"
    assert '"b":"wü"' in report


def test_top_level_and_nested_key_order():
    times = (0, 1800, 3600)
    before = [_row(t, c, 1.0) for t in times for c in ("a", "b")]
    after = _table(
        ("a", "b"),
        times,
        {(0, "a"): 0.0, (0, "b"): 0.0,
         (1800, "a"): 1.0, (1800, "b"): 0.0,
         (3600, "a"): 2.0, (3600, "b"): 0.0},
    )
    report = scenario_shift(
        before, after, [("a", "b")], [1], {"a": "u", "b": "u"},
        {0: "w1", 1800: "w1", 3600: "w2"}, minutes=30, alpha=0.1,
    )
    parsed = json.loads(report)
    assert list(parsed) == ["minutes", "alpha", "groups"]
    assert parsed["minutes"] == 30
    assert list(parsed["groups"][0]) == ["key", "comparisons"]
    assert list(parsed["groups"][0]["comparisons"][0]) == ["a", "b", "lags"]
    assert list(parsed["groups"][0]["comparisons"][0]["lags"][0]) == [
        "lag", "n_a", "n_b", "diff", "p", "q", "reject",
    ]
    assert '"minutes":30' in report
    assert '"alpha":0.100000' in report


def test_minutes_controls_bucketing():
    # With minutes=30 the 1800-second rows form their own buckets, so the
    # lag-1 pairings and the window assignment change accordingly.
    times = (0, 1800, 3600)
    before = [_row(t, c, 1.0) for t in times for c in ("a", "b")]
    after = _table(
        ("a", "b"),
        times,
        {(0, "a"): 0.0, (0, "b"): 0.0,
         (1800, "a"): 1.0, (1800, "b"): 0.0,
         (3600, "a"): 3.0, (3600, "b"): 0.0},
    )
    report = scenario_shift(
        before, after, [("a", "b")], [1], {"a": "u", "b": "u"},
        {0: "w1", 1800: "w1", 3600: "w2"}, minutes=30,
    )
    lag = json.loads(report)["groups"][0]["comparisons"][0]["lags"][0]
    # e at bucket 1800 is 1, at bucket 3600 is 2 -> w1 = [1], w2 = [2].
    assert lag["n_a"] == 1
    assert lag["n_b"] == 1
    assert lag["diff"] == -1.0


def test_tables_not_list_raise_type_error():
    with pytest.raises(TypeError):
        scenario_shift("not a list", [], [], [1], {}, {})
    with pytest.raises(TypeError):
        scenario_shift([], "not a list", [], [1], {}, {})


def test_neighbors_lags_labels_windows_type_errors():
    with pytest.raises(TypeError):
        scenario_shift([], [], "not a list", [1], {}, {})
    with pytest.raises(TypeError):
        scenario_shift([], [], [], "not a list", {}, {})
    with pytest.raises(TypeError):
        scenario_shift([], [], [], [1], "not a dict", {})
    with pytest.raises(TypeError):
        scenario_shift([], [], [], [1], {}, "not a dict")


def test_differing_key_sets_raise_value_error():
    before = [_row(0, "a", 1.0)]
    after = [_row(3600, "a", 1.0)]
    with pytest.raises(ValueError):
        scenario_shift(before, after, [], [1], {"a": "u"}, {})
    other_cell = [_row(0, "b", 1.0)]
    with pytest.raises(ValueError):
        scenario_shift(before, other_cell, [], [1], {"a": "u"}, {})
    bigger = [_row(0, "a", 1.0), _row(3600, "a", 1.0)]
    with pytest.raises(ValueError):
        scenario_shift(before, bigger, [], [1], {"a": "u"}, {})
    with pytest.raises(ValueError):
        scenario_shift(bigger, before, [], [1], {"a": "u"}, {})


def test_invalid_rows_raise_value_error():
    good = [_row(0, "a", 1.0)]
    bad_row = [0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # not a tuple
    with pytest.raises(ValueError):
        scenario_shift([bad_row], good, [], [1], {"a": "u"}, {})
    with pytest.raises(ValueError):
        scenario_shift(good, [bad_row], [], [1], {"a": "u"}, {})
    duplicate = [_row(0, "a", 1.0), _row(0, "a", 2.0)]
    with pytest.raises(ValueError):
        scenario_shift(duplicate, good, [], [1], {"a": "u"}, {})
    with pytest.raises(ValueError):
        scenario_shift(good, duplicate, [], [1], {"a": "u"}, {})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes": 0},
        {"minutes": 7},
        {"minutes": 1441},
        {"minutes": True},
        {"alpha": 0},
        {"alpha": 1.5},
        {"alpha": True},
        {"alpha": float("nan")},
    ],
)
def test_invalid_minutes_alpha_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        scenario_shift([], [], [], [1], {}, {}, **kwargs)


@pytest.mark.parametrize("lag", [0, -1, 1.5, True, "1"])
def test_invalid_lags_raise_value_error(lag):
    with pytest.raises(ValueError):
        scenario_shift([], [], [], [lag], {}, {})
    with pytest.raises(ValueError):
        scenario_shift([], [], [], [1, 1], {}, {})


def test_windows_contract_violations_raise_value_error():
    before = [_row(t, "a", 1.0) for t in (0, 3600)]
    after = [_row(t, "a", 2.0) for t in (0, 3600)]
    with pytest.raises(ValueError):
        scenario_shift(before, after, [], [1], {"a": "u"}, {0: "w1"})
    with pytest.raises(ValueError):
        scenario_shift(
            before, after, [], [1], {"a": "u"}, {0: "w1", 3600: ""}
        )
    with pytest.raises(ValueError):
        scenario_shift(
            before, after, [], [1], {"a": "u"}, {0: "w1", 3600: 2}
        )


def test_labels_contract_violations_raise_value_error():
    before = [_row(0, "a", 1.0)]
    after = [_row(0, "a", 2.0)]
    with pytest.raises(ValueError):
        scenario_shift(before, after, [], [1], {}, {0: "w1"})
    with pytest.raises(ValueError):
        scenario_shift(before, after, [], [1], {"a": ""}, {0: "w1"})


def test_more_than_16_combined_values_raises_value_error():
    cells = tuple(f"c{i}" for i in range(10))
    neighbors = [(cells[i], cells[i + 1]) for i in range(9)]
    labels = {c: "u" for c in cells}
    times = (0, 3600, 7200)
    before = [_row(t, c, 1.0) for t in times for c in cells]
    after = [
        _row(t, c, float(i) if t == 7200 else 0.0)
        for t in times
        for i, c in enumerate(cells)
    ]
    windows = {0: "w1", 3600: "w1", 7200: "w2"}
    with pytest.raises(ValueError):
        scenario_shift(before, after, neighbors, [1], labels, windows)


def test_16_combined_values_allowed():
    cells = tuple(f"c{i}" for i in range(9))
    neighbors = [(cells[i], cells[i + 1]) for i in range(8)]
    labels = {c: "u" for c in cells}
    times = (0, 3600, 7200)
    before = [_row(t, c, 1.0) for t in times for c in cells]
    after = [
        _row(t, c, float(i) if t == 7200 else 0.0)
        for t in times
        for i, c in enumerate(cells)
    ]
    windows = {0: "w1", 3600: "w1", 7200: "w2"}
    report = scenario_shift(before, after, neighbors, [1], labels, windows)
    lag = json.loads(report)["groups"][0]["comparisons"][0]["lags"][0]
    assert lag["n_a"] == 8
    assert lag["n_b"] == 8


def test_no_comparison_when_window_sample_empty():
    # Only one window has lag pairings, so there is nothing to compare.
    before = [_row(t, c, 1.0) for t in (0, 3600) for c in ("a", "b")]
    after = [_row(t, c, 2.0) for t in (0, 3600) for c in ("a", "b")]
    report = scenario_shift(
        before, after, [("a", "b")], [1], {"a": "u", "b": "u"},
        {0: "w1", 3600: "w2"},
    )
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}\n'
