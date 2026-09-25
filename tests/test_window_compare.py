"""Tests for urban_micro.window_compare."""

import json
from fractions import Fraction
from itertools import combinations

import pytest

from urban_micro import window_compare


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _permutation(sample_a, sample_b):
    """Reference diff and exact two-sided permutation p in Fraction."""
    values_a = [Fraction(value) for value in sample_a]
    values_b = [Fraction(value) for value in sample_b]
    n_a, n_b = len(values_a), len(values_b)
    sum_a, sum_b = sum(values_a, Fraction(0)), sum(values_b, Fraction(0))
    diff = sum_a / n_a - sum_b / n_b
    threshold = abs(sum_a * n_b - sum_b * n_a)
    pooled = values_a + values_b
    pooled_sum = sum_a + sum_b
    assignments = 0
    hits = 0
    for combo in combinations(range(n_a + n_b), n_a):
        perm_a = sum((pooled[position] for position in combo), Fraction(0))
        perm_b = pooled_sum - perm_a
        if abs(perm_a * n_b - perm_b * n_a) >= threshold:
            hits += 1
        assignments += 1
    return diff, Fraction(hits, assignments)


def _groups(details, neighbors, labels, windows, lags=(1,), **kwargs):
    report = window_compare(
        details, list(neighbors), list(lags), labels, windows, **kwargs
    )
    return report, json.loads(report)


THREE_BUCKETS = {
    0: {"a1": 0, "a2": 0, "b1": 0, "b2": 0},
    3600: {"a1": 1, "a2": 4, "b1": 2, "b2": 5},
    7200: {"a1": 3, "a2": 10, "b1": 1, "b2": 2},
}
NEIGHBORS = [("a1", "a2"), ("b1", "b2")]
LABELS = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}


def _details(values=THREE_BUCKETS):
    rows = []
    for timestamp, cells in values.items():
        for cell_id, delta in cells.items():
            rows.append(_row(timestamp, cell_id, float(delta)))
    return rows


def test_empty_details_requires_empty_windows_and_gives_empty_groups():
    report = window_compare([], [], [], {}, {})
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}\n'
    assert json.loads(report) == {
        "minutes": 60,
        "alpha": 0.05,
        "groups": [],
    }


def test_windows_non_dict_type_error():
    with pytest.raises(TypeError):
        window_compare(_details(), NEIGHBORS, [1], LABELS, [(0, "w")])


@pytest.mark.parametrize(
    "windows",
    [
        {0: "w1", 3600: "w1"},  # missing bucket 7200
        {0: "w1", 3600: "w1", 7200: "w1", 10800: "w1"},  # extra bucket
    ],
)
def test_windows_keys_must_match_buckets(windows):
    with pytest.raises(ValueError):
        window_compare(_details(), NEIGHBORS, [1], LABELS, windows)


@pytest.mark.parametrize("bad_value", ["", 3, None])
def test_window_values_must_be_non_empty_strings(bad_value):
    windows = {0: "w1", 3600: "w1", 7200: bad_value}
    with pytest.raises(ValueError):
        window_compare(_details(), NEIGHBORS, [1], LABELS, windows)


def test_window_concatenates_buckets_in_ascending_order():
    # One window spans all buckets: A sample = [x@3600, x@7200] = [-3, -4]
    # and B sample = [-3, 2].
    windows = {0: "w", 3600: "w", 7200: "w"}
    report, parsed = _groups(_details(), NEIGHBORS, LABELS, windows)
    assert report.endswith("\n")
    group = parsed["groups"][0]
    assert group["key"] == "w"
    assert list(group) == ["key", "comparisons"]
    comparison = group["comparisons"][0]
    assert list(comparison) == ["a", "b", "lags"]
    assert comparison["a"] == "A" and comparison["b"] == "B"
    entry = comparison["lags"][0]
    assert list(entry) == ["lag", "n_a", "n_b", "diff", "p", "q", "reject"]
    assert entry["lag"] == 1 and entry["n_a"] == 2 and entry["n_b"] == 2

    diff, p_value = _permutation([-3, -4], [-3, 2])
    assert entry["diff"] == pytest.approx(float(diff))
    assert entry["p"] == pytest.approx(float(p_value))
    assert entry["q"] == pytest.approx(float(p_value))  # N = 1
    assert entry["reject"] is False


def test_bucket_boundaries_split_windows():
    windows = {0: "w1", 3600: "w1", 7200: "w2"}
    _, parsed = _groups(_details(), NEIGHBORS, LABELS, windows)
    groups = parsed["groups"]
    assert [group["key"] for group in groups] == ["w1", "w2"]
    first = groups[0]["comparisons"][0]["lags"][0]
    second = groups[1]["comparisons"][0]["lags"][0]
    # w1: only B=3600 pairs with B=0 -> A=-3, B=-3, diff 0.
    assert (first["n_a"], first["n_b"], first["diff"]) == (1, 1, 0.0)
    # w2: B=7200 pairs across the boundary with B=3600 -> A=-4, B=2, diff -6.
    assert (second["n_a"], second["n_b"], second["diff"]) == (1, 1, -6.0)


def test_lags_render_in_ascending_order():
    windows = {0: "w", 3600: "w", 7200: "w"}
    _, parsed = _groups(
        _details(), NEIGHBORS, LABELS, windows, lags=(2, 1)
    )
    lags = parsed["groups"][0]["comparisons"][0]["lags"]
    assert [entry["lag"] for entry in lags] == [1, 2]
    # lag 2 only pairs B=7200 with B=0: A x = -7, B x = -1.
    diff, p_value = _permutation([-7], [-1])
    entry = lags[1]
    assert (entry["n_a"], entry["n_b"]) == (1, 1)
    assert entry["diff"] == pytest.approx(float(diff))
    assert entry["p"] == pytest.approx(float(p_value))


def test_global_bh_across_windows_and_lags():
    # Two windows each yield one lag-1 comparison, so N = 2 and q values
    # come from the shared BH ranking, not per-window ranks.
    windows = {0: "w1", 3600: "w1", 7200: "w2"}
    _, parsed = _groups(_details(), NEIGHBORS, LABELS, windows)
    entries = [
        group["comparisons"][0]["lags"][0] for group in parsed["groups"]
    ]
    p_values = sorted(Fraction(str(entry["p"])) for entry in entries)
    n_total = len(entries)
    expected_q = [
        min(Fraction(1), min(
            Fraction(n_total) * p_values[l - 1] / l
            for l in range(j, n_total + 1)
        ))
        for j in range(1, n_total + 1)
    ]
    for entry in entries:
        rank = p_values.index(Fraction(str(entry["p"])))
        assert entry["q"] == pytest.approx(float(expected_q[rank]))


def test_combined_sample_over_16_raises():
    rows = []
    windows = {}
    for k in range(9):  # nine independent lag pairings per label -> 9 + 9
        base = 14400 * k
        for cell_id in ("a1", "a2", "b1", "b2"):
            delta = 1.0 if cell_id == "a1" else (2.0 if cell_id == "b1" else 0.0)
            rows.append(_row(base, cell_id, 0.0))
            rows.append(_row(base + 3600, cell_id, delta))
        windows[base] = "w"
        windows[base + 3600] = "w"
    with pytest.raises(ValueError):
        window_compare(rows, NEIGHBORS, [1], LABELS, windows)


def test_combined_sample_of_16_is_allowed():
    rows = []
    windows = {}
    for k in range(8):
        base = 14400 * k
        for cell_id in ("a1", "a2", "b1", "b2"):
            delta = 1.0 if cell_id == "a1" else (2.0 if cell_id == "b1" else 0.0)
            rows.append(_row(base, cell_id, 0.0))
            rows.append(_row(base + 3600, cell_id, delta))
        windows[base] = "w"
        windows[base + 3600] = "w"
    report = window_compare(rows, NEIGHBORS, [1], LABELS, windows)
    entry = json.loads(report)["groups"][0]["comparisons"][0]["lags"][0]
    assert (entry["n_a"], entry["n_b"]) == (8, 8)


def test_single_label_yields_no_comparisons():
    windows = {0: "w1", 3600: "w1", 7200: "w2"}
    labels = {cell_id: "A" for cell_id in LABELS}
    report = window_compare(_details(), NEIGHBORS, [1], labels, windows)
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}\n'


def test_empty_lags_yields_no_comparisons():
    windows = {0: "w", 3600: "w", 7200: "w"}
    report = window_compare(_details(), NEIGHBORS, [], LABELS, windows)
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}\n'


def test_unicode_window_name_preserved_unescaped():
    windows = {0: "窗α", 3600: "窗α", 7200: "窗α"}
    report, parsed = _groups(_details(), NEIGHBORS, LABELS, windows)
    assert parsed["groups"][0]["key"] == "窗α"
    assert "窗α" in report
