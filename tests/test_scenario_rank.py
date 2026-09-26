"""Tests for urban_micro.scenario_rank and urban_micro.scenario_shift."""

import json
from fractions import Fraction
from itertools import combinations

import pytest

from urban_micro import scenario_rank, scenario_shift


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _three_bucket_tables():
    """Three buckets with same-label cells a/b/c and edges ab/ac.

    Lag-1 pairings exist at buckets 3600 and 7200. The three tables differ
    only in cell ``a``'s delta, so every ``e`` below is the change in
    ``a``'s bucket-to-bucket delta along each edge.
    """
    base_rows = [
        (0, "a", 1.0), (0, "b", 4.0), (0, "c", 10.0),
        (3600, "a", 3.0), (3600, "b", 10.0), (3600, "c", 12.0),
        (7200, "a", 2.0), (7200, "b", 8.0), (7200, "c", 20.0),
    ]
    lo_rows = [
        (0, "a", 2.0), (0, "b", 4.0), (0, "c", 10.0),
        (3600, "a", 5.0), (3600, "b", 10.0), (3600, "c", 12.0),
        (7200, "a", 4.0), (7200, "b", 8.0), (7200, "c", 20.0),
    ]
    hi_rows = [
        (0, "a", 1.0), (0, "b", 4.0), (0, "c", 10.0),
        (3600, "a", 6.0), (3600, "b", 10.0), (3600, "c", 12.0),
        (7200, "a", 5.0), (7200, "b", 8.0), (7200, "c", 20.0),
    ]
    make = lambda rows: [_row(t, c, d) for t, c, d in rows]
    neighbors = [("a", "b"), ("a", "c")]
    labels = {"a": "L", "b": "L", "c": "L"}
    windows = {0: "p", 3600: "p", 7200: "q"}
    return (
        make(base_rows), make(lo_rows), make(hi_rows),
        neighbors, labels, windows,
    )


# ---------------------------------------------------------------------------
# scenario_rank: statistics
# ---------------------------------------------------------------------------


def _reference_rank(
    base, sets, neighbors, lags, labels, windows, *, minutes=60
):
    """Fraction-exact reference: returns {group: {"ranks": [...],
    "tests": {(a,b): (diff, p, q)}}}, with q filled via the same BH rule."""
    width = minutes * 60

    def bucket_means(table):
        totals = {}
        for t, c, _, _, delta, *_ in table:
            b = (t // width) * width
            acc = totals.setdefault((b, c), [Fraction(0), 0])
            acc[0] += Fraction(delta)
            acc[1] += 1
        return {key: total / n for key, (total, n) in totals.items()}

    means_tables = {"": bucket_means(base)}
    for name, table in sets.items():
        means_tables[name] = bucket_means(table)

    cell_edges = {}
    for a, b in neighbors:
        edge = (a, b) if a < b else (b, a)
        cell_edges.setdefault(labels[edge[0]], set())
        if labels[a] == labels[b]:
            cell_edges.setdefault(labels[a], set()).add(edge)

    def x_values(means, label, lag, bucket):
        previous = {
            c: v for (b, c), v in means.items() if b == bucket - lag * width
        }
        current = {c: v for (b, c), v in means.items() if b == bucket}
        out = []
        for a, b in sorted(cell_edges.get(label, ())):
            if (
                a in current and b in current
                and a in previous and b in previous
            ):
                out.append(
                    (current[a] - previous[a]) - (current[b] - previous[b])
                )
        return out

    samples = {}
    for name in sets:
        for label in cell_edges:
            for lag in lags:
                buckets = sorted({b for (b, _c) in means_tables[name]})
                for bucket in buckets:
                    x_set = x_values(means_tables[name], label, lag, bucket)
                    if not x_set:
                        continue
                    x_base = x_values(means_tables[""], label, lag, bucket)
                    group = (label, windows[bucket], lag)
                    pocket = samples.setdefault(group, {}).setdefault(name, [])
                    pocket.extend(
                        value - base_value
                        for value, base_value in zip(x_set, x_base)
                    )

    groups = {}
    records = []
    for group, pocket in samples.items():
        if len(pocket) < 2:
            continue
        mean = {
            name: sum(values, Fraction(0)) / len(values)
            for name, values in pocket.items()
        }
        ordered = sorted(mean, key=lambda name: (mean[name], name))
        groups[group] = {
            "ranks": [(n, mean[n], i) for i, n in enumerate(ordered, 1)]
        }
        names = sorted(pocket)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa, sb = pocket[a], pocket[b]
                na, nb = len(sa), len(sb)
                diff = sum(sa, Fraction(0)) / na - sum(sb, Fraction(0)) / nb
                pooled = sa + sb
                total = sum(pooled, Fraction(0))
                threshold = abs(sum(sa) * nb - sum(sb) * na)
                hits = 0
                for combo in combinations(range(na + nb), na):
                    pa = sum(pooled[pos] for pos in combo)
                    pb = total - pa
                    if abs(pa * nb - pb * na) >= threshold:
                        hits += 1
                count_assignments = len(
                    list(combinations(range(na + nb), na))
                )
                p_value = Fraction(hits, count_assignments)
                record = [group, a, b, diff, p_value, None]
                records.append(record)
                groups[group].setdefault("tests", []).append(record)

    count = len(records)
    ordered_records = sorted(
        records, key=lambda r: (r[4], r[0][0], r[0][1], r[0][2], r[1], r[2])
    )
    running = Fraction(1)
    for rank in range(count, 0, -1):
        record = ordered_records[rank - 1]
        running = min(running, Fraction(count) * record[4] / rank)
        record[5] = running
    return groups


def test_rank_matches_fraction_reference_across_groups():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    mid_rows = [
        (0, "a", 0.0), (0, "b", 4.0), (0, "c", 10.0),
        (3600, "a", 4.0), (3600, "b", 11.0), (3600, "c", 12.0),
        (7200, "a", 9.0), (7200, "b", 5.0), (7200, "c", 21.0),
    ]
    mid = [_row(t, c, d) for t, c, d in mid_rows]
    sets = {"hi": hi, "lo": lo, "mid": mid}
    lags = [1, 2]
    report = scenario_rank(
        base, sets, neighbors, lags, labels, windows, minutes=60, alpha=0.2
    )
    payload = json.loads(report)
    reference = _reference_rank(base, sets, neighbors, lags, labels, windows)

    assert len(payload["groups"]) == len(reference)
    for group_object, (group, expected) in zip(
        payload["groups"], sorted(reference.items())
    ):
        label, window, lag = group
        assert group_object["label"] == label
        assert group_object["window"] == window
        assert group_object["lag"] == lag
        assert isinstance(group_object["lag"], int)

        assert [r["key"] for r in group_object["ranks"]] == [
            name for name, _m, _r in expected["ranks"]
        ]
        for rank_object, (name, mean_value, rank_value) in zip(
            group_object["ranks"], expected["ranks"]
        ):
            assert rank_object["key"] == name
            assert rank_object["rank"] == rank_value
            assert isinstance(rank_object["rank"], int)
            assert rank_object["mean"] == pytest.approx(
                float(mean_value), abs=5e-7
            )

        by_pair = {(r[1], r[2]): r for r in expected["tests"]}
        pairs = [(t["a"], t["b"]) for t in group_object["tests"]]
        assert pairs == sorted(by_pair)
        for test_object in group_object["tests"]:
            record = by_pair[(test_object["a"], test_object["b"])]
            assert test_object["diff"] == pytest.approx(
                float(record[3]), abs=5e-7
            )
            assert test_object["p"] == pytest.approx(
                float(record[4]), abs=5e-7
            )
            assert test_object["q"] == pytest.approx(
                float(record[5]), abs=5e-7
            )
            assert test_object["reject"] == (record[5] <= Fraction("0.2"))
            assert isinstance(test_object["reject"], bool)


def test_rank_exact_two_set_report():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    report = scenario_rank(
        base, {"hi": hi, "lo": lo}, neighbors, [1], labels, windows
    )
    assert report == (
        '{"minutes":60,"alpha":0.050000,"groups":['
        '{"label":"L","window":"p","lag":1,'
        '"ranks":[{"key":"lo","mean":1.000000,"rank":1},'
        '{"key":"hi","mean":3.000000,"rank":2}],'
        '"tests":[{"a":"hi","b":"lo","diff":2.000000,"p":0.333333,'
        '"q":0.666667,"reject":false}]},'
        '{"label":"L","window":"q","lag":1,'
        '"ranks":[{"key":"hi","mean":0.000000,"rank":1},'
        '{"key":"lo","mean":0.000000,"rank":2}],'
        '"tests":[{"a":"hi","b":"lo","diff":0.000000,"p":1.000000,'
        '"q":1.000000,"reject":false}]}'
        ']}\n'
    )


def test_ranks_tie_on_mean_fall_back_to_key():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    # Window q has zero e in both sets, so means tie at 0.
    report = json.loads(
        scenario_rank(
            base, {"hi": hi, "lo": lo}, neighbors, [1], labels, windows
        )
    )
    q_group = next(g for g in report["groups"] if g["window"] == "q")
    assert [r["key"] for r in q_group["ranks"]] == ["hi", "lo"]
    assert [r["rank"] for r in q_group["ranks"]] == [1, 2]


def test_groups_ordered_by_label_window_lag():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    labels = {"a": "Z", "b": "Z", "c": "Z"}
    report = json.loads(
        scenario_rank(
            base, {"hi": hi, "lo": lo}, neighbors, [2, 1], labels, windows
        )
    )
    assert [
        (g["label"], g["window"], g["lag"]) for g in report["groups"]
    ] == [
        ("Z", "p", 1),
        ("Z", "q", 1),
        ("Z", "q", 2),
    ]
    # Ranks within a group are (rank, key) sorted and tests (a, b) sorted.
    for group in report["groups"]:
        assert [r["rank"] for r in group["ranks"]] == list(
            range(1, len(group["ranks"]) + 1)
        )
        pairs = [(t["a"], t["b"]) for t in group["tests"]]
        assert pairs == sorted(pairs)


def test_samples_concatenate_over_window_buckets():
    # Window "p" covers buckets 0 and 3600; lag-1 values exist only at
    # bucket 3600 here, giving two edge values per set.
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    report = json.loads(
        scenario_rank(
            base, {"hi": hi, "lo": lo}, neighbors, [1], labels, windows
        )
    )
    p_group = next(g for g in report["groups"] if g["window"] == "p")
    means = {r["key"]: r["mean"] for r in p_group["ranks"]}
    assert means == {"hi": 3.0, "lo": 1.0}


def test_empty_lags_gives_empty_groups():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    report = scenario_rank(
        base, {"hi": hi, "lo": lo}, neighbors, [], labels, windows
    )
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}\n'


def test_no_same_label_edges_gives_empty_groups():
    base, lo, hi, _neighbors, _labels, windows = _three_bucket_tables()
    # Cross-label edges never produce pairings.
    neighbors = [("a", "x"), ("b", "y"), ("c", "z")]
    labels = {"a": "L", "b": "L", "c": "L", "x": "M", "y": "M", "z": "M"}
    for extra in (_row(0, "x", 1.0), _row(0, "y", 2.0), _row(0, "z", 3.0)):
        for table in (base, lo, hi):
            table.append(extra)
    report = scenario_rank(
        base, {"hi": hi, "lo": lo}, neighbors, [1], labels, windows
    )
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}\n'


def test_alpha_one_rejects_everything():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    report = json.loads(
        scenario_rank(
            base, {"hi": hi, "lo": lo}, neighbors, [1], labels,
            windows, alpha=1,
        )
    )
    assert all(
        test["reject"]
        for group in report["groups"]
        for test in group["tests"]
    )


def test_more_than_sixteen_combined_values_raises_value_error():
    # Ten cells, nine edges contribute at two paired buckets within one
    # window -> 18 values per set -> 36 combined.
    cells = [f"c{i}" for i in range(10)]
    edges = [(cells[0], cells[i]) for i in range(1, 10)]
    labels = {cell: "L" for cell in cells}

    def table(shift):
        rows = []
        for cell in cells:
            rows.append(_row(0, cell, 1.0))
            rows.append(_row(3600, cell, 1.0 + shift))
            rows.append(_row(7200, cell, 1.0))
        return rows

    base = table(0.0)
    set_a = table(0.1)
    set_b = table(0.2)
    windows = {0: "p", 3600: "q", 7200: "q"}
    with pytest.raises(ValueError, match="scenario rank"):
        scenario_rank(
            base, {"a": set_a, "b": set_b}, edges, [1], labels, windows
        )


def test_unicode_set_name_is_escaped_compactly():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    report = scenario_rank(
        base, {"sét": hi, "lö": lo}, neighbors, [1], labels, windows
    )
    assert '"key":"sét"' in report
    assert " " not in report.rstrip("\n")
    assert report.endswith("\n") and not report.endswith("\n\n")


# ---------------------------------------------------------------------------
# scenario_rank: error contract
# ---------------------------------------------------------------------------


def test_base_not_list_raises_type_error():
    _base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    with pytest.raises(TypeError, match="base must be a list"):
        scenario_rank(
            tuple(_base), {"hi": hi, "lo": lo}, neighbors, [1],
            labels, windows,
        )


def test_sets_not_dict_raises_type_error():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    with pytest.raises(TypeError, match="sets must be a dict"):
        scenario_rank(
            base, [("hi", hi), ("lo", lo)], neighbors, [1],
            labels, windows,
        )


def test_fewer_than_two_sets_raises_value_error():
    base, _lo, hi, neighbors, labels, windows = _three_bucket_tables()
    with pytest.raises(ValueError, match="at least two"):
        scenario_rank(base, {"hi": hi}, neighbors, [1], labels, windows)


def test_empty_set_key_raises_value_error():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    with pytest.raises(ValueError, match="non-empty string"):
        scenario_rank(
            base, {"": lo, "hi": hi}, neighbors, [1], labels, windows
        )


def test_non_string_set_key_raises_value_error():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    with pytest.raises(ValueError, match="non-empty string"):
        scenario_rank(base, {1: lo, "hi": hi}, neighbors, [1], labels, windows)


def test_set_value_not_list_raises_value_error():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    with pytest.raises(ValueError, match="must be a list"):
        scenario_rank(
            base, {"hi": tuple(hi), "lo": lo}, neighbors, [1],
            labels, windows,
        )


def test_set_key_mismatch_raises_value_error():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    broken = hi[:-1]
    with pytest.raises(ValueError, match="share the same"):
        scenario_rank(
            base, {"hi": broken, "lo": lo}, neighbors, [1], labels, windows
        )


def test_windows_not_dict_raises_type_error():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    with pytest.raises(TypeError, match="windows must be a dict"):
        scenario_rank(
            base, {"hi": hi, "lo": lo}, neighbors, [1], labels, [(0, "p")]
        )


def test_windows_wrong_keys_raise_value_error():
    base, lo, hi, neighbors, labels, _windows = _three_bucket_tables()
    with pytest.raises(ValueError, match="bucket starts"):
        scenario_rank(
            base, {"hi": hi, "lo": lo}, neighbors, [1], labels,
            {0: "p", 3600: "p"},
        )


def test_empty_window_name_raises_value_error():
    base, lo, hi, neighbors, labels, _windows = _three_bucket_tables()
    windows = {0: "p", 3600: "p", 7200: ""}
    with pytest.raises(ValueError, match="window name"):
        scenario_rank(
            base, {"hi": hi, "lo": lo}, neighbors, [1], labels, windows
        )


def test_invalid_alpha_raises_value_error():
    base, lo, hi, neighbors, labels, windows = _three_bucket_tables()
    with pytest.raises(ValueError):
        scenario_rank(
            base, {"hi": hi, "lo": lo}, neighbors, [1], labels,
            windows, alpha=0,
        )


# ---------------------------------------------------------------------------
# scenario_shift
# ---------------------------------------------------------------------------


def test_shift_exact_report():
    base, _lo, hi, neighbors, labels, windows = _three_bucket_tables()
    report = scenario_shift(base, hi, neighbors, [1], labels, windows)
    assert report == (
        '{"minutes":60,"alpha":0.050000,"groups":['
        '{"key":"L","comparisons":['
        '{"a":"p","b":"q","lags":['
        '{"lag":1,"n_a":2,"n_b":2,"diff":3.000000,"p":0.333333,'
        '"q":0.333333,"reject":false}]}]}]}\n'
    )


def test_shift_lag_and_comparison_ordering():
    # Four buckets: window p covers 0/3600/7200, window q covers 10800.
    # Lag 1 pairs at 3600 (p), 7200 (p) and 10800 (q); lag 2 pairs at
    # 7200 (p) and 10800 (q), so both windows carry both lags.
    base_rows = [
        (t, c, 1.0)
        for t in (0, 3600, 7200, 10800)
        for c in ("a", "b", "c")
    ]
    after_rows = [
        (t, "a", 1.0 + (0.1 if t else 0.0))
        for t in (0, 3600, 7200, 10800)
    ] + [
        (t, c, 1.0)
        for t in (0, 3600, 7200, 10800)
        for c in ("b", "c")
    ]
    base = [_row(t, c, d) for t, c, d in base_rows]
    after = [_row(t, c, d) for t, c, d in after_rows]
    neighbors = [("a", "b"), ("a", "c")]
    labels = {"a": "L", "b": "L", "c": "L"}
    windows = {0: "p", 3600: "p", 7200: "p", 10800: "q"}
    report = json.loads(
        scenario_shift(base, after, neighbors, [2, 1], labels, windows)
    )
    group = report["groups"][0]
    assert group["key"] == "L"
    pairs = [(c["a"], c["b"]) for c in group["comparisons"]]
    assert pairs == [("p", "q")]
    seen_lags = [lag["lag"] for c in group["comparisons"] for lag in c["lags"]]
    assert seen_lags == sorted(seen_lags) == [1, 2]
    for comparison in group["comparisons"]:
        for lag in comparison["lags"]:
            assert isinstance(lag["lag"], int)
            assert isinstance(lag["n_a"], int)
            assert isinstance(lag["n_b"], int)
            assert isinstance(lag["reject"], bool)


def test_shift_before_not_list_raises_type_error():
    base, _lo, hi, neighbors, labels, windows = _three_bucket_tables()
    with pytest.raises(TypeError, match="before must be a list"):
        scenario_shift(tuple(base), hi, neighbors, [1], labels, windows)
    with pytest.raises(TypeError, match="after must be a list"):
        scenario_shift(base, tuple(hi), neighbors, [1], labels, windows)


def test_shift_key_mismatch_raises_value_error():
    base, lo, _hi, neighbors, labels, windows = _three_bucket_tables()
    with pytest.raises(ValueError, match="share the same"):
        scenario_shift(base, lo[:-1], neighbors, [1], labels, windows)


def test_shift_empty_lags_gives_empty_groups():
    base, _lo, hi, neighbors, labels, windows = _three_bucket_tables()
    report = scenario_shift(base, hi, neighbors, [], labels, windows)
    assert report == '{"minutes":60,"alpha":0.050000,"groups":[]}\n'


def test_shift_windows_wrong_keys_raise_value_error():
    base, _lo, hi, neighbors, labels, _windows = _three_bucket_tables()
    with pytest.raises(ValueError, match="bucket starts"):
        scenario_shift(base, hi, neighbors, [1], labels, {0: "p", 3600: "p"})


def test_shift_more_than_sixteen_combined_raises_value_error():
    cells = [f"c{i}" for i in range(10)]
    edges = [(cells[0], cells[i]) for i in range(1, 10)]
    labels = {cell: "L" for cell in cells}

    def table(shift):
        rows = []
        for cell in cells:
            rows.append(_row(0, cell, 1.0))
            rows.append(_row(3600, cell, 1.0 + shift))
            rows.append(_row(7200, cell, 2.0))
            rows.append(_row(10800, cell, 2.0 + shift))
        return rows

    base = table(0.0)
    after = table(0.1)
    windows = {0: "p", 3600: "p", 7200: "q", 10800: "q"}
    with pytest.raises(ValueError, match="scenario shift"):
        scenario_shift(base, after, edges, [1], labels, windows)


def test_shift_reject_fires_at_small_alpha_threshold():
    # Two fully separated samples of size 2 each give p = 2/6 = 1/3; with a
    # single comparison q = 1/3, so alpha >= 1/3 rejects.
    base, lo, _hi, neighbors, labels, windows = _three_bucket_tables()
    rejected = json.loads(
        scenario_shift(base, lo, neighbors, [1], labels, windows, alpha=0.34)
    )
    lag = rejected["groups"][0]["comparisons"][0]["lags"][0]
    assert lag["p"] == pytest.approx(1 / 3, abs=5e-7)
    assert lag["reject"] is True
