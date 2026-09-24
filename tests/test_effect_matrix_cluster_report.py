"""Tests for urban_micro.effect_matrix_cluster_report."""

import itertools
import json

import pytest

from urban_micro import effect_matrix_cluster_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def _bruteforce(details, neighbors, *, minutes=60, threshold=0.0):
    """Reference bucketed cell means, connected components, kinds and
    sign-enumeration p-values."""
    buckets = {}
    for t, c, delta in ((r[0], r[1], r[4]) for r in details):
        bucket = (t // (minutes * 60)) * (minutes * 60)
        buckets.setdefault(bucket, {}).setdefault(c, [0.0, 0])
        buckets[bucket][c][0] += delta
        buckets[bucket][c][1] += 1

    result = []
    for bucket in sorted(buckets):
        means = {
            c: total / count
            for c, (total, count) in buckets[bucket].items()
            if abs(total / count) >= threshold
        }
        adjacency = {c: set() for c in means}
        for a, b in neighbors:
            if a in means and b in means:
                adjacency[a].add(b)
                adjacency[b].add(a)

        visited = set()
        clusters = []
        for c in sorted(means):
            if c in visited:
                continue
            stack = [c]
            visited.add(c)
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for other in adjacency[node]:
                    if other not in visited:
                        visited.add(other)
                        stack.append(other)
            component.sort()
            values = [means[c] for c in component]
            k = len(values)
            mean = sum(values) / k
            peak = values[0]
            for value in values[1:]:
                if abs(value) > abs(peak):
                    peak = value
            hits = 0
            for signs in itertools.product((-1, 1), repeat=k):
                signed = sum(sign * value for sign, value in zip(signs, values))
                if abs(signed / k) >= abs(mean) - 1e-12:
                    hits += 1
            p_value = hits / 2**k
            kind = "hot" if mean > 0 else "cold" if mean < 0 else "mixed"
            clusters.append((component, mean, peak, p_value, kind))
        result.append((bucket, clusters))
    return result


def _clusters(report):
    return [(group["key"], group["clusters"]) for group
            in json.loads(report)["groups"]]


def test_empty_details_gives_empty_groups():
    report = effect_matrix_cluster_report([], [])
    assert report == '{"minutes":60,"threshold":0.000000,"groups":[]}'
    assert json.loads(report) == {
        "minutes": 60, "threshold": 0.0, "groups": []
    }


def test_empty_with_explicit_minutes_and_threshold():
    report = effect_matrix_cluster_report(
        [], [], minutes=30, threshold=0.25
    )
    assert report == '{"minutes":30,"threshold":0.250000,"groups":[]}'


def test_path_matches_bruteforce():
    # d = 1, 2, -3 on a path a-b-c; only a and b are connected (c shares
    # no edge with them either, but here a-b-c is one component).
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0), _row(0, "c", -3.0)]
    neighbors = [("a", "b"), ("b", "c")]
    report = effect_matrix_cluster_report(details, neighbors)
    groups = _clusters(report)
    expected = _bruteforce(details, neighbors)
    assert [(b, len(cs)) for b, cs in groups] == [(0, 1)]
    ((bucket, clusters),) = groups
    ((ebucket, e_clusters),) = expected
    assert bucket == ebucket
    assert len(clusters) == len(e_clusters) == 1
    cluster = clusters[0]
    e_cells, e_mean, e_peak, e_p, e_kind = e_clusters[0]
    assert cluster["cells"] == e_cells == ["a", "b", "c"]
    assert cluster["key"] == "a"
    assert cluster["n"] == 3
    assert cluster["mean"] == pytest.approx(e_mean, abs=1e-6) == 0.0
    assert cluster["peak"] == pytest.approx(e_peak, abs=1e-6) == -3.0
    assert cluster["p"] == pytest.approx(e_p, abs=1e-6)
    assert cluster["kind"] == e_kind == "mixed"


def test_isolated_cells_become_singleton_clusters_sorted():
    details = [_row(0, "c", 3.0), _row(0, "a", 1.0), _row(0, "b", 2.0)]
    report = effect_matrix_cluster_report(details, [])
    groups = json.loads(report)["groups"]
    assert [c["key"] for c in groups[0]["clusters"]] == ["a", "b", "c"]
    for cluster, delta in zip(groups[0]["clusters"], (1.0, 2.0, 3.0)):
        assert cluster["cells"] == [cluster["key"]]
        assert cluster["n"] == 1
        assert cluster["mean"] == delta
        assert cluster["peak"] == delta
        assert cluster["p"] == 1.0
        assert cluster["kind"] == "hot"


def test_hot_and_cold_kinds_and_peak_sign():
    report = effect_matrix_cluster_report(
        [_row(0, "a", -1.0), _row(0, "b", -3.0)], [("a", "b")]
    )
    cluster = json.loads(report)["groups"][0]["clusters"][0]
    assert cluster["kind"] == "cold"
    assert cluster["mean"] == -2.0
    assert cluster["peak"] == -3.0


def test_peak_absolute_tie_prefers_smaller_cell():
    # |a| == |b|; the first cell a wins and keeps its (positive) delta.
    report = effect_matrix_cluster_report(
        [_row(0, "a", 2.0), _row(0, "b", -2.0)], [("a", "b")]
    )
    cluster = json.loads(report)["groups"][0]["clusters"][0]
    assert cluster["peak"] == 2.0
    assert cluster["kind"] == "mixed"


def test_sign_enumeration_p_for_three_cells():
    # values 1, 2, 3 -> total 6; only the all-plus and all-minus sign
    # vectors reach |signed sum| >= 6, so p = 2/8 = 0.25.
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0), _row(0, "c", 3.0)]
    report = effect_matrix_cluster_report(
        details, [("a", "b"), ("b", "c")]
    )
    cluster = json.loads(report)["groups"][0]["clusters"][0]
    assert cluster["p"] == 0.25


def test_threshold_keeps_cells_and_breaks_edges():
    # b = 0.2 is dropped; the path a-b-c falls apart into singletons a, c.
    details = [
        _row(0, "a", 1.0),
        _row(0, "b", 0.2),
        _row(0, "c", -2.0),
    ]
    neighbors = [("a", "b"), ("b", "c")]
    report = effect_matrix_cluster_report(
        details, neighbors, threshold=0.5
    )
    clusters = json.loads(report)["groups"][0]["clusters"]
    assert [c["cells"] for c in clusters] == [["a"], ["c"]]
    assert clusters[1]["kind"] == "cold"


def test_threshold_is_inclusive_and_zero_mean_cell_removed():
    details = [_row(0, "a", 0.5), _row(0, "b", 0.0)]
    report = effect_matrix_cluster_report(
        details, [("a", "b")], threshold=0.5
    )
    clusters = json.loads(report)["groups"][0]["clusters"]
    assert [c["cells"] for c in clusters] == [["a"]]


def test_bucket_without_active_cells_is_omitted():
    report = effect_matrix_cluster_report(
        [_row(0, "a", 0.1)], [], threshold=0.5
    )
    assert json.loads(report)["groups"] == []


def test_rows_average_within_bucket_cell_before_clustering():
    # a: deltas 0 and 2 in bucket 0 -> mean 1; b at 7200 -> bucket 7200.
    details = [
        _row(0, "a", 0.0),
        _row(1800, "a", 2.0),
        _row(7200, "b", 1.0),
    ]
    groups = json.loads(effect_matrix_cluster_report(details, []))["groups"]
    assert [g["key"] for g in groups] == [0, 7200]
    first = groups[0]["clusters"][0]
    assert first["key"] == "a" and first["mean"] == 1.0


def test_groups_and_clusters_ascending_across_buckets():
    details = [
        _row(7200, "a", 1.0),
        _row(3600, "b", 2.0),
        _row(0, "c", 3.0),
    ]
    groups = json.loads(effect_matrix_cluster_report(details, []))["groups"]
    assert [g["key"] for g in groups] == [0, 3600, 7200]
    for group in groups:
        cells = [c["key"] for c in group["clusters"]]
        assert cells == sorted(cells)


def test_more_than_twelve_cells_in_cluster_raises_value_error():
    cells = [f"c{i:02d}" for i in range(13)]
    edges = [(cells[i], cells[i + 1]) for i in range(12)]
    details = [_row(0, c, 1.0) for c in cells]
    with pytest.raises(ValueError):
        effect_matrix_cluster_report(details, edges)


def test_twelve_cells_allowed():
    cells = [f"c{i:02d}" for i in range(12)]
    edges = [(cells[i], cells[i + 1]) for i in range(11)]
    details = [_row(0, c, 1.0) for c in cells]
    report = effect_matrix_cluster_report(details, edges)
    cluster = json.loads(report)["groups"][0]["clusters"][0]
    assert cluster["n"] == 12
    # only all-plus and all-minus survive: 2/4096
    assert cluster["p"] == pytest.approx(2 / 4096, abs=1e-6)


def test_large_graph_split_into_small_components_is_allowed():
    cells = [f"c{i:02d}" for i in range(13)]
    edges = [(cells[i], cells[i + 1]) for i in range(10)]
    edges.append((cells[11], cells[12]))
    details = [_row(0, c, 1.0) for c in cells]
    groups = json.loads(
        effect_matrix_cluster_report(details, edges)
    )["groups"]
    sizes = sorted(c["n"] for c in groups[0]["clusters"])
    assert sizes == [2, 11]


def test_threshold_filtering_can_bring_oversize_cluster_under_limit():
    cells = [f"c{i:02d}" for i in range(13)]
    edges = [(cells[i], cells[i + 1]) for i in range(12)]
    details = [_row(0, c, 1.0) for c in cells[:12]]
    details.append(_row(0, cells[12], 0.0))
    report = effect_matrix_cluster_report(
        details, edges, threshold=0.5
    )
    cluster = json.loads(report)["groups"][0]["clusters"][0]
    assert cluster["n"] == 12


def test_compact_utf8_no_spaces_no_trailing_newline():
    details = [_row(0, "céll", 1.0), _row(0, "λ", 2.0)]
    report = effect_matrix_cluster_report(details, [("céll", "λ")])
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report
    assert '"cells":["céll","λ"]' in report


def test_key_orders():
    report = effect_matrix_cluster_report(
        [_row(0, "a", 1.0), _row(0, "b", 2.0)], [("a", "b")]
    )
    assert list(json.loads(report)) == ["minutes", "threshold", "groups"]
    group = json.loads(report)["groups"][0]
    assert list(group) == ["key", "clusters"]
    assert list(group["clusters"][0]) == [
        "key", "cells", "n", "mean", "peak", "p", "kind"
    ]


def test_negative_zero_normalized():
    report = effect_matrix_cluster_report(
        [_row(0, "a", 1.0), _row(0, "b", -1.0)], [("a", "b")]
    )
    assert "-0.000000" not in report
    cluster = json.loads(report)["groups"][0]["clusters"][0]
    assert cluster["mean"] == 0.0


def test_integer_bucket_key_and_str_cell_types():
    report = effect_matrix_cluster_report([_row(0, "a", 1.0)], [])
    group = json.loads(report)["groups"][0]
    assert isinstance(group["key"], int)
    cluster = group["clusters"][0]
    assert isinstance(cluster["key"], str)
    assert all(isinstance(c, str) for c in cluster["cells"])
    assert isinstance(cluster["n"], int)


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_cluster_report("not a list", [])


def test_neighbors_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_cluster_report([], (("a", "b"),))


@pytest.mark.parametrize("minutes", [0, 7, 1441, True, "60", 60.0])
def test_invalid_minutes_raise_value_error(minutes):
    with pytest.raises(ValueError):
        effect_matrix_cluster_report([], [], minutes=minutes)


@pytest.mark.parametrize(
    "threshold", [-0.01, True, False, "0.0", 0.5 + 0j,
                  float("nan"), float("inf"), -float("inf")]
)
def test_invalid_threshold_raises_value_error(threshold):
    with pytest.raises(ValueError):
        effect_matrix_cluster_report([], [], threshold=threshold)


@pytest.mark.parametrize(
    "neighbor",
    [
        ("a",),
        ["a", "b"],
        ("a", "b", "c"),
        ("", "b"),
        ("a", 1),
        ("a", "a"),
    ],
)
def test_invalid_neighbors_raise_value_error(neighbor):
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_cluster_report(details, [neighbor])


def test_unknown_and_duplicate_neighbor_endpoints_raise_value_error():
    details = [_row(0, "a", 1.0), _row(0, "b", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_cluster_report(details, [("a", "z")])
    with pytest.raises(ValueError):
        effect_matrix_cluster_report(details, [("a", "b"), ("b", "a")])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    details = [_row(0, "a", 1.0), _row(0, "a", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_cluster_report(details, [])
