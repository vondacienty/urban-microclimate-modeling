"""Tests for urban_micro.effect_matrix_cluster_report."""

import json

import pytest

from urban_micro import effect_matrix_cluster_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def test_empty_details_yields_empty_groups():
    assert effect_matrix_cluster_report([], []) == (
        '{"minutes":60,"threshold":0.000000,"groups":[]}'
    )


def test_two_clusters_in_one_bucket():
    details = [
        _row(0, "A", 1.0),
        _row(0, "B", 2.0),
        _row(0, "C", -1.0),
    ]
    report = effect_matrix_cluster_report(details, [("A", "B")])
    assert report == (
        '{"minutes":60,"threshold":0.000000,"groups":['
        '{"key":0,"clusters":['
        '{"key":"A","cells":["A","B"],"n":2,"mean":1.500000,'
        '"peak":2.000000,"p":0.500000,"kind":"hot"},'
        '{"key":"C","cells":["C"],"n":1,"mean":-1.000000,'
        '"peak":-1.000000,"p":1.000000,"kind":"cold"}'
        "]}]}"
    )
    # Compact JSON: no spaces, no trailing newline.
    assert " " not in report
    assert not report.endswith("\n")


def test_bucket_averaging_and_group_order():
    details = [
        _row(1800, "A", 3.0),
        _row(0, "A", 1.0),
        _row(3600, "A", -4.0),
    ]
    report = effect_matrix_cluster_report(details, [])
    assert report == (
        '{"minutes":60,"threshold":0.000000,"groups":['
        '{"key":0,"clusters":['
        '{"key":"A","cells":["A"],"n":1,"mean":2.000000,'
        '"peak":2.000000,"p":1.000000,"kind":"hot"}]},'
        '{"key":3600,"clusters":['
        '{"key":"A","cells":["A"],"n":1,"mean":-4.000000,'
        '"peak":-4.000000,"p":1.000000,"kind":"cold"}]}'
        "]}"
    )


def test_minutes_controls_bucketing():
    details = [_row(0, "A", 1.0), _row(1800, "A", 3.0)]
    report = effect_matrix_cluster_report(details, [], minutes=30)
    groups = json.loads(report)["groups"]
    assert [group["key"] for group in groups] == [0, 1800]


def test_threshold_filters_cells_and_empty_buckets():
    details = [
        _row(0, "A", 1.0),
        _row(0, "B", 2.0),
        _row(0, "C", -1.0),
        _row(3600, "A", 0.5),
    ]
    neighbors = [("A", "B"), ("B", "C")]
    report = effect_matrix_cluster_report(details, neighbors, threshold=1.5)
    assert report == (
        '{"minutes":60,"threshold":1.500000,"groups":['
        '{"key":0,"clusters":['
        '{"key":"B","cells":["B"],"n":1,"mean":2.000000,'
        '"peak":2.000000,"p":1.000000,"kind":"hot"}]}'
        "]}"
    )
    # Everything filtered out -> no groups at all.
    assert effect_matrix_cluster_report(details, neighbors, threshold=3.0) == (
        '{"minutes":60,"threshold":3.000000,"groups":[]}'
    )


def test_threshold_boundary_is_inclusive():
    details = [_row(0, "A", -1.5)]
    report = effect_matrix_cluster_report(details, [], threshold=1.5)
    clusters = json.loads(report)["groups"][0]["clusters"]
    assert clusters[0]["mean"] == -1.5
    assert clusters[0]["kind"] == "cold"


def test_zero_mean_cluster_is_mixed_and_peak_tie_takes_smaller_cell():
    details = [_row(0, "A", 1.0), _row(0, "B", -1.0)]
    report = effect_matrix_cluster_report(details, [("A", "B")])
    assert report == (
        '{"minutes":60,"threshold":0.000000,"groups":['
        '{"key":0,"clusters":['
        '{"key":"A","cells":["A","B"],"n":2,"mean":0.000000,'
        '"peak":1.000000,"p":1.000000,"kind":"mixed"}]}'
        "]}"
    )


def test_negative_zero_is_normalized():
    report = effect_matrix_cluster_report([_row(0, "A", -0.0)], [])
    assert report == (
        '{"minutes":60,"threshold":0.000000,"groups":['
        '{"key":0,"clusters":['
        '{"key":"A","cells":["A"],"n":1,"mean":0.000000,'
        '"peak":0.000000,"p":1.000000,"kind":"mixed"}]}'
        "]}"
    )


def test_sign_flip_p_value():
    # means 1, 2, 3: signed sums are +/-6, +/-4, +/-2, 0; only |6| >= 6.
    details = [_row(0, "A", 1.0), _row(0, "B", 2.0), _row(0, "C", 3.0)]
    neighbors = [("A", "B"), ("B", "C")]
    clusters = json.loads(
        effect_matrix_cluster_report(details, neighbors)
    )["groups"][0]["clusters"]
    assert clusters[0]["n"] == 3
    assert clusters[0]["mean"] == 2.0
    assert clusters[0]["peak"] == 3.0
    assert clusters[0]["p"] == 0.25
    assert clusters[0]["kind"] == "hot"


def test_cluster_order_by_first_cell():
    details = [_row(0, "B", 1.0), _row(0, "A", 2.0), _row(0, "C", 3.0)]
    report = effect_matrix_cluster_report(details, [])
    clusters = json.loads(report)["groups"][0]["clusters"]
    assert [cluster["key"] for cluster in clusters] == ["A", "B", "C"]


def test_cluster_with_more_than_twelve_cells_raises():
    cells = [f"c{i:02d}" for i in range(13)]
    details = [_row(0, cell, 1.0) for cell in cells]
    neighbors = list(zip(cells, cells[1:]))
    with pytest.raises(ValueError):
        effect_matrix_cluster_report(details, neighbors)
    # Twelve connected cells are still accepted.
    details = [_row(0, cell, 1.0) for cell in cells[:12]]
    neighbors = list(zip(cells[:12], cells[1:12]))
    clusters = json.loads(
        effect_matrix_cluster_report(details, neighbors)
    )["groups"][0]["clusters"]
    assert clusters[0]["n"] == 12


def test_non_list_arguments_raise_type_error():
    with pytest.raises(TypeError):
        effect_matrix_cluster_report((), [])
    with pytest.raises(TypeError):
        effect_matrix_cluster_report([], ())
    with pytest.raises(TypeError):
        effect_matrix_cluster_report(None, [])


@pytest.mark.parametrize("minutes", [True, 0, -60, 7, 1441, 90.0, "60"])
def test_invalid_minutes_raise_value_error(minutes):
    with pytest.raises(ValueError):
        effect_matrix_cluster_report([], [], minutes=minutes)


@pytest.mark.parametrize("threshold", [True, -0.5, float("nan"), float("inf"), "1"])
def test_invalid_threshold_raises_value_error(threshold):
    with pytest.raises(ValueError):
        effect_matrix_cluster_report([], [], threshold=threshold)


def test_duplicate_timestamp_cell_pair_raises_value_error():
    details = [_row(0, "A", 1.0), _row(0, "A", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_cluster_report(details, [])


@pytest.mark.parametrize(
    "neighbors",
    [
        [("A", "A")],               # self-loop
        [("A", "B"), ("B", "A")],   # duplicate edge, either orientation
        [("A", "B"), ("A", "B")],   # duplicate edge
        [("A", "Z")],               # unknown cell id
        [("A",)],                   # not a two-tuple
        [("A", "")],                # empty endpoint
        [("A", 1)],                 # non-string endpoint
        ["AB"],                     # not a tuple
    ],
)
def test_invalid_neighbors_raise_value_error(neighbors):
    details = [_row(0, "A", 1.0), _row(0, "B", 2.0)]
    with pytest.raises(ValueError):
        effect_matrix_cluster_report(details, neighbors)


def test_invalid_detail_rows_raise_value_error():
    with pytest.raises(ValueError):
        effect_matrix_cluster_report([_row(-1, "A", 1.0)], [])
    with pytest.raises(ValueError):
        effect_matrix_cluster_report([_row(0, "", 1.0)], [])
    with pytest.raises(ValueError):
        effect_matrix_cluster_report([_row(0, "A", float("nan"))], [])
    with pytest.raises(ValueError):
        effect_matrix_cluster_report([_row(0, "A", True)], [])
    with pytest.raises(ValueError):
        effect_matrix_cluster_report([(0, "A", 0.0, 1.0, 1.0)], [])
