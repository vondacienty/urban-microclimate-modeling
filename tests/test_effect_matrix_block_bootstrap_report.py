"""Tests for urban_micro.effect_matrix_block_bootstrap_report."""

import json

import pytest

from urban_micro import effect_matrix_block_bootstrap_report


def _row(t, c, delta, *, base=0.0, post=None, cg=0.0, cr=0.0, cm=0.0):
    """Build one scenario eight-tuple; post defaults to base + delta."""
    if post is None:
        post = base + delta
    return (t, c, base, post, delta, cg, cr, cm)


def test_empty_details_gives_empty_groups():
    report = effect_matrix_block_bootstrap_report([])
    assert report == '{"minutes":60,"block":2,"confidence":0.950000,"groups":[]}'
    assert json.loads(report) == {
        "minutes": 60,
        "block": 2,
        "confidence": 0.95,
        "groups": [],
    }


def test_defaults_echoed_in_empty_report():
    report = effect_matrix_block_bootstrap_report(
        [], minutes=30, block=3, confidence=0.9
    )
    assert report == '{"minutes":30,"block":3,"confidence":0.900000,"groups":[]}'


def test_single_bucket_cell_omitted():
    # A cell seen in only one bucket (even with several rows) has n = 1 and
    # is dropped; a cell spanning two buckets is kept.
    details = [
        _row(0, "a", 2.0),
        _row(100, "a", 4.0),
        _row(0, "b", 1.0),
        _row(3600, "b", 3.0),
    ]
    report = effect_matrix_block_bootstrap_report(details, block=2)
    groups = json.loads(report)["groups"]
    assert [g["key"] for g in groups] == ["b"]
    assert groups[0]["n"] == 2


def test_all_cells_single_bucket_yields_empty_groups():
    report = effect_matrix_block_bootstrap_report(
        [_row(0, "a", 1.0), _row(100, "a", 2.0)]
    )
    assert json.loads(report)["groups"] == []


def test_groups_sorted_by_cell_id():
    details = [
        _row(3600, "c", 1.0),
        _row(0, "b", 1.0),
        _row(3600, "b", 2.0),
        _row(0, "a", 1.0),
        _row(3600, "a", 2.0),
        _row(0, "c", 2.0),
    ]
    report = effect_matrix_block_bootstrap_report(details)
    assert [g["key"] for g in json.loads(report)["groups"]] == ["a", "b", "c"]


def test_series_is_bucket_means_in_ascending_bucket_order():
    # Bucket 0 mean delta = (0 + 2) / 2 = 1, bucket 3600 = 3,
    # bucket 7200 mean = (5 + 5) / 2 = 5 -> series y = [1, 3, 5], mean 3.
    details = [
        _row(0, "a", 0.0),
        _row(100, "a", 2.0),
        _row(3600, "a", 3.0),
        _row(7200, "a", 5.0),
        _row(7250, "a", 5.0),
    ]
    report = effect_matrix_block_bootstrap_report(details, block=2)
    group = json.loads(report)["groups"][0]
    assert group["n"] == 3
    assert group["delta"] == 3.0


def test_n2_block1_interval():
    # y = [1, 3], k = 2, block = 1: sorted resample means [1, 2, 2, 3];
    # q = 0.025, r = 0.075 -> 1 + 0.075 * (2 - 1) = 1.075;
    # upper = 3 - 0.075 = 2.925.
    report = effect_matrix_block_bootstrap_report(
        [_row(0, "a", 1.0), _row(3600, "a", 3.0)], block=1
    )
    group = json.loads(report)["groups"][0]
    assert group == {
        "key": "a",
        "n": 2,
        "delta": 2.0,
        "lower": 1.075,
        "upper": 2.925,
    }
    assert '"lower":1.075000' in report
    assert '"upper":2.925000' in report


def test_n2_block2_degenerate_interval():
    # y = [1, 3], k = 1, block = 2: every circular block averages to 2, so
    # both bounds equal the mean.
    report = effect_matrix_block_bootstrap_report(
        [_row(0, "a", 1.0), _row(3600, "a", 3.0)], block=2
    )
    group = json.loads(report)["groups"][0]
    assert group["lower"] == 2.0
    assert group["upper"] == 2.0


def test_n3_block2_circular_blocks():
    # y = [1, 3, 5], k = 2; each draw concatenates a length-2 circular block
    # then truncates to the first 3 terms, i.e. block(start) + y[next].
    # Numerators {6:2, 9:3, 10:2, 13:2}; means /3 sorted
    # [2,2,3,3,3,10/3,10/3,13/3,13/3], r = 0.2 -> lower 1.8, upper 4.2.
    report = effect_matrix_block_bootstrap_report(
        [_row(0, "a", 1.0), _row(3600, "a", 3.0), _row(7200, "a", 5.0)],
        block=2,
    )
    group = json.loads(report)["groups"][0]
    assert group["n"] == 3
    assert group["delta"] == 3.0
    assert group["lower"] == 1.8
    assert group["upper"] == 4.2


def test_integral_r_takes_value_directly():
    # confidence = 1/3 -> q = 1/3, r = (4 - 1) / 3 = 1 integral; sorted
    # means [1, 2, 2, 3] -> both bounds 2 with no interpolation.
    report = effect_matrix_block_bootstrap_report(
        [_row(0, "a", 1.0), _row(3600, "a", 3.0)],
        block=1,
        confidence=1.0 / 3.0,
    )
    group = json.loads(report)["groups"][0]
    assert group["lower"] == 2.0
    assert group["upper"] == 2.0


def test_block_larger_than_n_truncates():
    # n = 2, block = 8 -> k = 1, the single block is truncated after 2 terms,
    # which is the whole of y: degenerate interval.
    report = effect_matrix_block_bootstrap_report(
        [_row(0, "a", 1.0), _row(3600, "a", 4.0)], block=8
    )
    group = json.loads(report)["groups"][0]
    assert group["n"] == 2
    assert group["delta"] == 2.5
    assert group["lower"] == 2.5
    assert group["upper"] == 2.5


def test_eight_buckets_allowed():
    details = [_row(i * 3600, "a", 1.0) for i in range(8)]
    report = effect_matrix_block_bootstrap_report(details, block=1)
    assert json.loads(report)["groups"][0]["n"] == 8


def test_more_than_eight_buckets_raises_value_error():
    details = [_row(i * 3600, "a", 1.0) for i in range(9)]
    with pytest.raises(ValueError):
        effect_matrix_block_bootstrap_report(details)


def test_more_than_eight_rows_in_one_bucket_is_fine():
    # Many rows but a single bucket -> n = 1 cell, omitted, no error.
    details = [_row(i * 60, "a", 1.0) for i in range(20)]
    assert json.loads(effect_matrix_block_bootstrap_report(details))["groups"] == []


def test_minutes_changes_bucket_count():
    # 3 one-hour rows: minutes=60 gives 3 buckets, minutes=180 gives 1.
    details = [_row(0, "a", 1.0), _row(3600, "a", 2.0), _row(7200, "a", 3.0)]
    assert json.loads(
        effect_matrix_block_bootstrap_report(details, minutes=60)
    )["groups"][0]["n"] == 3
    assert (
        json.loads(
            effect_matrix_block_bootstrap_report(details, minutes=180)
        )["groups"]
        == []
    )


def test_negative_zero_normalized():
    report = effect_matrix_block_bootstrap_report(
        [_row(0, "a", -0.0), _row(3600, "a", -0.0)]
    )
    assert "-0.000000" not in report
    assert report == (
        '{"minutes":60,"block":2,"confidence":0.950000,"groups":['
        '{"key":"a","n":2,"delta":0.000000,"lower":0.000000,"upper":0.000000}'
        ']}'
    )


def test_compact_utf8_no_spaces_no_trailing_newline():
    report = effect_matrix_block_bootstrap_report(
        [_row(0, "céll", 1.0), _row(3600, "céll", 2.0)]
    )
    assert " " not in report
    assert not report.endswith("\n")
    assert '"key":"céll"' in report


def test_top_level_and_group_key_order():
    report = effect_matrix_block_bootstrap_report(
        [_row(0, "a", 1.0), _row(3600, "a", 2.0)],
        minutes=30,
        block=4,
        confidence=0.9,
    )
    assert list(json.loads(report)) == ["minutes", "block", "confidence", "groups"]
    assert list(json.loads(report)["groups"][0]) == [
        "key", "n", "delta", "lower", "upper",
    ]
    assert '"minutes":30' in report
    assert '"block":4' in report
    assert '"confidence":0.900000' in report


def test_details_not_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_matrix_block_bootstrap_report("not a list")
    with pytest.raises(TypeError):
        effect_matrix_block_bootstrap_report((_row(0, "a", 1.0),))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minutes": 0},
        {"minutes": 7},
        {"minutes": 1441},
        {"minutes": True},
        {"minutes": 60.0},
        {"minutes": "60"},
        {"block": 0},
        {"block": 9},
        {"block": -1},
        {"block": True},
        {"block": False},
        {"block": 2.0},
        {"block": "2"},
        {"confidence": 0},
        {"confidence": 1},
        {"confidence": -0.1},
        {"confidence": 1.0000001},
        {"confidence": True},
        {"confidence": "0.95"},
        {"confidence": float("nan")},
        {"confidence": float("inf")},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_matrix_block_bootstrap_report([], **kwargs)


@pytest.mark.parametrize("minutes", [1, 30, 60, 1440])
def test_minutes_boundaries(minutes):
    report = effect_matrix_block_bootstrap_report([], minutes=minutes)
    assert json.loads(report)["minutes"] == minutes


@pytest.mark.parametrize("block", [1, 2, 8])
def test_block_boundaries(block):
    report = effect_matrix_block_bootstrap_report([], block=block)
    assert json.loads(report)["block"] == block


@pytest.mark.parametrize(
    "row",
    [
        [0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # not a tuple
        (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0),  # wrong length
        (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # too long
        (True, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # bool timestamp
        (-1, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # negative timestamp
        (0, "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # empty cell id
        (0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # non-string cell id
        (0, "a", 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0),  # non-finite
        (0, "a", 0.0, 0.0, True, 0.0, 0.0, 0.0),  # bool value
    ],
)
def test_invalid_rows_raise_value_error(row):
    with pytest.raises(ValueError):
        effect_matrix_block_bootstrap_report([row])


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = _row(0, "a", 0.0)
    with pytest.raises(ValueError):
        effect_matrix_block_bootstrap_report([row, row])
