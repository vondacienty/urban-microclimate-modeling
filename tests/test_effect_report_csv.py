"""Tests for urban_micro.effect_report_csv."""

import csv
import io

import pytest

from urban_micro import effect_report, effect_report_csv


def test_empty_details_gives_header_only():
    report = effect_report_csv([], by="time")
    assert report == "key,n,base,post,delta,cg,cr,cm,p,se,lower,upper\r\n"


def test_time_buckets_and_interval():
    details = [
        (0, "a", 1.0, 2.0, 1.0, 0.5, 0.25, 0.25),
        (3599, "b", 3.0, 6.0, 3.0, 1.5, 0.75, 0.75),
        (3600, "a", 0.0, -1.0, -1.0, 0.0, -1.0, 0.0),
    ]
    report = effect_report_csv(details, by="time", minutes=60)
    assert report == (
        "key,n,base,post,delta,cg,cr,cm,p,se,lower,upper\r\n"
        "0,2,2.000000,4.000000,2.000000,1.000000,0.500000,0.500000,"
        "0.500000,1.000000,0.040000,3.960000\r\n"
        "3600,1,0.000000,-1.000000,-1.000000,0.000000,-1.000000,0.000000,"
        "1.000000,0.000000,-1.000000,-1.000000\r\n"
    )


def test_cell_grouping_sorted_string_keys():
    details = [
        (0, "b", 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        (1, "a", 0.0, 2.0, 2.0, 0.0, 0.0, 0.0),
        (2, "b", 0.0, 3.0, 3.0, 0.0, 0.0, 0.0),
    ]
    report = effect_report_csv(details, by="cell", z=0.0)
    assert report == (
        "key,n,base,post,delta,cg,cr,cm,p,se,lower,upper\r\n"
        "a,1,0.000000,2.000000,2.000000,0.000000,0.000000,0.000000,"
        "1.000000,0.000000,2.000000,2.000000\r\n"
        "b,2,0.000000,2.000000,2.000000,0.000000,0.000000,0.000000,"
        "0.500000,1.000000,2.000000,2.000000\r\n"
    )


def test_cell_keys_with_commas_quotes_and_newlines_are_quoted():
    details = [
        (0, 'a,"b"\nc', 0.0, 1.0, 1.0, 0.0, 0.0, 0.0),
        (1, "plain", 0.0, 2.0, 2.0, 0.0, 0.0, 0.0),
    ]
    report = effect_report_csv(details, by="cell")
    lines = report.split("\r\n")
    assert lines[1].startswith('"a,""b""\nc",1,')
    assert lines[2].startswith("plain,1,")
    rows = list(csv.reader(io.StringIO(report)))
    assert rows[1][0] == 'a,"b"\nc'
    assert rows[2][0] == "plain"


def test_negative_zero_normalized():
    report = effect_report_csv(
        [(0, "a", -0.0, -0.0, -0.0, -0.0, -0.0, -0.0)], by="time"
    )
    assert "-0.000000" not in report
    assert report == (
        "key,n,base,post,delta,cg,cr,cm,p,se,lower,upper\r\n"
        "0,1,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,"
        "1.000000,0.000000,0.000000,0.000000\r\n"
    )


def test_crlf_line_endings_and_trailing_crlf():
    report = effect_report_csv(
        [(0, "a", 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)], by="cell"
    )
    assert report.endswith("\r\n")
    assert "\n" not in report.replace("\r\n", "")


def test_values_match_effect_report():
    details = [
        (0, "a", 1.0, 2.0, 1.0, 0.5, 0.25, 0.25),
        (1, "b", 3.0, 6.0, 3.0, 1.5, 0.75, 0.75),
        (3600, "a", 0.0, -1.0, -1.0, 0.0, -1.0, 0.0),
    ]
    csv_rows = list(csv.reader(io.StringIO(effect_report_csv(details, by="time"))))
    json_groups = __import__("json").loads(
        effect_report(details, by="time")
    )["groups"]
    assert csv_rows[0] == [
        "key", "n", "base", "post", "delta", "cg", "cr", "cm",
        "p", "se", "lower", "upper",
    ]
    for csv_row, group in zip(csv_rows[1:], json_groups):
        assert csv_row[0] == str(group["key"])
        assert int(csv_row[1]) == group["n"]
        for field, cell in zip(
            ("base", "post", "delta", "cg", "cr", "cm", "p", "se", "lower", "upper"),
            csv_row[2:],
        ):
            assert cell == f"{group[field]:.6f}"


def test_details_not_a_list_raises_type_error():
    with pytest.raises(TypeError):
        effect_report_csv(((0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),), by="time")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"by": "row"},
        {"by": "time", "minutes": 0},
        {"by": "time", "minutes": 7},
        {"by": "time", "minutes": True},
        {"by": "time", "z": -1.0},
        {"by": "time", "z": True},
        {"by": "time", "z": float("nan")},
        {"by": "time", "z": "1.96"},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        effect_report_csv([], **kwargs)


@pytest.mark.parametrize(
    "row",
    [
        [0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # not a tuple
        (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0),  # wrong length
        (True, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # bool timestamp
        (-1, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # negative timestamp
        (0, "", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # empty cell id
        (0, "a", 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0),  # non-finite
        (0, "a", 0.0, 0.0, True, 0.0, 0.0, 0.0),  # bool value
    ],
)
def test_invalid_rows_raise_value_error(row):
    with pytest.raises(ValueError):
        effect_report_csv([row], by="time")


def test_duplicate_timestamp_cell_pair_raises_value_error():
    row = (0, "a", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        effect_report_csv([row, row], by="cell")
