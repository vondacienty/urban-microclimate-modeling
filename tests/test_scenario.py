"""Tests for urban_micro.scenario."""

import math

import pytest

from urban_micro import scenario


def _row(t, c, h=0.0, b=0.0, i=0.0, g=0.0, d=0.0):
    return (t, c, 1.0, 2.0, d, h, b, i, g)


def test_base_model_without_actions():
    model = (2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a"), _row(1, "b")]
    details, summary = scenario(model, rows, [])
    assert details == [
        (0, "a", 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        (1, "b", 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    ]
    assert summary == (2, 1.0, 1.0, 0.0)


def test_full_action_formula():
    # b0=1, bi=-1, bg=2 -> cg = (bg-bi)*G = 3*G
    model = (1, 1.0, 0.0, 0.0, -1.0, 2.0, 0.5)
    rows = [_row(0, "a", b=0.5, i=0.6)]
    actions = [("a", 0.5, 0.2, 0.1)]
    details, summary = scenario(
        model, rows, actions, roof_c=0.3, material_c=0.4
    )
    # base = 1 + (-1)*0.6 + 2*0 = 0.4
    # cg = 3*0.5 = 1.5; cr = -0.3*0.2 = -0.06; cm = -0.4*0.1 = -0.04
    # post = 0.4 + 1.5 - 0.06 - 0.04 = 1.8; delta = 1.4
    assert details == [(0, "a", 0.4, 1.8, 1.4, 1.5, -0.06, -0.04)]
    assert summary == (1, 0.4, 1.8, 1.4)


def test_all_model_features_enter_base():
    # base = b0 + bh*h + bb*b + bi*i + bg*g
    model = (1, 1.0, 2.0, 3.0, 4.0, 5.0, 0.9)
    rows = [_row(0, "a", h=1.0, b=0.1, i=0.2, g=0.3)]
    details, summary = scenario(model, rows, [])
    # 1 + 2*1 + 3*0.1 + 4*0.2 + 5*0.3 = 5.6
    assert details == [(0, "a", 5.6, 5.6, 0.0, 0.0, 0.0, 0.0)]
    assert summary == (1, 5.6, 5.6, 0.0)


def test_details_sorted_by_timestamp_cell_id():
    model = (3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(5, "z", i=1.0), _row(0, "b"), _row(0, "a", i=1.0)]
    actions = [("z", 1.0, 0.0, 0.0), ("a", 0.5, 0.0, 0.0)]
    details, _ = scenario(model, rows, actions)
    assert [(d[0], d[1]) for d in details] == [(0, "a"), (0, "b"), (5, "z")]


def test_action_applies_to_every_row_of_cell():
    model = (2, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0)
    rows = [_row(0, "x", i=1.0), _row(1, "x", i=1.0), _row(2, "y")]
    actions = [("x", 0.5, 0.0, 0.0)]
    details, _ = scenario(model, rows, actions)
    # cg = (1 - (-1))*0.5 = 1.0 on both x rows; y untouched
    assert [d[4] for d in details] == [1.0, 1.0, 0.0]
    assert [d[5] for d in details] == [1.0, 1.0, 0.0]


def test_unlisted_cells_get_zero_actions():
    model = (1, 1.0, 0.0, 0.0, -10.0, 10.0, 0.0)
    rows = [_row(0, "a", i=1.0, g=1.0)]
    details, _ = scenario(model, rows, [])
    assert details[0][5:8] == (0.0, 0.0, 0.0)
    assert details[0][4] == 0.0


def test_summary_means_unquantized():
    # bases 1/3 and 2/3 style: use coefficients giving repeating decimals
    model = (2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a"), _row(1, "b")]
    # one action of G=1/3 with cg coefficient 1 -> delta 1/3 on one row
    model2 = (2, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    rows2 = [_row(0, "a", i=1.0), _row(1, "b", i=1.0)]
    details, summary = scenario(model2, rows2, [("a", 1.0 / 3.0, 0.0, 0.0)])
    # base means are 0; delta mean = (1/3)/2 = 1/6
    assert details[0][5] == round(1.0 / 3.0, 6)
    assert summary == (2, 0.0, round(1.0 / 6.0, 6), round(1.0 / 6.0, 6))


def test_zero_action_values_allowed_at_boundaries():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a", b=1.0, i=1.0)]
    details, _ = scenario(model, rows, [("a", 0.5, 1.0, 0.5)])
    assert details[0][5:8] == (0.0, 0.0, 0.0)


def test_negative_zero_normalized():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a", b=1.0)]
    # roof_c default 0 with R>0 gives cr = -0.0 * R -> negative zero
    details, summary = scenario(model, rows, [("a", 0.0, 0.5, 0.0)])
    for value in details[0][2:]:
        assert value == 0.0
        assert math.copysign(1.0, value) == 1.0
    for value in summary[1:]:
        assert value == 0.0
        assert math.copysign(1.0, value) == 1.0


def test_rows_must_be_list():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(TypeError):
        scenario(model, (_row(0, "a"),), [])
    with pytest.raises(TypeError):
        scenario(model, {}, [])


def test_actions_must_be_list():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(TypeError):
        scenario(model, [_row(0, "a")], (("a", 0.0, 0.0, 0.0),))
    with pytest.raises(TypeError):
        scenario(model, [_row(0, "a")], {"a": (0.0, 0.0, 0.0)})


def test_empty_rows_rejected():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        scenario(model, [], [])


def test_invalid_rows_rejected():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # duplicate (timestamp, cell_id)
    with pytest.raises(ValueError):
        scenario(model, [_row(0, "a"), _row(0, "a")], [])
    # fraction out of range
    with pytest.raises(ValueError):
        scenario(model, [_row(0, "a", i=1.5)], [])


def test_model_must_be_seven_tuple():
    rows = [_row(0, "a")]
    for bad in (
        (1, 0.0, 0.0, 0.0, 0.0, 0.0),
        (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1),
        [1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "nope",
    ):
        with pytest.raises(ValueError):
            scenario(bad, rows, [])


def test_model_n_must_be_positive_int():
    rows = [_row(0, "a")]
    coefs = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for bad_n in (0, -1, 1.5, True, False, "1"):
        with pytest.raises(ValueError):
            scenario((bad_n, *coefs), rows, [])


def test_model_coefficients_must_be_finite_numbers():
    rows = [_row(0, "a")]
    base = [1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for idx, bad in (
        (1, True),
        (2, "0.0"),
        (3, float("nan")),
        (4, float("inf")),
        (5, float("-inf")),
        (6, 1 + 0j),
    ):
        model = list(base)
        model[idx] = bad
        with pytest.raises(ValueError):
            scenario(tuple(model), rows, [])


def test_costs_must_be_non_negative_finite():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a", b=1.0)]
    actions = [("a", 0.0, 0.5, 0.0)]
    for bad in (-0.000001, True, False, float("nan"), float("inf"), "0.1"):
        with pytest.raises(ValueError):
            scenario(model, rows, actions, roof_c=bad)
    actions2 = [("a", 0.0, 0.0, 0.5)]
    for bad in (-0.000001, True, float("-inf")):
        with pytest.raises(ValueError):
            scenario(model, rows, actions2, material_c=bad)


def test_action_must_be_four_tuple():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a")]
    for bad in (
        ["a", 0.0, 0.0, 0.0],
        ("a", 0.0, 0.0),
        ("a", 0.0, 0.0, 0.0, 0.0),
        "nope",
    ):
        with pytest.raises(ValueError):
            scenario(model, rows, [bad])


def test_action_cell_must_occur_in_rows():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a")]
    with pytest.raises(ValueError):
        scenario(model, rows, [("b", 0.0, 0.0, 0.0)])


def test_action_cell_must_be_unique():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a")]
    with pytest.raises(ValueError):
        scenario(
            model,
            rows,
            [("a", 0.1, 0.0, 0.0), ("a", 0.2, 0.0, 0.0)],
        )


def test_action_values_validation():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a", b=1.0, i=1.0)]
    for bad in (-0.000001, 1.000001, True, False, float("nan"),
                float("inf"), "0.5"):
        for position in range(1, 4):
            values = [0.0, 0.0, 0.0]
            values[position - 1] = bad
            with pytest.raises(ValueError):
                scenario(model, rows, [("a", *values)])


def test_green_plus_material_must_not_exceed_impervious():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a", i=0.5)]
    with pytest.raises(ValueError):
        scenario(model, rows, [("a", 0.3, 0.0, 0.3)])
    # equality is allowed
    scenario(model, rows, [("a", 0.3, 0.0, 0.2)])


def test_roof_must_not_exceed_building():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a", b=0.4)]
    with pytest.raises(ValueError):
        scenario(model, rows, [("a", 0.0, 0.5, 0.0)])
    # equality is allowed
    scenario(model, rows, [("a", 0.0, 0.4, 0.0)])


def test_feasibility_checked_per_row_of_cell():
    model = (2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # same cell at two timestamps; one row has smaller i -> violation
    rows = [_row(0, "x", i=0.8), _row(1, "x", i=0.3)]
    with pytest.raises(ValueError):
        scenario(model, rows, [("x", 0.5, 0.0, 0.0)])


def test_costs_keyword_only():
    model = (1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    rows = [_row(0, "a", b=1.0)]
    with pytest.raises(TypeError):
        scenario(model, rows, [], 0.1)  # type: ignore[misc]


def test_independent_of_default_context_precision():
    from decimal import getcontext

    previous = getcontext().prec
    getcontext().prec = 2
    try:
        model = (1, 1e300, 1e300, 0.0, 0.0, 0.0, 0.0)
        rows = [_row(0, "a", h=1.0, b=1.0, i=1.0)]
        details, summary = scenario(
            model, rows, [("a", 0.5, 0.5, 0.0)], roof_c=1.0
        )
        assert len(details) == 1
        assert summary[0] == 1
        assert all(isinstance(v, float) for v in details[0][2:])
    finally:
        getcontext().prec = previous


def test_int_inputs_accepted():
    model = (1, 0, 0, 0, 0, 0, 0)
    rows = [_row(0, "a", h=1, b=0, i=1, g=0, d=0)]
    details, summary = scenario(
        model, rows, [("a", 1, 0, 0)], roof_c=0, material_c=0
    )
    assert details[0][3] == 0.0
    assert summary[0] == 1
