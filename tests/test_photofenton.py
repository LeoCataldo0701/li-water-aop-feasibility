import pytest
from photofenton import (
    MW, T_REF_K, arrhenius_factor, check_ph, simulate,
    reagent_cost_per_litre, cost_per_ug_removed,
)

COSTS = {"H2O2_per_mmol": 0.014, "Fe_per_mmol": 0.0557, "UV_per_min": 0.008}


def test_molar_conversion_uses_compound_mw():
    r = simulate("PFBA", 1000.0, 3.0, 25.0, 10.0, 5.0, t_end=600)
    assert r["pfas0_mol_l"] == pytest.approx(1000e-9 / MW["PFBA"])


def test_iron_is_conserved():
    r = simulate("PFOA", 100.0, 3.0, 25.0, 10.0, 5.0, t_end=1800)
    assert r["fe_balance_error"] < 1e-6


def test_no_reagents_means_no_removal():
    r = simulate("PFOA", 100.0, 3.0, 25.0, 0.0, 0.0, t_end=600)
    assert r["removal_final"] == pytest.approx(0.0, abs=1e-12)


def test_out_of_range_ph_raises():
    with pytest.raises(ValueError):
        simulate("PFOA", 10.0, 5.83, 12.9, 10.0, 5.0)


def test_arrhenius_is_one_at_reference():
    assert arrhenius_factor(T_REF_K - 273.15) == pytest.approx(1.0)


def test_arrhenius_suppresses_rate_at_field_temperature():
    assert 0.40 < arrhenius_factor(12.9) < 0.45


def test_reagent_cost_matches_hand_calculation():
    # 10 mM H2O2 at 0.014 $/mmol, 5 uM Fe at 0.0557 $/mmol, 10 min UV at 0.008 $/min
    expected = 10 * 0.014 + 5e-3 * 0.0557 + 10 * 0.008
    assert reagent_cost_per_litre(10.0, 5.0, 10.0, COSTS) == pytest.approx(expected)


def test_cost_is_none_when_target_not_reached():
    assert reagent_cost_per_litre(10.0, 5.0, None, COSTS) is None
    assert cost_per_ug_removed(10.0, None) is None


def test_cost_per_ug_hand_calculation():
    # 10 ng/L = 0.01 ug/L; 90% removal = 0.009 ug/L removed
    assert cost_per_ug_removed(10.0, 0.2) == pytest.approx(0.2 / 0.009)
