"""
Tests for the photofenton package.

Covers:
- Compound database loading
- Parameter database loading
- Arrhenius correction
- ODE iron conservation (mass balance)
- Molar conversion per compound
- pH validation
- Zero-dose edge case
- Cost calculations
- Parameter sweep
- Output key completeness
"""
import pytest
import numpy as np
from photofenton import (
    simulate,
    reagent_cost_per_litre,
    cost_per_ug_removed,
    arrhenius_factor,
    parameter_sweep,
    load_compounds,
    get_compound,
    load_parameters,
)

COSTS = {"H2O2_per_mmol": 0.014, "Fe_per_mmol": 0.0557, "UV_per_min": 0.008}


# ── Compound database ────────────────────────────────────────────────────────

class TestCompoundDatabase:
    def test_known_compounds_load(self):
        compounds = load_compounds()
        for name in ["PFOA", "PFOS", "PFBA"]:
            assert name in compounds

    def test_molar_masses_are_positive(self):
        for name, props in load_compounds().items():
            assert props["molar_mass"] > 0, f"{name} molar mass must be positive"

    def test_gac_removal_is_fraction(self):
        for name, props in load_compounds().items():
            assert 0.0 <= props["gac_removal_fraction"] <= 1.0

    def test_k_oh_is_positive(self):
        for name, props in load_compounds().items():
            assert props["k_OH"] > 0

    def test_unknown_compound_raises(self):
        with pytest.raises(KeyError):
            get_compound("NotAPFAS")

    def test_pfba_molar_mass_correct(self):
        c = get_compound("PFBA")
        assert abs(c["molar_mass"] - 214.04) < 0.1

    def test_pfoa_molar_mass_correct(self):
        c = get_compound("PFOA")
        assert abs(c["molar_mass"] - 414.07) < 0.1


# ── Parameters ───────────────────────────────────────────────────────────────

class TestParameterDatabase:
    def test_required_keys_present(self):
        params = load_parameters()
        for key in ["k_f_ref", "k_r_ref", "k_scav", "k_Fe", "k_I", "k_s",
                    "k_hv_h2o2", "k_hv_fe", "Ea_J_mol", "T_ref_K"]:
            assert key in params, f"Missing parameter: {key}"

    def test_all_values_are_positive(self):
        for k, v in load_parameters().items():
            assert v > 0, f"Parameter {k} must be positive"


# ── Arrhenius ────────────────────────────────────────────────────────────────

class TestArrhenius:
    def test_unity_at_reference(self):
        assert arrhenius_factor(25.0) == pytest.approx(1.0, rel=1e-6)

    def test_suppressed_at_field_temperature(self):
        # Suffolk County field temp 12.9 C -> ~43% of 25 C rate
        f = arrhenius_factor(12.9)
        assert 0.40 < f < 0.46

    def test_increases_with_temperature(self):
        assert arrhenius_factor(30.0) > arrhenius_factor(20.0)

    def test_never_negative(self):
        for t in [-10, 0, 5, 12.9, 25, 35, 60]:
            assert arrhenius_factor(t) > 0


# ── simulate() ───────────────────────────────────────────────────────────────

class TestSimulate:
    def test_iron_is_conserved(self):
        r = simulate("PFOA", 120.0, 3.0, 25.0, 10.0, 5.0)
        assert r["fe_balance_error"] < 1e-6

    def test_molar_conversion_uses_compound_mw(self):
        # PFBA MW ~214, PFOA MW ~414. Same ng/L -> different mol/L.
        r_pfba = simulate("PFBA", 1000.0, 3.0, 25.0, 10.0, 5.0, t_end=600)
        r_pfoa = simulate("PFOA", 1000.0, 3.0, 25.0, 10.0, 5.0, t_end=600)
        from photofenton import get_compound
        pfba = get_compound("PFBA")
        pfoa = get_compound("PFOA")
        assert r_pfba["pfas0_mol_l"] == pytest.approx(1000e-9 / pfba["molar_mass"])
        assert r_pfoa["pfas0_mol_l"] == pytest.approx(1000e-9 / pfoa["molar_mass"])

    def test_pfba_molar_concentration_differs_from_pfoa(self):
        r_pfba = simulate("PFBA", 1000.0, 3.0, 25.0, 10.0, 5.0, t_end=600)
        r_pfoa = simulate("PFOA", 1000.0, 3.0, 25.0, 10.0, 5.0, t_end=600)
        # PFBA mol/L should be ~1.94x PFOA mol/L (414/214)
        ratio = r_pfba["pfas0_mol_l"] / r_pfoa["pfas0_mol_l"]
        assert 1.8 < ratio < 2.1

    def test_no_reagents_means_no_removal(self):
        r = simulate("PFOA", 100.0, 3.0, 25.0, 0.0, 0.0, t_end=600)
        assert r["removal_final"] == pytest.approx(0.0, abs=1e-10)

    def test_out_of_range_ph_raises(self):
        with pytest.raises(ValueError, match="outside the validated range"):
            simulate("PFOA", 10.0, 5.83, 12.9, 10.0, 5.0)

    def test_low_ph_raises(self):
        with pytest.raises(ValueError):
            simulate("PFOA", 10.0, 1.0, 12.9, 10.0, 5.0)

    def test_negative_concentration_raises(self):
        with pytest.raises(ValueError):
            simulate("PFOA", -1.0, 3.0, 25.0, 10.0, 5.0)

    def test_zero_concentration_raises(self):
        with pytest.raises(ValueError):
            simulate("PFOA", 0.0, 3.0, 25.0, 10.0, 5.0)

    def test_unknown_compound_raises(self):
        with pytest.raises(KeyError):
            simulate("NotPFAS", 100.0, 3.0, 25.0, 10.0, 5.0)

    def test_removal_is_between_zero_and_one(self):
        r = simulate("PFOA", 120.0, 3.0, 25.0, 10.0, 5.0)
        assert 0.0 <= r["removal_final"] <= 1.0

    def test_output_keys_complete(self):
        r = simulate("PFOA", 120.0, 3.0, 12.9, 10.0, 5.0)
        for key in ["compound", "pH", "temp_C", "k_f", "k_OH", "pfas0_mol_l",
                    "t_min", "removal", "removal_final", "t90_min", "fe_balance_error"]:
            assert key in r, f"Missing output key: {key}"

    def test_t90_is_none_with_zero_reagents(self):
        r = simulate("PFOA", 100.0, 3.0, 25.0, 0.0, 0.0, t_end=600)
        assert r["t90_min"] is None

    def test_temperature_affects_rate(self):
        r_cold = simulate("PFOA", 120.0, 3.0, 5.0,  10.0, 5.0)
        r_hot  = simulate("PFOA", 120.0, 3.0, 35.0, 10.0, 5.0)
        # Warmer water should give higher or equal removal
        assert r_hot["removal_final"] >= r_cold["removal_final"]

    def test_all_three_core_compounds_run(self):
        for compound in ["PFOA", "PFOS", "PFBA"]:
            r = simulate(compound, 100.0, 3.0, 25.0, 10.0, 5.0, t_end=600)
            assert r["removal_final"] >= 0.0


# ── Cost functions ───────────────────────────────────────────────────────────

class TestCostFunctions:
    def test_cost_matches_hand_calculation(self):
        # 10 mM H2O2 * 0.014 + 0.005 mmol Fe * 0.0557 + 10 min * 0.008
        expected = 10 * 0.014 + 5e-3 * 0.0557 + 10 * 0.008
        assert reagent_cost_per_litre(10.0, 5.0, 10.0, COSTS) == pytest.approx(expected)

    def test_none_when_t90_not_reached(self):
        assert reagent_cost_per_litre(10.0, 5.0, None, COSTS) is None

    def test_cost_per_ug_none_propagates(self):
        assert cost_per_ug_removed(100.0, None) is None

    def test_cost_per_ug_hand_calculation(self):
        # 100 ng/L * 90% removal = 90 ng/L = 0.09 ug/L removed
        assert cost_per_ug_removed(100.0, 0.18) == pytest.approx(0.18 / 0.09)

    def test_cost_per_ug_uses_mass_not_moles(self):
        # For 10 ng/L: 90% removal = 0.009 ug/L removed
        assert cost_per_ug_removed(10.0, 0.2) == pytest.approx(0.2 / 0.009)


# ── Parameter sweep ──────────────────────────────────────────────────────────

class TestParameterSweep:
    def test_returns_correct_number_of_rows(self):
        rows = parameter_sweep(
            "PFOA", 120.0, 3.0, 25.0,
            H2O2_mM_values=[5, 10],
            Fe_uM_values=[2, 5, 10],
            costs=COSTS,
            t_end=600,
        )
        assert len(rows) == 6  # 2 H2O2 * 3 Fe values

    def test_each_row_has_dose_and_cost_keys(self):
        rows = parameter_sweep(
            "PFOA", 120.0, 3.0, 25.0,
            H2O2_mM_values=[10],
            Fe_uM_values=[5],
            costs=COSTS,
            t_end=600,
        )
        assert "H2O2_mM" in rows[0]
        assert "Fe_uM" in rows[0]
        assert "cost_per_L" in rows[0]
        assert "cost_per_ug" in rows[0]

    def test_higher_dose_does_not_reduce_removal(self):
        rows = parameter_sweep(
            "PFOA", 120.0, 3.0, 25.0,
            H2O2_mM_values=[5, 50],
            Fe_uM_values=[5],
            costs=COSTS,
            t_end=1800,
        )
        low_dose  = next(r for r in rows if r["H2O2_mM"] == 5)
        high_dose = next(r for r in rows if r["H2O2_mM"] == 50)
        assert high_dose["removal_final"] >= low_dose["removal_final"]
def test_sensitivity_analysis_shape_and_direction():
    from photofenton import sensitivity_analysis
    rows = sensitivity_analysis(
        "PFOA", 120.0, 3.0, 25.0, 10.0, 5.0,
        param_name="k_s", factors=(0.1, 10.0), t_end=1800,
    )
    assert len(rows) == 2
    assert rows[1]["removal_final"] <= rows[0]["removal_final"]
