"""
photofenton: batch photo-Fenton kinetics for PFAS screening.

This is a screening model. Its rate constants are partly literature values,
partly placeholders, and partly uncalibrated. It has not been validated
against measured PFAS degradation data. See PARAM_STATUS.
"""
import numpy as np
from scipy.integrate import solve_ivp

# Molar masses, g/mol
MW = {"PFOA": 414.07, "PFOS": 500.13, "PFBA": 214.04}

# OH-radical rate constants, L/(mol*s)
K_OH = {"PFOA": 3e5, "PFOS": 1e5, "PFBA": 9.31e7}

# Arrhenius temperature correction
T_REF_K = 298.15
EA_J_MOL = 50_000.0
R_GAS = 8.314

PARAMS = {
    "k_f_ref": 63.0,     # Fe2+ + H2O2 near 25 C, pH ~3 (Walling 1975)
    "k_r_ref": 8.4e-6,   # Fe3+ + H2O2 (Walling 1975)
    "k_scav": 2.7e7,     # OH + H2O2 (Buxton et al. 1988)
    "k_Fe": 3.0e6,       # Fe2+ + OH (Walling 1975)
    "k_I": 3.0e8,        # PLACEHOLDER: lumped intermediates
    "k_s": 2.0e4,        # PLACEHOLDER: background OH scavenging, 1/s
    "k_hv_h2o2": 5e-4,   # UNCALIBRATED: needs lamp photon flux, 1/s
    "k_hv_fe": 1e-2,     # UNCALIBRATED: Fe3+ photoreduction, 1/s
}

PARAM_STATUS = {
    "literature": ["K_OH", "k_scav", "k_Fe", "k_f_ref", "k_r_ref", "Ea"],
    "placeholder": ["k_I", "k_s"],
    "uncalibrated": ["k_hv_h2o2", "k_hv_fe"],
}

# The model has no validated pH dependence. k_f_ref was measured near pH 3,
# so results are only reported inside this window.
PH_VALID_RANGE = (2.5, 3.5)

T_END_S = 7200.0
TARGET_REMOVAL = 0.90


def arrhenius_factor(temp_C):
    """Scale rate constants from T_REF_K to temp_C."""
    temp_K = temp_C + 273.15
    return float(np.exp(-EA_J_MOL / R_GAS * (1.0 / temp_K - 1.0 / T_REF_K)))


def check_ph(pH):
    lo, hi = PH_VALID_RANGE
    if not lo <= pH <= hi:
        raise ValueError(
            f"pH {pH} is outside the validated range {lo}-{hi}. "
            "The model has no validated pH dependence there, so no result is reported."
        )


def rhs(t, y, k_f, k_r, k_pfas, p):
    """Right-hand side of the 6-species ODE system.
    Species order: H2O2, OH, PFAS, I (intermediates), Fe2+, Fe3+. Concentrations in mol/L."""
    H2O2, OH, PFAS, I, Fe2, Fe3 = np.maximum(y, 0.0)

    r_fenton = k_f * Fe2 * H2O2
    r_photo_fe = p["k_hv_fe"] * Fe3
    r_photo_h2o2 = p["k_hv_h2o2"] * H2O2
    r_pfas = k_pfas * OH * PFAS
    r_I = p["k_I"] * OH * I
    r_scav_h2o2 = p["k_scav"] * OH * H2O2
    r_fe_oh = p["k_Fe"] * Fe2 * OH

    dH2O2 = -r_fenton - r_photo_h2o2 - r_scav_h2o2 - k_r * Fe3 * H2O2
    dOH = (r_fenton + 2 * r_photo_h2o2 + r_photo_fe
           - r_pfas - r_I - p["k_s"] * OH - r_scav_h2o2 - r_fe_oh)
    dPFAS = -r_pfas
    dI = r_pfas - r_I
    dFe2 = -r_fenton + k_r * Fe3 * H2O2 + r_photo_fe - r_fe_oh
    dFe3 = -dFe2  # total iron is conserved
    return [dH2O2, dOH, dPFAS, dI, dFe2, dFe3]


def simulate(pfas_name, pfas_ng_l, pH, temp_C, H2O2_mM, Fe_uM,
             params=PARAMS, t_end=T_END_S):
    """Run one batch simulation for one compound at one set of conditions.

    Returns a dict. Raises ValueError for out-of-range pH and RuntimeError
    if the ODE solver fails.
    """
    check_ph(pH)
    if pfas_name not in MW:
        raise KeyError(f"Unknown compound {pfas_name}. Known: {list(MW)}")
    if pfas_ng_l <= 0:
        raise ValueError("pfas_ng_l must be positive")

    arr = arrhenius_factor(temp_C)
    k_f = params["k_f_ref"] * arr
    k_r = params["k_r_ref"] * arr
    k_pfas = K_OH[pfas_name]

    # ng/L -> mol/L, using this compound's own molar mass
    pfas0 = pfas_ng_l * 1e-9 / MW[pfas_name]
    h0 = H2O2_mM * 1e-3
    fe0 = Fe_uM * 1e-6

    t_eval = np.linspace(0.0, t_end, int(t_end) + 1)
    sol = solve_ivp(
        rhs, (0.0, t_end), [h0, 0.0, pfas0, 0.0, fe0, 0.0],
        method="BDF", t_eval=t_eval, args=(k_f, k_r, k_pfas, params),
        rtol=1e-8, atol=1e-14,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    t_min = sol.t / 60.0
    pfas_t = np.maximum(sol.y[2], 0.0)
    removal = 1.0 - pfas_t / pfas0

    hit = np.nonzero(removal >= TARGET_REMOVAL)[0]
    t90_min = float(t_min[hit[0]]) if hit.size else None

    fe_total = sol.y[4] + sol.y[5]
    fe_balance_error = float(np.max(np.abs(fe_total - fe0)) / fe0) if fe0 > 0 else 0.0

    return {
        "compound": pfas_name,
        "pH": pH,
        "temp_C": temp_C,
        "k_f": k_f,
        "k_OH": k_pfas,
        "pfas0_mol_l": pfas0,
        "t_min": t_min,
        "removal": removal,
        "removal_final": float(removal[-1]),
        "t90_min": t90_min,
        "fe_balance_error": fe_balance_error,
    }


def reagent_cost_per_litre(H2O2_mM, Fe_uM, t90_min, costs):
    """Dollars per litre to reach the target removal. None if the target is not reached.

    costs keys: H2O2_per_mmol, Fe_per_mmol, UV_per_min.
    """
    if t90_min is None:
        return None
    return (H2O2_mM * costs["H2O2_per_mmol"]
            + Fe_uM * 1e-3 * costs["Fe_per_mmol"]
            + t90_min * costs["UV_per_min"])


def cost_per_ug_removed(pfas_ng_l, cost_per_L):
    """Dollars per microgram removed at the target removal. Uses mass units, so no molar-mass error."""
    if cost_per_L is None:
        return None
    removed_ug_per_L = TARGET_REMOVAL * pfas_ng_l * 1e-3
    return cost_per_L / removed_ug_per_L
