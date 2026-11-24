"""
Core simulation engine for photofenton.

Implements a 6-species batch photo-Fenton ODE system and reagent cost functions.
All compound properties and kinetic parameters are loaded from the JSON data files
in photofenton/data/, so new compounds can be added without editing source code.

Model limitations (see paper.md for full discussion)
----------------------------------------------------
- Validated pH range: 2.5-3.5 only. k_f_ref was measured at pH ~3.
- Batch reactor only. No flow, residence time, or mass transfer.
- k_I and k_s are placeholders. k_hv_h2o2 and k_hv_fe are uncalibrated.
- Not validated against experimental PFAS degradation data.
- Results are for feasibility screening only.
"""
import numpy as np
from scipy.integrate import solve_ivp
from typing import Optional

from photofenton.compounds import get_compound
from photofenton.parameters import load_parameters

# Valid pH range. k_f_ref was measured near pH 3 (Walling 1975).
# No validated pH correction exists for this model above pH 3.5.
PH_VALID_RANGE = (2.5, 3.5)

# Target fractional removal for t90 and cost calculations.
TARGET_REMOVAL = 0.90

# Default simulation duration, seconds (2 hours).
T_END_S = 7200.0


def arrhenius_factor(temp_C: float, params: Optional[dict] = None) -> float:
    """Arrhenius temperature correction factor relative to 25 C reference.

    Scales k_f and k_r from their 25 C reference values to the given temperature.
    Activation energy Ea = 50 kJ/mol (De Laat & Gallard 1999).

    Parameters
    ----------
    temp_C : float
        Water temperature in degrees Celsius.
    params : dict, optional
        Parameter dict. Defaults to values loaded from parameters.json.

    Returns
    -------
    float
        Dimensionless factor. 1.0 at 25 C, ~0.43 at 12.9 C (Suffolk County field value).

    Examples
    --------
    >>> arrhenius_factor(25.0)
    1.0
    >>> round(arrhenius_factor(12.9), 2)
    0.43
    """
    if params is None:
        params = load_parameters()
    temp_K = temp_C + 273.15
    return float(np.exp(
        -params["Ea_J_mol"] / 8.314 * (1.0 / temp_K - 1.0 / params["T_ref_K"])
    ))


def _rhs(t, y, k_f, k_r, k_pfas, params):
    """ODE right-hand side. Not part of the public API; called by simulate().

    Species vector y = [H2O2, OH, PFAS, I, Fe2+, Fe3+], all in mol/L.
    Negative concentrations are clamped to zero before each evaluation.
    """
    H2O2, OH, PFAS, I, Fe2, Fe3 = np.maximum(y, 0.0)

    r_fenton     = k_f * Fe2 * H2O2
    r_photo_fe   = params["k_hv_fe"] * Fe3
    r_photo_h2o2 = params["k_hv_h2o2"] * H2O2
    r_pfas       = k_pfas * OH * PFAS
    r_I          = params["k_I"] * OH * I
    r_scav_h2o2  = params["k_scav"] * OH * H2O2
    r_fe_oh      = params["k_Fe"] * Fe2 * OH

    dH2O2 = -r_fenton - r_photo_h2o2 - r_scav_h2o2 - k_r * Fe3 * H2O2
    dOH   = (r_fenton + 2 * r_photo_h2o2 + r_photo_fe
             - r_pfas - r_I - params["k_s"] * OH - r_scav_h2o2 - r_fe_oh)
    dPFAS = -r_pfas
    dI    =  r_pfas - r_I
    dFe2  = -r_fenton + k_r * Fe3 * H2O2 + r_photo_fe - r_fe_oh
    dFe3  = -dFe2  # iron is conserved: d[Fe2+]/dt + d[Fe3+]/dt = 0

    return [dH2O2, dOH, dPFAS, dI, dFe2, dFe3]


def simulate(
    pfas_name: str,
    pfas_ng_l: float,
    pH: float,
    temp_C: float,
    H2O2_mM: float,
    Fe_uM: float,
    params: Optional[dict] = None,
    t_end: float = T_END_S,
) -> dict:
    """Run a batch photo-Fenton simulation for one PFAS compound.

    Integrates the 6-species ODE system using SciPy's stiff BDF solver
    (rtol=1e-8, atol=1e-14) and returns removal kinetics and cost inputs.

    Parameters
    ----------
    pfas_name : str
        Compound name, e.g. 'PFOA'. Must exist in compounds.json.
    pfas_ng_l : float
        Initial PFAS concentration in ng/L (nanograms per litre).
    pH : float
        Reactor pH. Must be in PH_VALID_RANGE (2.5-3.5).
    temp_C : float
        Water temperature in degrees Celsius. Arrhenius correction applied.
    H2O2_mM : float
        Hydrogen peroxide dose in millimolar (mM).
    Fe_uM : float
        Ferrous iron (Fe2+) dose in micromolar (uM).
    params : dict, optional
        Override kinetic parameters. Defaults to parameters.json values.
        Useful for sensitivity analysis.
    t_end : float, optional
        Simulation duration in seconds. Default 7200 (2 hours).

    Returns
    -------
    dict with keys:
        compound (str)         : compound name
        pH (float)             : pH used
        temp_C (float)         : temperature used
        k_f (float)            : effective Fenton rate constant at this temperature
        k_OH (float)           : OH radical rate constant for this compound, L/(mol*s)
        pfas0_mol_l (float)    : initial PFAS concentration, mol/L
        t_min (ndarray)        : time array, minutes
        removal (ndarray)      : fractional removal at each time point (0-1)
        removal_final (float)  : fractional removal at t_end
        t90_min (float|None)   : time to 90% removal in minutes, or None if not reached
        fe_balance_error (float): max relative iron conservation error (quality check)

    Raises
    ------
    ValueError
        If pH is outside PH_VALID_RANGE or pfas_ng_l <= 0.
    KeyError
        If pfas_name is not in the compound database.
    RuntimeError
        If the ODE solver fails to converge.

    Examples
    --------
    >>> result = simulate("PFOA", pfas_ng_l=120.0, pH=3.0, temp_C=12.9,
    ...                   H2O2_mM=10.0, Fe_uM=5.0)
    >>> result["removal_final"]   # fraction removed at 2 hours
    >>> result["t90_min"]         # minutes to 90% removal
    """
    lo, hi = PH_VALID_RANGE
    if not lo <= pH <= hi:
        raise ValueError(
            f"pH {pH:.2f} is outside the validated range {lo}–{hi}. "
            "k_f_ref was measured near pH 3 (Walling 1975). "
            "This model has no validated pH correction outside this window."
        )
    if pfas_ng_l <= 0:
        raise ValueError(f"pfas_ng_l must be positive, got {pfas_ng_l}")

    compound = get_compound(pfas_name)
    if params is None:
        params = load_parameters()

    arr  = arrhenius_factor(temp_C, params)
    k_f  = params["k_f_ref"] * arr
    k_r  = params["k_r_ref"] * arr
    k_OH = compound["k_OH"]

    # Convert ng/L → mol/L using this compound's own molar mass.
    pfas0 = pfas_ng_l * 1e-9 / compound["molar_mass"]
    h0    = H2O2_mM * 1e-3
    fe0   = Fe_uM * 1e-6

    t_eval = np.linspace(0.0, t_end, int(t_end) + 1)
    sol = solve_ivp(
        _rhs, (0.0, t_end),
        y0=[h0, 0.0, pfas0, 0.0, fe0, 0.0],
        method="BDF", t_eval=t_eval,
        args=(k_f, k_r, k_OH, params),
        rtol=1e-8, atol=1e-14,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    t_min  = sol.t / 60.0
    pfas_t = np.maximum(sol.y[2], 0.0)
    removal = 1.0 - pfas_t / pfas0

    hit = np.nonzero(removal >= TARGET_REMOVAL)[0]
    t90_min = float(t_min[hit[0]]) if hit.size else None

    fe_total = sol.y[4] + sol.y[5]
    fe_balance_error = (
        float(np.max(np.abs(fe_total - fe0)) / fe0) if fe0 > 0 else 0.0
    )

    return {
        "compound":        pfas_name,
        "pH":              pH,
        "temp_C":          temp_C,
        "k_f":             k_f,
        "k_OH":            k_OH,
        "pfas0_mol_l":     pfas0,
        "t_min":           t_min,
        "removal":         removal,
        "removal_final":   float(removal[-1]),
        "t90_min":         t90_min,
        "fe_balance_error": fe_balance_error,
        "H2O2_mM_t":       sol.y[0] * 1e3,
    }


def reagent_cost_per_litre(
    H2O2_mM: float,
    Fe_uM: float,
    t90_min: Optional[float],
    costs: dict,
) -> Optional[float]:
    """Reagent cost in dollars per litre of water treated to 90% removal.

    Does not include capital costs, labor, sludge disposal, or energy beyond UV.
    Returns None if the target removal was not reached (t90_min is None).

    Parameters
    ----------
    H2O2_mM : float
        H2O2 dose in mM.
    Fe_uM : float
        Fe2+ dose in uM.
    t90_min : float or None
        Time to 90% removal in minutes. Pass result['t90_min'] from simulate().
    costs : dict
        Cost rates with keys:
        - 'H2O2_per_mmol': dollars per mmol H2O2
        - 'Fe_per_mmol':   dollars per mmol Fe2+
        - 'UV_per_min':    dollars per minute of UV lamp operation

    Returns
    -------
    float or None
        Reagent cost in $/L, or None if target not reached.

    Examples
    --------
    >>> costs = {"H2O2_per_mmol": 0.014, "Fe_per_mmol": 0.0557, "UV_per_min": 0.008}
    >>> reagent_cost_per_litre(10.0, 5.0, 45.0, costs)
    """
    if t90_min is None:
        return None
    return (
        H2O2_mM * costs["H2O2_per_mmol"]
        + Fe_uM * 1e-3 * costs["Fe_per_mmol"]
        + t90_min * costs["UV_per_min"]
    )


def cost_per_ug_removed(pfas_ng_l: float, cost_per_L: Optional[float]) -> Optional[float]:
    """Reagent cost in dollars per microgram of PFAS removed.

    Uses mass units (ng/L → ug/L) so molar mass does not enter the calculation.

    Parameters
    ----------
    pfas_ng_l : float
        Initial PFAS concentration in ng/L.
    cost_per_L : float or None
        Cost per litre from reagent_cost_per_litre(). Pass None if target not reached.

    Returns
    -------
    float or None
        Cost in $/ug removed, or None if target not reached.
    """
    if cost_per_L is None:
        return None
    removed_ug_per_L = TARGET_REMOVAL * pfas_ng_l * 1e-3  # ng→ug: divide by 1000
    return cost_per_L / removed_ug_per_L


def parameter_sweep(
    pfas_name: str,
    pfas_ng_l: float,
    pH: float,
    temp_C: float,
    H2O2_mM_values,
    Fe_uM_values,
    costs: dict,
    t_end: float = T_END_S,
) -> list:
    """Run simulate() over a grid of H2O2 and Fe2+ doses.

    Useful for sensitivity analysis and dose optimization before bench-scale testing.

    Parameters
    ----------
    pfas_name : str
        Compound name.
    pfas_ng_l : float
        Initial PFAS concentration in ng/L.
    pH : float
        Reactor pH, must be in PH_VALID_RANGE.
    temp_C : float
        Temperature in degrees Celsius.
    H2O2_mM_values : iterable of float
        H2O2 doses to sweep, in mM.
    Fe_uM_values : iterable of float
        Fe2+ doses to sweep, in uM.
    costs : dict
        Cost dict as in reagent_cost_per_litre().
    t_end : float, optional
        Simulation duration, seconds.

    Returns
    -------
    list of dict
        One entry per (H2O2, Fe) pair, each containing the simulate() result
        plus 'H2O2_mM', 'Fe_uM', 'cost_per_L', and 'cost_per_ug' keys.

    Examples
    --------
    >>> costs = {"H2O2_per_mmol": 0.014, "Fe_per_mmol": 0.0557, "UV_per_min": 0.008}
    >>> rows = parameter_sweep("PFOA", 120.0, 3.0, 12.9,
    ...                        H2O2_mM_values=[5, 10, 20],
    ...                        Fe_uM_values=[2, 5, 10],
    ...                        costs=costs)
    >>> # find the cheapest dose that reaches 90%
    >>> viable = [r for r in rows if r["cost_per_L"] is not None]
    >>> best = min(viable, key=lambda r: r["cost_per_L"])
    """
    results = []
    for h in H2O2_mM_values:
        for fe in Fe_uM_values:
            r = simulate(pfas_name, pfas_ng_l, pH, temp_C, h, fe, t_end=t_end)
            cost = reagent_cost_per_litre(h, fe, r["t90_min"], costs)
            r["H2O2_mM"]     = h
            r["Fe_uM"]        = fe
            r["cost_per_L"]   = cost
            r["cost_per_ug"]  = cost_per_ug_removed(pfas_ng_l, cost)
            results.append(r)
    return results

def sensitivity_analysis(
    pfas_name, pfas_ng_l, pH, temp_C, H2O2_mM, Fe_uM,
    param_name, factors=(0.1, 0.5, 2.0, 10.0), t_end=T_END_S,
):
    """Vary one kinetic parameter by multiplicative factors and report removal.

    Parameters
    ----------
    param_name : str
        Key in parameters.json, e.g. 'k_s', 'k_I', 'k_hv_fe'.
    factors : tuple of float
        Multipliers applied to the base value.

    Returns
    -------
    list of dict with keys 'factor', 'param_value', 'removal_final'.
    """
    base = load_parameters()
    if param_name not in base:
        raise KeyError(f"Unknown parameter '{param_name}'. Known: {sorted(base)}")
    results = []
    for factor in factors:
        modified = dict(base)
        modified[param_name] = base[param_name] * factor
        r = simulate(pfas_name, pfas_ng_l, pH, temp_C, H2O2_mM, Fe_uM,
                     params=modified, t_end=t_end)
        results.append({
            "factor": factor,
            "param_value": modified[param_name],
            "removal_final": r["removal_final"],
        })
    return results
