import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="LI PFAS Treatment Optimizer",
    page_icon="💧",
    layout="wide"
)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("💧 Long Island PFAS Treatment Optimizer")
st.subheader("Photo-Fenton Advanced Oxidation Process Feasibility Model")
st.caption(
    "Developed by Leo Cataldo | CUNY MEC Vittadello Lab | "
    "Initial conditions: Finkelstein et al. 2025 USGS Suffolk County data"
)
st.warning(
    "⚠️ **Validation pending.** Rate constants are literature-derived (Hori et al. 2004; "
    "De Laat & Gallard 1999). Experimental validation against field data is in progress. "
    "Results are for feasibility screening only."
)

st.divider()

# ─────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────
st.sidebar.title("⚙️ Treatment Parameters")
st.sidebar.markdown("Adjust conditions and click **Run Simulation**.")

preset = st.sidebar.selectbox(
    "📍 Load Preset Conditions",
    ["Custom", "Suffolk County Field Conditions", "Lab Optimal (pH 3, 25°C)"]
)

if preset == "Suffolk County Field Conditions":
    default_pH   = 5.83
    default_temp = 12.9
    default_H2O2 = 10.0
    default_Fe   = 100.0
    default_PFOA = 7.75
elif preset == "Lab Optimal (pH 3, 25°C)":
    default_pH   = 3.0
    default_temp = 25.0
    default_H2O2 = 10.0
    default_Fe   = 100.0
    default_PFOA = 7.75
else:
    default_pH   = 5.83
    default_temp = 12.9
    default_H2O2 = 10.0
    default_Fe   = 100.0
    default_PFOA = 7.75

st.sidebar.markdown("### 🌍 Field Conditions")
user_pH   = st.sidebar.slider("pH", 3.0, 8.0, default_pH, 0.1,
                               help="Optimal Photo-Fenton pH is 3–4. Suffolk County avg: 5.83")
user_temp = st.sidebar.slider("Temperature (°C)", 5.0, 35.0, default_temp, 0.5,
                               help="Suffolk County groundwater avg: 12.9°C")
user_PFOA = st.sidebar.slider("Initial PFOA (ng/L)", 1.0, 50.0, default_PFOA, 0.5,
                               help="Suffolk County range: 3.8–13 ng/L (Finkelstein et al. 2025)")

st.sidebar.markdown("### 🧪 Reagent Doses")
user_H2O2 = st.sidebar.slider("H₂O₂ Dose (mM)", 1.0, 100.0, default_H2O2, 1.0,
                               help="Typical municipal range: 5–50 mM")
user_Fe   = st.sidebar.slider("Fe²⁺ Dose (µM)", 1.0, 5000.0, default_Fe, 10.0,
                               help="Optimal Fe varies by H₂O₂ dose and pH")

st.sidebar.markdown("### 💰 Cost Parameters")
cost_H2O2_per_mmol = st.sidebar.number_input("H₂O₂ cost ($/mmol)", value=0.014, format="%.4f")
cost_Fe_per_mmol   = st.sidebar.number_input("Fe²⁺ cost ($/mmol)", value=0.0557, format="%.4f")
cost_UV_per_min    = st.sidebar.number_input("UV cost ($/min)", value=0.008, format="%.4f")

run_button = st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────
# MODEL CONSTANTS (recalculated from inputs)
# ─────────────────────────────────────────
MW_PFOA  = 414.07
T_REF    = 298.15
Ea_J     = 50_000
R_gas    = 8.314

k_PFOA      = 1.2e7
k_I         = 3e8
k_s         = 2e4
k_H2O2_scav = 2.7e7
k_Fe        = 4.3e8
k_hv_Fe     = 3.5e-4
k_hv_h2o2   = 6.6e-6

def build_rate_constants(pH, temp_C):
    temp_K = temp_C + 273.15
    arr    = np.exp(-Ea_J / R_gas * (1/temp_K - 1/T_REF))
    pH_cor = 10 ** (-(pH - 3.0) * 0.5)
    k_f    = 63.0   * arr * pH_cor
    k_r    = 8.4e-6 * arr
    return k_f, k_r

def photo_fenton_odes(t, y, Fe_total, k_f, k_r):
    H2O2, OH, PFOA, I, Fe2, Fe3 = [max(v, 0) for v in y]
    dH2O2 = (-k_hv_h2o2 * H2O2 - k_H2O2_scav * OH * H2O2
              - k_f * Fe2 * H2O2 - k_r * Fe3 * H2O2)
    dOH   = (2 * k_hv_h2o2 * H2O2 + k_f * Fe2 * H2O2
             - k_PFOA * OH * PFOA - k_I * OH * I
             - k_s * OH - k_H2O2_scav * OH * H2O2
             - k_Fe * Fe2 * OH)
    dPFOA = -k_PFOA * OH * PFOA
    dI    =  k_PFOA * OH * PFOA - k_I * OH * I
    dFe2  = (-k_f * Fe2 * H2O2 + k_r * Fe3 * H2O2
              - k_Fe * Fe2 * OH + k_hv_Fe * Fe3)
    dFe3  = -dFe2
    return [dH2O2, dOH, dPFOA, dI, dFe2, dFe3]

def run_simulation(H2O2_init, Fe_total, PFOA_0, k_f, k_r):
    t_eval = np.linspace(0, 7200, 3600)
    y0 = [H2O2_init, 0.0, PFOA_0, 0.0, Fe_total, 0.0]
    sol = solve_ivp(
        fun=lambda t, y: photo_fenton_odes(t, y, Fe_total, k_f, k_r),
        t_span=(0, 7200), y0=y0, method='BDF',
        t_eval=t_eval, rtol=1e-8, atol=1e-12
    )
    return sol

def time_to_target(sol, PFOA_0, target=90):
    removal = (1 - np.maximum(sol.y[2], 0) / PFOA_0) * 100
    idx = np.where(removal >= target)[0]
    return sol.t[idx[0]] / 60.0 if len(idx) > 0 else None

# ─────────────────────────────────────────
# MAIN PANEL — DEFAULT STATE
# ─────────────────────────────────────────
if not run_button:
    st.info("👈 Set your parameters in the sidebar and click **Run Simulation** to begin.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Suffolk County PFOA (avg)", "7.75 ng/L", "USGS 2021-2023")
    col2.metric("Groundwater pH (avg)", "5.83", "Range: 5.5–6.2")
    col3.metric("Groundwater Temp (avg)", "12.9°C", "Range: 12.1–14.6°C")

    st.markdown("### About This Tool")
    st.markdown("""
    This model simulates **Photo-Fenton Advanced Oxidation** for PFAS (PFOA) degradation
    in Long Island groundwater. It uses a system of ordinary differential equations (ODEs)
    to predict hydroxyl radical (·OH) generation and PFOA breakdown over time.

    **Key features:**
    - Real Suffolk County field conditions from USGS data (Finkelstein et al. 2025)
    - Temperature and pH corrections to Fenton kinetics (De Laat & Gallard 1999)
    - Cost analysis per liter treated
    - Comparison across H₂O₂ dose scenarios

    **Why Photo-Fenton for PFAS?**  
    Carbon adsorption — the standard treatment — fails for ultra-short chain PFAS variants.
    Photo-Fenton AOP is a candidate alternative, particularly for these recalcitrant compounds.
    This tool screens feasibility before costly bench-scale experiments.

    **Data source:** Finkelstein et al. 2025, USGS Suffolk County NY groundwater PFAS survey.  
    **GitHub:** [LeoCataldo0701/li-water-aop-feasibility](https://github.com/LeoCataldo0701/li-water-aop-feasibility)
    """)

# ─────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────
else:
    PFOA_0   = user_PFOA * 1e-9 / MW_PFOA
    H2O2_mol = user_H2O2 * 1e-3
    Fe_mol   = user_Fe   * 1e-6
    k_f, k_r = build_rate_constants(user_pH, user_temp)

    with st.spinner("Running simulation..."):
        sol = run_simulation(H2O2_mol, Fe_mol, PFOA_0, k_f, k_r)

    t_min      = sol.t / 60
    PFOA_t     = np.maximum(sol.y[2], 0)
    removal    = (1 - PFOA_t / PFOA_0) * 100
    final_rem  = removal[-1]
    t90        = time_to_target(sol, PFOA_0, 90)
    t99        = time_to_target(sol, PFOA_0, 99)

    # Cost calculation
    PFOA_removed_ug = (PFOA_0 - PFOA_t[-1]) * MW_PFOA * 1e6
    t_for_cost      = t90 if t90 else 120.0
    cost_h2o2  = H2O2_mol * 1000 * cost_H2O2_per_mmol
    cost_iron  = Fe_mol   * 1000 * cost_Fe_per_mmol
    cost_uv    = t_for_cost * cost_UV_per_min
    cost_total = cost_h2o2 + cost_iron + cost_uv
    cost_per_ug = cost_total / PFOA_removed_ug if PFOA_removed_ug > 0 else 999

    # ── KPI METRICS ──
    st.markdown("### 📊 Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Final PFOA Removal", f"{final_rem:.1f}%",
              "✅ Above 90%" if final_rem >= 90 else "❌ Below 90%")
    m2.metric("Time to 90% Removal",
              f"{t90:.1f} min" if t90 else "Not reached",
              "Within 2 hrs" if t90 else "Increase dose")
    m3.metric("Cost per Liter Treated", f"${cost_total:.4f}")
    m4.metric("Cost per µg PFOA Removed",
              f"${cost_per_ug:.4f}" if cost_per_ug < 999 else "N/A")

    st.divider()

    # ── PLOTS ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### PFOA Removal Over Time")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(t_min, removal, color='#2196F3', linewidth=2.5)
        ax1.axhline(90, color='gray', linestyle='--', alpha=0.6, label='90% target')
        ax1.axhline(99, color='gray', linestyle=':',  alpha=0.4, label='99% target')
        if t90:
            ax1.axvline(t90, color='green', linestyle='--', alpha=0.5,
                        label=f't₉₀ = {t90:.1f} min')
        ax1.set_xlabel('Time (min)')
        ax1.set_ylabel('PFOA Removal (%)')
        ax1.set_ylim(0, 105)
        ax1.set_xlim(0, 120)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)

    with col_right:
        st.markdown("#### PFOA Concentration Over Time")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(t_min, PFOA_t * 1e12, color='#F44336', linewidth=2.5)
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('[PFOA] (pmol/L)')
        ax2.set_xlim(0, 120)
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f'Initial: {user_PFOA:.1f} ng/L → Final: {PFOA_t[-1]*MW_PFOA*1e12:.2f} pmol/L')
        st.pyplot(fig2)

    # ── COST BREAKDOWN ──
    st.markdown("#### 💰 Cost Breakdown (per liter treated)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("H₂O₂ Cost", f"${cost_h2o2:.4f}")
    c2.metric("Fe²⁺ Cost",  f"${cost_iron:.4f}")
    c3.metric("UV Cost",    f"${cost_uv:.4f}")
    c4.metric("Total",      f"${cost_total:.4f}")

    # ── CONDITIONS USED ──
    st.divider()
    st.markdown("#### 🔬 Conditions Used in This Run")
    arr_factor = np.exp(-Ea_J / R_gas * (1/(user_temp+273.15) - 1/T_REF))
    pH_cor     = 10 ** (-(user_pH - 3.0) * 0.5)
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | pH | {user_pH} |
    | Temperature | {user_temp}°C |
    | PFOA₀ | {user_PFOA} ng/L ({PFOA_0:.3e} mol/L) |
    | H₂O₂ dose | {user_H2O2} mM |
    | Fe²⁺ dose | {user_Fe} µM |
    | Arrhenius factor | {arr_factor:.3f} |
    | pH correction | {pH_cor:.4f} |
    | k_f (corrected) | {k_f:.4f} L/(mol·s) |
    | Rate constants source | Hori et al. 2004; De Laat & Gallard 1999 |
    | Field data source | Finkelstein et al. 2025, USGS |
    """)

    st.caption("⚠️ Validation pending. Results are theoretical — experimental confirmation required before use in treatment planning.")
