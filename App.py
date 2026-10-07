import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Import your local well data module
from well_data import BOMARC_WELLS, FIELD_CONDITIONS, get_well_data

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
    "Initial conditions: Finkelstein et al. 2025 USGS & Suffolk County Dept. of Health Services"
)
st.warning(
    "⚠️ **Validation pending.** Rate constants are literature-derived (Hori et al. 2004; "
    "De Laat & Gallard 1999). Experimental validation against field data is in progress. "
    "Results are for feasibility screening only."
)

st.divider()

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Screen",
    ["🏠 Home & Overview", "📊 Simulation Optimizer", "⚡ Treatment Comparison", "📧 Action & Private Well Portal"]
)

# ─────────────────────────────────────────
# DYNAMIC SIDEBAR INPUTS (Only shown on relevant screens)
# ─────────────────────────────────────────
# Initialize default values so variables exist globally
default_pH, default_temp, default_H2O2, default_Fe, default_PFOA = 5.83, 12.9, 10.0, 100.0, 7.75
cost_H2O2_per_mmol, cost_Fe_per_mmol, cost_UV_per_min = 0.014, 0.0557, 0.008
run_button = False

if page == "📊 Simulation Optimizer":
    st.sidebar.markdown("---")
    st.sidebar.title("⚙️ Treatment Parameters")
    
    mode = st.sidebar.selectbox(
        "📍 Select Data Source",
        ["Custom / Manual Input", "Suffolk County BOMARC Wells", "Lab Optimal (pH 3, 25°C)"]
    )

    if mode == "Suffolk County BOMARC Wells":
        selected_well_name = st.sidebar.selectbox("Choose Well Site", list(BOMARC_WELLS.keys()))
        well_info = get_well_data(selected_well_name)
        
        default_pH   = FIELD_CONDITIONS["pH"]
        default_temp = FIELD_CONDITIONS["temp_C"]
        default_H2O2 = 10.0
        default_Fe   = 100.0
        default_PFOA = float(well_info["level_ng_l"])
        
        st.sidebar.info(
            f"**Site:** {well_info['location']}\n\n"
            f"**Contaminant:** {well_info['contaminant']} ({well_info['level_ng_l']} ng/L)\n\n"
            f"**Safe Limit:** {well_info['safe_limit_ng_l']} ng/L"
        )
    elif mode == "Lab Optimal (pH 3, 25°C)":
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
    user_pH   = st.sidebar.slider("pH", 3.0, 8.0, default_pH, 0.1)
    user_temp = st.sidebar.slider("Temperature (°C)", 5.0, 35.0, default_temp, 0.5)
    user_PFOA = st.sidebar.slider("Initial Concentration (ng/L)", 1.0, 150.0, default_PFOA, 0.5)

    st.sidebar.markdown("### 🧪 Reagent Doses")
    user_H2O2 = st.sidebar.slider("H₂O₂ Dose (mM)", 1.0, 100.0, default_H2O2, 1.0)
    user_Fe   = st.sidebar.slider("Fe²⁺ Dose (µM)", 1.0, 5000.0, default_Fe, 10.0)

    st.sidebar.markdown("### 💰 Cost Parameters")
    cost_H2O2_per_mmol = st.sidebar.number_input("H₂O₂ cost ($/mmol)", value=0.014, format="%.4f")
    cost_Fe_per_mmol   = st.sidebar.number_input("Fe²⁺ cost ($/mmol)", value=0.0557, format="%.4f")
    cost_UV_per_min    = st.sidebar.number_input("UV cost ($/min)", value=0.008, format="%.4f")

    run_button = st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────
# MODEL CONSTANTS & KINETICS
# ─────────────────────────────────────────
MW_PFOA  = 414.07
T_REF    = 298.15
Ea_J     = 50_000
R_gas    = 8.314

k_PFOA     = 1.2e7
k_I        = 3e8
k_s        = 2e4
k_H2O2_scav = 2.7e7
k_Fe       = 4.3e8
k_hv_Fe    = 3.5e-4
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
    if PFOA_0 <= 0:
        return 0.0
    removal = (1 - np.maximum(sol.y[2], 0) / PFOA_0) * 100
    idx = np.where(removal >= target)[0]
    return sol.t[idx[0]] / 60.0 if len(idx) > 0 else None

# ─────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────

if page == "🏠 Home & Overview":
    st.info("👈 Use the sidebar navigation menu to explore simulations, treatability comparisons, and action tools.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Suffolk County PFOA (avg)", "7.75 ng/L", "USGS 2021-2023")
    col2.metric("Groundwater pH (avg)", "5.83", "Range: 5.5–6.2")
    col3.metric("Groundwater Temp (avg)", "12.9°C", "Range: 12.1–14.6°C")
    
    st.markdown("### About This Tool")
    st.markdown("""
    This model simulates **Photo-Fenton Advanced Oxidation** for PFAS degradation
    in Long Island groundwater using real well data from the Suffolk County Department of Health Services and USGS.

    **Key features:**
    - Real BOMARC and municipal well contamination records
    - Temperature and pH corrections to Fenton kinetics
    - Cost analysis per liter treated
    - Dedicated private well reporting portal for unsewered sectors
    """)

elif page == "📊 Simulation Optimizer":
    if not run_button:
        st.info("👈 Adjust your parameters in the sidebar and click **Run Simulation**.")
    else:
        PFOA_0   = user_PFOA * 1e-9 / MW_PFOA
        H2O2_mol = user_H2O2 * 1e-3
        Fe_mol   = user_Fe   * 1e-6
        k_f, k_r = build_rate_constants(user_pH, user_temp)

        with st.spinner("Running simulation..."):
            sol = run_simulation(H2O2_mol, Fe_mol, PFOA_0, k_f, k_r)

        t_min      = sol.t / 60
        PFOA_t     = np.maximum(sol.y[2], 0)
        removal    = (1 - PFOA_t / PFOA_0) * 100 if PFOA_0 > 0 else np.zeros_like(PFOA_t)
        final_rem  = removal[-1]
        t90        = time_to_target(sol, PFOA_0, 90)

        # Cost calculation
        PFOA_removed_ug = (PFOA_0 - PFOA_t[-1]) * MW_PFOA * 1e6
        t_for_cost      = t90 if t90 else 120.0
        cost_h2o2  = H2O2_mol * 1000 * cost_H2O2_per_mmol
        cost_iron  = Fe_mol   * 1000 * cost_Fe_per_mmol
        cost_uv    = t_for_cost * cost_UV_per_min
        cost_total = cost_h2o2 + cost_iron + cost_uv
        cost_per_ug = cost_total / PFOA_removed_ug if PFOA_removed_ug > 0 else 0.0

        st.markdown("### 📊 Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Contaminant Removal", f"{final_rem:.1f}%",
                  "✅ Above 90%" if final_rem >= 90 else "❌ Below 90%")
        m2.metric("Time to 90% Removal",
                  f"{t90:.1f} min" if t90 else "Not reached",
                  "Within 2 hrs" if t90 else "Increase dose")
        m3.metric("Cost per Liter Treated", f"${cost_total:.4f}")
        m4.metric("Cost per µg Removed",
                  f"${cost_per_ug:.4f}" if cost_per_ug > 0 else "N/A")

        st.divider()

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Removal Over Time")
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(t_min, removal, color='#2196F3', linewidth=2.5)
            ax1.axhline(90, color='gray', linestyle='--', alpha=0.6, label='90% target')
            ax1.axhline(99, color='gray', linestyle=':',  alpha=0.4, label='99% target')
            if t90:
                ax1.axvline(t90, color='green', linestyle='--', alpha=0.5,
                            label=f't₉₀ = {t90:.1f} min')
            ax1.set_xlabel('Time (min)')
            ax1.set_ylabel('Removal (%)')
            ax1.set_ylim(0, 105)
            ax1.set_xlim(0, 120)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

        with col_right:
            st.markdown("#### Concentration Over Time")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(t_min, PFOA_t * 1e12, color='#F44336', linewidth=2.5)
            ax2.set_xlabel('Time (min)')
            ax2.set_ylabel('Concentration (pmol/L)')
            ax2.set_xlim(0, 120)
            ax2.grid(True, alpha=0.3)
            
            final_ng_l = PFOA_t[-1] * MW_PFOA * 1e9
            ax2.set_title(f'Initial: {default_PFOA:.1f} ng/L → Final: {final_ng_l:.2f} ng/L')
            st.pyplot(fig2)

elif page == "⚡ Treatment Comparison":
    st.markdown("### ⚡ Granular Activated Carbon (GAC) vs. Photo-Fenton AOP")
    st.markdown("Compare operational parameters, removal efficiencies, and economic feasibility.")

    col_gac, col_aop = st.columns(2)

    with col_gac:
        st.subheader("🔘 Granular Activated Carbon (GAC)")
        gac_ebct = st.slider("GAC Bed Contact Time (EBCT in min)", 5.0, 30.0, 15.0, 1.0)
        gac_cost_lb = st.number_input("Carbon media cost ($/lb)", value=2.25)
        st.metric("Estimated Removal Efficiency", "75% - 85%", "Struggles with ultra-short chains")
        st.metric("Media Spent Frequency", "High (Requires disposal)")

    with col_aop:
        st.subheader("⚡ Photo-Fenton AOP")
        st.metric("Estimated Removal Efficiency", "90%+", "Destroys molecular structure")
        st.metric("Destruction Speed", "45–60 min batch/flow")
        st.metric("Byproduct Management", "Mineralized end-products")

    # Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4))
    treatments = ["GAC Adsorption", "Photo-Fenton AOP"]
    removals = [80, 92]
    colors = ["#FFA500", "#4CAF50"]
    bars = ax.bar(treatments, removals, color=colors, alpha=0.8)
    ax.set_ylabel("Expected Efficacy %", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h, f'{h}%', ha='center', va='bottom', fontweight='bold')
    st.pyplot(fig)

elif page == "📧 Action & Private Well Portal":
    st.markdown("### 🚨 Private Well Contamination Action & Reporting Portal")
    st.markdown("""
    *Private wells in unsewered sectors of Long Island frequently fly under municipal radar.* 
    Use this customized reporting generator to draft a formal notification packet for local health officials and town supervisors.
    """)

    with st.form("private_well_form"):
        c1, c2 = st.columns(2)
        with c1:
            resident_name = st.text_input("Property Owner / Resident Name", "Jane Doe")
            property_address = st.text_input("Property Address / Town", "Westhampton, NY 11977")
            well_depth = st.number_input("Well Depth (ft)", value=65)
        with c2:
            official_name = st.text_input("Town Official / Health Dept Contact", "Town Supervisor / SCDHS Rep")
            official_email = st.text_input("Official Email Address", "waterquality@suffolkcountyny.gov")
            detected_pfoa = st.number_input("Detected PFOA/PFOS Level (ng/L)", value=85.0)

        submitted = st.form_submit_button("Generate Formal Advocacy Notification Draft")

    if submitted:
        st.success("✅ Formal Notification Packet Generated!")
        letter_draft = f"""SUBJECT: URGENT: Private Drinking Well PFAS Detection - {property_address}

Dear {official_name},

I am writing to formally report elevated PFAS contamination detected at a private well location in {property_address}. 

Because private wells lack the routine municipal testing oversight of public water districts, this location requires immediate official awareness and technical verification.

- Resident / Owner: {resident_name}
- Well Depth: {well_depth} ft
- Detected Concentration: {detected_pfoa} ng/L (Exceeding NYS / EPA health guidelines)

We request confirmation sampling by the Suffolk County Department of Health Services and technical guidance on advanced mitigation options (such as GAC or Photo-Fenton treatment systems).

Sincerely,
{resident_name}
{property_address}
"""
        st.text_area("Copy and paste this official notification draft:", letter_draft, height=280)
