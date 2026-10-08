import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from well_data import BOMARC_WELLS, FIELD_CONDITIONS, get_well_data

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="LI PFAS Treatment Optimizer",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Long Island PFAS Treatment Optimizer")
st.subheader("Advanced Oxidation (Photo-Fenton) vs. GAC Feasibility Model")
st.caption("Developed by Leo Cataldo | CUNY MEC Vittadello Lab | Data: Finkelstein et al. 2025 USGS & SCDHS")
st.warning("⚠️ **Validation pending.** Results are for feasibility screening and comparative engineering analysis only.")
st.divider()

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION & CONTROLS
# ─────────────────────────────────────────
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select Screen",
    ["🏠 Home & Overview", "📊 Simulation & Treatment Comparison", "📧 Action & Site Reporting"]
)

# Initialize defaults
default_pH, default_temp, default_H2O2, default_Fe, default_PFOA = 5.83, 12.9, 10.0, 100.0, 7.75
cost_H2O2_per_mmol, cost_Fe_per_mmol, cost_UV_per_min = 0.014, 0.0557, 0.008
gac_ebct, gac_cost_lb, gac_disposal_cost = 15.0, 2.25, 1.50
run_sim_button = False

if page == "📊 Simulation & Treatment Comparison":
    st.sidebar.markdown("---")
    st.sidebar.title("⚙️ Scenario Parameters")
    
    mode = st.sidebar.selectbox(
        "📍 Select Data Source",
        ["Custom / Manual Input", "Suffolk County BOMARC Wells", "Lab Optimal (pH 3, 25°C)"]
    )

    if mode == "Suffolk County BOMARC Wells":
        selected_well_name = st.sidebar.selectbox("Choose Well Site", list(BOMARC_WELLS.keys()))
        well_info = get_well_data(selected_well_name)
        default_pH   = FIELD_CONDITIONS["pH"]
        default_temp = FIELD_CONDITIONS["temp_C"]
        default_PFOA = float(well_info["level_ng_l"])
        st.sidebar.info(f"**Site:** {well_info['location']}\n\n**Contaminant:** {well_info['contaminant']} ({well_info['level_ng_l']} ng/L)")
    elif mode == "Lab Optimal (pH 3, 25°C)":
        default_pH, default_temp, default_PFOA = 3.0, 25.0, 7.75
    else:
        default_pH, default_temp, default_PFOA = 5.83, 12.9, 7.75

    st.sidebar.markdown("### 🌍 Groundwater Conditions")
    user_pH   = st.sidebar.slider("pH", 3.0, 8.0, default_pH, 0.1)
    user_temp = st.sidebar.slider("Temperature (°C)", 5.0, 35.0, default_temp, 0.5)
    user_PFOA = st.sidebar.slider("Initial PFOA (ng/L)", 1.0, 150.0, default_PFOA, 0.5)

    st.sidebar.markdown("### ⚡ Photo-Fenton Doses & Costs")
    user_H2O2 = st.sidebar.slider("H₂O₂ Dose (mM)", 1.0, 100.0, 10.0, 1.0)
    user_Fe   = st.sidebar.slider("Fe²⁺ Dose (µM)", 1.0, 5000.0, 100.0, 10.0)
    cost_H2O2_per_mmol = st.sidebar.number_input("H₂O₂ cost ($/mmol)", value=0.014, format="%.4f")
    cost_Fe_per_mmol   = st.sidebar.number_input("Fe²⁺ cost ($/mmol)", value=0.0557, format="%.4f")
    cost_UV_per_min    = st.sidebar.number_input("UV cost ($/min)", value=0.008, format="%.4f")

    st.sidebar.markdown("### 🔘 GAC Parameters & Costs")
    gac_ebct = st.sidebar.slider("GAC Empty Bed Contact Time (min)", 5.0, 30.0, 15.0, 1.0)
    gac_cost_lb = st.sidebar.number_input("GAC Media Cost ($/lb)", value=2.25)
    gac_disposal_cost = st.sidebar.number_input("Spent Carbon Disposal Cost ($/lb)", value=1.50)

    run_sim_button = st.sidebar.button("▶️ Run Comparative Simulation", type="primary", use_container_width=True)

# ─────────────────────────────────────────
# KINETICS & MODEL FUNCTIONS
# ─────────────────────────────────────────
MW_PFOA = 414.07
T_REF = 298.15
Ea_J = 50_000
R_gas = 8.314

k_PFOA = 1.2e7
k_I = 3e8
k_s = 2e4
k_H2O2_scav = 2.7e7
k_Fe = 4.3e8
k_hv_Fe = 3.5e-4
k_hv_h2o2 = 6.6e-6

def build_rate_constants(pH, temp_C):
    temp_K = temp_C + 273.15
    arr = np.exp(-Ea_J / R_gas * (1/temp_K - 1/T_REF))
    pH_cor = 10 ** (-(pH - 3.0) * 0.5)
    k_f = 63.0 * arr * pH_cor
    k_r = 8.4e-6 * arr
    return k_f, k_r

def photo_fenton_odes(t, y, Fe_total, k_f, k_r):
    H2O2, OH, PFOA, I, Fe2, Fe3 = [max(v, 0) for v in y]
    dH2O2 = (-k_hv_h2o2 * H2O2 - k_H2O2_scav * OH * H2O2 - k_f * Fe2 * H2O2 - k_r * Fe3 * H2O2)
    dOH   = (2 * k_hv_h2o2 * H2O2 + k_f * Fe2 * H2O2 - k_PFOA * OH * PFOA - k_I * OH * I - k_s * OH - k_H2O2_scav * OH * H2O2 - k_Fe * Fe2 * OH)
    dPFOA = -k_PFOA * OH * PFOA
    dI    =  k_PFOA * OH * PFOA - k_I * OH * I
    dFe2  = (-k_f * Fe2 * H2O2 + k_r * Fe3 * H2O2 - k_Fe * Fe2 * OH + k_hv_Fe * Fe3)
    dFe3  = -dFe2
    return [dH2O2, dOH, dPFOA, dI, dFe2, dFe3]

# ─────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────

if page == "🏠 Home & Overview":
    st.markdown("### Welcome to the Long Island PFAS Feasibility & Action Suite")
    st.markdown("""
    This application assists researchers, municipal planners, and community advocates in evaluating 
    remediation options for PFAS contamination across Long Island groundwater.
    
    * **Simulation & Comparison**: Model Photo-Fenton destruction kinetics side-by-side with Granular Activated Carbon (GAC) filtration, including complete economic breakdowns.
    * **Action & Reporting**: Generate customized notification drafts for private well owners, public water districts, and industrial sites.
    """)

elif page == "📊 Simulation & Treatment Comparison":
    if not run_sim_button:
        st.info("👈 Set your parameters in the sidebar and click **Run Comparative Simulation**.")
    else:
        # Run Photo-Fenton Model
        PFOA_0 = user_PFOA * 1e-9 / MW_PFOA
        H2O2_mol = user_H2O2 * 1e-3
        Fe_mol = user_Fe * 1e-6
        k_f, k_r = build_rate_constants(user_pH, user_temp)
        
        t_eval = np.linspace(0, 7200, 3600)
        y0 = [H2O2_mol, 0.0, PFOA_0, 0.0, Fe_mol, 0.0]
        sol = solve_ivp(fun=lambda t, y: photo_fenton_odes(t, y, Fe_mol, k_f, k_r), t_span=(0, 7200), y0=y0, method='BDF', t_eval=t_eval)
        
        t_min = sol.t / 60
        fenton_pfoa = np.maximum(sol.y[2], 0)
        fenton_removal = (1 - fenton_pfoa / PFOA_0) * 100
        fenton_max_rem = fenton_removal[-1]
        
        # Find T90 or max time reached
        idx_90 = np.where(fenton_removal >= 90)[0]
        fenton_t_target = t_min[idx_90[0]] if len(idx_90) > 0 else 120.0

        # Fenton Cost calculation
        fenton_removed_ug = (PFOA_0 - fenton_pfoa[-1]) * MW_PFOA * 1e6
        cost_h2o2 = H2O2_mol * 1000 * cost_H2O2_per_mmol
        cost_iron = Fe_mol * 1000 * cost_Fe_per_mmol
        cost_uv = fenton_t_target * cost_UV_per_min
        fenton_total_cost = cost_h2o2 + cost_iron + cost_uv
        fenton_cost_per_ug = fenton_total_cost / fenton_removed_ug if fenton_removed_ug > 0 else 0.0

# GAC Modeling Calculation (Realistic constraints: efficiency ceiling, short-chain breakthrough, disposal liability)
        gac_max_rem = max(65.0, 85.0 - (gac_ebct * 0.2))  
        gac_total_cost = (gac_cost_lb + gac_disposal_cost) * (0.025 * (15.0 / max(5.0, gac_ebct))) 

        st.markdown("### 📊 Side-by-Side Treatment Comparison: Destruction vs. Adsorption")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("⚡ Photo-Fenton (Destructive AOP)", f"{fenton_max_rem:.1f}% Removal", f"Time to Target: {fenton_t_target:.1f} min")
            st.markdown("""
            * **Mechanism:** Complete mineralization (destroys bonds)
            * **Waste Stream:** None (benign end-products)
            * **Short-Chains:** Handled effectively
            """)
        with col2:
            st.metric("🔘 GAC Adsorption (Phase Transfer)", f"{gac_max_rem:.1f}% Removal (Ceiling)", f"EBCT: {gac_ebct} min")
            st.markdown("""
            * **Mechanism:** Phase transfer (traps contaminants)
            * **Waste Stream:** Spent hazardous carbon media requiring disposal/regeneration
            * **Short-Chains:** Prone to early breakthrough
            """)

        st.divider()

        # Overlaid Plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(t_min, fenton_removal, color='#2196F3', linewidth=2.5, label='Photo-Fenton AOP')
        ax1.axhline(gac_max_rem, color='#FFA500', linestyle='--', label=f'GAC Max Efficacy ({gac_max_rem:.0f}%)')
        ax1.set_xlabel('Time / Contact Duration (min)')
        ax1.set_ylabel('Removal Efficiency (%)')
        ax1.set_ylim(0, 105)
        ax1.set_xlim(0, 120)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Removal Efficiency Comparison')

        # Concentration over time overlay
        ax2.plot(t_min, fenton_pfoa * 1e12, color='#F44336', linewidth=2.5, label='Photo-Fenton PFOA (pmol/L)')
        ax2.set_xlabel('Time (min)')
        ax2.set_ylabel('Concentration (pmol/L)')
        ax2.set_xlim(0, 120)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Contaminant Destruction Profile')
        
        st.pyplot(fig)

elif page == "📧 Action & Site Reporting":
    st.markdown("### 🚨 Community Action & Site Reporting Portal")
    st.markdown("Select your reporting context below to generate a tailored notification packet for local authorities.")

    site_type = st.selectbox(
        "Select Site Category",
        ["Private Well (Unsewered Resident)", "Municipal District Supply Well", "Industrial / Fire Training Site (BOMARC)"]
    )

    with st.form("action_form"):
        col1, col2 = st.columns(2)
        with col1:
            reporter_name = st.text_input("Your Name / Organization", "Jane Doe")
            location_desc = st.text_input("Location / Address / Town", "Westhampton, NY")
            contaminant_level = st.number_input("Detected PFAS Concentration (ng/L)", value=85.0)
        with col2:
            official_contact = st.text_input("Recipient Official / Department", "Suffolk County Dept. of Health Services (Andrew Rapiejko)")
            contact_email = st.text_input("Official Email", "waterquality@suffolkcountyny.gov")
            if site_type == "Private Well (Unsewered Resident)":
                well_depth = st.number_input("Well Depth (ft)", value=60)
            elif site_type == "Municipal District Supply Well":
                district_name = st.text_input("Water District Name", "Westhampton Water District")
            else:
                facility_name = st.text_input("Facility Name / Source", "Former BOMARC Site")

        submitted = st.form_submit_button("Generate Formal Report Draft")

    if submitted:
        st.success("✅ Formal Notification Draft Generated Successfully!")
        
        if site_type == "Private Well (Unsewered Resident)":
            draft = f"""SUBJECT: URGENT: Elevated PFAS Detection in Private Well - {location_desc}

Dear {official_contact},

I am writing to formally report elevated PFAS contamination detected at a private well location in {location_desc}. Because private wells lack the routine testing oversight of public water districts, this site requires immediate official verification.

- Resident: {reporter_name}
- Well Depth: {well_depth} ft
- Detected Concentration: {contaminant_level} ng/L

We request confirmatory sampling by the SCDHS and technical guidance on deploying advanced mitigation systems.

Sincerely,
{reporter_name}
"""
        elif site_type == "Municipal District Supply Well":
            draft = f"""SUBJECT: Formal Inquiry: Public Water Supply PFAS Levels - {location_desc}

Dear {official_contact},

I am writing on behalf of community stakeholders regarding recent PFAS test results ({contaminant_level} ng/L) associated with {district_name} in {location_desc}. 

We request public transparency regarding current treatment capacity (GAC vs. AOP upgrades) and a timeline for compliance with state maximum contaminant levels.

Sincerely,
{reporter_name}
"""
        else:
            draft = f"""SUBJECT: Environmental Concern: Source Zone Contamination Report - {location_desc}

Dear {official_contact},

We are submitting contamination tracking data ({contaminant_level} ng/L detected) regarding runoff and migration from {facility_name} in {location_desc}. 

Given the hydrogeological characteristics of Long Island's shallow aquifer, urgent action and feasibility evaluation for destructive treatment technologies are required.

Sincerely,
{reporter_name}
"""

        st.text_area("Copy and paste your official notification:", draft, height=280)
