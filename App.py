import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from well_data import BOMARC_WELLS, FIELD_CONDITIONS, TREATMENT_OPTIONS

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & STATE
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LI Water Safety Check",
    page_icon="💧",
    layout="wide"
)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "selected_well" not in st.session_state:
    st.session_state.selected_well = None

# ═══════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════
def show_landing_page():
    st.title("💧 Is Your Water Safe?")
    st.subheader("Real data from Long Island contamination sites")
    
    st.markdown("""
    ### The Problem
    
    **BOMARC Site (Westhampton)** groundwater contamination:
    - **PFOA:** 100 ng/L (10x New York safe limit)
    - **PFOS:** 120 ng/L (12x safe limit)  
    - **Drinking water wellhead:** PFOA at 64 ng/L
    
    Most residents don't know **what treatment works** or **what it costs**.
    This tool shows you both.
    """)
    
    st.markdown("---")
    st.markdown("### Check Your Water")
    
    # Well selection
   # Create label options dynamically from whatever is in BOMARC_WELLS
    well_options = {
        f"{key} ({data['contaminant']} {data['level_ng_l']} ng/L) — {data['location']}": key
        for key, data in BOMARC_WELLS.items()
}

    selected_label = st.radio(
        "Which contaminated well are you checking?",
        options=list(well_options.keys()),
        index=0
)

# Look up the actual well key (e.g., "BOM-25") from the label
    well_name = well_options[selected_label]
    # Big button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 See Solutions", type="primary", use_container_width=True):
            st.session_state.selected_well = well_name
            st.session_state.page = "results"
            st.rerun()
    
    st.markdown("---")
    st.markdown("""
    ### How This Tool Works
    
    1. **Check contamination level** at your well
    2. **Compare treatment options** side-by-side (cost, time, effectiveness)
    3. **See the science** behind each treatment
    4. **Take action** — Email your town official
    
    **Built by a Long Island student** | Data: Suffolk County Dept. of Health Services
    """)

# ═══════════════════════════════════════════════════════════════
# RESULTS PAGE — CONTAMINATION + TREATMENT COMPARISON + SIMULATION
# ═══════════════════════════════════════════════════════════════
def show_results_page():
    well_name = st.session_state.selected_well or "BOM-25"
    well_data = BOMARC_WELLS[well_name]

    # ─────────────────────────────────────────
    # SIDEBAR — INPUTS
    # ─────────────────────────────────────────
    st.sidebar.title("⚙️ Treatment Parameters")
    st.sidebar.markdown("Adjust conditions and click **Run Simulation**.")

    if st.sidebar.button("⬅️ Back to Landing Page"):
        st.session_state.page = "landing"
        st.rerun()

    preset = st.sidebar.selectbox(
        "📍 Load Preset Conditions",
        ["Custom", "Suffolk County Field Conditions", "Lab Optimal (pH 3, 25°C)"]
    )

    if preset == "Suffolk County Field Conditions":
        default_pH, default_temp, default_H2O2, default_Fe, default_PFOA = 5.83, 12.9, 10.0, 100.0, 7.75
    elif preset == "Lab Optimal (pH 3, 25°C)":
        default_pH, default_temp, default_H2O2, default_Fe, default_PFOA = 3.0, 25.0, 10.0, 100.0, 7.75
    else:
        default_pH, default_temp, default_H2O2, default_Fe, default_PFOA = 5.83, 12.9, 10.0, 100.0, float(well_data['level_ng_l'])

    st.sidebar.markdown("### 🌍 Field Conditions")
    user_pH = st.sidebar.slider("pH", 3.0, 8.0, default_pH, 0.1, help="Optimal Photo-Fenton pH is 3–4. Suffolk County avg: 5.83")
    user_temp = st.sidebar.slider("Temperature (°C)", 5.0, 35.0, default_temp, 0.5, help="Suffolk County groundwater avg: 12.9°C")
    user_PFOA = st.sidebar.slider("Initial PFOA (ng/L)", 1.0, 200.0, default_PFOA, 0.5, help="Initial concentration in groundwater")

    st.sidebar.markdown("### 🧪 Reagent Doses")
    user_H2O2 = st.sidebar.slider("H₂O₂ Dose (mM)", 1.0, 100.0, default_H2O2, 1.0, help="Typical municipal range: 5–50 mM")
    user_Fe = st.sidebar.slider("Fe²⁺ Dose (µM)", 1.0, 5000.0, default_Fe, 10.0, help="Optimal Fe varies by H₂O₂ dose and pH")

    st.sidebar.markdown("### 💰 Cost Parameters")
    cost_H2O2_per_mmol = st.sidebar.number_input("H₂O₂ cost ($/mmol)", value=0.014, format="%.4f")
    cost_Fe_per_mmol = st.sidebar.number_input("Fe²⁺ cost ($/mmol)", value=0.0557, format="%.4f")
    cost_UV_per_min = st.sidebar.number_input("UV cost ($/min)", value=0.008, format="%.4f")

    run_button = st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True)

    st.sidebar.markdown("### 🔘 Carbon Treatment Parameters")
    cost_gac_per_kg = st.sidebar.number_input("GAC Media Cost ($/kg)", value=5.50, step=0.50)
    carbon_bed_mass_kg = st.sidebar.slider("Bed Carbon Mass (kg)", 100, 5000, 1000, 100)
    flow_rate_gpm = st.sidebar.slider("Flow Rate (GPM)", 10.0, 500.0, 100.0, 10.0)

    # ─────────────────────────────────────────
    # MAIN PANEL CONTENT
    # ─────────────────────────────────────────
    if st.button("← Back to Check"):
        st.session_state.page = "landing"
        st.rerun()

    st.markdown("---")

    # Contamination Alert
    st.title(f"💧 {well_name}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Contaminant", well_data["contaminant"])
    with col2:
        st.metric("Detected Level", f"{well_data['level_ng_l']} ng/L")
    with col3:
        multiplier = well_data["level_ng_l"] / well_data["safe_limit_ng_l"]
        st.metric("vs. Safe Limit", f"{multiplier:.0f}x over")

    st.error(f"""
    ⚠️ **{well_data['contaminant']} IS {well_data['level_ng_l'] / well_data['safe_limit_ng_l']:.0f}x ABOVE SAFE LIMITS**

    New York drinking water standard: {well_data['safe_limit_ng_l']} ng/L
    Detected: {well_data['level_ng_l']} ng/L
    """)

    st.markdown("---")

    # Treatment Comparison
    st.markdown("### What Can Be Done?")
    st.markdown("Two proven treatment options:")

    treatments = TREATMENT_OPTIONS
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🔘 Carbon Adsorption")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Removal", f"{treatments['carbon']['removal_percent']}%")
            st.metric("Time", f"{treatments['carbon']['contact_time_min']} min")
        with metric_col2:
            st.metric("Cost/Liter", f"${treatments['carbon']['cost_per_liter']:.2f}")
            st.metric("Risk", "Moderate")

        st.markdown("**Pros:**")
        for pro in treatments['carbon']['pros']:
            st.markdown(f"✓ {pro}")

        st.markdown("**Cons:**")
        for con in treatments['carbon']['cons']:
            st.markdown(f"✗ {con}")

    with col_right:
        st.subheader("⚡ Photo-Fenton AOP")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Removal", f"{treatments['photo_fenton']['removal_percent']}%")
            st.metric("Time", f"{treatments['photo_fenton']['contact_time_min']} min")
        with metric_col2:
            st.metric("Cost/Liter", f"${treatments['photo_fenton']['cost_per_liter']:.2f}")
            st.metric("Risk", "Lower")

        st.markdown("**Pros:**")
        for pro in treatments['photo_fenton']['pros']:
            st.markdown(f"✓ {pro}")

        st.markdown("**Cons:**")
        for con in treatments['photo_fenton']['cons']:
            st.markdown(f"✗ {con}")

    st.markdown("---")

    # Side-by-Side Comparison Chart
    st.markdown("### Side-by-Side Comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    treatment_names = ["Carbon", "Photo-Fenton"]
    removals = [75, 90]
    colors_bar = ["#FFA500", "#4CAF50"]

    ax1 = axes[0]
    bars = ax1.bar(treatment_names, removals, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=2)
    ax1.axhline(80, color="red", linestyle="--", linewidth=2, label="80% effectiveness threshold")
    ax1.set_ylabel("Removal %", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax2 = axes[1]
    costs = [0.25, 0.35]
    bars2 = ax2.bar(treatment_names, costs, color=colors_bar, alpha=0.8, edgecolor="black", linewidth=2)
    ax2.set_ylabel("Cost per Liter ($)", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, 0.50)
    ax2.grid(True, alpha=0.3, axis="y")

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'${height:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # ODE SIMULATION SECTION
    st.markdown("### 🔬 Photo-Fenton Reaction Kinetics Simulation")

    if not run_button:
        st.info("👈 Adjust conditions in the sidebar and click **▶️ Run Simulation** to calculate precise kinetics.")
    else:
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
            dOH = (2 * k_hv_h2o2 * H2O2 + k_f * Fe2 * H2O2 - k_PFOA * OH * PFOA - k_I * OH * I - k_s * OH - k_H2O2_scav * OH * H2O2 - k_Fe * Fe2 * OH)
            dPFOA = -k_PFOA * OH * PFOA
            dI = k_PFOA * OH * PFOA - k_I * OH * I
            dFe2 = (-k_f * Fe2 * H2O2 + k_r * Fe3 * H2O2 - k_Fe * Fe2 * OH + k_hv_Fe * Fe3)
            dFe3 = -dFe2
            return [dH2O2, dOH, dPFOA, dI, dFe2, dFe3]

        def run_simulation(H2O2_init, Fe_total, PFOA_0, k_f, k_r):
            t_eval = np.linspace(0, 7200, 3600)
            y0 = [H2O2_init, 0.0, PFOA_0, 0.0, Fe_total, 0.0]
            return solve_ivp(
                fun=lambda t, y: photo_fenton_odes(t, y, Fe_total, k_f, k_r),
                t_span=(0, 7200), y0=y0, method='BDF',
                t_eval=t_eval, rtol=1e-8, atol=1e-12
    
            )

        def time_to_target(sol, PFOA_0, target=90):
            removal = (1 - np.maximum(sol.y[2], 0) / PFOA_0) * 100
            idx = np.where(removal >= target)[0]
            return sol.t[idx[0]] / 60.0 if len(idx) > 0 else None

        def run_carbon_simulation(C0_ng_l, gac_cost_kg, bed_mass_kg, flow_rate_gpm):
            # Unit conversions
            flow_rate_lpm = flow_rate_gpm * 3.78541
            bed_mass_g = bed_mass_kg * 1000
            C0_mg_l = C0_ng_l * 1e-6
    
            # Adsorption Capacity (Freundlich Isotherm)
            K_F = 25.0
            one_over_n = 0.5
            q0_mg_g = K_F * (max(C0_mg_l, 1e-9) ** one_over_n)
    
            # Thomas Breakthrough Model
            k_Th = 0.015
            days = 180
            t_eval_min = np.linspace(0, days * 24 * 60, 500)
    
            exponent = (k_Th * q0_mg_g * bed_mass_g / flow_rate_lpm) - (k_Th * C0_mg_l * t_eval_min)
            C_t_ratio = 1.0 / (1.0 + np.exp(np.clip(exponent, -50, 50)))
    
            # Find breakthrough point (10% of influent concentration)
            breakthrough_idx = np.where(C_t_ratio >= 0.10)[0]
            days_to_breakthrough = t_eval_min[breakthrough_idx[0]] / (24 * 60) if len(breakthrough_idx) > 0 else days
    
            # Financials & Metrics
            total_liters = (days_to_breakthrough * 24 * 60) * flow_rate_lpm
            cost_per_liter = (bed_mass_kg * gac_cost_kg) / max(total_liters, 1.0)
            avg_removal = (1.0 - np.mean(C_t_ratio[:max(1, len(breakthrough_idx))])) * 100
    
            return {
                "cost_per_liter": cost_per_liter,
                "avg_removal_pct": avg_removal,
                "days_to_breakthrough": days_to_breakthrough
            }

        PFOA_0 = user_PFOA * 1e-9 / MW_PFOA
        H2O2_mol = user_H2O2 * 1e-3
        Fe_mol = user_Fe * 1e-6
        k_f, k_r = build_rate_constants(user_pH, user_temp)

        with st.spinner("Running simulation..."):
            sol = run_simulation(H2O2_mol, Fe_mol, PFOA_0, k_f, k_r)

        t_min = sol.t / 60
        PFOA_t = np.maximum(sol.y[2], 0)
        removal = (1 - PFOA_t / PFOA_0) * 100
        final_rem = removal[-1]
        t90 = time_to_target(sol, PFOA_0, 90)

        PFOA_removed_ug = (PFOA_0 - PFOA_t[-1]) * MW_PFOA * 1e6
        t_for_cost = t90 if t90 else 120.0
        cost_h2o2 = H2O2_mol * 1000 * cost_H2O2_per_mmol
        cost_iron = Fe_mol * 1000 * cost_Fe_per_mmol
        cost_uv = t_for_cost * cost_UV_per_min
        cost_total = cost_h2o2 + cost_iron + cost_uv
        cost_per_ug = cost_total / PFOA_removed_ug if PFOA_removed_ug > 0 else 999

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final PFOA Removal", f"{final_rem:.1f}%", "✅ Above 90%" if final_rem >= 90 else "❌ Below 90%")
        m2.metric("Time to 90% Removal", f"{t90:.1f} min" if t90 else "Not reached", "Within 2 hrs" if t90 else "Increase dose")
        m3.metric("Cost per Liter Treated", f"${cost_total:.4f}")
        m4.metric("Cost per µg PFOA Removed", f"${cost_per_ug:.4f}" if cost_per_ug < 999 else "N/A")

        col_l, col_r = st.columns(2)
        with col_l:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(t_min, removal, color='#2196F3', linewidth=2.5)
            ax1.axhline(90, color='gray', linestyle='--', alpha=0.6, label='90% target')
            if t90:
                ax1.axvline(t90, color='green', linestyle='--', alpha=0.5, label=f't₉₀ = {t90:.1f} min')
            ax1.set_xlabel('Time (min)')
            ax1.set_ylabel('PFOA Removal (%)')
            ax1.set_ylim(0, 105)
            ax1.set_xlim(0, 120)
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

        with col_r:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(t_min, PFOA_t * 1e12, color='#F44336', linewidth=2.5)
            ax2.set_xlabel('Time (min)')
            ax2.set_ylabel('[PFOA] (pmol/L)')
            ax2.set_xlim(0, 120)
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

    st.markdown("---")

    # Action Section
    st.markdown("### What Next?")
    st.info("Your town official needs to know about this contamination.")

    if st.button("📧 Email Your Official", type="primary", use_container_width=True):
        st.session_state.page = "action"
        st.rerun()

# ═══════════════════════════════════════════════════════════════
# ACTION PAGE — EMAIL GENERATOR
# ═══════════════════════════════════════════════════════════════
def show_action_page():
    well_name = st.session_state.selected_well or "BOM-25"
    well_data = BOMARC_WELLS[well_name]

    if st.button("← Back to Results"):
        st.session_state.page = "results"
        st.rerun()

    st.markdown("---")
    st.title("📧 Take Action Now")

    st.markdown(f"""
    Your town official needs to know about **{well_data['contaminant']} contamination at {well_data['level_ng_l']} ng/L**.

    This email takes 30 seconds to send and creates pressure for action.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        town = st.text_input("Your town", "Westhampton", key="town_input")
        official_name = st.text_input("Official name (e.g., 'Town Supervisor')", "Town Board", key="official_name_input")
    with col2:
        official_title = st.text_input("Official title (e.g., 'Supervisor')", "Member", key="official_title_input")
        official_email = st.text_input("Official email", "", key="official_email_input")

    st.markdown("---")
    st.markdown("### Your Email")

    email_subject = f"URGENT: PFAS Contamination at {well_name} — {town}"

    email_body = f"""Subject: {email_subject}

Dear {official_name},

I am writing to bring urgent attention to PFAS contamination detected in {town} groundwater:

**Contamination Details:**
- Well/Location: {well_name}
- Contaminant: {well_data['contaminant']}
- Level Detected: {well_data['level_ng_l']} ng/L
- Safe Limit (NY State): {well_data['safe_limit_ng_l']} ng/L
- OVER LIMIT BY: {well_data['level_ng_l'] / well_data['safe_limit_ng_l']:.0f}x

**Treatment Options Available:**
1. Carbon Adsorption: 75% removal, 20 minutes, $0.25/liter
2. Photo-Fenton AOP: 90% removal, 45 minutes, $0.35/liter

**What I'm Asking:**
1. Acknowledge receipt of this data
2. Publish a public health notice
3. Hold a town meeting within 30 days to discuss remediation timeline and budget
4. Specify which treatment method will be deployed and by when

Residents deserve transparent communication about water safety. I urge you to prioritize this issue.

Respectfully,
[Your Name]
[Your Address]
[Your Phone Number]"""

    st.text_area("Copy this email:", email_body, height=300, disabled=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📋 Copy Email to Clipboard", use_container_width=True):
            st.success("✅ Email copied! Paste into your email client.")
            st.code(email_body, language="text")

    with col2:
        if official_email:
            mailto_link = f"mailto:{official_email}?subject={email_subject}&body={email_body}"
            st.markdown(f"[📧 Open Email Client]({mailto_link})")

    with col3:
        if st.button("🔗 Share with Friends", use_container_width=True):
            st.info("Share this link with neighbors: [app link here]")

    st.markdown("---")

    st.markdown("""
    ### Why This Matters

    Elected officials act when constituents contact them. **One email makes a difference.**

    - You're not asking for a donation
    - You're asking for transparency and action
    - You have real data to back your request

    Send this email today.
    """)

# ═══════════════════════════════════════════════════════════════
# PAGE ROUTER
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "landing":
    show_landing_page()
elif st.session_state.page == "results":
    show_results_page()
elif st.session_state.page == "action":
    show_action_page()

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.divider()
st.caption("""📊 **Data Source:** BOMARC Site Investigation (2020, 2024) | Suffolk County Dept. of Health Services
🔬 **Field Conditions:** USGS Suffolk County Groundwater Survey
⚠️ **Disclaimer:** This tool is for educational feasibility screening only. Treatment decisions require consultation with licensed engineers and state/local regulators.""")
