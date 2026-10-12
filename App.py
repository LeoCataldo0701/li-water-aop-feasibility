import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from well_data import BOMARC_WELLS, FIELD_CONDITIONS, get_well_data

st.set_page_config(
    page_title="LI Water Safety — PFAS Action Tool",
    page_icon="💧",
    layout="wide"
)

# ─────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
.crisis-banner {
    background: #b91c1c;
    color: white;
    padding: 18px 24px;
    border-radius: 8px;
    margin-bottom: 16px;
    font-size: 1.1rem;
    font-weight: 600;
}
.well-card {
    background: #1e293b;
    color: white;
    border-radius: 8px;
    padding: 20px;
    margin: 8px 0;
}
.treatment-card-fenton {
    background: #0f172a;
    border-left: 4px solid #22c55e;
    border-radius: 6px;
    padding: 18px;
    color: white;
}
.treatment-card-gac {
    background: #0f172a;
    border-left: 4px solid #f59e0b;
    border-radius: 6px;
    padding: 18px;
    color: white;
}
.stat-number {
    font-size: 2.2rem;
    font-weight: 700;
    color: #ef4444;
}
.stat-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Flag_of_Suffolk_County%2C_New_York.svg/200px-Flag_of_Suffolk_County%2C_New_York.svg.png", width=80)
st.sidebar.title("LI PFAS Tool")
st.sidebar.markdown("*Long Island Water Safety*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Crisis Overview", "📊 Compare Treatments", "📧 Take Action"],
    label_visibility="collapsed"
)

# Default values used across pages
default_pH   = FIELD_CONDITIONS["pH"]
default_temp = FIELD_CONDITIONS["temp_C"]
default_PFOA = 100.0
default_H2O2 = 10.0
default_Fe   = 100.0
cost_H2O2_per_mmol = 0.014
cost_Fe_per_mmol   = 0.0557
cost_UV_per_min    = 0.008
gac_ebct           = 15.0
gac_cost_lb        = 2.25
gac_disposal_cost  = 1.50
run_sim_button     = False
user_pH            = default_pH
user_temp          = default_temp
user_PFOA          = default_PFOA
user_H2O2          = default_H2O2
user_Fe            = default_Fe

if page == "📊 Compare Treatments":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Well Site**")
    selected_well_name = st.sidebar.selectbox(
        "Choose contaminated well",
        list(BOMARC_WELLS.keys()),
        label_visibility="collapsed"
    )
    well_info  = get_well_data(selected_well_name)
    user_PFOA = float(well_info["compounds"]["PFOA"]["level_ng_l"])
    user_pH    = FIELD_CONDITIONS["pH"]
    user_temp  = FIELD_CONDITIONS["temp_C"]

    st.sidebar.markdown("**Adjust Conditions**")
    user_PFOA = st.sidebar.slider(
        "Initial PFAS (ng/L)", 1.0, 150.0, user_PFOA, 0.5,
        help="Pre-loaded from BOMARC field data"
    )
    user_pH   = st.sidebar.slider("pH", 3.0, 8.0, user_pH, 0.1)
    user_temp = st.sidebar.slider("Temp (°C)", 5.0, 35.0, user_temp, 0.5)

    with st.sidebar.expander("Advanced: Photo-Fenton"):
        user_H2O2          = st.slider("H₂O₂ Dose (mM)", 1.0, 100.0, 10.0, 1.0)
        user_Fe            = st.slider("Fe²⁺ Dose (µM)", 1.0, 5000.0, 100.0, 10.0)
        cost_H2O2_per_mmol = st.number_input("H₂O₂ cost ($/mmol)", value=0.014, format="%.4f")
        cost_Fe_per_mmol   = st.number_input("Fe²⁺ cost ($/mmol)", value=0.0557, format="%.4f")
        cost_UV_per_min    = st.number_input("UV cost ($/min)", value=0.008, format="%.4f")

    with st.sidebar.expander("Advanced: GAC"):
        gac_ebct          = st.slider("Contact Time (min)", 5.0, 30.0, 15.0, 1.0)
        gac_cost_lb       = st.number_input("GAC Media ($/lb)", value=2.25)
        gac_disposal_cost = st.number_input("Disposal ($/lb)", value=1.50)

    run_sim_button = st.sidebar.button(
        "▶ Run Comparison", type="primary", use_container_width=True
    )

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: BOMARC Site Investigations (2020, 2024)\n"
    "Suffolk County Dept. of Health Services\n"
    "USGS Finkelstein et al. 2025"
)

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

def simulate_all_contaminants(well_info, pH, temp, H2O2, Fe, costs):
    """Simulate Fenton + GAC for all contaminants at a well"""
    results = {}
    
    for contaminant, data in well_info["compounds"].items():
        level_ng_l = data["level_ng_l"]
        gac_removal_pct = data["gac_removal"] * 100
        
        # Run Fenton ODE (same for all — hydroxyl radicals don't distinguish)
        PFOA_0 = level_ng_l * 1e-9 / MW_PFOA  # Convert to mol/L
        H2O2_mol = H2O2 * 1e-3
        Fe_mol = Fe * 1e-6
        k_f, k_r = build_rate_constants(pH, temp)
        
        t_eval = np.linspace(0, 7200, 3600)
        y0 = [H2O2_mol, 0.0, PFOA_0, 0.0, Fe_mol, 0.0]
        sol = solve_ivp(
            fun=lambda t, y: photo_fenton_odes(t, y, Fe_mol, k_f, k_r),
            t_span=(0, 7200), y0=y0, method='BDF', t_eval=t_eval,
            rtol=1e-8, atol=1e-12
        )
        
        t_min = sol.t / 60
        fenton_pfoa = np.maximum(sol.y[2], 0)
        fenton_removal = (1 - fenton_pfoa / PFOA_0) * 100
        
        idx_90 = np.where(fenton_removal >= 90)[0]
        t90 = t_min[idx_90[0]] if len(idx_90) > 0 else None
        
        # Fenton cost (same for all)
        cost_h2o2 = H2O2_mol * 1000 * costs["H2O2"]
        cost_iron = Fe_mol * 1000 * costs["Fe"]
        cost_uv = (t90 if t90 else 120.0) * costs["UV"]
        fenton_cost = cost_h2o2 + cost_iron + cost_uv
        
        # GAC cost (different removal per compound)
        gac_cost = (costs["gac_lb"] + costs["gac_disposal"]) * 0.025
        
        results[contaminant] = {
            "level_ng_l": level_ng_l,
            "fenton_removal": fenton_removal[-1],
            "fenton_t90": t90,
            "fenton_cost": fenton_cost,
            "gac_removal": gac_removal_pct,
            "gac_cost": gac_cost,
            "gac_fails": gac_removal_pct < 50,
            "t_min": t_min,
            "fenton_removal_curve": fenton_removal,
        }
    
    return results

# ─────────────────────────────────────────
# PAGE ROUTING
# ─────────────────────────────────────────

if page == "🏠 Crisis Overview":

    st.markdown("""
    <div class="crisis-banner">
    ⚠️ PFAS contamination at BOMARC site (Westhampton) exceeds New York safe limits at every monitored well.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## What's in Long Island's groundwater?")
    st.markdown(
        "PFAS chemicals — used in firefighting foam at military installations — "
        "have been detected in Suffolk County groundwater. "
        "These compounds do not break down naturally and accumulate in the body over time."
    )

    # Well data cards
    col1, col2, col3 = st.columns(3)
    wells = list(BOMARC_WELLS.values())
    cols = [col1, col2, col3]

    for i, (col, well) in enumerate(zip(cols, wells)):
        with col:
            # Find the compound with the highest level to feature on the card
            max_comp_name, max_comp_data = max(well["compounds"].items(), key=lambda x: x[1]["level_ng_l"])
            level = max_comp_data["level_ng_l"]
            safe_limit = well["safe_limit_ng_l"]
        
            mult = level / safe_limit
            color = "#ef4444" if mult >= 10 else "#f59e0b"
        
            st.markdown(f"""
            <div style="background:#1e293b;border-radius:8px;padding:20px;color:white;border-top:4px solid {color}">
                <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:4px">{well['location']}</div>
                <div style="font-size:2rem;font-weight:700;color:{color}">{level} ng/L</div>
                <div style="font-size:1rem;color:white;margin-top:4px">{max_comp_name}</div>
                <div style="font-size:0.85rem;color:#94a3b8;margin-top:8px"> 
                    Safe limit: {safe_limit_ng_l} ng/L<br>
                    <b style="color:{color}">{mult:.0f}x over</b> 
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Why does this matter?")
        st.markdown("""
        - **PFAS don't break down.** They persist in soil and water indefinitely.
        - **They accumulate.** Even low levels build up in the body over years.
        - **Long Island's shallow aquifer** is particularly vulnerable — sandy soil means contaminants travel fast.
        - **Current treatment fails.** Standard carbon filters are ineffective against short-chain PFAS like PFNA.
        """)

    with col_right:
        st.markdown("### New York Safe Limits")
        st.markdown("""
        | Compound | Limit |
        |---|---|
        | PFOA | 10 ng/L |
        | PFOS | 10 ng/L |
        | PFNA | 10 ng/L |
        """)
        st.caption("NYS MCL, effective 2023")

    st.markdown("---")
    st.markdown("### What this tool does")

    a, b, c = st.columns(3)
    with a:
        st.markdown("**1. Compare treatments**")
        st.markdown("See Photo-Fenton AOP vs. carbon filtration side-by-side — cost, speed, effectiveness.")
    with b:
        st.markdown("**2. Use real field data**")
        st.markdown("All contamination levels come from BOMARC site investigations by Suffolk County DOHS.")
    with c:
        st.markdown("**3. Take action**")
        st.markdown("Generate a formal letter to your town official or water district in 30 seconds.")

    st.markdown("---")
    st.caption(
        "Data sources: BOMARC Groundwater & Soil Investigation (Sept 2020), "
        "Suffolk County Dept. of Health Services | "
        "Groundwater field conditions: USGS Finkelstein et al. 2025"
    )

elif page == "📊 Compare Treatments":

    if not run_sim_button:
        st.markdown("## Compare Treatment Options")
        st.info("Select a contaminated well in the sidebar and click **▶ Run Comparison**.")

        st.markdown("### How the two treatments work")
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("""
            **🟢 Photo-Fenton Advanced Oxidation**

            Generates hydroxyl radicals (·OH) using iron + hydrogen peroxide + UV light.
            These radicals chemically **destroy** PFAS molecules — breaking carbon-fluorine bonds
            until only harmless end-products remain.

            - Works on all PFAS variants including short-chain PFNA
            - No hazardous waste stream
            - Higher upfront cost
            """)

        with col_right:
            st.markdown("""
            **🟡 Granular Activated Carbon (GAC)**

            Passes water through porous carbon media.
            PFAS molecules **adsorb** (stick) to the surface — but are not destroyed.
            Spent carbon becomes **hazardous waste** requiring regulated disposal.

            - Proven, widely deployed
            - Lower upfront cost
            - Fails on short-chain PFAS (PFNA, PFBA)
            - Disposal cost adds up over time
            """)

    else:
        # ── RUN MULTI-COMPOUND SIMULATION ──
        well_info = get_well_data(selected_well_name)
        
        costs = {
            "H2O2": cost_H2O2_per_mmol,
            "Fe": cost_Fe_per_mmol,
            "UV": cost_UV_per_min,
            "gac_lb": gac_cost_lb,
            "gac_disposal": gac_disposal_cost,
        }
        
        results = simulate_all_contaminants(well_info, user_pH, user_temp, user_H2O2, user_Fe, costs)
        
        st.markdown(f"## Results: {selected_well_name}")
        st.markdown("### All Contaminants at This Well")
        
        # Show results for each compound
        num_compounds = len(results)
        cols = st.columns(num_compounds)
        
        for col, (contaminant, data) in zip(cols, results.items()):
            with col:
                mult = data["level_ng_l"] / well_info["safe_limit_ng_l"]
                gac_fails = data["gac_fails"]
                
                st.markdown(f"""
                <div style="background:#1e293b;border-radius:8px;padding:16px;color:white">
                    <div style="font-size:1.2rem;font-weight:700">{contaminant}</div>
                    <div style="color:#94a3b8;font-size:0.85rem">{data['level_ng_l']} ng/L ({mult:.0f}x)</div>
                    <hr style="border-color:#334155;margin:12px 0">
                    <div style="font-size:0.9rem">
                        <b>Photo-Fenton:</b> {data['fenton_removal']:.0f}%<br>
                        <b>GAC:</b> {data['gac_removal']:.0f}% {'❌ FAILS' if gac_fails else '✅'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Side-by-side comparison chart
        compounds_list = list(results.keys())
        fenton_removals = [results[c]["fenton_removal"] for c in compounds_list]
        gac_removals = [results[c]["gac_removal"] for c in compounds_list]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        
        x = np.arange(len(compounds_list))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, fenton_removals, width, label='Photo-Fenton', color='#22c55e', alpha=0.85)
        bars2 = ax.bar(x + width/2, gac_removals, width, label='GAC', color='#f59e0b', alpha=0.85)
        
        ax.set_ylabel('Removal %', color='#94a3b8')
        ax.set_title('Removal Efficiency by Contaminant', color='white')
        ax.set_xticks(x)
        ax.set_xticklabels(compounds_list, color='#94a3b8')
        ax.set_ylim(0, 105)
        ax.legend(facecolor='#1e293b', labelcolor='white')
        ax.tick_params(colors='#94a3b8')
        ax.grid(True, alpha=0.15, axis='y', color='white')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}%',
                       ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Big callout for PFNA
        pfna_data = results.get("PFNA")
        if pfna_data and pfna_data["gac_removal"] < 50:
            st.error(f"""
            ⚠️ **CRITICAL: GAC FAILS ON SHORT-CHAIN PFAS**
            
            At {selected_well_name}, GAC can only remove **{pfna_data['gac_removal']:.0f}% of PFNA**.
            
            **Photo-Fenton destroys {pfna_data['fenton_removal']:.0f}% of PFNA** and all other PFAS compounds.
            
            For mixed-PFAS contamination like BOMARC, Photo-Fenton is the only proven solution.
            """)
        
        # Show which compounds fail GAC
        st.markdown("### Treatment Effectiveness Summary")
        
        summary_data = []
        for compound, data in results.items():
            summary_data.append({
                "Compound": compound,
                "Level (ng/L)": f"{data['level_ng_l']:.1f}",
                "Fenton Removal": f"{data['fenton_removal']:.0f}%",
                "GAC Removal": f"{data['gac_removal']:.0f}%",
                "GAC Status": "❌ FAILS" if data["gac_fails"] else "✅"
            })
        
        summary_df = st.dataframe(summary_data, use_container_width=True)
        
        st.markdown("---")
        
        # Cost breakdown
        st.markdown("### Cost Comparison")
        
        fenton_avg_cost = np.mean([results[c]["fenton_cost"] for c in compounds_list])
        gac_avg_cost = np.mean([results[c]["gac_cost"] for c in compounds_list])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Photo-Fenton (avg)", f"${fenton_avg_cost:.4f}/L")
        with col2:
            st.metric("GAC (avg)", f"${gac_avg_cost:.4f}/L")
        
        st.markdown("---")
        
        # Conditions table
        with st.expander("View conditions used in this simulation"):
            arr_factor = np.exp(-Ea_J / R_gas * (1/(user_temp+273.15) - 1/T_REF))
            pH_cor     = 10 ** (-(user_pH - 3.0) * 0.5)
            st.markdown(f"""
            | Parameter | Value | Source |
            |---|---|---|
            | pH | {user_pH} | Field/USGS data |
            | Temperature | {user_temp}°C | USGS groundwater survey |
            | H₂O₂ dose | {user_H2O2} mM | Lab standard |
            | Fe²⁺ dose | {user_Fe} µM | Optimized |
            | Arrhenius factor | {arr_factor:.3f} | De Laat & Gallard 1999 |
            | pH correction | {pH_cor:.4f} | De Laat & Gallard 1999 |
            | PFOA k-rate | 1.2×10⁷ L/(mol·s) | Hori et al. 2004 |
            | Photo-Fenton removal | 90% (all PFAS) | Model — hydroxyl mechanism |
            | GAC removal (PFOA/PFOS) | 75%/70% | EPA PFAS Fact Sheet 2020 |
            | GAC removal (PFNA) | 22% | Long et al. 2023 — short-chain limitation |
            """)

elif page == "📧 Take Action":

    st.markdown("## Tell your officials — in 30 seconds")
    st.markdown(
        "Elected officials respond to constituent pressure. "
        "This page generates a formal letter you can send today."
    )

    st.markdown("---")

    # Step 1: Who are you?
    st.markdown("### Step 1: Your situation")
    site_type = st.radio(
        "What best describes you?",
        [
            "I'm a resident near a contaminated well",
            "I want to contact my water district",
            "I want to report contamination near an industrial site"
        ],
        horizontal=True
    )

    st.markdown("---")

    # Step 2: Fill in details
    st.markdown("### Step 2: Fill in details")
    col1, col2 = st.columns(2)

    with col1:
        your_name   = st.text_input("Your name", "Your Name")
        your_town   = st.text_input("Your town", "Westhampton")
        pfas_level  = st.number_input("PFAS level detected (ng/L)", value=100.0, step=1.0,
                                       help="Pre-filled from BOMARC data. Adjust if needed.")
        contaminant = st.selectbox("Contaminant", ["PFOA", "PFOS", "PFNA", "General PFAS"])

    with col2:
        official_name  = st.text_input("Official name / department",
                                        "Suffolk County Dept. of Health Services")
        official_email = st.text_input("Official email",
                                        "waterquality@suffolkcountyny.gov")
        safe_limit_ng_l     = 10.0
        multiplier     = pfas_level / safe_limit_ng_l

    st.markdown("---")

    # Step 3: Generate letter
    st.markdown("### Step 3: Your letter")

    if site_type == "I'm a resident near a contaminated well":
        subject = f"URGENT: PFAS Contamination Near My Home — {your_town}"
        body = f"""Subject: {subject}

Dear {official_name},

I am writing as a concerned resident of {your_town} to formally request action on PFAS contamination affecting groundwater in my community.

Recent testing at the BOMARC Site (Westhampton) has detected {contaminant} at {pfas_level:.0f} ng/L — {multiplier:.0f}x New York's safe limit of {safe_limit_ng_l:.0f} ng/L.

As a resident who relies on this water supply, I am requesting:

1. Immediate public notification of affected residents
2. Free confirmatory water testing for properties within 1 mile of affected wells
3. A clear timeline for remediation
4. A public meeting within 30 days to present findings

Two treatment options are available:
- Photo-Fenton Advanced Oxidation: 90% removal, destroys PFAS completely, no hazardous waste
- Granular Activated Carbon: 75% removal, but creates hazardous spent carbon and fails on PFNA

I urge the county to evaluate Photo-Fenton AOP for sites with mixed PFAS contamination.

Sincerely,
{your_name}
{your_town}, NY"""

    elif site_type == "I want to contact my water district":
        subject = f"Request: PFAS Treatment Plan Disclosure — {your_town} Water District"
        body = f"""Subject: {subject}

Dear {official_name},

I am writing to request transparency regarding the current treatment capacity of our water supply in response to {contaminant} contamination detected at {pfas_level:.0f} ng/L ({multiplier:.0f}x the NYS safe limit).

Specifically, I am requesting:

1. Confirmation of which treatment technology is currently deployed (GAC vs. AOP)
2. Current removal efficiency data for PFOA, PFOS, and PFNA
3. A timeline for upgrading to advanced oxidation if GAC is the current method
4. Cost and funding plan for treatment upgrades

Our community deserves to know its water is safe and that treatment is keeping pace with updated NYS standards.

Sincerely,
{your_name}
{your_town}, NY"""

    else:
        subject = f"Environmental Report: PFAS Migration from Industrial Site — {your_town}"
        body = f"""Subject: {subject}

Dear {official_name},

I am writing to report evidence of PFAS migration from an industrial source site in {your_town}.

{contaminant} has been detected at {pfas_level:.0f} ng/L — {multiplier:.0f}x New York's safe limit. Given Long Island's sandy aquifer and shallow water table, contaminant migration is rapid.

I request:

1. Source investigation to confirm migration pathway
2. Groundwater monitoring well installation
3. Immediate assessment of Photo-Fenton AOP feasibility for source-zone treatment
4. Notification of downstream well owners

This contamination poses an active public health risk requiring urgent response.

Sincerely,
{your_name}
{your_town}, NY"""

    st.text_area("Your letter (copy and paste):", body, height=320, disabled=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📋 Copy Letter", type="primary", use_container_width=True):
            st.success("✅ Copied! Paste into your email client.")
            st.code(body, language="text")
    with col_b:
        if official_email:
            st.markdown(
                f"[📧 Open in Email Client](mailto:{official_email}"
                f"?subject={subject})",
                unsafe_allow_html=False
            )

    st.markdown("---")
    st.markdown("### Other ways to act")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**📞 Call**")
        st.markdown("Suffolk County DoHS: (631) 853-3000")
    with col2:
        st.markdown("**🌐 Report online**")
        st.markdown("[NYSDEC Environmental Complaint](https://www.dec.ny.gov/chemical/8428.html)")
    with col3:
        st.markdown("**📢 Share**")
        st.markdown("Share this tool with neighbors so they can send letters too.")

    st.markdown("---")
    st.caption(
        "PFAS data: BOMARC Site Investigation 2020-2024, Suffolk County Dept. of Health Services | "
        "Treatment data: EPA PFAS Treatment Technology Fact Sheets"
    )
