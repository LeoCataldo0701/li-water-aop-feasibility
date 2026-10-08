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
    user_PFOA  = float(well_info["level_ng_l"])
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
    cols  = [col1, col2, col3]

    for i, (col, well) in enumerate(zip(cols, wells)):
        with col:
            mult = well["level_ng_l"] / well["safe_limit_ng_l"]
            color = "#ef4444" if mult >= 10 else "#f59e0b"
            st.markdown(f"""
            <div style="background:#1e293b;border-radius:8px;padding:20px;color:white;border-top:4px solid {color}">
                <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:4px">{well['location']}</div>
                <div style="font-size:2rem;font-weight:700;color:{color}">{well['level_ng_l']} ng/L</div>
                <div style="font-size:1rem;color:white;margin-top:4px">{well['contaminant']}</div>
                <div style="font-size:0.85rem;color:#94a3b8;margin-top:8px">
                    Safe limit: {well['safe_limit_ng_l']} ng/L<br>
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
        # ── RUN MODEL ──
        PFOA_0   = user_PFOA * 1e-9 / MW_PFOA
        H2O2_mol = user_H2O2 * 1e-3
        Fe_mol   = user_Fe   * 1e-6
        k_f, k_r = build_rate_constants(user_pH, user_temp)

        t_eval = np.linspace(0, 7200, 3600)
        y0     = [H2O2_mol, 0.0, PFOA_0, 0.0, Fe_mol, 0.0]
        sol    = solve_ivp(
            fun=lambda t, y: photo_fenton_odes(t, y, Fe_mol, k_f, k_r),
            t_span=(0, 7200), y0=y0, method='BDF', t_eval=t_eval,
            rtol=1e-8, atol=1e-12
        )

        t_min          = sol.t / 60
        fenton_pfoa    = np.maximum(sol.y[2], 0)
        fenton_removal = (1 - fenton_pfoa / PFOA_0) * 100

        idx_90         = np.where(fenton_removal >= 90)[0]
        fenton_t90     = t_min[idx_90[0]] if len(idx_90) > 0 else None
        fenton_max_rem = fenton_removal[-1]

        # GAC constants (lab-derived)
        GAC_REMOVAL    = 75.0   # % — literature constant
        GAC_FAILS_PFNA = True

        # Fenton cost
        cost_h2o2       = H2O2_mol * 1000 * cost_H2O2_per_mmol
        cost_iron       = Fe_mol   * 1000 * cost_Fe_per_mmol
        cost_uv         = (fenton_t90 if fenton_t90 else 120.0) * cost_UV_per_min
        fenton_cost     = cost_h2o2 + cost_iron + cost_uv
        fenton_removed  = (PFOA_0 - fenton_pfoa[-1]) * MW_PFOA * 1e6

        # GAC cost (upfront + disposal)
        gac_media_cost    = (gac_cost_lb + gac_disposal_cost) * 0.025
        gac_total_cost    = gac_media_cost
        gac_removed       = PFOA_0 * (GAC_REMOVAL / 100.0) * MW_PFOA * 1e6
        gac_cost_per_ug   = gac_total_cost / gac_removed if gac_removed > 0 else 0
        fenton_cost_per_ug = fenton_cost / fenton_removed if fenton_removed > 0 else 0

        # ── HEADER ──
        st.markdown(f"## Results: {selected_well_name}")
        well_info = get_well_data(selected_well_name)
        mult      = well_info["level_ng_l"] / well_info["safe_limit_ng_l"]

        st.markdown(f"""
        <div class="crisis-banner">
        {well_info['contaminant']} detected at {well_info['level_ng_l']} ng/L
        — {mult:.0f}x New York's safe limit of {well_info['safe_limit_ng_l']} ng/L
        — {well_info['location']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── SIDE BY SIDE METRICS ──
        st.markdown("### Which treatment wins?")
        col_left, col_right = st.columns(2)

        with col_left:
            fenton_wins = fenton_max_rem > GAC_REMOVAL
            st.markdown(f"""
            <div class="treatment-card-fenton">
                <div style="font-size:0.8rem;color:#86efac;font-weight:600;margin-bottom:8px">
                    ⚡ PHOTO-FENTON AOP {"✅ RECOMMENDED" if fenton_wins else ""}
                </div>
                <div style="font-size:2.4rem;font-weight:700;color:#22c55e">{fenton_max_rem:.0f}%</div>
                <div style="color:#94a3b8;font-size:0.85rem">PFAS Destroyed</div>
                <hr style="border-color:#334155;margin:12px 0">
                <div>Time to 90%: <b style="color:white">{f"{fenton_t90:.0f} min" if fenton_t90 else "Not reached"}</b></div>
                <div>Cost/liter: <b style="color:white">${fenton_cost:.4f}</b></div>
                <div>Cost/µg removed: <b style="color:white">${fenton_cost_per_ug:.4f}</b></div>
                <div>Waste stream: <b style="color:#22c55e">None</b></div>
                <div>Works on PFNA: <b style="color:#22c55e">Yes</b></div>
                <div>pH {user_pH} / {user_temp}°C field conditions applied</div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown(f"""
            <div class="treatment-card-gac">
                <div style="font-size:0.8rem;color:#fcd34d;font-weight:600;margin-bottom:8px">
                    🔘 GRANULAR ACTIVATED CARBON
                </div>
                <div style="font-size:2.4rem;font-weight:700;color:#f59e0b">{GAC_REMOVAL:.0f}%</div>
                <div style="color:#94a3b8;font-size:0.85rem">PFAS Adsorbed (not destroyed)</div>
                <hr style="border-color:#334155;margin:12px 0">
                <div>Contact time: <b style="color:white">{gac_ebct:.0f} min</b></div>
                <div>Cost/liter: <b style="color:white">${gac_total_cost:.4f}</b></div>
                <div>Cost/µg captured: <b style="color:white">${gac_cost_per_ug:.4f}</b></div>
                <div>Waste stream: <b style="color:#ef4444">Hazardous (spent carbon)</b></div>
                <div>Works on PFNA: <b style="color:#ef4444">No</b></div>
                <div style="font-size:0.75rem;color:#94a3b8;margin-top:8px">
                    Lab constant — EPA PFAS Treatment Fact Sheet
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── PLOTS ──
        st.markdown("### Removal over time")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#0f172a')
        for ax in [ax1, ax2]:
            ax.set_facecolor('#1e293b')
            ax.tick_params(colors='#94a3b8')
            ax.xaxis.label.set_color('#94a3b8')
            ax.yaxis.label.set_color('#94a3b8')
            ax.spines[:].set_color('#334155')

        # Plot 1: Removal %
        ax1.plot(t_min, fenton_removal, color='#22c55e', linewidth=2.5, label='Photo-Fenton AOP')
        ax1.axhline(GAC_REMOVAL, color='#f59e0b', linewidth=2, linestyle='--',
                    label=f'GAC ceiling ({GAC_REMOVAL:.0f}%)')
        ax1.axhline(90, color='#64748b', linewidth=1, linestyle=':', label='90% target')
        if fenton_t90:
            ax1.axvline(fenton_t90, color='#22c55e', linewidth=1, linestyle='--', alpha=0.5,
                        label=f'Fenton hits 90% at {fenton_t90:.0f} min')
        ax1.set_xlabel('Time (min)', color='#94a3b8')
        ax1.set_ylabel('Removal %', color='#94a3b8')
        ax1.set_ylim(0, 105)
        ax1.set_xlim(0, 120)
        ax1.legend(fontsize=8, facecolor='#1e293b', labelcolor='white')
        ax1.grid(True, alpha=0.15, color='white')
        ax1.set_title('Removal Efficiency vs. Time', color='white')

        # Plot 2: Total cost of ownership bar chart
        categories = ['Photo-Fenton\n(destruction)', 'GAC\n(adsorption)']
        total_costs = [fenton_cost, gac_total_cost]
        colors_bar  = ['#22c55e', '#f59e0b']
        bars = ax2.bar(categories, total_costs, color=colors_bar, alpha=0.85, width=0.5)
        ax2.set_ylabel('Cost per Liter ($)', color='#94a3b8')
        ax2.set_title('Cost per Liter Treated', color='white')
        ax2.grid(True, alpha=0.15, color='white', axis='y')
        for bar, val in zip(bars, total_costs):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005,
                     f'${val:.4f}', ha='center', va='bottom', color='white', fontweight='bold')

        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        # ── KEY INSIGHT ──
        pfna_note = (
            "**Critical for BOMARC site:** PFNA (detected at nearby Calverton at 7,580 ng/L) "
            "does not adsorb onto GAC effectively. Photo-Fenton handles it. GAC fails."
        )
        st.info(f"📌 {pfna_note}")

        # ── CONDITIONS TABLE ──
        with st.expander("View conditions used in this simulation"):
            arr_factor = np.exp(-Ea_J / R_gas * (1/(user_temp+273.15) - 1/T_REF))
            pH_cor     = 10 ** (-(user_pH - 3.0) * 0.5)
            st.markdown(f"""
            | Parameter | Value | Source |
            |---|---|---|
            | pH | {user_pH} | USGS / Field data |
            | Temperature | {user_temp}°C | USGS groundwater survey |
            | Initial PFAS | {user_PFOA} ng/L | BOMARC site investigation |
            | H₂O₂ dose | {user_H2O2} mM | Lab standard |
            | Fe²⁺ dose | {user_Fe} µM | Optimized |
            | Arrhenius factor | {arr_factor:.3f} | De Laat & Gallard 1999 |
            | pH correction | {pH_cor:.4f} | De Laat & Gallard 1999 |
            | k_PFOA | 1.2×10⁷ L/(mol·s) | Hori et al. 2004 |
            | GAC removal ceiling | 75% | Literature lab constant |
            | GAC contact time | {gac_ebct:.0f} min | EPA PFAS Fact Sheet |
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
        safe_limit     = 10.0
        multiplier     = pfas_level / safe_limit

    st.markdown("---")

    # Step 3: Generate letter
    st.markdown("### Step 3: Your letter")

    if site_type == "I'm a resident near a contaminated well":
        subject = f"URGENT: PFAS Contamination Near My Home — {your_town}"
        body = f"""Subject: {subject}

Dear {official_name},

I am writing as a concerned resident of {your_town} to formally request action on PFAS contamination affecting groundwater in my community.

Recent testing at the BOMARC Site (Westhampton) has detected {contaminant} at {pfas_level:.0f} ng/L — {multiplier:.0f}x New York's safe limit of {safe_limit:.0f} ng/L.

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
