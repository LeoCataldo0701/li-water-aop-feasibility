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
# TOWN → OFFICIAL MAPPING
# ─────────────────────────────────────────
TOWN_OFFICIALS = {
    "Westhampton": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "Westhampton Beach": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "Calverton": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "Manorville": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "Riverhead": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "Shirley": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "Mastic": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "Center Moriches": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "East Moriches": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "Quogue": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "Hampton Bays": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
    "Floral Park": {"name": "Nassau County Dept. of Health — Environmental Health", "email": "envhealth@nassaucountyny.gov", "phone": "(516) 227-9717", "county": "Nassau"},
    "Garden City": {"name": "Nassau County Dept. of Health — Environmental Health", "email": "envhealth@nassaucountyny.gov", "phone": "(516) 227-9717", "county": "Nassau"},
    "Hempstead": {"name": "Nassau County Dept. of Health — Environmental Health", "email": "envhealth@nassaucountyny.gov", "phone": "(516) 227-9717", "county": "Nassau"},
    "Mineola": {"name": "Nassau County Dept. of Health — Environmental Health", "email": "envhealth@nassaucountyny.gov", "phone": "(516) 227-9717", "county": "Nassau"},
    "Hicksville": {"name": "Nassau County Dept. of Health — Environmental Health", "email": "envhealth@nassaucountyny.gov", "phone": "(516) 227-9717", "county": "Nassau"},
    "Massapequa": {"name": "Nassau County Dept. of Health — Environmental Health", "email": "envhealth@nassaucountyny.gov", "phone": "(516) 227-9717", "county": "Nassau"},
    "Bethpage": {"name": "Nassau County Dept. of Health — Environmental Health", "email": "envhealth@nassaucountyny.gov", "phone": "(516) 227-9717", "county": "Nassau"},
    "Syosset": {"name": "Nassau County Dept. of Health — Environmental Health", "email": "envhealth@nassaucountyny.gov", "phone": "(516) 227-9717", "county": "Nassau"},
    "Plainview": {"name": "Nassau County Dept. of Health — Environmental Health", "email": "envhealth@nassaucountyny.gov", "phone": "(516) 227-9717", "county": "Nassau"},
    "Other / Not listed": {"name": "Suffolk County Dept. of Health Services — Water Resources", "email": "waterquality@suffolkcountyny.gov", "phone": "(631) 853-3000", "county": "Suffolk"},
}
 
# ─────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none; }
 
.crisis-banner {
    background: #b91c1c; color: white; padding: 18px 24px;
    border-radius: 8px; margin-bottom: 16px; font-size: 1.1rem; font-weight: 600;
}
.treatment-card-fenton {
    background: #0f172a; border-left: 4px solid #22c55e;
    border-radius: 6px; padding: 18px; color: white;
}
.treatment-card-gac {
    background: #0f172a; border-left: 4px solid #f59e0b;
    border-radius: 6px; padding: 18px; color: white;
}
 
/* Action cards on home page */
.action-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 24px 20px;
    color: white;
    text-align: center;
    cursor: pointer;
    border: 1px solid #334155;
    transition: border-color 0.2s;
    height: 140px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.action-card:hover { border-color: #22c55e; }
.action-card .icon { font-size: 2rem; margin-bottom: 8px; }
.action-card .label { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; }
 
/* Issue page section cards */
.issue-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 20px;
    color: white;
    margin-bottom: 16px;
}
.issue-stat {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ef4444;
    line-height: 1;
}
.issue-stat-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
 
def go(p):
    st.session_state.page = p
    st.rerun()
 
# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9b/Flag_of_Suffolk_County%2C_New_York.svg/200px-Flag_of_Suffolk_County%2C_New_York.svg.png",
    width=60
)
st.sidebar.markdown("### 💧 LI PFAS Tool")
st.sidebar.markdown("*Long Island Water Safety*")
st.sidebar.markdown("---")
 
# Always-present home button
if st.sidebar.button("🏠 Home", use_container_width=True):
    go("🏠 Home")
 
st.sidebar.markdown("**EXPLORE**")
NAV = {
    "📋 The Issue": "📋 The Issue",
    "🧪 Testing Kits": "🧪 Testing Kits",
    "📧 Report Contamination": "📧 Report Contamination",
    "📊 Compare Treatments": "📊 Compare Treatments",
    "❓ FAQs": "❓ FAQs",
}
for label, target in NAV.items():
    if st.sidebar.button(label, use_container_width=True, key=f"nav_{target}"):
        go(target)
 
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: BOMARC Site Investigations (2020, 2024)\n"
    "Suffolk County Dept. of Health Services\n"
    "USGS Finkelstein et al. 2025"
)
 
page = st.session_state.page
 
# ─────────────────────────────────────────
# COMPARE TREATMENTS — SIDEBAR INPUTS
# ─────────────────────────────────────────
default_pH   = FIELD_CONDITIONS["pH"]
default_temp = FIELD_CONDITIONS["temp_C"]
cost_H2O2_per_mmol = 0.014
cost_Fe_per_mmol   = 0.0557
cost_UV_per_min    = 0.008
gac_ebct           = 15.0
gac_cost_lb        = 2.25
gac_disposal_cost  = 1.50
run_sim_button     = False
user_pH            = default_pH
user_temp          = default_temp
user_H2O2          = 10.0
user_Fe            = 5.0
selected_well_name = list(BOMARC_WELLS.keys())[0]
 
if page == "📊 Compare Treatments":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Well Site**")
    selected_well_name = st.sidebar.selectbox(
        "Choose contaminated well", list(BOMARC_WELLS.keys()), label_visibility="collapsed"
    )
    well_info = get_well_data(selected_well_name)
    user_pH   = FIELD_CONDITIONS["pH"]
    user_temp = FIELD_CONDITIONS["temp_C"]
    st.sidebar.markdown("**Adjust Conditions**")
    user_pH   = st.sidebar.slider("pH", 3.0, 8.0, user_pH, 0.1)
    user_temp = st.sidebar.slider("Temp (°C)", 5.0, 35.0, user_temp, 0.5)
    with st.sidebar.expander("Advanced: Photo-Fenton"):
        user_H2O2          = st.slider("H₂O₂ Dose (mM)", 1.0, 100.0, 10.0, 1.0)
        user_Fe            = st.slider("Fe²⁺ Dose (µM)", 0.1, 50.0, 5.0, 0.1)
        cost_H2O2_per_mmol = st.number_input("H₂O₂ cost ($/mmol)", value=0.014, format="%.4f")
        cost_Fe_per_mmol   = st.number_input("Fe²⁺ cost ($/mmol)", value=0.0557, format="%.4f")
        cost_UV_per_min    = st.number_input("UV cost ($/min)", value=0.008, format="%.4f")
    with st.sidebar.expander("Advanced: GAC"):
        gac_ebct          = st.slider("Contact Time (min)", 5.0, 30.0, 15.0, 1.0)
        gac_cost_lb       = st.number_input("GAC Media ($/lb)", value=2.25)
        gac_disposal_cost = st.number_input("Disposal ($/lb)", value=1.50)
    run_sim_button = st.sidebar.button("▶ Run Comparison", type="primary", use_container_width=True)
 
# ─────────────────────────────────────────
# KINETICS & MODEL
# ─────────────────────────────────────────
MW_PFOA=414.07; T_REF=298.15; Ea_J=50_000; R_gas=8.314
k_PFOA=1.2e7; k_I=3e8; k_s=2e4; k_H2O2_scav=2.7e7
k_Fe=2.3e8; k_hv_Fe=3.5e-4; k_hv_h2o2=5e-4
 
def build_rate_constants(pH, temp_C):
    temp_K = temp_C + 273.15
    arr    = np.exp(-Ea_J / R_gas * (1/temp_K - 1/T_REF))
    pH_cor = 10 ** (-(pH - 3.0) * 0.5)
    return 63.0 * arr * pH_cor, 8.4e-6 * arr
 
def photo_fenton_odes(t, y, Fe_total, k_f, k_r):
    H2O2, OH, PFOA, I, Fe2, Fe3 = [max(v, 0) for v in y]
    dH2O2 = (-k_hv_h2o2*H2O2 - k_H2O2_scav*OH*H2O2 - k_f*Fe2*H2O2 - k_r*Fe3*H2O2)
    dOH   = (2*k_hv_h2o2*H2O2 + k_f*Fe2*H2O2 - k_PFOA*OH*PFOA - k_I*OH*I - k_s*OH - k_H2O2_scav*OH*H2O2 - k_Fe*Fe2*OH)
    dPFOA = -k_PFOA*OH*PFOA
    dI    =  k_PFOA*OH*PFOA - k_I*OH*I
    dFe2  = (-k_f*Fe2*H2O2 + k_r*Fe3*H2O2 - k_Fe*Fe2*OH + k_hv_Fe*Fe3)
    return [dH2O2, dOH, dPFOA, dI, dFe2, -dFe2]
 
def simulate_all_contaminants(well_info, pH, temp, H2O2, Fe, costs):
    results = {}
    for contaminant, data in well_info["compounds"].items():
        level_ng_l = data["level_ng_l"]
        gac_pct    = data["gac_removal"] * 100
        PFOA_0     = level_ng_l * 1e-9 / MW_PFOA
        H2O2_mol   = H2O2 * 1e-3
        Fe_mol     = Fe * 1e-6
        k_f, k_r   = build_rate_constants(pH, temp)
        t_eval     = np.linspace(0, 7200, 3600)
        sol = solve_ivp(
            fun=lambda t, y: photo_fenton_odes(t, y, Fe_mol, k_f, k_r),
            t_span=(0,7200), y0=[H2O2_mol,0.0,PFOA_0,0.0,Fe_mol,0.0],
            method='BDF', t_eval=t_eval, rtol=1e-8, atol=1e-12
        )
        t_min   = sol.t / 60
        f_pfoa  = np.maximum(sol.y[2], 0)
        f_rem   = (1 - f_pfoa / PFOA_0) * 100
        idx_90  = np.where(f_rem >= 90)[0]
        t90     = t_min[idx_90[0]] if len(idx_90) > 0 else None
        fc      = H2O2_mol*1000*costs["H2O2"] + Fe_mol*1000*costs["Fe"] + (t90 or 120.0)*costs["UV"]
        gc      = (costs["gac_lb"] + costs["gac_disposal"]) * 0.025
        results[contaminant] = {
            "level_ng_l": level_ng_l, "fenton_removal": f_rem[-1], "fenton_t90": t90,
            "fenton_cost": fc, "gac_removal": gac_pct, "gac_cost": gc,
            "gac_fails": gac_pct < 50, "t_min": t_min, "fenton_removal_curve": f_rem,
        }
    return results
 
# ═══════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════
if page == "🏠 Home":
 
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px 20px 20px;">
        <div style="font-size:0.95rem; color:#94a3b8; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:12px;">
            ● PFAS CONTAMINATION SELF-REPORTING TOOL
        </div>
        <div style="font-size:3.2rem; font-weight:800; color:white; line-height:1.1; margin-bottom:16px;">
            Long Island Water Safety
        </div>
        <div style="font-size:1.15rem; color:#94a3b8; max-width:520px; margin:0 auto 40px auto; line-height:1.6;">
            Real contamination data. Local officials. Actionable steps.<br>
            Built for Long Island residents affected by PFAS in their groundwater.
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # Action cards
    c1, c2, c3, c4 = st.columns(4)
 
    with c1:
        st.markdown("""
        <div class="action-card">
            <div class="icon">📧</div>
            <div class="label">Connect with a Local Official</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go →", key="home_report", use_container_width=True):
            go("📧 Report Contamination")
 
    with c2:
        st.markdown("""
        <div class="action-card">
            <div class="icon">📊</div>
            <div class="label">Determine Feasible Treatment Options</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go →", key="home_compare", use_container_width=True):
            go("📊 Compare Treatments")
 
    with c3:
        st.markdown("""
        <div class="action-card">
            <div class="icon">📋</div>
            <div class="label">Learn More About the Issue</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go →", key="home_issue", use_container_width=True):
            go("📋 The Issue")
 
    with c4:
        st.markdown("""
        <div class="action-card">
            <div class="icon">🧪</div>
            <div class="label">Check Your Own Property</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go →", key="home_kits", use_container_width=True):
            go("🧪 Testing Kits")
 
    st.markdown("---")
 
    # Stats strip
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="issue-stat">10x</div><div class="issue-stat-label">Over NYS Safe Limit at BOMARC</div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="issue-stat">3M+</div><div class="issue-stat-label">Long Islanders on the aquifer</div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="issue-stat">70+</div><div class="issue-stat-label">Years of PFAS accumulation</div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="issue-stat">0</div><div class="issue-stat-label">Natural breakdown of PFAS in soil</div>', unsafe_allow_html=True)
 
    st.markdown("---")
    st.caption("Data: BOMARC Site Investigations 2020–2024, Suffolk County Dept. of Health Services | USGS Finkelstein et al. 2025")
 
# ═══════════════════════════════════════════════════
# PAGE: THE ISSUE
# ═══════════════════════════════════════════════════
elif page == "📋 The Issue":
 
    st.markdown("## 📋 The Issue")
    st.markdown("*What are PFAS, why do they matter, and what is happening on Long Island?*")
    st.markdown("---")
 
    # ── SECTION 1: What are PFAS? ──
    col_img, col_text = st.columns([1, 2])
    with col_img:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/PFOA_structure.svg/320px-PFOA_structure.svg.png",
                 caption="Chemical structure of PFOA — a common PFAS compound")
    with col_text:
        st.markdown("### 🧪 What are PFAS?")
        st.markdown("""
        **Per- and polyfluoroalkyl substances (PFAS)** are a group of ~12,000 synthetic chemicals
        used since the 1950s in products including:
 
        - Non-stick cookware (Teflon)
        - Waterproof clothing and carpets
        - Food packaging
        - **Military firefighting foam (AFFF)** ← source of Long Island contamination
 
        The carbon-fluorine bond is one of the strongest in chemistry — which is why PFAS
        **never break down** in soil, water, or the human body. They are called **"forever chemicals."**
        """)
    
    st.markdown("---")
 
    # ── SECTION 2: How are they harmful? ──
    col_text2, col_img2 = st.columns([2, 1])
    with col_text2:
        st.markdown("### ⚠️ How are they harmful?")
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown("""
            <div class="issue-card" style="border-top:3px solid #ef4444;">
                <div style="font-size:1.6rem;">🎗️</div>
                <div style="font-weight:700; margin:8px 0 4px 0;">Cancer</div>
                <div style="font-size:0.85rem; color:#94a3b8;">Linked to kidney, testicular, and thyroid cancer at prolonged exposures</div>
            </div>
            """, unsafe_allow_html=True)
        with h2:
            st.markdown("""
            <div class="issue-card" style="border-top:3px solid #f59e0b;">
                <div style="font-size:1.6rem;">🧒</div>
                <div style="font-weight:700; margin:8px 0 4px 0;">Child Development</div>
                <div style="font-size:0.85rem; color:#94a3b8;">Disrupts hormones and immune development in children and infants</div>
            </div>
            """, unsafe_allow_html=True)
        with h3:
            st.markdown("""
            <div class="issue-card" style="border-top:3px solid #3b82f6;">
                <div style="font-size:1.6rem;">🫀</div>
                <div style="font-weight:700; margin:8px 0 4px 0;">Heart & Liver</div>
                <div style="font-size:0.85rem; color:#94a3b8;">Elevated cholesterol, liver enzyme abnormalities at low chronic doses</div>
            </div>
            """, unsafe_allow_html=True)
    with col_img2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/Anatomy_of_the_Human_Body_-_Kidney.jpg/240px-Anatomy_of_the_Human_Body_-_Kidney.jpg",
                 caption="Kidney cancer is among the cancers linked to PFOA exposure")
 
    st.markdown("---")
 
    # ── SECTION 3: Long Island ──
    st.markdown("### 🗺️ What has happened on Long Island?")
 
    col_li1, col_li2 = st.columns([2, 1])
    with col_li1:
        st.markdown("""
        Long Island sits entirely on a **sole-source aquifer** — there is no backup water supply.
        Sandy, porous glacial soil means contaminants travel fast and deep.
 
        The **BOMARC Missile Site** in Westhampton was a Cold War-era anti-aircraft installation
        active from the 1950s–1970s. Firefighting foam (AFFF) containing PFOA and PFOS was used
        extensively on-site and soaked into the soil for decades.
 
        Suffolk County Dept. of Health Services investigations in **2020 and 2024** confirmed
        PFAS levels at surrounding monitoring wells **up to 10x above New York's safe limit**.
        """)
 
        # Contamination data cards
        col1, col2, col3 = st.columns(3)
        wells = list(BOMARC_WELLS.values())
        for col, well in zip([col1, col2, col3], wells):
            with col:
                max_comp, max_data = max(well["compounds"].items(), key=lambda x: x[1]["level_ng_l"])
                level = max_data["level_ng_l"]
                mult  = level / well["safe_limit_ng_l"]
                color = "#ef4444" if mult >= 10 else "#f59e0b"
                st.markdown(f"""
                <div style="background:#0f172a;border-radius:8px;padding:16px;color:white;border-left:4px solid {color}">
                    <div style="font-size:0.75rem;color:#94a3b8">{well['location']}</div>
                    <div style="font-size:1.8rem;font-weight:700;color:{color}">{level} ng/L</div>
                    <div style="font-size:0.9rem;color:white">{max_comp}</div>
                    <div style="font-size:0.8rem;color:{color};margin-top:4px">{mult:.0f}x over limit</div>
                </div>
                """, unsafe_allow_html=True)
 
    with col_li2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Bomarc_missile_site.jpg/320px-Bomarc_missile_site.jpg",
                 caption="BOMARC missile site — source of decades of PFAS contamination")
 
    st.markdown("---")
 
    # ── SECTION 4: How are they treated? ──
    st.markdown("### 🔬 How are they treated?")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        <div class="issue-card" style="border-left:4px solid #f59e0b;">
            <div style="font-weight:700; font-size:1.1rem; margin-bottom:8px;">🟡 Granular Activated Carbon (GAC)</div>
            <div style="font-size:0.9rem; color:#94a3b8; line-height:1.6;">
            Water passes through porous carbon — PFAS stick to the surface.<br><br>
            <b style="color:white;">Problem:</b> It doesn't destroy PFAS. Spent carbon becomes hazardous waste.
            Short-chain PFAS like PFNA barely adsorb at all. And it needs replacement every 8 months.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="issue-card" style="border-left:4px solid #22c55e;">
            <div style="font-weight:700; font-size:1.1rem; margin-bottom:8px;">🟢 Photo-Fenton Advanced Oxidation (AOP)</div>
            <div style="font-size:0.9rem; color:#94a3b8; line-height:1.6;">
            Iron + hydrogen peroxide + UV light generates hydroxyl radicals (·OH) that
            <b style="color:white;">chemically destroy</b> PFAS molecules — breaking every carbon-fluorine bond.<br><br>
            Works on all PFAS variants. No hazardous waste. No replacement cycle.
            </div>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("---")
 
    # ── Recent news ──
    st.markdown("### 📰 Read More")
    n1, n2, n3 = st.columns(3)
    with n1:
        st.markdown("""
        <div class="issue-card">
            <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:6px;">NEWSDAY</div>
            <div style="font-weight:600;margin-bottom:8px;">"PFAS in Long Island Drinking Water: What You Need to Know"</div>
            <a href="https://www.newsday.com/long-island/pfas-water-contamination" target="_blank"
               style="color:#22c55e;font-size:0.85rem;">Read article →</a>
        </div>
        """, unsafe_allow_html=True)
    with n2:
        st.markdown("""
        <div class="issue-card">
            <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:6px;">EPA</div>
            <div style="font-weight:600;margin-bottom:8px;">"EPA Sets First-Ever PFAS National Drinking Water Standards"</div>
            <a href="https://www.epa.gov/newsreleases/biden-harris-administration-finalizes-first-ever-national-drinking-water-standard-pfas"
               target="_blank" style="color:#22c55e;font-size:0.85rem;">Read article →</a>
        </div>
        """, unsafe_allow_html=True)
    with n3:
        st.markdown("""
        <div class="issue-card">
            <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:6px;">SUFFOLK COUNTY DOHS</div>
            <div style="font-weight:600;margin-bottom:8px;">BOMARC Site Investigation Report (2024)</div>
            <a href="https://www.suffolkcountyny.gov/Departments/Health-Services/Environmental-Quality"
               target="_blank" style="color:#22c55e;font-size:0.85rem;">View report →</a>
        </div>
        """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.caption("Sources: Suffolk County DOHS, USGS Finkelstein et al. 2025, EPA, NYS DEC")
 
# ═══════════════════════════════════════════════════
# PAGE: TESTING KITS
# ═══════════════════════════════════════════════════
elif page == "🧪 Testing Kits":
 
    st.markdown("## 🧪 Get Your Water Tested")
    st.markdown("If you are on a **private well**, your water is not tested by the county. You are responsible.")
    st.info("🔧 **Links to NY-certified testing labs coming soon.**")
    st.markdown("---")
    st.markdown("""
    ### What to know
    - **Public water:** Your district tests it. Request their annual Consumer Confidence Report (CCR).
    - **Private well:** Arrange testing yourself (~$150–$400 for a full PFAS panel).
    - **NY State certified labs only** — results from uncertified labs cannot be used for legal action.
 
    ### What to request
    Ask for a PFAS panel including: PFOA, PFOS, PFNA, PFBS, PFHxS, PFHxA
    """)
    st.caption("NY State certified lab list: https://www.health.ny.gov/environmental/water/drinking/")
 
# ═══════════════════════════════════════════════════
# PAGE: REPORT CONTAMINATION
# ═══════════════════════════════════════════════════
elif page == "📧 Report Contamination":
 
    st.markdown("## 📧 Report Contamination to Your Officials")
    st.markdown("Generate a formal letter to **your specific town's officials** — in 30 seconds.")
    st.markdown("---")
 
    st.markdown("### Step 1: Enter your town")
    selected_town = st.selectbox("Start typing your town name:", list(TOWN_OFFICIALS.keys()), index=0)
    official = TOWN_OFFICIALS[selected_town]
    county_color = "#3b82f6" if official["county"] == "Nassau" else "#8b5cf6"
    st.markdown(f"""
    <div style="background:#1e293b;border-radius:8px;padding:16px;color:white;margin:12px 0;border-left:4px solid {county_color}">
        <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:4px">{official['county']} County</div>
        <div style="font-size:1rem;font-weight:600">{official['name']}</div>
        <div style="font-size:0.9rem;color:#94a3b8;margin-top:4px">📧 {official['email']}</div>
        <div style="font-size:0.9rem;color:#94a3b8">📞 {official['phone']}</div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("### Step 2: What are you reporting?")
    site_type = st.radio("", ["I'm a resident near a contaminated well", "I want to contact my water district", "I want to report contamination near an industrial site"], horizontal=True, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Step 3: Your details")
    col1, col2 = st.columns(2)
    with col1:
        your_name   = st.text_input("Your name", "Your Name")
        pfas_level  = st.number_input("PFAS level detected (ng/L)", value=100.0, step=1.0)
        contaminant = st.selectbox("Contaminant", ["PFOA", "PFOS", "PFNA", "General PFAS"])
    with col2:
        official_name  = st.text_input("Official / department", official["name"])
        official_email = st.text_input("Official email", official["email"])
        multiplier = pfas_level / 10.0
 
    st.markdown("---")
    st.markdown("### Step 4: Your letter")
 
    if site_type == "I'm a resident near a contaminated well":
        subject = f"URGENT: PFAS Contamination Near My Home — {selected_town}"
        body = f"""Subject: {subject}\n\nDear {official_name},\n\nI am writing as a concerned resident of {selected_town} to formally request action on PFAS contamination affecting groundwater in my community.\n\nRecent testing at the BOMARC Site (Westhampton) has detected {contaminant} at {pfas_level:.0f} ng/L — {multiplier:.0f}x New York's safe limit of 10 ng/L.\n\nAs a resident who relies on this water supply, I am requesting:\n\n1. Immediate public notification of affected residents\n2. Free confirmatory water testing for properties within 1 mile of affected wells\n3. A clear timeline for remediation\n4. A public meeting within 30 days to present findings\n\nTwo treatment options are available:\n- Photo-Fenton Advanced Oxidation: 90% removal, destroys PFAS completely, no hazardous waste\n- Granular Activated Carbon: 75% removal, but creates hazardous spent carbon and fails on PFNA\n\nI urge the county to evaluate Photo-Fenton AOP for sites with mixed PFAS contamination.\n\nSincerely,\n{your_name}\n{selected_town}, NY"""
    elif site_type == "I want to contact my water district":
        subject = f"Request: PFAS Treatment Plan Disclosure — {selected_town} Water District"
        body = f"""Subject: {subject}\n\nDear {official_name},\n\nI am writing to request transparency regarding PFAS treatment in our water supply. {contaminant} has been detected at {pfas_level:.0f} ng/L ({multiplier:.0f}x the NYS safe limit).\n\nI am requesting:\n1. Confirmation of current treatment technology deployed (GAC vs. AOP)\n2. Current removal efficiency data for PFOA, PFOS, and PFNA\n3. A timeline for upgrading to advanced oxidation if GAC is the current method\n4. Cost and funding plan for treatment upgrades\n\nSincerely,\n{your_name}\n{selected_town}, NY"""
    else:
        subject = f"Environmental Report: PFAS Migration from Industrial Site — {selected_town}"
        body = f"""Subject: {subject}\n\nDear {official_name},\n\nI am writing to report evidence of PFAS migration from an industrial source site in {selected_town}.\n\n{contaminant} has been detected at {pfas_level:.0f} ng/L — {multiplier:.0f}x New York's safe limit. Given Long Island's sandy aquifer and shallow water table, contaminant migration is rapid.\n\nI request:\n1. Source investigation to confirm migration pathway\n2. Groundwater monitoring well installation\n3. Immediate assessment of Photo-Fenton AOP feasibility\n4. Notification of downstream well owners\n\nSincerely,\n{your_name}\n{selected_town}, NY"""
 
    st.text_area("Your letter:", body, height=280, disabled=True)
    cola, colb = st.columns(2)
    with cola:
        if st.button("📋 Copy Letter", type="primary", use_container_width=True):
            st.success("✅ Copy from the box above.")
            st.code(body, language="text")
    with colb:
        st.markdown(f"[📧 Open in Email Client](mailto:{official_email}?subject={subject})")
 
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**📞 Call**\n\n{official['name']}: {official['phone']}")
    with col2:
        st.markdown("**🌐 Report online**\n\n[NYSDEC Environmental Complaint](https://www.dec.ny.gov/chemical/8428.html)")
    with col3:
        st.markdown("**📢 Share**\n\nShare this tool with neighbors.")
    st.caption("PFAS data: BOMARC Site Investigation 2020-2024, Suffolk County DOHS")
 
# ═══════════════════════════════════════════════════
# PAGE: COMPARE TREATMENTS
# ═══════════════════════════════════════════════════
elif page == "📊 Compare Treatments":
 
    if not run_sim_button:
        st.markdown("## Compare Treatment Options")
        st.info("Select a contaminated well in the sidebar and click **▶ Run Comparison**.")
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**🟢 Photo-Fenton Advanced Oxidation**\n\nGenerates hydroxyl radicals using iron + H₂O₂ + UV. Chemically **destroys** PFAS.\n\n- Works on all PFAS including short-chain PFNA\n- No hazardous waste\n- Higher upfront cost")
        with col_right:
            st.markdown("**🟡 Granular Activated Carbon (GAC)**\n\nPFAS **adsorb** onto carbon surface — not destroyed. Spent carbon = hazardous waste.\n\n- Proven, widely deployed\n- Lower upfront cost\n- Fails on short-chain PFAS (PFNA, PFBA)\n- Replacement every 8 months")
    else:
        well_info = get_well_data(selected_well_name)
        costs = {"H2O2": cost_H2O2_per_mmol, "Fe": cost_Fe_per_mmol, "UV": cost_UV_per_min, "gac_lb": gac_cost_lb, "gac_disposal": gac_disposal_cost}
        results = simulate_all_contaminants(well_info, user_pH, user_temp, user_H2O2, user_Fe, costs)
 
        st.markdown(f"## Results: {selected_well_name}")
        st.markdown("### All Contaminants at This Well")
        cols = st.columns(len(results))
        for col, (contaminant, data) in zip(cols, results.items()):
            with col:
                mult = data["level_ng_l"] / well_info["safe_limit_ng_l"]
                st.markdown(f"""
                <div style="background:#1e293b;border-radius:8px;padding:16px;color:white">
                    <div style="font-size:1.2rem;font-weight:700">{contaminant}</div>
                    <div style="color:#94a3b8;font-size:0.85rem">{data['level_ng_l']} ng/L ({mult:.0f}x)</div>
                    <hr style="border-color:#334155;margin:12px 0">
                    <div style="font-size:0.9rem">
                        <b>Photo-Fenton:</b> {data['fenton_removal']:.0f}%<br>
                        <b>GAC:</b> {data['gac_removal']:.0f}% {'❌ FAILS' if data['gac_fails'] else '✅'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
 
        st.markdown("---")
        compounds_list  = list(results.keys())
        fenton_removals = [results[c]["fenton_removal"] for c in compounds_list]
        gac_removals    = [results[c]["gac_removal"]    for c in compounds_list]
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0f172a'); ax.set_facecolor('#1e293b')
        x = np.arange(len(compounds_list)); width = 0.35
        bars1 = ax.bar(x-width/2, fenton_removals, width, label='Photo-Fenton', color='#22c55e', alpha=0.85)
        bars2 = ax.bar(x+width/2, gac_removals,    width, label='GAC',          color='#f59e0b', alpha=0.85)
        ax.set_ylabel('Removal %', color='#94a3b8'); ax.set_title('Removal Efficiency by Contaminant', color='white')
        ax.set_xticks(x); ax.set_xticklabels(compounds_list, color='#94a3b8')
        ax.set_ylim(0, 105); ax.legend(facecolor='#1e293b', labelcolor='white')
        ax.tick_params(colors='#94a3b8'); ax.grid(True, alpha=0.15, axis='y', color='white')
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2., h, f'{int(h)}%', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig)
 
        st.markdown("---")
        pfna = results.get("PFNA")
        if pfna and pfna["gac_removal"] < 50:
            st.error(f"⚠️ **CRITICAL:** GAC only removes **{pfna['gac_removal']:.0f}% of PFNA** at this site. Photo-Fenton destroys **{pfna['fenton_removal']:.0f}%**.")
 
        st.markdown("### Treatment Effectiveness Summary")
        st.dataframe([{"Compound": c, "Level (ng/L)": f"{d['level_ng_l']:.1f}", "Fenton Removal": f"{d['fenton_removal']:.0f}%", "GAC Removal": f"{d['gac_removal']:.0f}%", "GAC Status": "❌ FAILS" if d["gac_fails"] else "✅"} for c, d in results.items()], use_container_width=True)
 
        st.markdown("---")
        st.markdown("### Cost Comparison")
        col1, col2 = st.columns(2)
        with col1: st.metric("Photo-Fenton (avg)", f"${np.mean([results[c]['fenton_cost'] for c in compounds_list]):.4f}/L")
        with col2: st.metric("GAC (avg)", f"${np.mean([results[c]['gac_cost'] for c in compounds_list]):.4f}/L")
 
        with st.expander("View conditions used"):
            arr = np.exp(-Ea_J/R_gas*(1/(user_temp+273.15)-1/T_REF))
            st.markdown(f"| Parameter | Value | Source |\n|---|---|---|\n| pH | {user_pH} | USGS |\n| Temp | {user_temp}°C | USGS |\n| Arrhenius | {arr:.3f} | De Laat & Gallard 1999 |\n| PFOA k | 1.2×10⁷ | Hori et al. 2004 |\n| GAC PFOA/PFOS | 75%/70% | EPA 2020 |\n| GAC PFNA | 22% | Long et al. 2023 |")
 
# ═══════════════════════════════════════════════════
# PAGE: FAQs
# ═══════════════════════════════════════════════════
elif page == "❓ FAQs":
 
    st.markdown("## ❓ Frequently Asked Questions")
    st.markdown("---")
    faqs = [
        ("What is PFAS?", "PFAS (per- and polyfluoroalkyl substances) are ~12,000 synthetic chemicals used since the 1950s in non-stick cookware, waterproof clothing, and military firefighting foam. Called 'forever chemicals' because they never break down in soil, water, or the body."),
        ("Why is Long Island particularly at risk?", "Long Island relies entirely on a shallow underground aquifer — no backup supply. Sandy, porous soil means contaminants move fast. Military AFFF foam use at BOMARC and other sites has been migrating into wells for decades."),
        ("What is the BOMARC site?", "A Cold War-era anti-aircraft missile base in Westhampton, NY. PFAS-containing firefighting foam was used extensively. Suffolk County DOHS confirmed contamination 10x+ above safe limits in 2020 and 2024 investigations."),
        ("What are NYS safe limits?", "New York set MCLs in 2023: PFOA, PFOS, PFNA, PFHxS, PFBS, and NEtFOSAA all at 10 ng/L — among the strictest in the US."),
        ("What is Photo-Fenton AOP?", "An Advanced Oxidation Process using iron + hydrogen peroxide + UV to generate hydroxyl radicals (·OH) that chemically destroy PFAS. Unlike carbon filtration, it destroys rather than captures PFAS, and works on short-chain compounds like PFNA."),
        ("Why does GAC fail on PFNA?", "GAC works by adsorption. Short-chain PFAS like PFNA have low affinity for carbon surfaces — only ~22% removal vs. 75%+ for PFOA. A major limitation at BOMARC-type mixed-contamination sites."),
        ("Is my tap water safe?", "Public supply: your district must test and notify you. Private well: you must test yourself. Use a NY State certified lab for a PFAS panel. See the Testing Kits page."),
        ("Who is responsible for BOMARC cleanup?", "The U.S. Army and NY Air National Guard are the responsible parties. Suffolk County DOHS monitors groundwater. NYSDEC oversees remediation."),
    ]
    for q, a in faqs:
        with st.expander(f"**{q}**"):
            st.markdown(a)
    st.caption("Sources: Suffolk County DOHS, NYS DEC, EPA, USGS Finkelstein et al. 2025")
