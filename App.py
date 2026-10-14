import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from well_data import BOMARC_WELLS, FIELD_CONDITIONS, get_well_data

st.set_page_config(
    page_title="WellGuard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none; }
[data-testid="collapsedControl"] {
    display: flex !important;
    background: #1e293b !important;
    border-radius: 0 8px 8px 0 !important;
    top: 1rem !important;
}
.action-card {
    background: #1e293b; border-radius: 10px; padding: 28px 20px;
    color: white; text-align: center; border: 1px solid #334155;
    min-height: 130px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.action-card .icon { font-size: 2rem; margin-bottom: 10px; }
.action-card .label { font-size: 0.95rem; font-weight: 600; color: #e2e8f0; line-height: 1.3; }
.issue-card { background: #1e293b; border-radius: 10px; padding: 20px; color: white; margin-bottom: 16px; }
.result-card { background: #1e293b; border-radius: 8px; padding: 16px; color: white; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

def go(p):
    st.session_state.page = p
    st.rerun()

with st.sidebar:
    st.markdown("### 💧 WellGuard")
    st.markdown("*Long Island Water Safety*")
    if st.button("🏠 Home", use_container_width=True):
        go("🏠 Home")
    st.markdown("---")
    for label in ["📋 The Issue", "📧 Report Contamination", "📊 Compare Treatments", "❓ FAQs", "🧪 Testing Kits"]:
        if st.button(label, use_container_width=True, key=f"nav_{label}"):
            go(label)
    st.markdown("---")
    st.caption("Data: BOMARC Site Investigations (2020, 2024)\nSuffolk County Dept. of Health Services\nUSGS Finkelstein et al. 2025")

page = st.session_state.page

# ── KINETICS ──
MW_PFOA=414.07; T_REF=298.15; Ea_J=50_000; R_gas=8.314
k_PFOA=1.2e7; k_I=3e8; k_s=2e4; k_H2O2_scav=2.7e7
k_Fe=2.3e8; k_hv_Fe=3.5e-4; k_hv_h2o2=5e-4

def build_rate_constants(pH, temp_C):
    temp_K = temp_C + 273.15
    arr    = np.exp(-Ea_J / R_gas * (1/temp_K - 1/T_REF))
    return 63.0 * arr * 10**(-(pH-3.0)*0.5), 8.4e-6 * arr

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
        t_min  = sol.t / 60
        f_rem  = (1 - np.maximum(sol.y[2], 0) / PFOA_0) * 100
        idx_90 = np.where(f_rem >= 90)[0]
        t90    = t_min[idx_90[0]] if len(idx_90) > 0 else None

        # Cost per liter
        fc = H2O2_mol*1000*costs["H2O2"] + Fe_mol*1000*costs["Fe"] + (t90 or 120.0)*costs["UV"]
        gc = (costs["gac_lb"] + costs["gac_disposal"]) * 0.025

        # Cost per µg PFOA removed
        pfoa_removed_ug = (PFOA_0 * f_rem[-1] / 100) * MW_PFOA * 1e6
        gac_removed_ug  = (PFOA_0 * gac_pct / 100)  * MW_PFOA * 1e6
        fc_per_ug = fc / pfoa_removed_ug if pfoa_removed_ug > 0 else 9999
        gc_per_ug = gc / gac_removed_ug  if gac_removed_ug  > 0 else 9999

        # Who wins on removal and cost?
        fenton_wins_removal = f_rem[-1] > gac_pct
        fenton_wins_cost    = fc < gc
        short_chain = gac_pct < 50  # GAC structurally fails

        results[contaminant] = {
            "level_ng_l":           level_ng_l,
            "fenton_removal":       f_rem[-1],
            "fenton_t90":           t90,
            "fenton_cost":          fc,
            "fenton_cost_per_ug":   fc_per_ug,
            "gac_removal":          gac_pct,
            "gac_cost":             gc,
            "gac_cost_per_ug":      gc_per_ug,
            "gac_fails":            short_chain,
            "fenton_wins_removal":  fenton_wins_removal,
            "fenton_wins_cost":     fenton_wins_cost,
            "t_min":                t_min,
            "fenton_removal_curve": f_rem,
        }
    return results

# ════════════════════════════════════════
# HOME
# ════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px 32px 20px;">
        <div style="font-size:0.85rem;color:#94a3b8;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:14px;">
            ● PFAS CONTAMINATION SELF-REPORTING TOOL
        </div>
        <div style="font-size:3.8rem;font-weight:900;color:lightblue;line-height:1.05;margin-bottom:14px;letter-spacing:-0.02em;">
            WellGuard
        </div>
        <div style="font-size:1.1rem;color:#94a3b8;max-width:500px;margin:0 auto 48px auto;line-height:1.65;">
            Real contamination data. Local officials. Actionable steps.<br>
            Built for Long Island residents affected by PFAS in their groundwater.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "📧", "Connect with a\nLocal Official",         "📧 Report Contamination"),
        (c2, "📊", "Determine Feasible\nTreatment Options",  "📊 Compare Treatments"),
        (c3, "📋", "Learn More\nAbout the Issue",            "📋 The Issue"),
        (c4, "🧪", "Check Your\nOwn Property",               "🧪 Testing Kits"),
    ]
    for col, icon, label, target in cards:
        with col:
            st.markdown(f"""
            <div class="action-card">
                <div class="icon">{icon}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Go →", key=f"home_{target}", use_container_width=True):
                go(target)

    st.markdown("---")
    st.caption("Data: BOMARC Site Investigations 2020–2024, Suffolk County DOHS | USGS Finkelstein et al. 2025")

# ════════════════════════════════════════
# THE ISSUE
# ════════════════════════════════════════
elif page == "📋 The Issue":
    st.markdown("## What's Happening to Long Island's Water?")
    st.markdown("---")

    st.markdown("### 🧪 What are PFAS?")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/PFOA_structure.svg/320px-PFOA_structure.svg.png",
                 caption="PFOA — a common PFAS compound", use_container_width=True)
    with col_b:
        st.markdown("""
        **Per- and polyfluoroalkyl substances (PFAS)** are man-made chemicals used since the 1950s
        in non-stick cookware, waterproof clothing, and — critically — **military firefighting foam**.

        The carbon-fluorine bond is the strongest in organic chemistry.
        PFAS never break down in soil, water, or the human body — hence **"forever chemicals."**

        > *Long Island's contamination source: AFFF firefighting foam used at the BOMARC missile site in Westhampton.*
        """)
    st.markdown("---")

    st.markdown("### ⚠️ How are they harmful?")
    h1, h2, h3 = st.columns(3)
    for col, icon, title, text in [
        (h1, "🎗️", "Cancer Risk",       "Linked to kidney, testicular, and thyroid cancers at prolonged exposures above safe limits."),
        (h2, "🧒", "Child Development", "Disrupts hormones and immune development. Infants are most vulnerable via breast milk and formula."),
        (h3, "🫀", "Heart & Liver",     "Associated with elevated cholesterol and liver enzyme abnormalities even at low chronic doses."),
    ]:
        with col:
            st.markdown(f"""
            <div class="issue-card" style="border-top:3px solid #ef4444;text-align:center;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="font-weight:700;margin:8px 0 6px 0;">{title}</div>
                <div style="font-size:0.85rem;color:#94a3b8;line-height:1.5;">{text}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🗺️ What Happened on Long Island?")
    col_li1, col_li2 = st.columns([3, 2])
    with col_li1:
        st.markdown("""
        Long Island's **sole-source aquifer** supplies 3 million people with no backup.
        Sandy glacial soil lets contaminants reach groundwater fast.

        The **BOMARC missile site** (Westhampton, 1950s–70s) used PFAS firefighting foam
        that soaked into the soil for decades. Suffolk County DOHS confirmed contamination
        **up to 10x above New York's safe limit** in both 2020 and 2024 investigations.
        """)
        w1, w2, w3 = st.columns(3)
        for col, well in zip([w1, w2, w3], list(BOMARC_WELLS.values())):
            with col:
                max_comp, max_data = max(well["compounds"].items(), key=lambda x: x[1]["level_ng_l"])
                level = max_data["level_ng_l"]; mult = level / well["safe_limit_ng_l"]
                color = "#ef4444" if mult >= 10 else "#f59e0b"
                st.markdown(f"""
                <div style="background:#0f172a;border-radius:8px;padding:14px;color:white;border-left:4px solid {color}">
                    <div style="font-size:0.7rem;color:#94a3b8">{well['location']}</div>
                    <div style="font-size:1.6rem;font-weight:700;color:{color}">{level} ng/L</div>
                    <div style="font-size:0.85rem;color:white">{max_comp}</div>
                    <div style="font-size:0.75rem;color:{color};margin-top:2px">{mult:.0f}x over limit</div>
                </div>
                """, unsafe_allow_html=True)
    with col_li2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/CIM-10_Bomarc_at_March_AFB_Museum.jpg/320px-CIM-10_Bomarc_at_March_AFB_Museum.jpg",
                 caption="BOMARC missile — PFAS foam used at Westhampton launch site", use_container_width=True)
    st.markdown("---")

    st.markdown("### 🔬 How Can It Be Treated?")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("""
        <div class="issue-card" style="border-left:4px solid #f59e0b;">
            <div style="font-weight:700;font-size:1.05rem;margin-bottom:8px;">🟡 Granular Activated Carbon (GAC) — Current Standard</div>
            <div style="font-size:0.875rem;color:#94a3b8;line-height:1.6;">
            PFAS stick to carbon media — <b style="color:white">not destroyed</b>. Spent carbon becomes
            <b style="color:#ef4444">hazardous waste</b> every 8 months. Short-chain PFAS like PFNA
            barely adsorb (~22% removal).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown("""
        <div class="issue-card" style="border-left:4px solid #22c55e;">
            <div style="font-weight:700;font-size:1.05rem;margin-bottom:8px;">🟢 Photo-Fenton AOP — Emerging Alternative</div>
            <div style="font-size:0.875rem;color:#94a3b8;line-height:1.6;">
            Iron + H₂O₂ + UV generates hydroxyl radicals that <b style="color:white">chemically destroy</b>
            PFAS — every C–F bond. Works on all PFAS variants.
            <b style="color:#22c55e">No hazardous waste. No replacement cycle.</b>
            Developed in part at CUNY MEC Vittadello Lab.
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📰 In the News")
    n1, n2, n3 = st.columns(3)
    for col, source, headline, url in [
        (n1, "NEWSDAY",            '"PFAS Detected in Long Island Wells Near Military Sites"',               "https://www.newsday.com/long-island/environment/pfas-water-contamination-suffolk-nassau-long-island-dvbkxn71"),
        (n2, "EPA",                '"First-Ever National PFAS Drinking Water Standards" (2024)',             "https://www.epa.gov/newsreleases/biden-harris-administration-finalizes-first-ever-national-drinking-water-standard-pfas"),
        (n3, "SUFFOLK COUNTY DOHS","BOMARC Site Investigation & Groundwater Reports (2020, 2024)",          "https://www.suffolkcountyny.gov/Departments/Health-Services/Environmental-Quality"),
    ]:
        with col:
            st.markdown(f"""
            <div class="issue-card">
                <div style="font-size:0.7rem;color:#94a3b8;margin-bottom:6px;">{source}</div>
                <div style="font-weight:600;font-size:0.9rem;margin-bottom:10px;">{headline}</div>
                <a href="{url}" target="_blank" style="color:#22c55e;font-size:0.85rem;">Read →</a>
            </div>
            """, unsafe_allow_html=True)
    st.caption("Sources: Suffolk County DOHS, USGS Finkelstein et al. 2025, EPA, NYS DEC")

# ════════════════════════════════════════
# REPORT CONTAMINATION
# ════════════════════════════════════════
elif page == "📧 Report Contamination":
    st.markdown("## 📧 Connect with Your Local Officials")
    st.markdown("---")

    # Step 1: Intent
    st.markdown("### Step 1: What describes your situation?")
    intent = st.radio("", [
        "I want to learn more — connect me with a local official",
        "I want to report contamination"
    ], label_visibility="collapsed")

    st.markdown("---")

    # Step 2: Town
    st.markdown("### Step 2: Enter your town")
    selected_town = st.selectbox("Start typing your town:", list(TOWN_OFFICIALS.keys()), index=0)
    official = TOWN_OFFICIALS[selected_town]
    county_color = "#3b82f6" if official["county"] == "Nassau" else "#8b5cf6"
    st.markdown(f"""
    <div style="background:#1e293b;border-radius:8px;padding:16px;color:white;margin:10px 0;border-left:4px solid {county_color}">
        <div style="font-size:0.75rem;color:#94a3b8">{official['county']} County</div>
        <div style="font-size:1rem;font-weight:600;margin-top:2px">{official['name']}</div>
        <div style="font-size:0.9rem;color:#94a3b8;margin-top:4px">📧 {official['email']} &nbsp;|&nbsp; 📞 {official['phone']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if intent == "I want to learn more — connect me with a local official":
        # ── LEARN MORE FLOW ──
        st.markdown("### Step 3: Your details")
        col1, col2 = st.columns(2)
        with col1:
            your_name = st.text_input("Your name", "Your Name")
        with col2:
            official_name  = st.text_input("Official / department", official["name"])
            official_email = st.text_input("Official email", official["email"])

        st.markdown("---")
        st.markdown("### Step 4: Your letter")
        subject = f"Question About PFAS Contamination and Water Safety — {selected_town}"
        body = f"""Subject: {subject}

Dear {official_name},

My name is {your_name}, and I am a resident of {selected_town}. I am writing because I have concerns about PFAS contamination in Long Island's groundwater and would like to learn more about the steps being taken to protect our community.

Specifically, I would appreciate information on:

1. Current PFAS monitoring results for wells near {selected_town}
2. What treatment technology is in use or planned for our water supply
3. Whether residents on private wells can access any county-supported testing
4. Who I should contact if I want to stay updated on remediation progress

Thank you for your time. I look forward to hearing from you.

Sincerely,
{your_name}
{selected_town}, NY"""

    else:
        # ── REPORT CONTAMINATION FLOW ──
        st.markdown("### Step 3: Your situation and details")
        site_type = st.radio("", [
            "Resident near a contaminated well",
            "Contacting my water district",
            "Contamination near an industrial site"
        ], horizontal=True, label_visibility="collapsed")

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

        if site_type == "Resident near a contaminated well":
            subject = f"URGENT: PFAS Contamination Near My Home — {selected_town}"
            body = f"""Subject: {subject}

Dear {official_name},

I am a resident of {selected_town} writing to request urgent action on PFAS contamination in our groundwater.

Recent BOMARC site testing detected {contaminant} at {pfas_level:.0f} ng/L — {multiplier:.0f}x New York's 10 ng/L safe limit.

I am requesting:
1. Immediate public notification of affected residents
2. Free water testing within 1 mile of affected wells
3. A clear remediation timeline
4. A public meeting within 30 days

I urge evaluation of Photo-Fenton Advanced Oxidation, which destroys all PFAS variants including PFNA that carbon filters cannot remove.

Sincerely,
{your_name}
{selected_town}, NY"""

        elif site_type == "Contacting my water district":
            subject = f"Request: PFAS Treatment Disclosure — {selected_town} Water District"
            body = f"""Subject: {subject}

Dear {official_name},

I am requesting transparency on PFAS treatment in our water supply. {contaminant} has been detected at {pfas_level:.0f} ng/L ({multiplier:.0f}x the NYS safe limit).

I request:
1. Confirmation of current treatment technology (GAC vs. AOP)
2. Removal efficiency data for PFOA, PFOS, and PFNA
3. Timeline for upgrading to advanced oxidation if currently using GAC
4. Funding plan for treatment upgrades

Sincerely,
{your_name}
{selected_town}, NY"""

        else:
            subject = f"Environmental Report: PFAS Near Industrial Site — {selected_town}"
            body = f"""Subject: {subject}

Dear {official_name},

I am reporting evidence of PFAS migration from an industrial source in {selected_town}.

{contaminant} has been detected at {pfas_level:.0f} ng/L — {multiplier:.0f}x New York's safe limit. Long Island's sandy aquifer means rapid contaminant migration.

I request:
1. Source investigation
2. Groundwater monitoring well installation
3. Photo-Fenton AOP feasibility assessment
4. Notification of downstream well owners

Sincerely,
{your_name}
{selected_town}, NY"""

    st.text_area("Your letter:", body, height=280, disabled=True)
    ca, cb = st.columns(2)
    with ca:
        if st.button("📋 Copy Letter", type="primary", use_container_width=True):
            st.success("✅ Copy from the box above.")
            st.code(body, language="text")
    with cb:
        st.markdown(f"[📧 Open in Email Client](mailto:{official_email}?subject={subject})")

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f"**📞 Call**\n\n{official['phone']}")
    with col2: st.markdown("**🌐 Report online**\n\n[NYSDEC Complaint](https://www.dec.ny.gov/chemical/8428.html)")
    with col3: st.markdown("**📢 Share**\n\nSend WellGuard to your neighbors.")
    st.caption("PFAS data: BOMARC Site Investigations 2020–2024, Suffolk County DOHS")

# ════════════════════════════════════════
# COMPARE TREATMENTS
# ════════════════════════════════════════
elif page == "📊 Compare Treatments":
    st.markdown("## 📊 Compare Treatment Options")
    st.markdown("Adjust parameters and run the simulation to compare Photo-Fenton AOP vs. GAC.")
    st.markdown("---")

    col_well, col_ph, col_temp = st.columns(3)
    with col_well:
        selected_well_name = st.selectbox("Well Site", list(BOMARC_WELLS.keys()))
        well_info = get_well_data(selected_well_name)
    with col_ph:
        user_pH = st.slider("pH", 3.0, 8.0, FIELD_CONDITIONS["pH"], 0.1)
    with col_temp:
        user_temp = st.slider("Temp (°C)", 5.0, 35.0, FIELD_CONDITIONS["temp_C"], 0.5)

    with st.expander("⚙️ Advanced: Reagent Doses & Costs"):
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            user_H2O2 = st.slider("H₂O₂ Dose (mM)", 1.0, 100.0, 10.0, 1.0)
            user_Fe   = st.slider("Fe²⁺ Dose (µM)", 0.1, 50.0, 5.0, 0.1)
        with ac2:
            cost_H2O2_per_mmol = st.number_input("H₂O₂ cost ($/mmol)", value=0.014, format="%.4f")
            cost_Fe_per_mmol   = st.number_input("Fe²⁺ cost ($/mmol)", value=0.0557, format="%.4f")
            cost_UV_per_min    = st.number_input("UV cost ($/min)", value=0.008, format="%.4f")
        with ac3:
            gac_cost_lb       = st.number_input("GAC Media ($/lb)", value=2.25)
            gac_disposal_cost = st.number_input("Disposal ($/lb)", value=1.50)

    run_sim_button = st.button("▶ Run Comparison", type="primary", use_container_width=True)
    st.markdown("---")

    if not run_sim_button:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("**🟢 Photo-Fenton AOP** — Generates ·OH radicals via Fe + H₂O₂ + UV. Chemically destroys PFAS. Works on all variants. No hazardous waste.")
        with col_right:
            st.markdown("**🟡 GAC** — PFAS adsorb onto carbon (not destroyed). Spent carbon = hazardous waste every 8 months. Fails on short-chain PFAS.")
    else:
        costs = {
            "H2O2": cost_H2O2_per_mmol, "Fe": cost_Fe_per_mmol, "UV": cost_UV_per_min,
            "gac_lb": gac_cost_lb, "gac_disposal": gac_disposal_cost,
        }
        results = simulate_all_contaminants(well_info, user_pH, user_temp, user_H2O2, user_Fe, costs)

        st.markdown(f"### Results: {selected_well_name}")

        # ── PER-COMPOUND RESULT CARDS ──
        cols = st.columns(len(results))
        for col, (compound, d) in zip(cols, results.items()):
            with col:
                mult = d["level_ng_l"] / well_info["safe_limit_ng_l"]

                # Determine winner dynamically
                if d["fenton_wins_removal"] and d["fenton_wins_cost"]:
                    winner_label = "✅ Photo-Fenton wins"
                    winner_color = "#22c55e"
                elif d["fenton_wins_removal"] and not d["fenton_wins_cost"]:
                    winner_label = "⚡ Fenton more effective, GAC cheaper"
                    winner_color = "#f59e0b"
                elif not d["fenton_wins_removal"] and d["fenton_wins_cost"]:
                    winner_label = "💰 GAC more effective, Fenton cheaper"
                    winner_color = "#f59e0b"
                else:
                    winner_label = "✅ GAC wins"
                    winner_color = "#f59e0b"

                # Short-chain note
                short_chain_note = ""
                if d["gac_fails"]:
                    short_chain_note = f"<div style='font-size:0.75rem;color:#ef4444;margin-top:6px;'>⚠️ GAC fails on {compound} — short-chain PFAS have poor affinity for carbon surfaces (~{d['gac_removal']:.0f}% adsorption)</div>"

                st.markdown(f"""
                <div class="result-card" style="border-left:4px solid {winner_color}">
                    <div style="font-size:1.1rem;font-weight:700">{compound}</div>
                    <div style="color:#94a3b8;font-size:0.8rem;margin-bottom:8px">{d['level_ng_l']} ng/L ({mult:.0f}x limit)</div>
                    <div style="font-size:0.85rem;margin-bottom:4px">
                        <b>Photo-Fenton removal:</b> {d['fenton_removal']:.0f}%<br>
                        <b>GAC removal:</b> {d['gac_removal']:.0f}%
                    </div>
                    <div style="font-size:0.85rem;margin-bottom:4px">
                        <b>Fenton cost/L:</b> ${d['fenton_cost']:.4f}<br>
                        <b>GAC cost/L:</b> ${d['gac_cost']:.4f}
                    </div>
                    <div style="font-size:0.85rem;color:#94a3b8">
                        <b>Fenton $/µg removed:</b> ${d['fenton_cost_per_ug']:.4f}<br>
                        <b>GAC $/µg removed:</b> ${d['gac_cost_per_ug']:.4f}
                    </div>
                    <div style="margin-top:8px;font-size:0.8rem;font-weight:600;color:{winner_color}">{winner_label}</div>
                    {short_chain_note}
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # ── BAR CHART ──
        compounds_list  = list(results.keys())
        fenton_removals = [results[c]["fenton_removal"] for c in compounds_list]
        gac_removals    = [results[c]["gac_removal"]    for c in compounds_list]

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor('#0f172a'); ax.set_facecolor('#1e293b')
        x = np.arange(len(compounds_list)); width = 0.35
        bars1 = ax.bar(x-width/2, fenton_removals, width, label='Photo-Fenton', color='#22c55e', alpha=0.85)
        bars2 = ax.bar(x+width/2, gac_removals,    width, label='GAC',          color='#f59e0b', alpha=0.85)
        ax.set_ylabel('Removal %', color='#94a3b8')
        ax.set_title('Removal Efficiency by Contaminant', color='white')
        ax.set_xticks(x); ax.set_xticklabels(compounds_list, color='#94a3b8')
        ax.set_ylim(0, 105); ax.legend(facecolor='#1e293b', labelcolor='white')
        ax.tick_params(colors='#94a3b8'); ax.grid(True, alpha=0.15, axis='y', color='white')
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2., h, f'{int(h)}%', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)

        # ── OVERALL VERDICT ──
        st.markdown("---")
        fenton_wins_overall = sum(1 for d in results.values() if d["fenton_wins_removal"])
        gac_wins_overall    = len(results) - fenton_wins_overall
        any_gac_fail        = any(d["gac_fails"] for d in results.values())

        if fenton_wins_overall >= gac_wins_overall:
            verdict = f"**Photo-Fenton is more effective** for {fenton_wins_overall}/{len(results)} contaminants at this well."
            if any_gac_fail:
                verdict += " GAC structurally fails on short-chain PFAS present here — short-chain compounds have too little affinity for carbon to adsorb efficiently."
            st.success(verdict)
        else:
            verdict = f"**GAC is more effective** for {gac_wins_overall}/{len(results)} contaminants at these conditions."
            if any_gac_fail:
                verdict += f" However, GAC fails on short-chain PFAS also present at this site."
                st.warning(verdict)
            else:
                st.info(verdict)

        # ── SUMMARY TABLE ──
        st.markdown("### Summary")
        summary = []
        for c, d in results.items():
            summary.append({
                "Compound":        c,
                "Level (ng/L)":    f"{d['level_ng_l']:.1f}",
                "Fenton Removal":  f"{d['fenton_removal']:.0f}%",
                "GAC Removal":     f"{d['gac_removal']:.0f}%",
                "Fenton $/L":      f"${d['fenton_cost']:.4f}",
                "GAC $/L":         f"${d['gac_cost']:.4f}",
                "Fenton $/µg":     f"${d['fenton_cost_per_ug']:.4f}",
                "GAC $/µg":        f"${d['gac_cost_per_ug']:.4f}",
                "Removal Winner":  "Photo-Fenton" if d["fenton_wins_removal"] else "GAC",
                "Cost Winner":     "Photo-Fenton" if d["fenton_wins_cost"] else "GAC",
            })
        st.dataframe(summary, use_container_width=True)

        with st.expander("Conditions used"):
            arr = np.exp(-Ea_J/R_gas*(1/(user_temp+273.15)-1/T_REF))
            st.markdown(f"pH: {user_pH} | Temp: {user_temp}°C | Arrhenius: {arr:.3f} | k_PFOA: 1.2×10⁷ (Hori 2004) | GAC PFOA: 75%, PFOS: 70%, PFNA: 22% (EPA 2020 / Long 2023)")

# ════════════════════════════════════════
# FAQs
# ════════════════════════════════════════
elif page == "❓ FAQs":
    st.markdown("## ❓ FAQs")
    st.markdown("---")
    faqs = [
        ("What is PFAS?", "Synthetic chemicals in non-stick cookware, waterproof clothing, and military firefighting foam since the 1950s. Called 'forever chemicals' because they never break down in soil, water, or the body."),
        ("Why is Long Island at risk?", "Long Island's sole-source aquifer sits under porous sandy soil — contaminants travel fast. Military sites like BOMARC used PFAS foam for decades, and it has been migrating into surrounding wells ever since."),
        ("What is the BOMARC site?", "A Cold War anti-aircraft missile base in Westhampton, NY. Suffolk County DOHS confirmed PFAS contamination 10x+ above safe limits in 2020 and 2024 investigations."),
        ("What are NYS safe limits?", "Effective 2023: PFOA, PFOS, PFNA, PFHxS, PFBS, and NEtFOSAA all at 10 ng/L — among the strictest PFAS standards in the US."),
        ("What is Photo-Fenton AOP?", "Fe + H₂O₂ + UV generates hydroxyl radicals that chemically destroy PFAS. Works on short-chain PFAS like PFNA that carbon filters cannot remove. No hazardous waste."),
        ("Why does GAC fail on PFNA?", "Short-chain PFAS have poor affinity for carbon surfaces — only ~22% removal vs 75%+ for PFOA. A serious limitation at sites like BOMARC with mixed PFAS contamination."),
        ("Is my tap water safe?", "Public supply: your district tests it — request their Consumer Confidence Report. Private well: test yourself with a NY State certified lab."),
        ("Who is responsible for BOMARC?", "The U.S. Army and NY Air National Guard are responsible parties. Suffolk County DOHS monitors groundwater. NYSDEC oversees remediation."),
    ]
    for q, a in faqs:
        with st.expander(f"**{q}**"):
            st.markdown(a)
    st.caption("Sources: Suffolk County DOHS, EPA, NYS DEC, USGS Finkelstein et al. 2025")

# ════════════════════════════════════════
# TESTING KITS
# ════════════════════════════════════════
elif page == "🧪 Testing Kits":
    st.markdown("## 🧪 Get Your Water Tested")
    st.markdown("---")
    st.info("🔧 **Links to NY State-certified PFAS testing labs coming soon.**")
    st.caption("NY State certified lab list: https://www.health.ny.gov/environmental/water/drinking/")
