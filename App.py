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

ELAP_LABS = [
    {"name": "Pace Analytical Services, LLC", "address": "575 Broad Hollow Rd, Melville, NY 11747", "county": ["Nassau", "Suffolk"], "phone": "(516) 370-6000", "url": "https://www.pacelabs.com/analytical-environmental/pfas/", "pfas_panel": "EPA 533 / 537.1", "notes": "Can mail in water samples"},
    {"name": "Suffolk County Private Well Water Testing Program", "address": "360 Yaphank Avenue, Suite 1C, Yaphank, NY 11980", "county": "Suffolk", "phone": "(631) 852-5810.", "url": "https://suffolkcountyny.gov/Departments/Health-Services/Environmental-Quality/Water-Resources/Private-Well-Water-Testing-Program?", "pfas_panel": "EPA 537.1", "notes": "County residents may qualify for subsidized testing."},
    {"name": "Long Island Analytical Laboratories, Inc.", "address": "110 Colin Dr, Holbrook, NY 11741", "county": ["Nassau", "Suffolk"], "phone": "(631) 472-3400", "url": "https://www.lialinc.com/what-we-do", "pfas_panel": "NYS ELAP-certified drinking water analysis", "notes": "Get shipped a kit or mail in a sample"},
    {"name": "Tap Score", "address": "1345 4th St. Berkeley, CA 94710", "county": "Regional", "phone": "(510)485-9396", "url": "https://mytapscore.com/products/pfas-water-test?country=US%C2%A4cy=USD&variant=29243769651297&utm_source=chatgpt.com", "pfas_panel": "EPA 537.1", "notes": "Full kit shipping option"},
]

# ─────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Light page background ── */
.stApp {
    background-color: #f1f5f9;
}

/* ── Hide default Streamlit chrome ── */
[data-testid="stSidebarNav"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Sidebar: dark navy ── */
[data-testid="stSidebar"] {
    background-color: #0d1b2a !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    border: none;
    color: #94a3b8 !important;
    text-align: left;
    font-size: 0.875rem;
    font-weight: 500;
    padding: 8px 12px;
    border-radius: 6px;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1b2f45;
    color: #f8fafc !important;
}
[data-testid="collapsedControl"] {
    display: flex !important;
    background: #0d1b2a !important;
    border-radius: 0 6px 6px 0 !important;
    top: 1rem !important;
}

/* ── Action cards (home page) ── */
.action-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 28px 24px;
    border-left: 4px solid #0ea5e9;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.action-card .card-label {
    font-size: 0.95rem;
    font-weight: 600;
    color: #0f172a;
    line-height: 1.4;
    margin-bottom: 6px;
}
.action-card .card-sub {
    font-size: 0.8rem;
    color: #64748b;
}

/* ── Content cards ── */
.data-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    color: #0f172a;
    margin-bottom: 12px;
}
.data-card.alert { border-left: 4px solid #dc2626; }
.data-card.warning { border-left: 4px solid #f59e0b; }
.data-card.info { border-left: 4px solid #0ea5e9; }
.data-card.success { border-left: 4px solid #16a34a; }

/* ── Contamination number cards ── */
.stat-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    border-top: 3px solid #dc2626;
}

/* ── Lab cards ── */
.lab-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    border-left: 3px solid #16a34a;
    margin-bottom: 10px;
}

/* ── Result cards (compare treatments) ── */
.result-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07);
    color: #0f172a;
    margin-bottom: 8px;
}

/* ── Math equation blocks ── */
.math-eq {
    background: #0f172a;
    border-radius: 6px;
    padding: 10px 16px;
    font-family: 'Courier New', monospace;
    color: #7dd3fc;
    font-size: 0.875rem;
    margin: 6px 0;
}

/* ── Page headings ── */
h1, h2, h3 { color: #0f172a; }

/* ── Streamlit override: make inputs look clean on light bg ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    color: #0f172a;
}

/* ── Remove default Streamlit markdown padding ── */
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# SESSION + NAV
# ─────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"

def go(p):
    st.session_state.page = p
    st.rerun()

with st.sidebar:
    st.markdown('<div style="padding:24px 16px 8px 16px;"><span style="font-size:1.1rem;font-weight:700;color:#f8fafc;letter-spacing:-0.01em;">WellGuard</span><br><span style="font-size:0.75rem;color:#64748b;">Long Island Water Safety</span></div>', unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#1b2f45;margin:8px 0 12px 0">', unsafe_allow_html=True)
    if st.button("Home", use_container_width=True, key="nav_Home"):
        go("Home")
    for label in ["The Issue", "Report Contamination", "Compare Treatments", "FAQs", "Testing Kits"]:
        if st.button(label, use_container_width=True, key=f"nav_{label}"):
            go(label)
    st.markdown('<hr style="border-color:#1b2f45;margin:16px 0 8px 0">', unsafe_allow_html=True)
    st.markdown('<div style="padding:0 8px;font-size:0.7rem;color:#475569;line-height:1.6;">Data: BOMARC Site Investigations (2020, 2024)<br>Suffolk County Dept. of Health Services<br>USGS Finkelstein et al. 2025</div>', unsafe_allow_html=True)

page = st.session_state.page

# ─────────────────────────────────────────
# KINETICS
# ─────────────────────────────────────────
MW_PFOA=414.07; T_REF=298.15; Ea_J=50_000; R_gas=8.314
k_PFOA=4e6; k_I=3e8; k_s=2e4; k_H2O2_scav=2.7e7
k_Fe=2.3e8; k_hv_Fe=3.5e-4; k_hv_h2o2=5e-4

def build_rate_constants(pH, temp_C):
    temp_K = temp_C + 273.15
    arr = np.exp(-Ea_J / R_gas * (1/temp_K - 1/T_REF))
    
    # Severe drop-off if pH exceeds optimal Fenton range due to iron precipitation
    if pH > 4.5:
        activity_factor = 0.01
    else:
        activity_factor = 10**(-(pH-3.0)*0.5)
        
    return 63.0 * arr * activity_factor, 8.4e-6 * arr

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
        H2O2_mol   = H2O2 * 1e-3; Fe_mol = Fe * 1e-6
        k_f, k_r   = build_rate_constants(pH, temp)
        sol = solve_ivp(
            fun=lambda t, y: photo_fenton_odes(t, y, Fe_mol, k_f, k_r),
            t_span=(0,7200), y0=[H2O2_mol,0.0,PFOA_0,0.0,Fe_mol,0.0],
            method='BDF', t_eval=np.linspace(0,7200,3600), rtol=1e-8, atol=1e-12
        )
        t_min = sol.t/60
        f_rem = (1 - np.maximum(sol.y[2],0)/PFOA_0)*100
        idx_90 = np.where(f_rem>=90)[0]
        t90 = t_min[idx_90[0]] if len(idx_90)>0 else None
        fc = H2O2_mol*1000*costs["H2O2"] + Fe_mol*1000*costs["Fe"] + (t90 or 120.0)*costs["UV"]
        gc = (costs["gac_lb"]+costs["gac_disposal"])*0.025
        pr = (PFOA_0*f_rem[-1]/100)*MW_PFOA*1e6
        gr = (PFOA_0*gac_pct/100)*MW_PFOA*1e6
        results[contaminant] = {
            "level_ng_l": level_ng_l, "fenton_removal": f_rem[-1], "fenton_t90": t90,
            "fenton_cost": fc, "fenton_cost_per_ug": fc/pr if pr>0 else 9999,
            "gac_removal": gac_pct, "gac_cost": gc,
            "gac_cost_per_ug": gc/gr if gr>0 else 9999,
            "gac_fails": gac_pct<50,
            "fenton_wins_removal": f_rem[-1]>gac_pct,
            "fenton_wins_cost": fc<gc,
            "t_min": t_min, "fenton_removal_curve": f_rem,
        }
    return results

# ═══════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════
if page == "Home":
    # Hero
    st.markdown("""
    <div style="background:#0d1b2a;border-radius:12px;padding:56px 48px 48px 48px;margin-bottom:32px;">
        <div style="font-size:0.75rem;font-weight:600;color:#0ea5e9;letter-spacing:0.08em;margin-bottom:16px;text-transform:uppercase;">
            PFAS Contamination — Long Island, New York
        </div>
        <div style="font-size:3rem;font-weight:800;color:#f8fafc;line-height:1.05;margin-bottom:16px;letter-spacing:-0.02em;">
            WellGuard
        </div>
        <div style="font-size:1rem;color:#94a3b8;max-width:520px;line-height:1.7;">
            Real contamination data from the BOMARC site investigation. Local health officials.
            Treatment feasibility modeling. Built for Long Island residents.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Contact a local official", "Generate a letter to your town's health department.", "Report Contamination"),
        (c2, "Compare treatment options", "Model Photo-Fenton AOP vs. carbon filtration at your well.", "Compare Treatments"),
        (c3, "Understand the issue", "What PFAS are, where they came from, and why they matter.", "The Issue"),
        (c4, "Test your property", "Find NYS-certified labs and step-by-step sampling guides.", "Testing Kits"),
    ]
    for col, label, sub, target in cards:
        with col:
            st.markdown(f'<div class="action-card"><div class="card-label">{label}</div><div class="card-sub">{sub}</div></div>', unsafe_allow_html=True)
            if st.button("Go", key=f"home_{target}", use_container_width=True):
                go(target)

    st.markdown("<br>", unsafe_allow_html=True)
    # Alert strip
    st.markdown("""
    <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;padding:14px 20px;display:flex;align-items:center;gap:12px;">
        <div style="width:8px;height:8px;border-radius:50%;background:#ea580c;flex-shrink:0;"></div>
        <div style="font-size:0.875rem;color:#9a3412;">
            <strong>Active contamination:</strong> PFOA and PFOS detected at the BOMARC site (Westhampton) at levels up to 12x above New York's 10 ng/L safe limit as of the 2024 Suffolk County investigation.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info("Data Notice: Nassau County contamination data retrieval is in progress")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Sources: BOMARC Site Investigations 2020–2024, Suffolk County DOHS | USGS Finkelstein et al. 2025")

# ═══════════════════════════════════════════
# THE ISSUE
# ═══════════════════════════════════════════
elif page == "The Issue":
    st.markdown("## What's Happening to Long Island's Water?")
    st.markdown('<hr style="border-color:#e2e8f0;margin-bottom:24px;">', unsafe_allow_html=True)

    st.markdown("### What are PFAS?")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.image("newPFOA.png", caption="PFOA molecular structure", use_container_width=True)
    with col_b:
        st.markdown("""
        Per- and polyfluoroalkyl substances (PFAS): a class of synthetic chemicals used since the 1950s in cookware, fabrics, food packaging, and military firefighting foam (AFFF), causing the contamination at BOMARC.

        The C-F bond is one of the strongest in chemistry. PFAS do not break down in soil, water, or the human body. Concentrations accumulate over decades of exposure. The body CANNOT breakdown PFAS faster than it accumulates. **Prevention is the best option**
        """)

    st.markdown('<hr style="border-color:#e2e8f0;margin:20px 0;">', unsafe_allow_html=True)
    st.markdown("### Health effects")

    h1, h2, h3 = st.columns(3)
    for col, title, text in [
        (h1, "Cancer",            "Studies link chronic PFAS exposure to elevated of kidney, testicular, and thyroid cancer rates. PFOA: classified as a likely carcinogen"),
        (h2, "Development","PFAS cross the placental barrier and are present in breast milk. Exposure as a baby causes immune suppression and hormone disruption in children."),
        (h3, "Cardiovascular",    "Chronic exposure correlates with elevated LDL cholesterol and liver enzyme abnormalities, risk factors for cardiovascular disease."),
    ]:
        with col:
            st.markdown(f'<div class="data-card alert"><div style="font-weight:600;font-size:0.9rem;margin-bottom:6px;color:#0f172a;">{title}</div><div style="font-size:0.82rem;color:#475569;line-height:1.6;">{text}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#e2e8f0;margin:20px 0;">', unsafe_allow_html=True)
    st.markdown("### What happened on Long Island?")

    col_li1, col_li2 = st.columns([3, 2])
    with col_li1:
        st.markdown("""
        Long Island uses a sole-source aquifer system: 1 aquifer is the only drinking water source for 3 million residents...no alternative supply. LI's sandy glacial soil is highly permeable: contaminants introduced at the surface can reach groundwater quickly

        The BOMARC site in Westhampton Beach was a Cold War-era Air National Guard installation from the 50s-70s. Firefighting foam containing PFAS was used routinely during that period and seeped into the surrounding soil. 
        """)
        w1, w2, w3 = st.columns(3)
        for col, well in zip([w1,w2,w3], list(BOMARC_WELLS.values())):
            with col:
                mc, md = max(well["compounds"].items(), key=lambda x: x[1]["level_ng_l"])
                lv = md["level_ng_l"]; mt = lv/well["safe_limit_ng_l"]
                clr = "#dc2626" if mt>=10 else "#d97706"
                st.markdown(f'<div class="stat-card" style="border-top-color:{clr};"><div style="font-size:0.7rem;color:#64748b;margin-bottom:4px;">{well["location"]}</div><div style="font-size:1.8rem;font-weight:800;color:{clr};line-height:1;">{lv}</div><div style="font-size:0.7rem;color:#64748b;">ng/L — {mc}</div><div style="font-size:0.75rem;font-weight:600;color:{clr};margin-top:4px;">{mt:.0f}x over limit</div></div>', unsafe_allow_html=True)
    with col_li2:
        st.image("AquiferSystem.jpg", caption="Long Island's 'lone aquier' system", use_container_width=True)

    st.markdown('<hr style="border-color:#e2e8f0;margin:20px 0;">', unsafe_allow_html=True)
    st.markdown("### Treatment options")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            '<div class="data-card warning">'
            '<div style="font-weight:600;margin-bottom:6px;">Granular Activated Carbon: current standard</div>'
            '<div style="font-size:0.85rem;color:#475569;line-height:1.65;">'
            '<ul style="margin:0;padding-left:18px;">'
            "<li><strong>PFAS</strong> molecules adsorb onto the carbon surface and are removed from the water, but not destroyed.</li>"
            "<li><strong>Spent</strong> carbon must be disposed of as hazardous waste.</li>"
            "<li><strong>Short</strong>-chain PFAS compounds such as PFBA adsorb poorly, achieving low removal.</li>"
            "</ul>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    with t2:
        st.markdown(
            '<div class="data-card success">'
            '<div style="font-weight:600;margin-bottom:6px;">Advanced Oxidation: emerging alternative</div>'
            '<div style="font-size:0.85rem;color:#475569;line-height:1.65;">'
            '<ul style="margin:0;padding-left:18px;">'
            "<li><strong>Generate</strong> hydroxyl radicals that break C-F bonds directly, mineralizing PFAS into non-toxic products.</li>"
            "<li><strong>Near</strong>-equal effectiveness across all chain lengths.</li>"
            "</ul>"
            "</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown('<hr style="border-color:#e2e8f0;margin:20px 0;">', unsafe_allow_html=True)
    st.markdown("### Recent coverage")
    n1, n2, n3 = st.columns(3)
    for col, src, hl, url in [
        (n1, "Newsday", "PFAS detected in Long Island wells near military sites", "https://www.newsday.com/long-island/environment/pfas-water-contamination-suffolk-nassau-long-island-dvbkxn71"),
        (n2, "U.S. EPA", "First-ever national PFAS drinking water standards finalized (2024)", "https://www.epa.gov/newsreleases/biden-harris-administration-finalizes-first-ever-national-drinking-water-standard-pfas"),
        (n3, "Suffolk County DOHS", "BOMARC site investigation and groundwater reports (2020, 2024)", "https://www.suffolkcountyny.gov/Departments/Health-Services/Environmental-Quality"),
    ]:
        with col:
            st.markdown(f'<div class="data-card info"><div style="font-size:0.7rem;font-weight:600;color:#0369a1;margin-bottom:6px;text-transform:uppercase;letter-spacing:0.04em;">{src}</div><div style="font-size:0.875rem;font-weight:500;color:#0f172a;margin-bottom:10px;">{hl}</div><a href="{url}" target="_blank" style="font-size:0.8rem;color:#0ea5e9;text-decoration:none;font-weight:500;">Read the report</a></div>', unsafe_allow_html=True)
    st.caption("Sources: Suffolk County DOHS, USGS Finkelstein et al. 2025, EPA, NYS DEC")

# ═══════════════════════════════════════════
# REPORT CONTAMINATION
# ═══════════════════════════════════════════
elif page == "Report Contamination":
    st.markdown("## Connect with Your Local Officials")
    st.markdown('<hr style="border-color:#e2e8f0;margin-bottom:24px;">', unsafe_allow_html=True)

    st.markdown("**What brings you here?**")
    intent = st.radio("", [
        "I want to learn more and connect with a local official",
        "I want to formally report contamination"
    ], label_visibility="collapsed")

    st.markdown('<hr style="border-color:#e2e8f0;margin:20px 0;">', unsafe_allow_html=True)
    st.markdown("**Your town**")
    selected_town = st.selectbox("", list(TOWN_OFFICIALS.keys()), index=0, label_visibility="collapsed")
    official = TOWN_OFFICIALS[selected_town]
    county_color = "#2563eb" if official["county"] == "Nassau" else "#7c3aed"
    st.markdown(f'<div class="data-card" style="border-left:3px solid {county_color};margin-top:8px;"><div style="font-size:0.7rem;font-weight:600;color:{county_color};text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">{official["county"]} County</div><div style="font-weight:600;color:#0f172a;">{official["name"]}</div><div style="font-size:0.85rem;color:#64748b;margin-top:4px;">{official["email"]} &nbsp;·&nbsp; {official["phone"]}</div></div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#e2e8f0;margin:20px 0;">', unsafe_allow_html=True)

    if intent == "I want to learn more and connect with a local official":
        col1, col2 = st.columns(2)
        with col1:
            your_name = st.text_input("Your name", "Your Name")
        with col2:
            official_name  = st.text_input("Official / department", official["name"])
            official_email = st.text_input("Email", official["email"])
        subject = f"Question about PFAS contamination — {selected_town}"
        body = f"Subject: {subject}\n\nDear {official_name},\n\nMy name is {your_name}, and I am a resident of {selected_town}. I am writing to request information about PFAS contamination in our area and the steps being taken to protect our water supply.\n\nI would appreciate answers to the following:\n\n1. Current PFAS monitoring results for wells near {selected_town}\n2. What treatment technology is in use or planned for our supply\n3. Whether private well owners can access county-supported testing\n4. The appropriate contact for updates on remediation progress\n\nThank you for your time.\n\nSincerely,\n{your_name}\n{selected_town}, NY"
    else:
        site_type = st.radio("", ["Resident near a contaminated well", "Contacting my water district", "Reporting an industrial source"], horizontal=True, label_visibility="collapsed")
        col1, col2 = st.columns(2)
        with col1:
            your_name   = st.text_input("Your name", "Your Name")
            pfas_level  = st.number_input("PFAS level detected (ng/L)", value=100.0, step=1.0)
            contaminant = st.selectbox("Contaminant detected", ["PFOA", "PFOS", "PFBA", "General PFAS"])
        with col2:
            official_name  = st.text_input("Official / department", official["name"])
            official_email = st.text_input("Email", official["email"])
            multiplier     = pfas_level / 10.0

        if site_type == "Resident near a contaminated well":
            subject = f"PFAS contamination near my home — {selected_town}"
            body = f"Subject: {subject}\n\nDear {official_name},\n\nI am a resident of {selected_town} requesting action on PFAS contamination affecting groundwater in my area.\n\nRecent testing detected {contaminant} at {pfas_level:.0f} ng/L — {multiplier:.0f}x New York's 10 ng/L limit.\n\nI am requesting:\n1. Public notification of affected residents\n2. Free confirmatory testing within 1 mile of affected wells\n3. A clear remediation timeline\n4. A public meeting within 30 days\n\nI urge the county to evaluate Photo-Fenton Advanced Oxidation as an alternative to carbon filtration, particularly given the presence of short-chain PFAS at this site.\n\nSincerely,\n{your_name}\n{selected_town}, NY"
        elif site_type == "Contacting my water district":
            subject = f"PFAS treatment disclosure request — {selected_town} Water District"
            body = f"Subject: {subject}\n\nDear {official_name},\n\nI am requesting information about how our water supply is being treated for PFAS contamination. {contaminant} has been detected at {pfas_level:.0f} ng/L ({multiplier:.0f}x the NYS limit).\n\nI request:\n1. Confirmation of the current treatment technology in use\n2. Removal efficiency data for PFOA, PFOS, and PFBA\n3. A timeline for any planned upgrades\n4. Cost and funding information for treatment improvements\n\nSincerely,\n{your_name}\n{selected_town}, NY"
        else:
            subject = f"PFAS source report — {selected_town}"
            body = f"Subject: {subject}\n\nDear {official_name},\n\nI am reporting suspected PFAS migration from an industrial source in {selected_town}.\n\n{contaminant} has been detected at {pfas_level:.0f} ng/L ({multiplier:.0f}x the NYS limit). Given Long Island's permeable soil and sole-source aquifer, I request:\n\n1. Source investigation\n2. Groundwater monitoring well installation\n3. Photo-Fenton AOP feasibility assessment\n4. Notification of downstream well owners\n\nSincerely,\n{your_name}\n{selected_town}, NY"

    st.markdown("**Your letter**")
    st.text_area("", body, height=260, disabled=True, label_visibility="collapsed")
    ca, cb = st.columns(2)
    with ca:
        if st.button("Copy letter", type="primary", use_container_width=True):
            st.success("Copy the text from the box above.")
            st.code(body, language="text")
    with cb:
        st.markdown(f"[Open in email client](mailto:{official_email}?subject={subject})")

    st.markdown('<hr style="border-color:#e2e8f0;margin:20px 0;">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Call directly**\n\n{official['phone']}")
    with col2:
        st.markdown("**File a complaint**\n\n[NYSDEC Environmental Complaint](https://www.dec.ny.gov/chemical/8428.html)")
    with col3:
        st.markdown("**Share this tool**\n\nPass WellGuard to neighbors on private wells.")
    st.caption("Data: BOMARC Site Investigations 2020–2024, Suffolk County DOHS")

# ═══════════════════════════════════════════
# COMPARE TREATMENTS
# ═══════════════════════════════════════════
elif page == "Compare Treatments":
    st.markdown("## Treatment Feasibility Model")
    st.markdown('<hr style="border-color:#e2e8f0;margin-bottom:24px;">', unsafe_allow_html=True)

    sub = st.radio("", ["Run simulation", "About the math"], horizontal=True, label_visibility="collapsed")
    st.markdown('<hr style="border-color:#e2e8f0;margin:16px 0;">', unsafe_allow_html=True)

    if sub == "About the math":
        st.markdown("### Model overview")
        st.markdown("The simulation integrates a 6-species ODE system over two hours using SciPy's stiff BDF solver. All parameters are sourced from peer-reviewed literature.")

        st.markdown("#### Species")
        for sym, desc in [
            ("H₂O₂",  "Hydrogen peroxide — the Fenton oxidant"),
            ("·OH",    "Hydroxyl radical — the species that reacts with PFAS"),
            ("PFAS",   "Target contaminant (PFOA, PFOS, or PFBA)"),
            ("I",      "Lumped intermediate degradation products"),
            ("Fe²⁺",   "Ferrous iron — Fenton catalyst"),
            ("Fe³⁺",   "Ferric iron — regenerated by UV photoreduction"),
        ]:
            st.markdown(f"**{sym}** — {desc}")

        st.markdown("---")
        st.markdown("#### Rate equations")
        for lhs, rhs in [
            ("d[H₂O₂]/dt", "= −k_hv·[H₂O₂] − k_scav·[OH]·[H₂O₂] − k_f·[Fe²⁺]·[H₂O₂] − k_r·[Fe³⁺]·[H₂O₂]"),
            ("d[OH]/dt",    "= 2·k_hv·[H₂O₂] + k_f·[Fe²⁺]·[H₂O₂] − k_PFAS·[OH]·[PFAS] − k_I·[OH]·[I] − k_s·[OH] − k_scav·[OH]·[H₂O₂] − k_Fe·[Fe²⁺]·[OH]"),
            ("d[PFAS]/dt",  "= −k_PFAS·[OH]·[PFAS]"),
            ("d[I]/dt",     "=  k_PFAS·[OH]·[PFAS] − k_I·[OH]·[I]"),
            ("d[Fe²⁺]/dt",  "= −k_f·[Fe²⁺]·[H₂O₂] + k_r·[Fe³⁺]·[H₂O₂] − k_Fe·[Fe²⁺]·[OH] + k_hv_Fe·[Fe³⁺]"),
            ("d[Fe³⁺]/dt",  "= −d[Fe²⁺]/dt"),
        ]:
            st.markdown(f'<div class="math-eq"><b>{lhs}</b>  {rhs}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Corrections applied")
        st.markdown("""
        **Temperature:** k_f is corrected from 25°C reference values via Arrhenius with Ea = 50 kJ/mol (De Laat & Gallard 1999). Suffolk County groundwater averages 12.9°C, suppressing the Fenton rate by ~57% vs. lab conditions.

        **pH:** An empirical iron-speciation correction penalizes k_f above pH 3, where Fe²⁺ increasingly precipitates as Fe(OH)₃ and becomes unavailable for the Fenton reaction.

        **GAC constants:** Literature-derived per compound. PFOA 75%, PFOS 70%, PFBA 22% (EPA PFAS Fact Sheet 2020; Long et al. 2023). PFBA's low value reflects weak van der Waals interactions between short-chain molecules and the activated carbon surface.
        """)

        st.markdown("---")
        st.markdown("#### Rate constants")
        st.dataframe([
            {"Symbol": "k_PFAS", "Value": "1 × 10⁷ L/(mol·s)", "Reaction": "OH + PFOA", "Source": "Hori et al. 2004"},
            {"Symbol": "k_I",    "Value": "3.0 × 10⁸ L/(mol·s)", "Reaction": "OH + intermediates (lumped)", "Source": "estimated"},
            {"Symbol": "k_s",    "Value": "2.0 × 10⁴ s⁻¹", "Reaction": "background OH scavenging", "Source": "field estimate"},
            {"Symbol": "k_scav", "Value": "2.7 × 10⁷ L/(mol·s)", "Reaction": "OH + H₂O₂", "Source": "Buxton et al. 1988"},
            {"Symbol": "k_Fe",   "Value": "2.3 × 10⁸ L/(mol·s)", "Reaction": "Fe²⁺ + OH", "Source": "Walling 1975"},
            {"Symbol": "k_hv",   "Value": "5 × 10⁻⁴ s⁻¹", "Reaction": "H₂O₂ photolysis", "Source": "UV-A lamp estimate"},
            {"Symbol": "k_hv_Fe","Value": "3.5 × 10⁻⁴ s⁻¹", "Reaction": "Fe³⁺ photoreduction", "Source": "De Laat & Gallard 1999"},
        ], use_container_width=True)
        st.caption("Validation pending. These results are for feasibility screening only.")

    else:
        col_well, col_ph, col_temp = st.columns(3)
        with col_well:
            selected_well_name = st.selectbox("Well site", list(BOMARC_WELLS.keys()))
            well_info = get_well_data(selected_well_name)
        with col_ph:
            user_pH = st.slider("pH", 3.0, 8.0, FIELD_CONDITIONS["pH"], 0.1)
        with col_temp:
            user_temp = st.slider("Temperature (°C)", 5.0, 35.0, FIELD_CONDITIONS["temp_C"], 0.5)

        with st.expander("Advanced — reagent doses and costs"):
            ac1, ac2, ac3 = st.columns(3)
            with ac1:
                user_H2O2 = st.slider("H₂O₂ dose (mM)", 1.0, 100.0, 10.0, 1.0)
                user_Fe   = st.slider("Fe²⁺ dose (µM)", 0.1, 50.0, 5.0, 0.1)
            with ac2:
                cost_H2O2_per_mmol = st.number_input("H₂O₂ ($/mmol)", value=0.014, format="%.4f")
                cost_Fe_per_mmol   = st.number_input("Fe²⁺ ($/mmol)", value=0.0557, format="%.4f")
                cost_UV_per_min    = st.number_input("UV ($/min)", value=0.008, format="%.4f")
            with ac3:
                gac_cost_lb       = st.number_input("GAC media ($/lb)", value=2.25)
                gac_disposal_cost = st.number_input("GAC disposal ($/lb)", value=1.50)

        if st.button("Run comparison", type="primary", use_container_width=True):
            costs = {"H2O2": cost_H2O2_per_mmol, "Fe": cost_Fe_per_mmol, "UV": cost_UV_per_min, "gac_lb": gac_cost_lb, "gac_disposal": gac_disposal_cost}
            results = simulate_all_contaminants(well_info, user_pH, user_temp, user_H2O2, user_Fe, costs)

            st.markdown(f"### {selected_well_name}")
            cols = st.columns(len(results))
            for col, (compound, d) in zip(cols, results.items()):
                with col:
                    mult = d["level_ng_l"] / well_info["safe_limit_ng_l"]
                    if d["fenton_wins_removal"] and d["fenton_wins_cost"]:
                        wl, wc = "Photo-Fenton preferred", "#16a34a"
                    elif d["fenton_wins_removal"]:
                        wl, wc = "Fenton more effective / GAC cheaper", "#d97706"
                    elif d["fenton_wins_cost"]:
                        wl, wc = "GAC more effective / Fenton cheaper", "#d97706"
                    else:
                        wl, wc = "GAC preferred", "#d97706"
                    sc = f'<div style="font-size:0.75rem;color:#dc2626;margin-top:6px;font-weight:500;">GAC removal limited on {compound} — short-chain PFAS adsorb poorly onto activated carbon (~{d["gac_removal"]:.0f}%)</div>' if d["gac_fails"] else ""
                    st.markdown(f'''<div class="result-card" style="border-left:3px solid {wc};">
                        <div style="font-weight:700;font-size:1rem;margin-bottom:4px;">{compound}</div>
                        <div style="font-size:0.78rem;color:#64748b;margin-bottom:10px;">{d["level_ng_l"]} ng/L &nbsp;·&nbsp; {mult:.0f}x over limit</div>
                        <div style="font-size:0.83rem;margin-bottom:6px;color:#0f172a;"><b>Photo-Fenton:</b> {d["fenton_removal"]:.0f}% removal &nbsp;·&nbsp; ${d["fenton_cost"]:.4f}/L &nbsp;·&nbsp; ${d["fenton_cost_per_ug"]:.4f}/µg</div>
                        <div style="font-size:0.83rem;margin-bottom:8px;color:#0f172a;"><b>GAC:</b> {d["gac_removal"]:.0f}% removal &nbsp;·&nbsp; ${d["gac_cost"]:.4f}/L &nbsp;·&nbsp; ${d["gac_cost_per_ug"]:.4f}/µg</div>
                        <div style="font-size:0.8rem;font-weight:600;color:{wc};">{wl}</div>{sc}</div>''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            compounds_list  = list(results.keys())
            fenton_removals = [results[c]["fenton_removal"] for c in compounds_list]
            gac_removals    = [results[c]["gac_removal"]    for c in compounds_list]

            fig, ax = plt.subplots(figsize=(10, 4))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#f8fafc')
            x = np.arange(len(compounds_list)); width = 0.35
            ax.bar(x-width/2, fenton_removals, width, label='Photo-Fenton', color='#0ea5e9', alpha=0.9)
            ax.bar(x+width/2, gac_removals,    width, label='GAC',          color='#f59e0b', alpha=0.9)
            ax.set_ylabel('Removal (%)', color='#374151'); ax.set_title('Removal by Contaminant', color='#0f172a', fontweight='600')
            ax.set_xticks(x); ax.set_xticklabels(compounds_list, color='#374151')
            ax.set_ylim(0, 105); ax.legend(facecolor='#ffffff', edgecolor='#e2e8f0')
            ax.tick_params(colors='#374151'); ax.grid(True, alpha=0.4, axis='y', color='#e2e8f0')
            for bars in [ax.containers[0], ax.containers[1]]:
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2., h+0.5, f'{int(h)}%', ha='center', va='bottom', color='#374151', fontsize=9, fontweight='600')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#e2e8f0'); ax.spines['bottom'].set_color('#e2e8f0')
            plt.tight_layout(); st.pyplot(fig)

            fw = sum(1 for d in results.values() if d["fenton_wins_removal"])
            gw = len(results) - fw
            any_fail = any(d["gac_fails"] for d in results.values())
            if fw >= gw:
                msg = f"Photo-Fenton achieves higher removal for {fw}/{len(results)} compounds at these conditions."
                if any_fail: msg += " GAC removal is structurally limited on short-chain PFAS present at this site."
                st.success(msg)
            else:
                msg = f"GAC achieves higher removal for {gw}/{len(results)} compounds at these conditions."
                if any_fail: msg += " Note: GAC fails on short-chain PFAS also present here."
                st.warning(msg) if any_fail else st.info(msg)

            st.markdown("### Full comparison")
            st.dataframe([{"Compound": c, "Level (ng/L)": f"{d['level_ng_l']:.1f}", "Fenton removal": f"{d['fenton_removal']:.0f}%", "GAC removal": f"{d['gac_removal']:.0f}%", "Fenton $/L": f"${d['fenton_cost']:.4f}", "GAC $/L": f"${d['gac_cost']:.4f}", "Fenton $/µg": f"${d['fenton_cost_per_ug']:.4f}", "GAC $/µg": f"${d['gac_cost_per_ug']:.4f}", "Removal winner": "Photo-Fenton" if d["fenton_wins_removal"] else "GAC", "Cost winner": "Photo-Fenton" if d["fenton_wins_cost"] else "GAC"} for c, d in results.items()], use_container_width=True)
            with st.expander("Conditions used in this run"):
                arr = np.exp(-Ea_J/R_gas*(1/(user_temp+273.15)-1/T_REF))
                st.markdown(f"pH {user_pH} · {user_temp}°C · Arrhenius factor {arr:.3f} · k_PFOA 1×10⁷ (Hori 2004) · GAC: PFOA 75%, PFOS 70%, PFBA 22% (EPA 2020 / Long 2023)")
        else:
            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown('<div class="data-card success"><b>Photo-Fenton AOP</b><br><span style="font-size:0.85rem;color:#475569;">Iron + hydrogen peroxide + UV generates hydroxyl radicals that break C–F bonds directly. Effective on all PFAS chain lengths. No hazardous waste stream.</span></div>', unsafe_allow_html=True)
            with col_right:
                st.markdown('<div class="data-card warning"><b>Granular Activated Carbon</b><br><span style="font-size:0.85rem;color:#475569;">PFAS adsorb onto carbon and are removed from water, but not destroyed. Spent carbon is hazardous waste. Adsorption efficiency drops significantly for short-chain PFAS.</span></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════
# FAQs
# ═══════════════════════════════════════════
elif page == "FAQs":
    st.markdown("## Frequently Asked Questions")
    st.markdown('<hr style="border-color:#e2e8f0;margin-bottom:24px;">', unsafe_allow_html=True)
    faqs = [
        ("Are private wells protected by the government?",
         "No. The federal Safe Drinking Water Act applies to public water systems only. Private well owners are solely responsible for testing and treating their own water. Suffolk and Nassau County health departments can provide guidance, but they do not test or remediate private wells."),
        ("How do I know if my well has PFAS contamination?",
         "PFAS have no detectable taste, odor, or color at any concentration found in groundwater. The only way to determine contamination is laboratory testing. A NYS ELAP-certified lab running EPA Method 533 or 537.1 on a water sample from your tap is the accepted standard. See the Testing Kits section for labs that serve Long Island."),
        ("What can be done about contamination in my well?",
         "In the short term, NSF/ANSI 58-certified reverse osmosis systems reduce PFAS at the point of use and can be installed under a kitchen sink. For long-term remediation, options include connection to a treated municipal supply, installation of whole-house AOP treatment, or formal remediation efforts by the responsible party (military or industrial). If a specific source is identifiable, legal remedies may also apply."),
        ("What is the BOMARC site?",
         "The BOMARC missile site in Westhampton Beach was an Air National Guard installation active from the late 1950s through the 1970s. PFAS-containing aqueous film-forming foam (AFFF) was used in fire training and suppression during that period. Suffolk County DOHS investigations in 2020 and 2024 confirmed PFAS contamination in surrounding groundwater monitoring wells at levels up to 12x above the NYS limit."),
        ("What are New York's legal limits for PFAS?",
         "New York State MCLs effective 2023: PFOA, PFOS, PFBA, PFHxS, PFBS, and NEtFOSAA, each at 10 ng/L. These are among the strictest PFAS drinking water standards in the US, and more stringent than the EPA's 2024 federal MCLs on some compounds."),
        ("Why does carbon filtration fail on some PFAS compounds?",
         "Granular Activated Carbon removes PFAS through adsorption: molecules adhere to the carbon surface via van der Waals forces. This mechanism favors larger, long-chain PFAS like PFOA (75% removal) and PFOS (70% removal). Short-chain compounds such as PFBA have lower molecular weight and weaker interactions with the carbon surface, yielding removal rates as low as 22%. Sites with mixed PFAS contamination, like BOMARC, present compounds that GAC cannot adequately address."),
        ("Can I take legal action over PFAS contamination?",
         "Potentially. Tort claims have been brought against PFAS manufacturers (3M, DuPont) and operators of contaminated sites (military installations, industrial facilities). Several class action suits have resulted in substantial settlements. Whether a claim is viable depends on the identifiable source of contamination and your jurisdiction. WellGuard does not provide legal advice — consult a private environmental attorney or the New York State Department of Environmental Conservation."),
        ("What is Photo-Fenton Advanced Oxidation?",
         "Photo-Fenton AOP is a water treatment technology that generates hydroxyl radicals (·OH) — among the most reactive oxidizing species in chemistry — through a combination of ferrous iron (Fe²⁺), hydrogen peroxide (H₂O₂), and UV light. These radicals attack and break carbon-fluorine bonds in PFAS molecules, mineralizing them to carbon dioxide, hydrogen fluoride, and water. Unlike GAC, it destroys PFAS rather than capturing them, is effective across all chain lengths, and produces no hazardous waste."),
    ]
    for q, a in faqs:
        with st.expander(f"**{q}**"):
            st.markdown(f'<div style="font-size:0.875rem;color:#374151;line-height:1.7;padding:4px 0;">{a}</div>', unsafe_allow_html=True)
    st.caption("Sources: Suffolk County DOHS, EPA, NYS DEC, SDWA, USGS Finkelstein et al. 2025")

# ═══════════════════════════════════════════
# TESTING KITS
# ═══════════════════════════════════════════
elif page == "Testing Kits":
    st.markdown("## Testing Your Private Well")
    st.markdown("If your home relies on a private well, county & state agencies do not test your water on your behalf. The steps below walk you through ordering a test, collecting a valid sample, and interpreting your results.")
    st.markdown('<hr style="border-color:#e2e8f0;margin-bottom:24px;">', unsafe_allow_html=True)

    kit_tab, checklist_tab, howto_tab = st.tabs(["Lab Directory", "Sampling Checklist", "Step-by-Step Guide"])

    with kit_tab:
        st.markdown("### NYS ELAP-Certified Labs — Long Island")
        st.markdown("These laboratories are certified by New York State for PFAS analysis and serve Long Island residents. All will ship HDPE collection bottles to your address on request.")

        county_filter = st.radio("Filter:", ["All", "Nassau", "Suffolk", "Regional"], horizontal=True)
        for lab in ELAP_LABS:
            if county_filter != "All" and county_filter not in lab["county"]:
                continue
            st.markdown(f"""<div class="lab-card">
                <div style="font-weight:600;font-size:0.95rem;color:#0f172a;margin-bottom:4px;">{lab['name']}</div>
                <div style="font-size:0.82rem;color:#64748b;margin-bottom:6px;">{lab['address']} &nbsp;·&nbsp; {lab['phone']}</div>
                <div style="font-size:0.82rem;color:#374151;margin-bottom:4px;"><b>Panel:</b> {lab['pfas_panel']}</div>
                <div style="font-size:0.8rem;color:#64748b;margin-bottom:8px;">{lab['notes']}</div>
                <a href="{lab['url']}" target="_blank" style="font-size:0.82rem;color:#16a34a;font-weight:500;text-decoration:none;">Visit website</a>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div style="margin-top:16px;"><a href="https://www.health.ny.gov/environmental/water/drinking/certified_lab/" target="_blank" style="font-size:0.875rem;color:#0ea5e9;">Search the full NYS ELAP certified lab directory</a></div>', unsafe_allow_html=True)
        st.caption("Verify ELAP certification directly with the lab before submitting samples for regulatory use.")

    with checklist_tab:
        st.markdown("### Before you collect a sample")
        st.markdown("Errors in sample collection are the most common reason results are invalidated. Work through this list before you start.")
        steps = [
            ("Order an HDPE kit from a certified lab", "Contact one of the labs in the directory. Request a PFAS sampling kit for EPA Method 533 or 537.1. They ship pre-preserved HDPE bottles to your address."),
            ("Do not use standard plastic bottles", "PFAS adsorb to ordinary plastic (PET, HDPE). Only use containers supplied by your lab. Glass is acceptable if the lab confirms it."),
            ("Flush the cold water tap for 5 minutes", "This clears stagnant water from the pipes and gives you a sample representative of your well, not pipe residuals."),
            ("Sample before any point-of-use filter", "If you have an under-sink RO system or carbon filter, sample the unfiltered tap. You want to measure the well, not the filter output."),
            ("Fill to the marked line and cap immediately", "Minimize air contact. Cap within seconds of filling."),
            ("Keep at 2–6°C and ship within 24 hours", "Use the ice pack your lab provides. Most labs accept FedEx or UPS overnight."),
            ("Complete the chain-of-custody form", "Note the sample date, time, tap location, and any filters bypassed. Include this form with your shipment."),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            with st.expander(f"{i}. {title}"):
                st.markdown(f'<div style="font-size:0.875rem;color:#374151;line-height:1.7;">{desc}</div>', unsafe_allow_html=True)
        st.info("Sample first thing in the morning, before any water use, for the most representative result.")

    with howto_tab:
        st.markdown("### From order to results")
        st.markdown("""
**1. Choose your panel**

At minimum, order a panel covering the 6 NYS-regulated compounds: PFOA, PFOS, PFBA, PFHxS, PFBS, NEtFOSAA. If you live near the BOMARC site, a military installation, or an industrial facility, a 40-compound panel (EPA Method 533) provides broader coverage.

**2. Order and receive your kit**

Contact a lab from the directory. Provide your address and request PFAS sampling materials. They will send HDPE bottles with a chain-of-custody form. Typical turnaround for kit delivery is 2–5 days. Testing cost ranges from $150–$400 depending on panel size.

**3. Collect your sample**

Follow the checklist on the previous tab. The most common errors: not flushing long enough, sampling downstream of a filter, using the wrong container.

**4. Ship overnight with the ice pack provided**

Most labs specify FedEx or UPS overnight. Results return as a PDF within 10–14 business days showing ng/L concentrations per compound.

**5. Interpret your results**

Compare each compound against NYS MCLs (10 ng/L). If any compound exceeds the limit:
- Contact your county health department (Suffolk DOHS: (631) 853-3000 / Nassau DOHS: (516) 227-9717)
- Use the Report Contamination page to contact your local official on record
- Install an NSF/ANSI 58-certified reverse osmosis system at the point of use as an interim measure
- Do not use unfiltered tap water for drinking or cooking until the situation is assessed
        """)

        st.markdown('<hr style="border-color:#e2e8f0;margin:16px 0;">', unsafe_allow_html=True)
        st.markdown("**Reference links**")
        for name, url in [
            ("NYS ELAP Certified Lab Finder", "https://www.health.ny.gov/environmental/water/drinking/certified_lab/"),
            ("EPA PFAS Testing Methods (537.1 and 533)", "https://www.epa.gov/water-research/pfas-analytical-methods-development-and-sampling-research"),
            ("NSF/ANSI 58 Certified Reverse Osmosis Filters", "https://www.nsf.org/consumer-resources/articles/certified-reverse-osmosis-systems"),
            ("Suffolk County Private Well Guidance", "https://www.suffolkcountyny.gov/Departments/Health-Services/Environmental-Quality/Water-Resources/Private-Well"),
        ]:
            st.markdown(f'<a href="{url}" target="_blank" style="display:block;font-size:0.875rem;color:#0ea5e9;margin-bottom:6px;text-decoration:none;">{name}</a>', unsafe_allow_html=True)
        st.caption("WellGuard provides this information for educational purposes. For health emergencies, contact your county health department.")
