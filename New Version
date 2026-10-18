import urllib.parse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp

from well_data import BOMARC_WELLS, FIELD_CONDITIONS, get_well_data

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="WellGuard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# REFERENCE DATA
# ──────────────────────────────────────────────────────────────

INK = "#12324A"
WATER = "#2F6F8F"
SAND = "#C9B48A"
PAPER = "#F7FAFA"
MUTED = "#5A6B78"
FLAG = "#9A5B12"

# Enforceable limits. NYS: 10 NYCRR Subpart 5-1 (2020). EPA: April 2024 PFAS rule.
# EPA proposed in May 2026 to rescind the PFNA and PFHxS limits; they remain in force until a final rule.
STANDARDS = {
    "PFOA": {"nys": 10.0, "epa": 4.0, "unit": "ng/L"},
    "PFOS": {"nys": 10.0, "epa": 4.0, "unit": "ng/L"},
    "PFNA": {"nys": None, "epa": 10.0, "unit": "ng/L", "note": "EPA proposed rescinding this limit in May 2026"},
    "PFHxS": {"nys": None, "epa": 10.0, "unit": "ng/L", "note": "EPA proposed rescinding this limit in May 2026"},
    "1,4-dioxane": {"nys": 1.0, "epa": None, "unit": "µg/L"},
}


def reference_limit(compound):
    s = STANDARDS.get(compound)
    if not s:
        return None, None
    if s["nys"] is not None:
        return s["nys"], f"NYS limit ({s['nys']:g} {s['unit']})"
    return s["epa"], f"EPA limit ({s['epa']:g} {s['unit']})"


COUNTY_CONTACTS = {
    "Nassau": {
        "agency": "Nassau County Department of Health, Bureau of Water Supply",
        "phone": "(516) 227-9692",
        "url": "https://www.nassaucountyny.gov/2970/Public-Water-Supply",
        "note": "Nassau has 46 public water systems and about 500 private wells. If you get a water bill, your water district is the first place to ask for test results.",
    },
    "Suffolk": {
        "agency": "Suffolk County Department of Health Services, Office of Water Resources",
        "phone": "(631) 852-5810",
        "url": None,
        "note": "This office answers private-well questions and runs no-charge well surveys near known or suspected contamination sites. The Suffolk County Water Authority (631-698-9500) can tell you whether you are on public water.",
    },
    "Queens": {
        "agency": "NYC Department of Environmental Protection, through 311",
        "phone": "311",
        "url": "https://portal.311.nyc.gov",
        "note": "Most of Queens is served by New York City's reservoir system; 311 routes drinking-water questions to DEP.",
    },
}

TOWNS = {
    "Bayville": "Nassau", "Bethpage": "Nassau", "East Norwich": "Nassau", "Farmingdale": "Nassau",
    "Floral Park": "Nassau", "Garden City": "Nassau", "Glen Cove": "Nassau", "Glen Head": "Nassau",
    "Great Neck": "Nassau", "Hempstead": "Nassau", "Hicksville": "Nassau", "Jericho": "Nassau",
    "Locust Valley": "Nassau", "Manhasset": "Nassau", "Massapequa": "Nassau", "Mineola": "Nassau",
    "New Hyde Park": "Nassau", "Old Bethpage": "Nassau", "Oyster Bay": "Nassau", "Plainview": "Nassau",
    "Port Washington": "Nassau", "Roslyn": "Nassau", "Sea Cliff": "Nassau", "Syosset": "Nassau",
    "Westbury": "Nassau", "Woodbury": "Nassau",
    "Calverton": "Suffolk", "Center Moriches": "Suffolk", "Dix Hills": "Suffolk", "East Moriches": "Suffolk",
    "Hampton Bays": "Suffolk", "Huntington": "Suffolk", "Manorville": "Suffolk", "Mastic": "Suffolk",
    "Melville": "Suffolk", "Quogue": "Suffolk", "Riverhead": "Suffolk", "Shirley": "Suffolk",
    "Westhampton": "Suffolk", "Westhampton Beach": "Suffolk",
    "Queens": "Queens",
}

LINKS = {
    "elap": "https://apps.health.ny.gov/pubdoh/applinks/wc/elappublicweb/",
    "nys_private_wells": "https://www.health.ny.gov/environmental/water/drinking/pfasinprivatewells.htm",
    "ucmr5": "https://www.epa.gov/dwucmr/fifth-unregulated-contaminant-monitoring-rule-data-finder",
    "bomarc_scdhs": "https://www.suffolkcountyny.gov/Departments/Health-Services/Environmental-Quality/Westhampton-BOMARC-Missile-BASE-Environmental-Investigation",
    "bomarc_dec": "https://extapps.dec.ny.gov/data/DecDocs/152262/",
    "suozzi_funding": "https://suozzi.house.gov/media/press-releases/congress-sends-funding-12-suozzi-requested-projects-president-signature",
}

PAGES = ["Home", "The issue", "Write to an agency", "Compare treatments", "Get tested", "Questions"]

# ──────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700;800&display=swap');
[data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="collapsedControl"] {{ display: none; }}
.block-container {{ max-width: 1040px; padding-top: 2rem; padding-bottom: 4rem; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
.stApp input, .stApp textarea, .stApp td, .stApp th, .stApp button p,
.stApp figcaption, .wg-mark, .wg-small, .wg-contact {{
    font-family: 'Public Sans', system-ui, sans-serif;
}}
.stApp h1 {{ font-weight: 800; font-size: 2.55rem; line-height: 1.1; letter-spacing: -0.015em; color: {INK}; max-width: 22ch; }}
.stApp h2 {{ font-weight: 700; font-size: 1.6rem; color: {INK}; margin-top: 2.2rem; }}
.stApp h3 {{ font-weight: 700; font-size: 1.15rem; color: {INK}; }}
.stApp p, .stApp li {{ line-height: 1.6; }}
.wg-mark {{ font-weight: 800; font-size: 1.05rem; color: {INK}; letter-spacing: -0.01em; }}
.wg-mark span {{ font-weight: 500; color: {MUTED}; margin-left: .6rem; font-size: .9rem; }}
.wg-lede {{ font-size: 1.15rem; max-width: 62ch; color: {INK}; }}
.wg-small {{ font-size: .88rem; color: {MUTED}; max-width: 70ch; }}
.wg-path h3 {{ margin: 0 0 .2rem 0; }}
.wg-path p {{ margin: 0; color: {MUTED}; max-width: 56ch; }}
.wg-rule {{ border: 0; border-top: 1px solid #D5DFE3; margin: 1.1rem 0; }}
.wg-table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
.wg-table th {{ text-align: left; font-weight: 600; color: {MUTED}; font-size: .85rem; border-bottom: 2px solid {INK}; padding: .45rem .6rem .45rem 0; }}
.wg-table td {{ border-bottom: 1px solid #D5DFE3; padding: .55rem .6rem .55rem 0; color: {INK}; vertical-align: top; }}
.wg-table td.num {{ text-align: right; }}
.wg-above {{ color: {FLAG}; font-weight: 600; }}
.wg-contact {{ border-left: 4px solid {WATER}; padding: .2rem 0 .2rem 1rem; margin: .8rem 0 1.2rem 0; }}
.wg-contact b {{ color: {INK}; }}
.wg-eq {{ font-family: 'Public Sans', system-ui, sans-serif; background: #E8EFF1; padding: .55rem .8rem; margin: .35rem 0; color: {INK}; font-size: .92rem; border-radius: 3px; }}
.wg-note {{ background: #EEF3F4; border-left: 4px solid {SAND}; padding: .8rem 1rem; color: {INK}; margin: 1rem 0; max-width: 75ch; }}
figure.wg-fig {{ margin: 0 0 1.5rem 0; }}
figure.wg-fig figcaption {{ font-size: .82rem; color: {MUTED}; margin-top: .3rem; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# DRAWN FIGURES (inline SVG, no external image hosting)
# ──────────────────────────────────────────────────────────────

AQUIFER_SVG = f"""
<figure class="wg-fig">
<svg viewBox="0 0 960 250" width="100%" role="img" aria-label="Simplified cross-section of Long Island's aquifers with a contaminant plume moving from a source area toward a supply well">
  <rect x="0" y="0" width="960" height="250" fill="{PAPER}"/>
  <path d="M0 40 Q 160 30 320 38 T 640 36 T 960 40 L960 112 L0 112 Z" fill="#E7DCBF"/>
  <rect x="0" y="112" width="960" height="78" fill="#D8C79F"/>
  <rect x="0" y="190" width="960" height="22" fill="#A3AFB4"/>
  <rect x="0" y="212" width="960" height="26" fill="#CDBC93"/>
  <rect x="0" y="238" width="960" height="12" fill="#6E7B82"/>
  <path d="M0 58 Q 240 52 480 60 T 960 62" fill="none" stroke="{WATER}" stroke-width="2" stroke-dasharray="7 5"/>
  <path d="M175 42 C 230 70, 300 90, 420 112 C 520 130, 600 150, 640 160 C 600 172, 470 160, 360 138 C 270 118, 200 92, 165 48 Z" fill="{FLAG}" fill-opacity="0.28" stroke="{FLAG}" stroke-width="1.2"/>
  <rect x="166" y="30" width="22" height="10" fill="{INK}"/>
  <line x1="640" y1="22" x2="640" y2="172" stroke="{INK}" stroke-width="3"/>
  <line x1="640" y1="140" x2="640" y2="172" stroke="{PAPER}" stroke-width="3" stroke-dasharray="3 3"/>
  <rect x="628" y="14" width="24" height="10" fill="{INK}"/>
  <line x1="820" y1="22" x2="820" y2="232" stroke="{INK}" stroke-width="3"/>
  <line x1="820" y1="214" x2="820" y2="232" stroke="{PAPER}" stroke-width="3" stroke-dasharray="3 3"/>
  <rect x="808" y="14" width="24" height="10" fill="{INK}"/>
  <g font-family="Public Sans, system-ui, sans-serif" font-size="13" fill="{INK}">
    <text x="150" y="22">Source area</text>
    <text x="560" y="10">Shallow supply well</text>
    <text x="760" y="10">Deep supply well</text>
    <text x="16" y="54" fill="{WATER}">water table</text>
    <text x="16" y="100">Upper Glacial aquifer</text>
    <text x="16" y="160">Magothy aquifer</text>
    <text x="16" y="205">Raritan clay</text>
    <text x="16" y="230">Lloyd aquifer</text>
    <text x="300" y="150" fill="{FLAG}">contaminant plume</text>
  </g>
</svg>
<figcaption>Long Island's drinking water comes from these stacked sand aquifers. Chemicals that reach the water table travel with groundwater toward wells. Not to scale.</figcaption>
</figure>
"""


def pfoa_svg():
    x0, dx, y_up, y_dn = 60, 52, 70, 100
    pts = [(x0 + i * dx, y_up if i % 2 == 0 else y_dn) for i in range(8)]
    bonds, labels = [], []
    for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
        bonds.append(f'<line x1="{xa}" y1="{ya}" x2="{xb}" y2="{yb}"/>')
    for i, (x, y) in enumerate(pts[:7]):
        sign = -1 if y == y_up else 1
        for ox in (-14, 14):
            fx, fy = x + ox, y + sign * 26
            bonds.append(f'<line x1="{x}" y1="{y}" x2="{fx}" y2="{fy - sign * 6}"/>')
            labels.append(f'<text x="{fx}" y="{fy + (sign * 6 if sign > 0 else 0)}" text-anchor="middle">F</text>')
        if i == 0:
            bonds.append(f'<line x1="{x}" y1="{y}" x2="{x - 26}" y2="{y + 10}"/>')
            labels.append(f'<text x="{x - 34}" y="{y + 18}" text-anchor="middle">F</text>')
    cx, cy = pts[7]
    bonds.append(f'<line x1="{cx - 3}" y1="{cy}" x2="{cx - 3}" y2="{cy + 28}"/>')
    bonds.append(f'<line x1="{cx + 3}" y1="{cy}" x2="{cx + 3}" y2="{cy + 28}"/>')
    labels.append(f'<text x="{cx}" y="{cy + 44}" text-anchor="middle">O</text>')
    bonds.append(f'<line x1="{cx}" y1="{cy}" x2="{cx + 34}" y2="{cy - 18}"/>')
    labels.append(f'<text x="{cx + 50}" y="{cy - 18}" text-anchor="middle">OH</text>')
    return f"""
<figure class="wg-fig">
<svg viewBox="0 0 520 170" width="100%" role="img" aria-label="Skeletal structure of PFOA: an eight-carbon chain with seven fully fluorinated carbons ending in a carboxylic acid group">
  <g stroke="{INK}" stroke-width="2" stroke-linecap="round">{''.join(bonds)}</g>
  <g font-family="Public Sans, system-ui, sans-serif" font-size="15" font-weight="600" fill="{WATER}">{''.join(labels)}</g>
</svg>
<figcaption>PFOA (perfluorooctanoic acid). Every hydrogen on the carbon chain is replaced by fluorine, which is why it resists breakdown.</figcaption>
</figure>
"""


# ──────────────────────────────────────────────────────────────
# NAVIGATION
# ──────────────────────────────────────────────────────────────

if "nav" not in st.session_state:
    st.session_state.nav = "Home"


def go(page):
    st.session_state.nav = page


top_l, top_r = st.columns([1, 3])
with top_l:
    st.markdown('<div class="wg-mark">WellGuard<span>Long Island groundwater</span></div>', unsafe_allow_html=True)
with top_r:
    st.radio("Section", PAGES, key="nav", horizontal=True, label_visibility="collapsed")

page = st.session_state.nav

# ──────────────────────────────────────────────────────────────
# MODEL (unchanged from the original app)
# ──────────────────────────────────────────────────────────────

MW_PFOA = 414.07; T_REF = 298.15; Ea_J = 50_000; R_gas = 8.314
k_PFOA = 1.2e7; k_I = 3e8; k_s = 2e4; k_H2O2_scav = 2.7e7
k_Fe = 2.3e8; k_hv_Fe = 3.5e-4; k_hv_h2o2 = 5e-4


def build_rate_constants(pH, temp_C):
    temp_K = temp_C + 273.15
    arr = np.exp(-Ea_J / R_gas * (1 / temp_K - 1 / T_REF))
    return 63.0 * arr * 10 ** (-(pH - 3.0) * 0.5), 8.4e-6 * arr


def photo_fenton_odes(t, y, Fe_total, k_f, k_r):
    H2O2, OH, PFOA, I, Fe2, Fe3 = [max(v, 0) for v in y]
    dH2O2 = (-k_hv_h2o2 * H2O2 - k_H2O2_scav * OH * H2O2 - k_f * Fe2 * H2O2 - k_r * Fe3 * H2O2)
    dOH = (2 * k_hv_h2o2 * H2O2 + k_f * Fe2 * H2O2 - k_PFOA * OH * PFOA - k_I * OH * I - k_s * OH
           - k_H2O2_scav * OH * H2O2 - k_Fe * Fe2 * OH)
    dPFOA = -k_PFOA * OH * PFOA
    dI = k_PFOA * OH * PFOA - k_I * OH * I
    dFe2 = (-k_f * Fe2 * H2O2 + k_r * Fe3 * H2O2 - k_Fe * Fe2 * OH + k_hv_Fe * Fe3)
    return [dH2O2, dOH, dPFOA, dI, dFe2, -dFe2]


def simulate_all_contaminants(well_info, pH, temp, H2O2, Fe, costs):
    results = {}
    for contaminant, data in well_info["compounds"].items():
        level_ng_l = data["level_ng_l"]
        gac_pct = data["gac_removal"] * 100
        PFOA_0 = level_ng_l * 1e-9 / MW_PFOA
        H2O2_mol = H2O2 * 1e-3
        Fe_mol = Fe * 1e-6
        k_f, k_r = build_rate_constants(pH, temp)
        t_eval = np.linspace(0, 7200, 3600)
        sol = solve_ivp(
            fun=lambda t, y: photo_fenton_odes(t, y, Fe_mol, k_f, k_r),
            t_span=(0, 7200), y0=[H2O2_mol, 0.0, PFOA_0, 0.0, Fe_mol, 0.0],
            method="BDF", t_eval=t_eval, rtol=1e-8, atol=1e-12,
        )
        t_min = sol.t / 60
        f_rem = (1 - np.maximum(sol.y[2], 0) / PFOA_0) * 100
        idx_90 = np.where(f_rem >= 90)[0]
        t90 = t_min[idx_90[0]] if len(idx_90) > 0 else None
        fc = H2O2_mol * 1000 * costs["H2O2"] + Fe_mol * 1000 * costs["Fe"] + (t90 or 120.0) * costs["UV"]
        gc = (costs["gac_lb"] + costs["gac_disposal"]) * 0.025
        pfoa_rem_ug = (PFOA_0 * f_rem[-1] / 100) * MW_PFOA * 1e6
        gac_rem_ug = (PFOA_0 * gac_pct / 100) * MW_PFOA * 1e6
        results[contaminant] = {
            "level_ng_l": level_ng_l,
            "fenton_removal": f_rem[-1],
            "fenton_t90": t90,
            "fenton_cost": fc,
            "fenton_cost_per_ug": fc / pfoa_rem_ug if pfoa_rem_ug > 0 else 9999,
            "gac_removal": gac_pct,
            "gac_cost": gc,
            "gac_cost_per_ug": gc / gac_rem_ug if gac_rem_ug > 0 else 9999,
            "gac_fails": gac_pct < 50,
            "fenton_wins_removal": f_rem[-1] > gac_pct,
            "fenton_wins_cost": fc < gc,
            "t_min": t_min,
            "fenton_removal_curve": f_rem,
        }
    return results


def well_rows_html(wells):
    rows = []
    for well in wells:
        for compound, d in well["compounds"].items():
            lim, lim_label = reference_limit(compound)
            lv = d["level_ng_l"]
            if lim:
                ratio = lv / lim
                status = f'<span class="wg-above">{ratio:.1f}× above</span>' if ratio > 1 else f"{ratio:.1f}× (at or below)"
            else:
                lim_label, status = "No enforceable limit", "—"
            rows.append(
                f"<tr><td>{well['location']}</td><td>{compound}</td><td class='num'>{lv:g}</td>"
                f"<td>{lim_label}</td><td>{status}</td></tr>"
            )
    return (
        "<table class='wg-table'><thead><tr><th>Sampling location</th><th>Compound</th>"
        "<th style='text-align:right'>Measured (ng/L)</th><th>Compared with</th><th>Result</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def contact_block(county):
    c = COUNTY_CONTACTS[county]
    link = f'<br><a href="{c["url"]}" target="_blank">Agency web page</a>' if c["url"] else ""
    st.markdown(
        f'<div class="wg-contact"><b>{c["agency"]}</b><br>Phone: {c["phone"]}{link}'
        f'<p class="wg-small" style="margin-top:.4rem">{c["note"]}</p></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════
if page == "Home":
    st.markdown(AQUIFER_SVG, unsafe_allow_html=True)
    st.markdown("# What is in the water under Long Island, and who is checking?")
    st.markdown(
        '<p class="wg-lede">Public water suppliers test on a schedule. Private wells, and chemicals without '
        "a legal limit, often go unchecked. WellGuard shows what government sampling has found, tells you "
        "who to contact, and helps you get your own water tested.</p>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="wg-rule">', unsafe_allow_html=True)
    for title, desc, target, label in [
        ("Contact the agency that oversees your water", "Pick your town and get a ready-to-send letter to the right county office.", "Write to an agency", "Write a letter"),
        ("Get your own water tested", "Find a state-certified lab and learn how to take a sample that will hold up.", "Get tested", "Testing steps"),
        ("See what sampling found", "Measured PFAS levels near the former BOMARC site in Westhampton, compared with legal limits.", "The issue", "See the data"),
        ("Compare treatment options", "Run the model comparing carbon filtration and photo-Fenton oxidation for a sampled well.", "Compare treatments", "Open the model"),
    ]:
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        with c1:
            st.markdown(f'<div class="wg-path"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
        with c2:
            st.button(label, key=f"home_{target}", on_click=go, args=(target,), width="stretch")
        st.markdown('<hr class="wg-rule">', unsafe_allow_html=True)
    st.caption("Data: Suffolk County Department of Health Services BOMARC investigations; NYS DEC site 152262; US EPA.")

# ══════════════════════════════════════════════════════════════
# THE ISSUE
# ══════════════════════════════════════════════════════════════
elif page == "The issue":
    st.markdown("# PFAS in Long Island groundwater")

    st.markdown("## What PFAS are")
    a, b = st.columns([2, 3])
    with a:
        st.markdown(pfoa_svg(), unsafe_allow_html=True)
    with b:
        st.markdown(
            "Per- and polyfluoroalkyl substances (PFAS) are synthetic chemicals used since the 1950s in "
            "non-stick coatings, stain- and water-repellent fabrics, and firefighting foam. The carbon–fluorine "
            "bonds that make them useful also make the fully fluorinated ones extremely persistent in soil, "
            "water, and the body. You cannot see, smell, or taste them at the levels that matter."
        )

    st.markdown("## Why they matter for health")
    st.markdown(
        "- In 2023 the WHO's cancer agency (IARC) classified **PFOA** as carcinogenic to humans (Group 1), with "
        "limited evidence in people for kidney and testicular cancer. **PFOS** was classified as possibly carcinogenic (Group 2B).\n"
        "- Long-term exposure to PFOA and PFOS is also associated with higher cholesterol, effects on the immune "
        "response to vaccines, and lower birth weight.\n"
        "- Drinking and cooking are the main exposure routes from water; bathing is not a significant one."
    )

    st.markdown("## The Westhampton BOMARC site")
    st.markdown(
        "The former BOMARC missile base on Old Country Road in Westhampton (Suffolk County) housed surface-to-air "
        "missiles until 1969. After PFOS turned up in a public supply well just south of the property, the "
        "Suffolk County Department of Health Services surveyed private wells starting in 2017 and installed 28 "
        "groundwater profile wells on the site. PFAS were found in 26 of them. In December 2019 NYS DEC listed the "
        "property as a potential inactive hazardous waste site (site 152262). Investigators have not yet pinned "
        "down the source; several other nearby sites may contribute."
    )
    st.markdown(
        '<p class="wg-small">These are groundwater monitoring results, not tap water. The county reports that the '
        "public supply in the area meets drinking water standards. The risk falls mainly on homes with private wells.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(well_rows_html(BOMARC_WELLS.values()), unsafe_allow_html=True)

    st.markdown("## Closer to home: north and central Nassau")
    st.markdown(
        "Nassau water districts face a second problem: **1,4-dioxane**, an industrial solvent stabilizer found in "
        "Long Island wells at some of the highest levels in the country, including in the Navy–Grumman plume under "
        "Bethpage. New York set the nation's first limit for it (1 µg/L) in 2020. Carbon filters barely remove it, "
        "so districts use UV advanced oxidation. In January 2026 Congress funded projects requested by Rep. Tom "
        "Suozzi for 1,4-dioxane treatment in the Bethpage and Westbury water districts."
    )

    st.markdown("## How it is treated")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("### Granular activated carbon (GAC)")
        st.markdown(
            "The standard for PFAS today. Water passes through carbon that holds onto PFAS. It works well for "
            "longer-chain compounds such as PFOS, PFNA, and PFOA when the carbon is replaced on schedule, and "
            "poorly for short-chain ones such as PFBA, PFBS, and GenX, which break through sooner. PFAS are "
            "captured, not destroyed, so spent carbon must be handled and disposed of or regenerated."
        )
    with t2:
        st.markdown("### Photo-Fenton oxidation")
        st.markdown(
            "Iron, hydrogen peroxide, and UV light generate hydroxyl radicals, which destroy many organic "
            "chemicals. Fully fluorinated PFAS react with hydroxyl radicals very slowly, so this is a research "
            "question for PFAS rather than an established treatment. The comparison tool on this site models it "
            "under stated assumptions."
        )

    st.markdown("## Primary sources")
    for name, key in [
        ("Suffolk County DHS: BOMARC environmental investigation", "bomarc_scdhs"),
        ("NYS DEC: site 152262 documents and reports", "bomarc_dec"),
        ("US EPA: UCMR 5 Data Finder (PFAS results for public water systems)", "ucmr5"),
        ("Rep. Suozzi: January 2026 water infrastructure funding", "suozzi_funding"),
    ]:
        st.markdown(f"- [{name}]({LINKS[key]})")

# ══════════════════════════════════════════════════════════════
# WRITE TO AN AGENCY
# ══════════════════════════════════════════════════════════════
elif page == "Write to an agency":
    st.markdown("# Write to the agency that oversees your water")

    town = st.selectbox("Your town", list(TOWNS.keys()), index=list(TOWNS.keys()).index("Plainview"))
    county = TOWNS[town]
    contact_block(county)

    intent = st.radio(
        "What do you want to do?",
        ["Ask what testing has found near me", "Report a test result from my own water", "Ask about contamination near an industrial or military site"],
    )

    your_name = st.text_input("Your name", placeholder="First and last name")
    signer = your_name.strip() or "[Your name]"
    agency = COUNTY_CONTACTS[county]["agency"]

    if intent == "Report a test result from my own water":
        c1, c2, c3 = st.columns(3)
        with c1:
            contaminant = st.selectbox("Compound", list(STANDARDS.keys()) + ["Other PFAS"])
        unit = STANDARDS.get(contaminant, {}).get("unit", "ng/L")
        with c2:
            level = st.number_input(f"Measured level ({unit})", min_value=0.0, value=0.0, step=0.1)
        with c3:
            source = st.selectbox("Water source", ["Private well", "Public water (I get a water bill)"])
        lim, lim_label = reference_limit(contaminant)
        comparison = ""
        if lim and level > 0:
            ratio = level / lim
            comparison = f" That is {ratio:.1f} times the {lim_label}." if ratio > 1 else f" That is at or below the {lim_label}."
        subject = f"Test result for {contaminant} in my drinking water, {town}"
        body = (
            f"To the {agency}:\n\n"
            f"I live in {town} and my home is served by {'a private well' if source == 'Private well' else 'public water'}. "
            f"A test of my tap water by a NYS ELAP-certified laboratory measured {contaminant} at {level:g} {unit}.{comparison}\n\n"
            "I am asking for:\n"
            "1. Guidance on what my household should do now, including whether to stop drinking the water\n"
            "2. Whether the county will confirm the result with its own sample\n"
            "3. Any other sampling results within a mile of my home\n"
            "4. Who I should contact for updates\n\n"
            "I can share the laboratory report on request.\n\n"
            f"Thank you,\n{signer}\n{town}, NY"
        )
    elif intent == "Ask about contamination near an industrial or military site":
        site = st.text_input("Site name or location", placeholder="For example, the former BOMARC site on Old Country Road")
        site_txt = site.strip() or "[site name or location]"
        subject = f"Question about groundwater near {site_txt}, {town}"
        body = (
            f"To the {agency}:\n\n"
            f"I live in {town} and I am concerned about groundwater contamination from {site_txt}.\n\n"
            "I am asking for:\n"
            "1. Any groundwater or private-well sampling results near the site\n"
            "2. Whether homes near the site have been offered testing\n"
            "3. Which agency is leading the investigation and its timeline\n"
            "4. How residents will be notified of new results\n\n"
            f"Thank you,\n{signer}\n{town}, NY"
        )
    else:
        subject = f"Question about PFAS and 1,4-dioxane testing near {town}"
        body = (
            f"To the {agency}:\n\n"
            f"I live in {town} and would like to understand what testing has found in the water that serves my area.\n\n"
            "I am asking for:\n"
            "1. Recent PFAS and 1,4-dioxane results for the water system or wells serving my address\n"
            "2. What treatment is in place or planned\n"
            "3. Whether private-well owners nearby can get testing, and at what cost\n"
            "4. Who I should contact for updates\n\n"
            f"Thank you,\n{signer}\n{town}, NY"
        )

    st.markdown("### Your letter")
    st.caption("Use the copy button in the corner of the box.")
    st.code(f"Subject: {subject}\n\n{body}", language=None, wrap_lines=True)
    mailto = "mailto:?" + urllib.parse.urlencode({"subject": subject, "body": body}, quote_via=urllib.parse.quote)
    st.link_button("Open in my email app", mailto)
    st.markdown(
        '<p class="wg-small">The email opens without an address. Ask the office for the right email when you call, or '
        "send the letter by mail. If you are on public water, send a copy to your water district too; its name is on your bill.</p>",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════
# COMPARE TREATMENTS
# ══════════════════════════════════════════════════════════════
elif page == "Compare treatments":
    st.markdown("# Compare treatment options")
    sub = st.radio("View", ["Run the model", "How the model works"], horizontal=True, label_visibility="collapsed")

    if sub == "How the model works":
        st.markdown("## What is simulated")
        st.markdown(
            "A six-species system of ordinary differential equations, integrated over two hours with SciPy's "
            "stiff BDF solver. Carbon filtration is not simulated; it uses a fixed removal fraction per compound "
            "taken from the well data file."
        )
        for sym, desc in [
            ("H₂O₂", "hydrogen peroxide, the oxidant supply"),
            ("·OH", "hydroxyl radical, the reactive species"),
            ("PFAS", "target compound"),
            ("I", "lumped intermediate products"),
            ("Fe²⁺", "ferrous iron, the Fenton catalyst"),
            ("Fe³⁺", "ferric iron, returned to Fe²⁺ by UV"),
        ]:
            st.markdown(f"- **{sym}**: {desc}")

        st.markdown("## Rate equations")
        for lhs, rhs in [
            ("d[H₂O₂]/dt", "= −k_hv[H₂O₂] − k_scav[·OH][H₂O₂] − k_f[Fe²⁺][H₂O₂] − k_r[Fe³⁺][H₂O₂]"),
            ("d[·OH]/dt", "= 2k_hv[H₂O₂] + k_f[Fe²⁺][H₂O₂] − k_PFAS[·OH][PFAS] − k_I[·OH][I] − k_s[·OH] − k_scav[·OH][H₂O₂] − k_Fe[Fe²⁺][·OH]"),
            ("d[PFAS]/dt", "= −k_PFAS[·OH][PFAS]"),
            ("d[I]/dt", "= k_PFAS[·OH][PFAS] − k_I[·OH][I]"),
            ("d[Fe²⁺]/dt", "= −k_f[Fe²⁺][H₂O₂] + k_r[Fe³⁺][H₂O₂] − k_Fe[Fe²⁺][·OH] + k_hv,Fe[Fe³⁺]"),
            ("d[Fe³⁺]/dt", "= −d[Fe²⁺]/dt   (total iron conserved)"),
        ]:
            st.markdown(f'<div class="wg-eq"><b>{lhs}</b> {rhs}</div>', unsafe_allow_html=True)

        st.markdown("## Corrections applied")
        st.markdown(
            "- **Temperature:** k_f is adjusted from 25 °C values with an Arrhenius factor, Ea = 50 kJ/mol.\n"
            "- **pH:** k_f is reduced above pH 3 with an empirical factor, because Fe³⁺ precipitates as hydroxide at higher pH."
        )

        st.markdown("## Rate constants")
        st.dataframe(
            [
                {"Symbol": "k_PFAS", "Value": "1.2 × 10⁷ L/(mol·s)", "Reaction": "·OH + PFOA", "Source": "Hori et al. 2004 (citation under review)"},
                {"Symbol": "k_I", "Value": "3.0 × 10⁸ L/(mol·s)", "Reaction": "·OH + intermediates (lumped)", "Source": "estimated"},
                {"Symbol": "k_s", "Value": "2.0 × 10⁴ s⁻¹", "Reaction": "background ·OH scavenging", "Source": "field estimate"},
                {"Symbol": "k_scav", "Value": "2.7 × 10⁷ L/(mol·s)", "Reaction": "·OH + H₂O₂", "Source": "Buxton et al. 1988"},
                {"Symbol": "k_Fe", "Value": "2.3 × 10⁸ L/(mol·s)", "Reaction": "Fe²⁺ + ·OH", "Source": "Walling 1975"},
                {"Symbol": "k_hv", "Value": "5 × 10⁻⁴ s⁻¹", "Reaction": "H₂O₂ photolysis", "Source": "UV-A lamp estimate"},
                {"Symbol": "k_hv,Fe", "Value": "3.5 × 10⁻⁴ s⁻¹", "Reaction": "Fe³⁺ photoreduction", "Source": "De Laat & Gallard 1999"},
            ],
            hide_index=True,
        )

        st.markdown("## Known limitations")
        st.markdown(
            "- Published upper bounds for the ·OH + PFOA rate constant are about 10⁵ L/(mol·s), roughly 100 times lower "
            "than the value used here. Modeled photo-Fenton removal is therefore likely overstated.\n"
            "- The model runs at the pH you set. Real well water is near neutral, and acidifying to pH 3 and neutralizing "
            "afterward adds cost and produces iron sludge that the cost estimate does not include.\n"
            "- Carbon filtration performance depends on how much water has passed through the media; a single fixed "
            "removal percentage per compound is a simplification."
        )

    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            well_name = st.selectbox("Sampled well", list(BOMARC_WELLS.keys()))
            well_info = get_well_data(well_name)
        with c2:
            user_pH = st.slider("pH", 3.0, 8.0, float(FIELD_CONDITIONS["pH"]), 0.1)
        with c3:
            user_temp = st.slider("Water temperature (°C)", 5.0, 35.0, float(FIELD_CONDITIONS["temp_C"]), 0.5)

        with st.expander("Reagent doses and unit costs"):
            a1, a2, a3 = st.columns(3)
            with a1:
                user_H2O2 = st.slider("H₂O₂ dose (mM)", 1.0, 100.0, 10.0, 1.0)
                user_Fe = st.slider("Fe²⁺ dose (µM)", 0.1, 50.0, 5.0, 0.1)
            with a2:
                cost_H2O2 = st.number_input("H₂O₂ ($/mmol)", value=0.014, format="%.4f")
                cost_Fe = st.number_input("Fe²⁺ ($/mmol)", value=0.0557, format="%.4f")
                cost_UV = st.number_input("UV ($/min)", value=0.008, format="%.4f")
            with a3:
                gac_lb = st.number_input("GAC media ($/lb)", value=2.25)
                gac_disp = st.number_input("GAC disposal ($/lb)", value=1.50)

        run = st.button("Run comparison", type="primary")
        st.markdown(
            '<p class="wg-small">Exploratory model. Read "How the model works" for assumptions and limitations before '
            "drawing conclusions.</p>",
            unsafe_allow_html=True,
        )

        if run:
            costs = {"H2O2": cost_H2O2, "Fe": cost_Fe, "UV": cost_UV, "gac_lb": gac_lb, "gac_disposal": gac_disp}
            with st.spinner("Solving the reaction model"):
                results = simulate_all_contaminants(well_info, user_pH, user_temp, user_H2O2, user_Fe, costs)

            st.markdown(f"## Results for {well_name}")

            rows = []
            for compound, d in results.items():
                lim, lim_label = reference_limit(compound)
                rows.append({
                    "Compound": compound,
                    "Measured (ng/L)": round(d["level_ng_l"], 1),
                    "Compared with": f"{d['level_ng_l'] / lim:.1f}× {lim_label}" if lim else "No enforceable limit",
                    "Photo-Fenton removal": f"{d['fenton_removal']:.0f}%",
                    "GAC removal (assumed)": f"{d['gac_removal']:.0f}%",
                    "Photo-Fenton cost": f"${d['fenton_cost']:.4f}",
                    "GAC cost": f"${d['gac_cost']:.4f}",
                    "Higher modeled removal": "Photo-Fenton" if d["fenton_wins_removal"] else "GAC",
                    "Lower modeled cost": "Photo-Fenton" if d["fenton_wins_cost"] else "GAC",
                })
            st.dataframe(rows, hide_index=True)

            low_gac = [c for c, d in results.items() if d["gac_fails"]]
            if low_gac:
                st.markdown(
                    f'<div class="wg-note">The well data assumes GAC removes less than half of '
                    f'{", ".join(low_gac)}. Check that value against treatment-plant data before relying on it.</div>',
                    unsafe_allow_html=True,
                )

            names = list(results.keys())
            fig, ax = plt.subplots(figsize=(9, 3.6))
            fig.patch.set_facecolor(PAPER)
            ax.set_facecolor(PAPER)
            x = np.arange(len(names)); w = 0.36
            b1 = ax.bar(x - w / 2, [results[n]["fenton_removal"] for n in names], w, label="Photo-Fenton (modeled)", color=WATER)
            b2 = ax.bar(x + w / 2, [results[n]["gac_removal"] for n in names], w, label="GAC (assumed)", color=SAND)
            ax.set_ylabel("Removal (%)", color=INK)
            ax.set_xticks(x); ax.set_xticklabels(names, color=INK)
            ax.set_ylim(0, 110)
            ax.tick_params(colors=INK, length=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.spines["bottom"].set_color(INK)
            ax.yaxis.grid(True, color="#D5DFE3"); ax.set_axisbelow(True)
            for bars in (b1, b2):
                for bar in bars:
                    h = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 2, f"{h:.0f}", ha="center", va="bottom", color=INK, fontsize=9)
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.18), ncol=2, labelcolor=INK)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            with st.expander("Conditions used"):
                arr = np.exp(-Ea_J / R_gas * (1 / (user_temp + 273.15) - 1 / T_REF))
                st.markdown(
                    f"pH {user_pH}, {user_temp} °C, Arrhenius factor {arr:.3f}, H₂O₂ {user_H2O2} mM, Fe²⁺ {user_Fe} µM, "
                    "k_PFAS 1.2 × 10⁷ L/(mol·s). GAC removal fractions come from the well data file."
                )

# ══════════════════════════════════════════════════════════════
# GET TESTED
# ══════════════════════════════════════════════════════════════
elif page == "Get tested":
    st.markdown("# Get your water tested")
    st.markdown(
        '<p class="wg-lede">Private wells are not covered by the federal Safe Drinking Water Act, and New York does '
        "not require owners to test them for PFAS. If you are on a private well, testing is up to you.</p>",
        unsafe_allow_html=True,
    )

    tab_lab, tab_sample, tab_results = st.tabs(["Find a lab", "Take the sample", "Read your results"])

    with tab_lab:
        st.markdown("## Start with the county")
        st.markdown(
            "Suffolk County samples private wells at no charge during surveys near known or suspected contamination "
            "sites. Call before paying for a test."
        )
        contact_block("Suffolk")
        contact_block("Nassau")
        st.markdown("## Use a state-certified lab")
        st.markdown(
            "Results only count if the lab holds NYS Environmental Laboratory Approval Program (ELAP) certification "
            "for PFAS in drinking water. Search the state directory, choose the potable water category, and ask the lab for "
            "**EPA Method 533** (25 PFAS) or **EPA Method 537.1** (18 PFAS). If you are near Bethpage or another solvent "
            "plume, ask about 1,4-dioxane as well. Commercial PFAS tests can cost more than $400 per sample; compare a few labs."
        )
        st.link_button("Search NYS ELAP-certified labs", LINKS["elap"])

    with tab_sample:
        st.markdown("## Before you start")
        st.markdown(
            "PFAS show up at parts per trillion, so traces from your hands, clothes, or the wrong bottle can ruin a sample. "
            "Follow the lab's instructions first; these steps match the EPA and state guidance they are based on."
        )
        steps = [
            ("Use only the lab's bottles", "Methods 533 and 537.1 use polypropylene bottles with preservative already inside. Never use glass, because PFAS stick to it, and never substitute your own container."),
            ("Keep the field blank", "The kit includes a bottle of PFAS-free water and an empty bottle. Open and pour it at the tap when you sample; it shows whether anything contaminated the sample in the air or on your hands."),
            ("Prepare your hands and clothes", "Wear the nitrile gloves from the kit. Skip waterproof or stain-resistant clothing, sunscreen, and lotion that day. Wash your hands before you start."),
            ("Pick the right tap", "Use a cold-water tap before any filter or softener, ideally the one closest to the well. Remove the aerator and avoid taps wrapped in Teflon tape."),
            ("Run the water", "Let cold water run until the temperature stops changing, usually about five minutes, then lower the flow so it doesn't splash."),
            ("Fill and cap", "Don't touch the inside of the bottle or cap, and don't set the cap down. Fill to the neck or the marked line, cap it, and gently shake to dissolve the preservative."),
            ("Chill and ship", "Keep the samples on regular ice and ship them the same day. The lab must receive them cold, at or below 10 °C. Fill in the chain-of-custody form with the date, time, and tap."),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f"**{i}. {title}**  \n{desc}")

    with tab_results:
        st.markdown("## Comparing your numbers")
        st.markdown(
            "<table class='wg-table'><thead><tr><th>Compound</th><th>New York limit</th><th>Federal limit</th></tr></thead><tbody>"
            "<tr><td>PFOA</td><td>10 ng/L</td><td>4 ng/L</td></tr>"
            "<tr><td>PFOS</td><td>10 ng/L</td><td>4 ng/L</td></tr>"
            "<tr><td>PFNA</td><td>none</td><td>10 ng/L (proposed for rescission)</td></tr>"
            "<tr><td>PFHxS</td><td>none</td><td>10 ng/L (proposed for rescission)</td></tr>"
            "<tr><td>1,4-dioxane</td><td>1 µg/L</td><td>none</td></tr>"
            "</tbody></table>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="wg-small">1 ng/L is one part per trillion. These limits legally apply to public water systems, not '
            "private wells, but they are the best yardstick available. Federal water systems have until 2029 to meet the "
            "4 ng/L limits; EPA has proposed extending that to 2031.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("## If a result is above a limit")
        st.markdown(
            "1. Use bottled water or a certified filter for drinking and cooking. Showering is not a significant exposure route.\n"
            "2. Call your county health department and share the lab report.\n"
            "3. Look for a filter certified to NSF/ANSI 53 or 58 with a PFOA/PFOS reduction claim, and replace cartridges on schedule.\n"
            "4. Use the letter tool on this site to put your request in writing."
        )
        st.markdown(f"[NYS Department of Health: PFAS in private wells]({LINKS['nys_private_wells']})")

    st.caption("WellGuard is an educational tool, not medical or legal advice. For urgent concerns, call your county health department.")

# ══════════════════════════════════════════════════════════════
# QUESTIONS
# ══════════════════════════════════════════════════════════════
elif page == "Questions":
    st.markdown("# Questions")
    faqs = [
        ("Does the government test my private well?",
         "Not routinely. Private wells fall outside the federal Safe Drinking Water Act, and New York does not require "
         "owners to test for PFAS. Suffolk County does sample private wells for free during surveys near known or suspected "
         "contamination sites, and its Office of Water Resources (631-852-5810) answers private-well questions. Nassau has "
         "only about 500 private wells; call the Bureau of Water Supply at 516-227-9692."),
        ("I'm on public water. Is it tested?",
         "Yes. Public systems must test for regulated contaminants and publish an Annual Water Quality Report. Many Long "
         "Island systems also tested for 29 PFAS under EPA's UCMR 5 program from 2023 to 2025, including compounds New York "
         "does not regulate. You can look up your system in EPA's UCMR 5 Data Finder."),
        ("What are the legal limits?",
         "New York: 10 ng/L each for PFOA and PFOS, and 1 µg/L for 1,4-dioxane (since 2020). Federal: 4 ng/L each for PFOA and "
         "PFOS, plus 10 ng/L for PFNA, PFHxS, and GenX. In May 2026 EPA proposed rescinding the PFNA, PFHxS, and GenX limits and "
         "giving systems until 2031 to meet the PFOA and PFOS limits. For PFOA and PFOS, the federal limit is the stricter one."),
        ("Why doesn't carbon filtration remove everything?",
         "Carbon holds onto PFAS mainly through their water-repelling fluorinated tails. Longer chains (PFOS, PFNA, PFOA) "
         "stick well; short-chain PFAS such as PFBA, PFBS, and GenX break through much sooner. 1,4-dioxane barely sticks at "
         "all, which is why Long Island districts treat it with UV advanced oxidation instead."),
        ("What is photo-Fenton oxidation?",
         "A process that uses iron, hydrogen peroxide, and UV light to make hydroxyl radicals. Those radicals destroy "
         "1,4-dioxane and many other organic chemicals, but react very slowly with fully fluorinated PFAS. Whether it can "
         "be made practical for PFAS is an open research question; WellGuard's model explores it under stated assumptions."),
        ("What is the BOMARC site?",
         "A former Air Force surface-to-air missile base on Old Country Road in Westhampton, active until 1969. Suffolk County "
         "found PFAS in 26 of 28 groundwater wells installed there, and NYS DEC lists it as a potential inactive hazardous "
         "waste site. The source of the PFAS has not been determined."),
        ("Can I take legal action?",
         "Possibly, if contamination is traced to a responsible party. Many Long Island water suppliers have sued PFAS and "
         "1,4-dioxane manufacturers. WellGuard can't give legal advice; an environmental attorney can."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.markdown(a)
    st.caption("Sources: NYS DOH, Suffolk County DHS, Nassau County DOH, US EPA, IARC Monographs vol. 135.")
