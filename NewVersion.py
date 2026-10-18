import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from well_data import BOMARC_WELLS, FIELD_CONDITIONS, get_well_data

st.set_page_config(
    page_title="WellGuard — Water Quality Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── TOWN OFFICIALS ──
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

# ── NYS ELAP LABS ──
ELAP_LABS = [
    {
        "name": "TestAmerica Laboratories (Pace Analytical)",
        "address": "100 Woodbury Rd, Woodbury, NY 11797",
        "county": "Nassau",
        "phone": "(516) 364-8040",
        "url": "https://www.pacelabs.com",
        "pfas_panel": "EPA Method 533 / 537.1",
        "notes": "Ships HDPE sample bottles on request. Accepts private well submissions.",
    },
    {
        "name": "Suffolk County Department of Health Services — Public Health Lab",
        "address": "360 Yaphank Ave, Yaphank, NY 11980",
        "county": "Suffolk",
        "phone": "(631) 852-5999",
        "url": "https://www.suffolkcountyny.gov",
        "pfas_panel": "EPA Method 537.1",
        "notes": "County facility. Local residents may qualify for subsidized testing programs.",
    },
    {
        "name": "Chemtech-Ford Laboratories",
        "address": "1750 New Highway, Farmingdale, NY 11735",
        "county": "Nassau/Suffolk border",
        "phone": "(631) 249-8088",
        "url": "https://www.chemtechford.com",
        "pfas_panel": "EPA Method 533 / 537.1",
        "notes": "Regional laboratory providing private well testing services and kits.",
    },
]

# Sidebar Navigation
st.sidebar.title("WellGuard")
st.sidebar.caption("Groundwater Analytics Platform")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "The Issue", "Report Contamination", "Compare Treatments", "Testing Kits"]
)

st.sidebar.divider()
st.sidebar.caption("Data: BOMARC Site Investigations (2020-2024)\nSuffolk County Dept. of Health Services\nUSGS Finkelstein et al. 2025")

# ── KINETICS ──
MW_PFOA = 414.07
T_REF = 298.15
Ea_J = 50000
R_gas = 8.314
k_PFOA = 1e7
k_I = 3e8
k_s = 2e4
k_H2O2_scav = 2.7e7
k_Fe = 2.3e8
k_hv_Fe = 3.5e-4
k_hv_h2o2 = 5e-4

def build_rate_constants(pH, temp_C):
    temp_K = temp_C + 273.15
    arr = np.exp(-Ea_J / R_gas * (1/temp_K - 1/T_REF))
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
        gac_pct = data["gac_removal"] * 100
        PFOA_0 = level_ng_l * 1e-9 / MW_PFOA
        H2O2_mol = H2O2 * 1e-3
        Fe_mol = Fe * 1e-6
        k_f, k_r = build_rate_constants(pH, temp)
        t_eval = np.linspace(0, 7200, 3600)
        sol = solve_ivp(
            fun=lambda t, y: photo_fenton_odes(t, y, Fe_mol, k_f, k_r),
            t_span=(0,7200), y0=[H2O2_mol, 0.0, PFOA_0, 0.0, Fe_mol, 0.0],
            method='BDF', t_eval=t_eval, rtol=1e-8, atol=1e-12
        )
        t_min = sol.t / 60
        f_rem = (1 - np.maximum(sol.y[2], 0) / PFOA_0) * 100
        idx_90 = np.where(f_rem >= 90)[0]
        t90 = t_min[idx_90[0]] if len(idx_90) > 0 else None
        fc = H2O2_mol*1000*costs["H2O2"] + Fe_mol*1000*costs["Fe"] + (t90 or 120.0)*costs["UV"]
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

# ════════════════════════════════════════
# HOME
# ════════════════════════════════════════
if page == "Home":
    st.title("WellGuard")
    st.subheader("PFAS Contamination Self-Reporting and Treatment Modeling Tool")
    st.write(
        "Real contamination data. Local officials. Actionable steps. "
        "Built for Long Island residents affected by PFAS in their groundwater."
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**Report Contamination**")
        st.write("Connect directly with local county officials.")
    with col2:
        st.markdown("**Compare Treatments**")
        st.write("Run simulation models for well treatment options.")
    with col3:
        st.markdown("**The Issue**")
        st.write("Learn background details on Long Island water safety.")
    with col4:
        st.markdown("**Testing Kits**")
        st.write("Find certified labs and sampling instructions.")

    st.divider()
    st.caption("Data: BOMARC Site Investigations 2020–2024, Suffolk County DOHS | USGS Finkelstein et al. 2025")

# ════════════════════════════════════════
# THE ISSUE
# ════════════════════════════════════════
elif page == "The Issue":
    st.title("What Is Happening to Long Island's Water?")
    st.divider()

    st.subheader("Per- and Polyfluoroalkyl Substances (PFAS)")
    st.write(
        "PFAS are man-made chemicals used since the 1950s in non-stick cookware, "
        "waterproof clothing, and military firefighting foam (AFFF). Because the carbon-fluorine bond "
        "is exceptionally strong, PFAS do not break down easily in soil or water, earning them the name "
        "'forever chemicals.'"
    )
    st.divider()

    st.subheader("Health Impacts")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Cancer Risk**")
        st.write("Linked to kidney, testicular, and thyroid cancers at prolonged exposures above safe limits.")
    with col2:
        st.markdown("**Child Development**")
        st.write("Disrupts hormones and immune development, posing risks to vulnerable populations.")
    with col3:
        st.markdown("**Heart and Liver**")
        st.write("Associated with elevated cholesterol and liver enzyme abnormalities at chronic doses.")

    st.divider()
    st.subheader("The Long Island Context")
    st.write(
        "Long Island relies on a sole-source aquifer system with sandy glacial soil that allows "
        "contaminants to reach groundwater rapidly. Historical use of AFFF at military installations "
        "like the BOMARC missile site in Westhampton has resulted in localized groundwater contamination."
    )

# ════════════════════════════════════════
# REPORT CONTAMINATION
# ════════════════════════════════════════
elif page == "Report Contamination":
    st.title("Connect with Local Officials")
    st.divider()

    intent = st.radio("Select your objective:", [
        "I want to learn more and connect with a local official",
        "I want to report contamination"
    ])
    st.divider()

    selected_town = st.selectbox("Select your town:", list(TOWN_OFFICIALS.keys()))
    official = TOWN_OFFICIALS[selected_town]
    
    st.write(f"**County:** {official['county']} County")
    st.write(f"**Department:** {official['name']}")
    st.write(f"**Email:** {official['email']} | **Phone:** {official['phone']}")
    st.divider()

    if intent == "I want to learn more and connect with a local official":
        your_name = st.text_input("Your Name", "Resident")
        subject = f"Question About PFAS Contamination and Water Safety — {selected_town}"
        body = f"Subject: {subject}\n\nDear {official['name']},\n\nMy name is {your_name}, and I am a resident of {selected_town}. I am writing to learn more about water safety and monitoring efforts in our community.\n\nSincerely,\n{your_name}\n{selected_town}, NY"
    else:
        your_name = st.text_input("Your Name", "Resident")
        pfas_level = st.number_input("Detected PFAS Level (ng/L)", value=100.0, step=1.0)
        contaminant = st.selectbox("Contaminant", ["PFOA", "PFOS", "PFBA", "General PFAS"])
        multiplier = pfas_level / 10.0
        
        subject = f"URGENT: PFAS Contamination Near My Home — {selected_town}"
        body = f"Subject: {subject}\n\nDear {official['name']},\n\nI am a resident of {selected_town} reporting test results showing {contaminant} at {pfas_level:.0f} ng/L ({multiplier:.0f}x New York's safe limit).\n\nSincerely,\n{your_name}\n{selected_town}, NY"

    st.text_area("Generated Letter Preview", body, height=200)
    st.download_button("Download Letter Text", data=body, file_name="contamination_report.txt")

# ════════════════════════════════════════
# COMPARE TREATMENTS
# ════════════════════════════════════════
elif page == "Compare Treatments":
    st.title("Treatment Options and Simulation Engine")
    st.divider()

    sub_tab = st.radio("View Mode", ["Run Simulation", "About the Math"], horizontal=True)
    st.divider()

    if sub_tab == "About the Math":
        st.subheader("Model Overview")
        st.write("The simulation solves a 6-species ordinary differential equation (ODE) system using SciPy's stiff BDF integrator.")
        st.write("- **H₂O₂:** Hydrogen peroxide oxidant supply")
        st.write("- **·OH:** Hydroxyl radicals")
        st.write("- **Target Contaminants:** PFOA, PFOS, or PFBA")
        st.write("- **Iron Catalysts:** Fe²⁺ and Fe³⁺")
    else:
        col_well, col_ph, col_temp = st.columns(3)
        with col_well:
            selected_well_name = st.selectbox("Well Site", list(BOMARC_WELLS.keys()))
            well_info = get_well_data(selected_well_name)
        with col_ph:
            user_pH = st.slider("pH", 3.0, 8.0, FIELD_CONDITIONS["pH"], 0.1)
        with col_temp:
            user_temp = st.slider("Temp (°C)", 5.0, 35.0, FIELD_CONDITIONS["temp_C"], 0.5)

        run_sim = st.button("Run Comparison", type="primary")
        st.divider()

        if run_sim:
            costs = {"H2O2": 0.014, "Fe": 0.0557, "UV": 0.008, "gac_lb": 2.25, "gac_disposal": 1.50}
            results = simulate_all_contaminants(well_info, user_pH, user_temp, 10.0, 5.0, costs)

            st.subheader(f"Results for {selected_well_name}")
            for compound, d in results.items():
                st.markdown(f"**Compound: {compound}**")
                st.write(f"Detected Level: {d['level_ng_l']} ng/L")
                st.write(f"Photo-Fenton Removal: {d['fenton_removal']:.1f}%")
                st.write(f"GAC Removal: {d['gac_removal']:.1f}%")
                st.divider()

# ════════════════════════════════════════
# TESTING KITS
# ════════════════════════════════════════
elif page == "Testing Kits":
    st.title("Certified Testing Laboratories & Protocols")
    st.divider()

    st.subheader("NYS ELAP-Certified Laboratories")
    for lab in ELAP_LABS:
        st.markdown(f"**{lab['name']}**")
        st.write(f"Address: {lab['address']} | Phone: {lab['phone']}")
        st.write(f"Approved Panel: {lab['pfas_panel']}")
        st.caption(lab['notes'])
        st.divider()

    st.subheader("Sampling Checklist")
    st.checkbox("Use high-density polyethylene (HDPE) collection bottles.")
    st.checkbox("Flush tap thoroughly with cold water prior to collection.")
    st.checkbox("Wear powder-free nitrile gloves to avoid sample contamination.")
