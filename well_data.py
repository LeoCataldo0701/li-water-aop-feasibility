# well_data.py
# Real BOMARC contamination data from Suffolk County investigations

BOMARC_WELLS = {
    "BOM-25": {
        "contaminant": "PFOA",
        "level_ng_l": 100,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
    "BOM-26": {
        "contaminant": "PFOA",
        "level_ng_l": 99,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
    "Pines MWG-2": {
        "contaminant": "PFOA",
        "level_ng_l": 64,
        "safe_limit_ng_l": 10,
        "location": "Drinking Water Wellhead",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    
  },
    "BOM-7": {
        "contaminant": "PFOS",
        "level_ng_l": 13,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
  
  },
    "BOM-11": {
        "contaminant": "PFOS",
        "level_ng_l": 11,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-17": {
        "contaminant": "PFOS",
        "level_ng_l": 13,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-19": {
        "contaminant": "PFOS",
        "level_ng_l": 15,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-21": {
        "contaminant": "PFOS",
        "level_ng_l": 21,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-22": {
        "contaminant": "PFOS",
        "level_ng_l": 13,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-23": {
        "contaminant": "PFOS",
        "level_ng_l": 10,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-24": {
        "contaminant": "PFOS",
        "level_ng_l": 22,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-28": {
        "contaminant": "PFOS",
        "level_ng_l": 10,
        "safe_limit_ng_l": 10,
        "location": "BOMARC Site, Westhampton",
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    }
}

# Fixed field conditions (real Suffolk County groundwater)
FIELD_CONDITIONS = {
    "pH": 5.83,
    "temp_C": 12.9,
    "location": "Suffolk County, Long Island",
    "source": "USGS Finkelstein et al. 2025"
}

# Lab-derived treatment constants
TREATMENT_OPTIONS = {
    "carbon": {
        "name": "Carbon Adsorption",
        "removal_percent": 75,
        "contact_time_min": 20,
        "cost_per_liter": 0.25,
        "description": "Standard municipal treatment",
        "pros": ["Proven technology", "Lower cost", "Standard approach"],
        "cons": ["Less effective on short-chain PFAS (PFNA)", "Requires frequent replacement"],
        "why": "Activated carbon adsorbs PFOA well but struggles with PFNA variants"
    },
    "photo_fenton": {
        "name": "Photo-Fenton Advanced Oxidation",
        "removal_percent": 90,
        "contact_time_min": 45,
        "cost_per_liter": 0.35,
        "description": "Advanced oxidation process",
        "pros": ["Handles all PFAS variants", "Faster degradation", "Destroys molecules (not just adsorbs)"],
        "cons": ["Higher cost", "Requires UV equipment"],
        "why": "Hydroxyl radicals destroy PFAS completely, including short-chain variants"
    }
}

def get_well_data(well_name):
    """Fetch well data by name"""
    return BOMARC_WELLS.get(well_name, None)

def get_treatment_comparison():
    """Return both treatment options"""
    return TREATMENT_OPTIONS

def format_crisis_message(well_name):
    """Generate crisis alert message"""
    well = BOMARC_WELLS[well_name]
    multiplier = well["level_ng_l"] / well["safe_limit_ng_l"]
    return f"{well['contaminant']} at {well['level_ng_l']} ng/L — **{multiplier:.0f}x above safe limits**"
