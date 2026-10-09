# well_data.py
# Real BOMARC contamination data from Suffolk County investigations

BOMARC_WELLS = {
    "BOM-25": {
        "location": "BOMARC Site, Westhampton",
        "compounds": { 
            "PFOA" : 100,
            "PFOS" : 27,
            "PFNA" : 8
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
    "BOM-26": {
        "location": "BOMARC Site, Westhampton",
        "compounds": { 
            "PFOA" : 27,
            "PFOS" : 120,
            "PFNA" : 99
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
    "Pines MWG-2": {
        "location": "Drinking Water Wellhead",
        "compounds": { 
            "PFOA" : 27,
            "PFOS" : 120,
            "PFNA" : 99
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    
  },
    "BOM-7": {
      "location": "BOMARC Site, Westhampton",
        "compounds": { 
            "PFOA" : 13,
            "PFOS" : 1.4,
            "PFNA" : .93
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
    "BOM-11": {
       "location": "BOMARC Site, Westhampton",
        "compounds": { 
            "PFOA" : 11,
            "PFOS" : 2,
            "PFNA" : 1.8
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-17": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : 99,
             "PFOS" : .73,
             "PFNA" : 1.8
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-19": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : 15,
             "PFOS" : 39,
             "PFNA" : 1.8
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-21": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : 21,
             "PFOS" : 4.2,
             "PFNA" : .36
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-22": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : 13,
             "PFOS" : 13,
             "PFNA" : 1.3
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-23": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : 10,
             "PFOS" : 13,
             "PFNA" : 2
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-24": {
       "location": "BOMARC Site, Westhampton",
       "compounds": { 
           "PFOA" : 22,
           "PFOS" : 36,
           "PFNA" : 45
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
      "BOM-28": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : 10,
             "PFOS" : 3.7,
             "PFNA" : .56
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation",
        "source": "Suffolk County Dept. of Health Services"
    },
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
