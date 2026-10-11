# well_data.py
# Real BOMARC contamination data from Suffolk County investigations

BOMARC_WELLS = {
    "BOM-25": {
        "location": "BOMARC Site, Westhampton",
        "compounds": { 
            "PFOA" : {"level_ng_l": 100, "gac_removal": 0.75},
            "PFOS" : {"level_ng_l": 27, "gac_removal": 0.70},
            "PFNA" : {"level_ng_l": 8, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    },
    "BOM-26": {
        "location": "BOMARC Site, Westhampton",
        "compounds": { 
            "PFOA" : {"level_ng_l": 27, "gac_removal": 0.75},
            "PFOS" : {"level_ng_l": 120, "gac_removal": 0.70},
            "PFNA" : {"level_ng_l": 99, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    },
    "Pines MWG-2": {
        "location": "Drinking Water Wellhead",
        "compounds": { 
            "PFOA" : {"level_ng_l": 27, "gac_removal": 0.75},
            "PFOS" : {"level_ng_l": 120, "gac_removal": 0.70},
            "PFNA" : {"level_ng_l": 99, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    
  },
    "BOM-7": {
      "location": "BOMARC Site, Westhampton",
        "compounds": { 
            "PFOA" : {"level_ng_l": 13, "gac_removal": 0.75},
            "PFOS" : {"level_ng_l": 1.4, "gac_removal": 0.70},
            "PFNA" : {"level_ng_l": .93, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    },
    "BOM-11": {
       "location": "BOMARC Site, Westhampton",
        "compounds": { 
            "PFOA" : {"level_ng_l": 11, "gac_removal": 0.75},
            "PFOS" : {"level_ng_l": 2, "gac_removal": 0.70},
            "PFNA" : {"level_ng_l": 1.8, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    },
      "BOM-17": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : {"level_ng_l": 99, "gac_removal": 0.75},
             "PFOS" : {"level_ng_l": .73, "gac_removal": 0.70},
             "PFNA" : {"level_ng_l": 1.8, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    },
      "BOM-19": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : {"level_ng_l": 15, "gac_removal": 0.75},
             "PFOS" : {"level_ng_l": 39, "gac_removal": 0.70},
             "PFNA" : {"level_ng_l": 1.8, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    },
      "BOM-21": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : {"level_ng_l": 21, "gac_removal": 0.75},
             "PFOS" : {"level_ng_l": 4.2, "gac_removal": 0.70},
             "PFNA" : {"level_ng_l": .36, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    },
      "BOM-22": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : {"level_ng_l": 13, "gac_removal": 0.75},
             "PFOS" : {"level_ng_l": 13, "gac_removal": 0.70},
             "PFNA" : {"level_ng_l": 1.3, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    },
      "BOM-23": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : {"level_ng_l": 10, "gac_removal": 0.75},
             "PFOS" : {"level_ng_l": 13, "gac_removal": 0.70},
             "PFNA" : {"level_ng_l": 2, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    },
      "BOM-24": {
       "location": "BOMARC Site, Westhampton",
       "compounds": { 
           "PFOA" : {"level_ng_l": 22, "gac_removal": 0.75},
           "PFOS" : {"level_ng_l": 36, "gac_removal": 0.70},
           "PFNA" : {"level_ng_l": 45, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
    },
      "BOM-28": {
         "location": "BOMARC Site, Westhampton",
         "compounds": { 
             "PFOA" : {"level_ng_l": 10, "gac_removal": 0.75},
             "PFOS" : {"level_ng_l": 3.7, "gac_removal": 0.70},
             "PFNA" : {"level_ng_l": .56, "gac_removal": 0.22},
        },
        "safe_limit_ng_l": 10,
        "date": "2024 Investigation"
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
