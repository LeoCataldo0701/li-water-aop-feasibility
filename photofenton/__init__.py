"""
photofenton
===========
Batch photo-Fenton kinetics and reagent cost screening for PFAS-contaminated water.

This is an *unvalidated screening model*. Rate constants are partly from peer-reviewed
literature, partly placeholders, and partly uncalibrated (see PARAM_STATUS and
photofenton/data/parameters.json). Results should not be used for engineering design
or regulatory decisions without independent validation against measured degradation data.

Validated pH range: 2.5 – 3.5 (where k_f_ref was measured).
Field pH (typically 5–8) is outside this range and is not modeled.

Quick start
-----------
>>> from photofenton import simulate, reagent_cost_per_litre, cost_per_ug_removed
>>> result = simulate("PFOA", pfas_ng_l=120.0, pH=3.0, temp_C=12.9,
...                   H2O2_mM=10.0, Fe_uM=5.0)
>>> print(f"Removal: {result['removal_final']*100:.1f}%")
>>> print(f"t90: {result['t90_min']} min")

References
----------
Walling 1975       Acc. Chem. Res. 8(4):125-131
Buxton et al. 1988 J. Phys. Chem. Ref. Data 17(2):513-886
Qian et al. 2016   Environ. Sci. Technol. 50(14):7854-7862
Vecitis et al. 2009 J. Phys. Chem. A 113(17):4071-4076
De Laat & Gallard 1999 Environ. Sci. Technol. 33(16):2726-2732
"""

from photofenton.core import (
    simulate,
    reagent_cost_per_litre,
    cost_per_ug_removed,
    arrhenius_factor,
    parameter_sweep,
    sensitivity_analysis,
)
from photofenton.compounds import load_compounds, get_compound
from photofenton.parameters import load_parameters, PARAM_STATUS

__version__ = "1.0.0"
__all__ = [
    "simulate",
    "reagent_cost_per_litre",
    "cost_per_ug_removed",
    "arrhenius_factor",
    "parameter_sweep",
    "load_compounds",
    "get_compound",
    "load_parameters",
    "PARAM_STATUS",
    "sensitivity_analysis",
]
