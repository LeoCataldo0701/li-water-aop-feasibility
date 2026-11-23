import numpy as np
from photofenton import simulate, load_parameters

PFOA_NG_L = 20e-6 * 414.07 * 1e9   # 20 uM -> 8.28e6 ng/L
BASE = dict(pfas_name="PFOA", pfas_ng_l=PFOA_NG_L, pH=3.0, temp_C=25.0,
            H2O2_mM=30.0, Fe_uM=2000.0, t_end=18000.0)

def at(t_min, arr, minutes):
    return arr[np.argmin(np.abs(t_min - minutes))]

def run(label, overrides=None):
    p = load_parameters()
    if overrides:
        p.update(overrides)
    r = simulate(**BASE, params=p)
    t = r["t_min"]
    print(f"{label:22s} removal 1h {at(t, r['removal'], 60)*100:6.1f}%  "
          f"5h {at(t, r['removal'], 300)*100:6.1f}%  "
          f"H2O2 left at 1h {at(t, r['H2O2_mM_t'], 60)/30*100:5.1f}%")

print("Paper (Tang 2012): removal 1h 87.9%, 5h 95.2%, H2O2 almost gone at 1h")
run("baseline")
run("dark (no UV)", {"k_hv_h2o2": 0.0, "k_hv_fe": 0.0})
run("no Fe3+ photolysis", {"k_hv_fe": 0.0})
run("k_hv_fe = 1e-3", {"k_hv_fe": 1e-3})