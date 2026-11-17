"""
BOMARC Site PFAS Treatment Screening — Worked Example
======================================================
This script reproduces a complete feasibility screening workflow using the
photofenton package. It requires no Streamlit and can be run from the terminal:

    pip install photofenton
    python examples/bomarc_screening.py

What this script does
---------------------
1. Loads compound properties from the database (no hardcoded values).
2. Runs photo-Fenton simulations for three PFAS compounds at BOMARC site
   conditions (USGS Finkelstein et al. 2025 field data).
3. Sweeps H2O2 and Fe doses to find the cheapest reagent combination that
   reaches 90% removal for each compound.
4. Compares photo-Fenton removal against literature GAC removal constants.
5. Saves a summary CSV and two figures to the working directory.

NOTE: This model is validated only at pH 2.5–3.5. The BOMARC site field pH
is 5.83, which is outside the validated range. This script uses pH 3.0 as a
conservative lab-condition proxy. Results are for feasibility screening only
and must not be used for engineering design without experimental validation.

Data sources
------------
Field conditions : USGS Finkelstein et al. (2025)
PFAS levels      : Suffolk County DOHS BOMARC Site Investigations (2020, 2024)
GAC removal      : EPA PFAS Fact Sheet (2020); Long et al. (2023)
"""
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from photofenton import (
    simulate,
    reagent_cost_per_litre,
    cost_per_ug_removed,
    parameter_sweep,
    get_compound,
    load_parameters,
    PARAM_STATUS,
)

# ── Site conditions (USGS Finkelstein et al. 2025) ──────────────────────────
FIELD_CONDITIONS = {
    "pH_field":    5.83,  # actual field pH — OUTSIDE model range, not used in simulation
    "pH_model":    3.0,   # pH used in simulation (within validated range 2.5–3.5)
    "temp_C":      12.9,  # °C, Suffolk County groundwater
}

# ── BOMARC site measured concentrations, ng/L (Suffolk County DOHS 2024) ────
BOMARC_CONCENTRATIONS = {
    "PFOA": 120.0,
    "PFOS":  45.0,
    "PFBA":  18.0,
}
NY_MCL_NG_L = 10.0  # NY maximum contaminant level for PFOA, PFOS

# ── Reagent unit costs (approximate commercial rates) ────────────────────────
COSTS = {
    "H2O2_per_mmol": 0.014,   # $/mmol H2O2 (~$0.47/L 30% H2O2)
    "Fe_per_mmol":   0.0557,  # $/mmol FeSO4
    "UV_per_min":    0.008,   # $/min UV lamp operation
}


def print_param_status():
    """Print a summary of which parameters are literature vs. estimated."""
    print("\n=== Parameter Status ===")
    meta = load_parameters()
    for status, description in PARAM_STATUS.items():
        print(f"\n  {status.upper()}: {description}")
    print()


def run_single_compound_demo():
    """Demonstrate a single simulate() call with detailed output."""
    print("=" * 60)
    print("STEP 1: Single compound simulation (PFOA at BOMARC conditions)")
    print("=" * 60)

    result = simulate(
        pfas_name="PFOA",
        pfas_ng_l=BOMARC_CONCENTRATIONS["PFOA"],
        pH=FIELD_CONDITIONS["pH_model"],
        temp_C=FIELD_CONDITIONS["temp_C"],
        H2O2_mM=10.0,
        Fe_uM=5.0,
    )

    arrh = result["k_f"] / load_parameters()["k_f_ref"]
    print(f"  Compound:        {result['compound']}")
    print(f"  Initial conc:    {BOMARC_CONCENTRATIONS['PFOA']} ng/L "
          f"({BOMARC_CONCENTRATIONS['PFOA']/NY_MCL_NG_L:.0f}x NY MCL)")
    print(f"  Temperature:     {result['temp_C']} °C  "
          f"(Arrhenius factor: {arrh:.3f} vs. 25 °C reference)")
    print(f"  pH used:         {result['pH']} (field pH {FIELD_CONDITIONS['pH_field']} "
          f"is outside model range 2.5–3.5)")
    print(f"  Removal at 2h:   {result['removal_final']*100:.1f}%")
    if result["t90_min"]:
        print(f"  t90:             {result['t90_min']:.0f} min")
    else:
        print(f"  t90:             NOT REACHED in 120 min")
    print(f"  Fe balance error:{result['fe_balance_error']:.2e}  (< 1e-6 is good)")

    cost = reagent_cost_per_litre(10.0, 5.0, result["t90_min"], COSTS)
    if cost:
        cpu = cost_per_ug_removed(BOMARC_CONCENTRATIONS["PFOA"], cost)
        print(f"  Reagent cost:    ${cost:.4f}/L  |  ${cpu:.4f}/µg removed")
    else:
        print(f"  Reagent cost:    N/A (target not reached)")
    return result


def run_all_compounds():
    """Run baseline simulation for all three BOMARC compounds."""
    print("\n" + "=" * 60)
    print("STEP 2: All three compounds at baseline doses (10 mM H2O2, 5 µM Fe)")
    print("=" * 60)

    baseline_results = {}
    for compound, conc in BOMARC_CONCENTRATIONS.items():
        r = simulate(
            pfas_name=compound,
            pfas_ng_l=conc,
            pH=FIELD_CONDITIONS["pH_model"],
            temp_C=FIELD_CONDITIONS["temp_C"],
            H2O2_mM=10.0,
            Fe_uM=5.0,
        )
        cost = reagent_cost_per_litre(10.0, 5.0, r["t90_min"], COSTS)
        cpu  = cost_per_ug_removed(conc, cost)
        c    = get_compound(compound)

        t90_str  = f"{r['t90_min']:.0f} min" if r["t90_min"] else "not reached"
        cost_str = f"${cost:.4f}/L" if cost else "N/A"

        print(f"\n  {compound} ({c['full_name']})")
        print(f"    Initial:          {conc} ng/L  ({conc/NY_MCL_NG_L:.0f}x NY MCL)")
        print(f"    Photo-Fenton:     {r['removal_final']*100:.1f}% removal  |  t90: {t90_str}")
        print(f"    GAC (literature): {c['gac_removal_fraction']*100:.0f}% removal  |  cost not modeled")
        print(f"    Reagent cost:     {cost_str}")

        baseline_results[compound] = {
            "result": r,
            "cost": cost,
            "cost_per_ug": cpu,
            "compound_props": c,
        }
    return baseline_results


def run_dose_sweep():
    """Parameter sweep over H2O2 and Fe doses for each compound."""
    print("\n" + "=" * 60)
    print("STEP 3: Dose sweep — finding the cheapest reagent combination")
    print("that reaches 90% removal for each compound")
    print("=" * 60)

    H2O2_range = [1, 2, 5, 10, 20, 50]
    Fe_range   = [0.5, 1, 2, 5, 10, 20]

    sweep_results = {}
    for compound, conc in BOMARC_CONCENTRATIONS.items():
        rows = parameter_sweep(
            pfas_name=compound,
            pfas_ng_l=conc,
            pH=FIELD_CONDITIONS["pH_model"],
            temp_C=FIELD_CONDITIONS["temp_C"],
            H2O2_mM_values=H2O2_range,
            Fe_uM_values=Fe_range,
            costs=COSTS,
        )

        viable = [r for r in rows if r["cost_per_L"] is not None]
        if viable:
            best = min(viable, key=lambda r: r["cost_per_L"])
            print(f"\n  {compound}: {len(viable)}/{len(rows)} dose combos reach 90% removal")
            print(f"    Cheapest:  H2O2={best['H2O2_mM']} mM, Fe={best['Fe_uM']} µM  "
                  f"→ ${best['cost_per_L']:.4f}/L  "
                  f"(${best['cost_per_ug']:.4f}/µg)")
        else:
            print(f"\n  {compound}: 0/{len(rows)} dose combos reach 90% removal in 2h")

        sweep_results[compound] = rows

    return sweep_results


def save_summary_csv(baseline_results, outpath="bomarc_summary.csv"):
    """Write a summary CSV file with per-compound results."""
    rows = []
    for compound, data in baseline_results.items():
        r = data["result"]
        c = data["compound_props"]
        rows.append({
            "compound":          compound,
            "initial_ng_l":      BOMARC_CONCENTRATIONS[compound],
            "ny_mcl_ng_l":       NY_MCL_NG_L,
            "multiples_over_mcl": BOMARC_CONCENTRATIONS[compound] / NY_MCL_NG_L,
            "ph_model":          r["pH"],
            "temp_C":            r["temp_C"],
            "fenton_removal_pct": round(r["removal_final"] * 100, 1),
            "t90_min":           r["t90_min"] if r["t90_min"] else "not_reached",
            "reagent_cost_per_L": round(data["cost"], 4) if data["cost"] else "N/A",
            "cost_per_ug":       round(data["cost_per_ug"], 4) if data["cost_per_ug"] else "N/A",
            "gac_removal_pct":   c["gac_removal_fraction"] * 100,
            "molar_mass_g_mol":  c["molar_mass"],
            "k_OH_L_mol_s":      c["k_OH"],
        })

    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Summary CSV saved → {outpath}")


def make_figures(baseline_results, sweep_results):
    """Generate two figures and save them as PNG files."""

    # ── Figure 1: Removal curves + bar chart ────────────────────────────────
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    ax_curves = fig.add_subplot(gs[0])
    ax_bars   = fig.add_subplot(gs[1])

    colors = {"PFOA": "#0ea5e9", "PFOS": "#8b5cf6", "PFBA": "#f59e0b"}
    compounds = list(BOMARC_CONCENTRATIONS.keys())

    for compound, data in baseline_results.items():
        r = data["result"]
        ax_curves.plot(
            r["t_min"], r["removal"] * 100,
            label=compound, color=colors[compound], linewidth=2,
        )

    ax_curves.axhline(90, color="#dc2626", linestyle="--", linewidth=1, label="90% target")
    ax_curves.set_xlabel("Time (min)", fontsize=11)
    ax_curves.set_ylabel("PFAS Removal (%)", fontsize=11)
    ax_curves.set_title(
        f"Photo-Fenton Removal Kinetics\n"
        f"pH {FIELD_CONDITIONS['pH_model']}, {FIELD_CONDITIONS['temp_C']} °C, "
        f"10 mM H₂O₂, 5 µM Fe²⁺",
        fontsize=10,
    )
    ax_curves.legend(fontsize=9)
    ax_curves.set_ylim(0, 105)
    ax_curves.grid(True, alpha=0.3)
    ax_curves.text(
        0.02, 0.02,
        "SCREENING MODEL — NOT VALIDATED",
        transform=ax_curves.transAxes,
        fontsize=7, color="#dc2626", alpha=0.6,
    )

    x = np.arange(len(compounds))
    width = 0.35
    fenton_pct = [baseline_results[c]["result"]["removal_final"] * 100 for c in compounds]
    gac_pct    = [get_compound(c)["gac_removal_fraction"] * 100 for c in compounds]

    b1 = ax_bars.bar(x - width/2, fenton_pct, width,
                     label="Photo-Fenton (model)", color="#0ea5e9", alpha=0.85)
    b2 = ax_bars.bar(x + width/2, gac_pct, width,
                     label="GAC (literature)", color="#f59e0b", alpha=0.85)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax_bars.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                     f"{int(h)}%", ha="center", va="bottom", fontsize=9, fontweight="600")

    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(compounds)
    ax_bars.set_ylabel("Removal (%)", fontsize=11)
    ax_bars.set_title("Photo-Fenton vs. GAC\nBaseline doses", fontsize=10)
    ax_bars.set_ylim(0, 115)
    ax_bars.legend(fontsize=9)
    ax_bars.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "BOMARC Site PFAS Feasibility Screening  |  photofenton v1.0.0",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig("bomarc_figure1_removal.png", dpi=150, bbox_inches="tight")
    print("  Figure 1 saved → bomarc_figure1_removal.png")
    plt.close(fig)

    # ── Figure 2: Cost heatmaps from dose sweep ──────────────────────────────
    H2O2_range = sorted(set(r["H2O2_mM"] for r in sweep_results["PFOA"]))
    Fe_range   = sorted(set(r["Fe_uM"]   for r in sweep_results["PFOA"]))

    fig2, axes = plt.subplots(1, len(compounds), figsize=(14, 4), sharey=True)

    for ax, compound in zip(axes, compounds):
        rows = sweep_results[compound]
        grid = np.full((len(Fe_range), len(H2O2_range)), np.nan)
        for row in rows:
            i = Fe_range.index(row["Fe_uM"])
            j = H2O2_range.index(row["H2O2_mM"])
            grid[i, j] = row["cost_per_L"] if row["cost_per_L"] is not None else np.nan

        im = ax.imshow(grid, aspect="auto", origin="lower",
                       cmap="YlOrRd_r", vmin=0)
        ax.set_xticks(range(len(H2O2_range)))
        ax.set_xticklabels([str(v) for v in H2O2_range], fontsize=8)
        ax.set_yticks(range(len(Fe_range)))
        ax.set_yticklabels([str(v) for v in Fe_range], fontsize=8)
        ax.set_xlabel("H₂O₂ (mM)", fontsize=9)
        ax.set_title(f"{compound}", fontsize=10, fontweight="600")
        plt.colorbar(im, ax=ax, label="$/L", shrink=0.8)

    axes[0].set_ylabel("Fe²⁺ (µM)", fontsize=9)
    fig2.suptitle(
        "Reagent Cost Heatmap ($/L to 90% removal)  |  NaN = target not reached",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig2.savefig("bomarc_figure2_cost_heatmap.png", dpi=150, bbox_inches="tight")
    print("  Figure 2 saved → bomarc_figure2_cost_heatmap.png")
    plt.close(fig2)


def main():
    print("\nphotofenton — BOMARC Site PFAS Screening Example")
    print("=" * 60)
    print(f"Field pH:   {FIELD_CONDITIONS['pH_field']}  "
          f"(NOTE: outside model range 2.5–3.5 — using pH {FIELD_CONDITIONS['pH_model']} instead)")
    print(f"Temp:       {FIELD_CONDITIONS['temp_C']} °C  "
          f"(Arrhenius suppresses k_f to ~43% of 25 °C value)")

    print_param_status()
    run_single_compound_demo()
    baseline = run_all_compounds()
    sweep = run_dose_sweep()

    print("\n" + "=" * 60)
    print("STEP 4: Saving outputs")
    print("=" * 60)
    save_summary_csv(baseline)
    make_figures(baseline, sweep)

    print("\n" + "=" * 60)
    print("Done. Files written to working directory:")
    print("  bomarc_summary.csv")
    print("  bomarc_figure1_removal.png")
    print("  bomarc_figure2_cost_heatmap.png")
    print("=" * 60)
    print("\nIMPORTANT: Results are unvalidated screening estimates.")
    print("k_I, k_s are placeholders. k_hv constants are uncalibrated.")
    print("Do not use for engineering design without experimental validation.")


if __name__ == "__main__":
    main()
