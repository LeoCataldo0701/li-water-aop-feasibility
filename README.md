# photofenton

Batch photo-Fenton screening model and reagent cost estimator for PFAS destruction in groundwater.

> **Status: screening model, not validated for design.** The model reproduces hydrogen peroxide depletion and dark-control behavior from published data, but underpredicts UV-driven PFOA removal by about 15 times. Read [Limitations](#limitations) before using any result.

## Statement of need

Firefighting foam used at the BOMARC site in Westhampton Beach, New York, released PFAS into groundwater. Measured PFOA and PFOS reach up to 12 times New York's 10 ng/L limit. Granular activated carbon captures PFAS rather than destroying them, and it removes short-chain compounds such as PFBA poorly. Photo-Fenton oxidation can destroy PFAS, but planning a bench test requires estimates of dose, time, and reagent cost first.

`photofenton` provides that first-pass estimate. It reports PFAS removal, time to 90% removal, and reagent cost per litre and per microgram removed, and it labels which inputs are literature values, placeholders, or uncalibrated.

## Installation

The package is not yet on PyPI. Install from GitHub:

```bash
pip install git+https://github.com/LeoCataldo0701/li-water-aop-feasibility.git
```

For development, clone the repository and install in editable mode inside a virtual environment:

```bash
git clone https://github.com/LeoCataldo0701/li-water-aop-feasibility.git
cd li-water-aop-feasibility
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Requirements: Python 3.9 or newer, numpy 1.24+, scipy 1.10+. Tests run on Python 3.11 and 3.12 in GitHub Actions.

## Quick start

```python
from photofenton import simulate, reagent_cost_per_litre, cost_per_ug_removed

result = simulate(
    pfas_name="PFOA",
    pfas_ng_l=120.0,   # initial concentration, ng/L
    pH=3.0,            # must be between 2.5 and 3.5
    temp_C=12.9,       # water temperature, °C
    H2O2_mM=10.0,      # hydrogen peroxide dose, mM
    Fe_uM=5.0,         # Fe(II) dose, µM
)

print(f"Removal after 2 h: {result['removal_final'] * 100:.1f}%")
print(f"Time to 90% removal: {result['t90_min']} min")

costs = {"H2O2_per_mmol": 0.014, "Fe_per_mmol": 0.0557, "UV_per_min": 0.008}
cost = reagent_cost_per_litre(10.0, 5.0, result["t90_min"], costs)
if cost is not None:
    print(f"Reagent cost: ${cost:.4f}/L")
```

Inputs outside the validated pH range raise a `ValueError`. Compounds available: PFOA, PFOS, PFBA, PFNA, PFHxS (see `photofenton/data/compounds.json`).

## Worked example

`examples/bomarc_screening.py` runs a complete screening of three BOMARC compounds with no graphical interface:

```bash
python examples/bomarc_screening.py
```

Expected output: a printed summary for each compound, `bomarc_summary.csv`, `bomarc_figure1_removal.png`, and `bomarc_figure2_cost_heatmap.png` in the working directory. At the default conditions (pH 3.0, 12.9 °C, 10 mM H₂O₂, 5 µM Fe²⁺), PFOA and PFOS remove less than 10% in 2 hours. That is the model's documented result for these conditions, not a failure. See [Validation](#validation).

## Running the tests

```bash
python -m pytest tests/ -v
```

Expected final line: `38 passed, 1 xfailed`. The expected failure documents a known gap against published data (see Validation).

## Parameter status

| Parameter | Status | Source |
|---|---|---|
| k_f_ref, k_r_ref, k_Fe, k_scav | literature | Walling 1975; Buxton et al. 1988 |
| Ea (activation energy) | literature | De Laat & Gallard 1999 |
| k_OH for PFOA, PFOS, PFBA | literature | Qian et al. 2016; Vecitis et al. 2009 |
| k_I (intermediate scavenging) | placeholder | Assumed |
| k_s (background scavenging) | placeholder | Field estimate |
| k_hv_h2o2, k_hv_fe (photolysis) | uncalibrated | Requires lamp photon flux |

All values live in `photofenton/data/parameters.json` with their sources.

## Validation

We compared the model against Tang et al. (2012), *Chem. Eng. J.* 184, 156–162, doi:10.1016/j.cej.2012.01.020, a homogeneous UV–Fenton study of PFOA at pH 3.0 with 30 mM H₂O₂ and 2.0 mM Fe²⁺.

| Behavior | Tang et al. (2012) | Model |
|---|---|---|
| Hydrogen peroxide depleted within 1 h | Yes | Yes |
| Dark Fenton control (no UV) | No degradation | 0.1% |
| PFOA removal at 1 h | 87.9% | 5.7% |
| PFOA removal at 5 h | 95.2% | 5.7% |

The gap is structural. Changing the photolysis constant tenfold moves the 5-hour result by less than one percentage point. The likely cause is two mechanisms the model omits: UV activation of the PFOA anion, and a Fe³⁺–PFOA complex that drives degradation after H₂O₂ is consumed. The test in `tests/test_validation.py` records this gap as an expected failure.

## Limitations

- **pH 2.5–3.5 only.** Groundwater at BOMARC is near pH 5.8. The model cannot be run at the conditions of the highest-removal PFOA study found so far (Schlesinger et al. 2022, pH 7–9, magnetite catalyst).
- **UV-driven PFOA destruction is underpredicted** by about a factor of fifteen. Do not use the model to predict PFOA destruction under UV.
- **Placeholder and uncalibrated parameters** (see table above).
- **Batch reactor only.** No flow, mass transfer, or residual reagent management.
- **One lumped intermediate.** Fluorinated byproducts and mineralization are not tracked.
- **No GAC simulation.** GAC removal values are fixed literature constants.
- **Concentration range.** Assumes pseudo-first-order kinetics. Validation used 20 µM PFOA, about five orders of magnitude above BOMARC levels.

## Contributing

Bug reports, feature requests, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

MIT. See [LICENSE](LICENSE).
