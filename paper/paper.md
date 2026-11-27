---
Title: "PhotoFenton: A Python Package for Batch Photo-Fenton Kinetic Screening and Reagent Cost Estimation for PFAS Destruction in Groundwater"
Tags:
  - Python
  - PFAS
  - advanced oxidation processes
  - photo-Fenton
  - water treatment
  - environmental engineering
  - groundwater remediation
Authors:
  - name: Leonzio Cataldo
    Orcid: 0009-0008-4494-3437
    Corresponding: true
    Affiliation: 1
Affiliations:
  - Name: Plainview-Old Bethpage JFK High School
    Index: 1
Bibliography: paper.bib
---

# Summary

Per- and polyfluoroalkyl substances (PFAS) are synthetic chemicals that resist
environmental and biological degradation. They have been measured in groundwater
at contaminated sites across the United States, including military installations
where aqueous film-forming foam (AFFF) was used for fire suppression. Photo-Fenton
oxidation, which generates hydroxyl radicals (·OH) from ferrous iron (Fe²⁺),
hydrogen peroxide (H₂O₂), and ultraviolet (UV) light, can destroy PFAS by breaking
carbon–fluorine bonds, unlike granular activated carbon (GAC) treatment, which
captures PFAS without destroying them.

`photofenton` is an open-source Python package that simulates batch photo-Fenton
treatment of PFAS-contaminated water and estimates the reagent cost of reaching
90% contaminant removal. A user specifies a compound name, its initial concentration,
reactor pH, water temperature, and hydrogen peroxide and iron doses. The package
integrates a six-species ordinary differential equation (ODE) system using
SciPy's stiff BDF solver, reports removal kinetics and the time to 90% removal,
and calculates reagent cost per litre treated and per microgram of PFAS removed.
A dose parameter sweep and a one-parameter sensitivity analysis support pre-bench-test
screening decisions. Compound properties and kinetic parameters are stored in
user-editable JSON files, so new PFAS compounds can be added without changing
source code.

The package is applied to publicly measured PFOA, PFOS, and PFBA contamination
data from the BOMARC missile site in Westhampton Beach, New York
[@suffolk2020; @suffolk2024], where PFAS concentrations in monitoring wells
reach up to 12 times New York's maximum contaminant level (MCL) of 10 ng/L
for PFOA and PFOS. Field groundwater conditions (pH 5.83, temperature 12.9°C)
are from @finkelstein2025.

# Statement of need

Long Island, New York, relies on a sole-source aquifer system as its only
drinking water supply for approximately three million residents. PFAS contamination
at military sites, including the BOMARC installation in Westhampton Beach, has
entered this aquifer through decades of AFFF use. GAC, the current treatment
standard, removes long-chain PFAS such as PFOA (approximately 75%) and PFOS
(approximately 70%), but removes short-chain compounds such as PFBA at only
about 22% because short-chain molecules interact weakly with activated carbon
surfaces [@schlesinger2022]. Sites with mixed PFAS contamination therefore
present compounds that GAC cannot adequately address.

Photo-Fenton advanced oxidation can destroy PFAS across all chain lengths in
principle. Deciding whether to pursue bench-scale testing requires first-pass
estimates of dose, reaction time, and reagent cost under site-specific conditions.
Those estimates are rarely published for the specific compound mixtures and
concentrations found at contaminated groundwater sites.

`photofenton` addresses this gap. It provides a transparent, testable, openly
documented screening tool for researchers and engineers comparing photo-Fenton
to GAC before committing resources to laboratory experiments. Its most important
design decision is explicit labeling of every kinetic parameter as `literature`,
`placeholder`, or `uncalibrated`, so users know exactly which inputs are measured
values and which are estimates. The package does not hide its limitations: it
enforces a validated pH range (2.5–3.5), raises errors on failed ODE integration,
reports iron mass balance error on each run, and documents a known 15-fold
underprediction of UV-driven PFOA removal in its test suite.

## Comparison with existing tools

General-purpose water treatment modeling frameworks do not implement
radical-driven PFAS oxidation kinetics. WaterTAP [@watertap2021] models
water treatment unit processes and techno-economics within the IDAES optimization
framework; it does not include photo-Fenton or PFAS-specific chemistry. QSDsan
[@qsdsan2022] supports quantitative sustainable design and life-cycle assessment
of sanitation and resource-recovery systems, again without PFAS oxidation kinetics.
PHREEQC [@phreeqc2013] computes aqueous geochemical equilibria and reaction-path
modeling but does not simulate radical attack on organic contaminants.

No open-source package known to the author implements a photo-Fenton ODE system
for PFAS with an extensible compound database and a coupled reagent cost calculation.
`photofenton` is intended to fill that specific gap at the screening stage of
treatment planning.

# Software design

## Reaction network

The model integrates a six-species batch ODE system over a user-specified
duration (default 7,200 s, or two hours) using `scipy.integrate.solve_ivp`
with the BDF method (relative tolerance 10⁻⁸, absolute tolerance 10⁻¹⁴).
The six species are hydrogen peroxide (H₂O₂), hydroxyl radical (·OH), the
target PFAS compound, a lumped intermediate (I), ferrous iron (Fe²⁺), and
ferric iron (Fe³⁺). Iron is conserved: d[Fe²⁺]/dt + d[Fe³⁺]/dt = 0 at every
time step, and the solver reports the maximum relative iron conservation error
as a quality check on each result.

The key PFAS degradation equation is:

$$\frac{d[\text{PFAS}]}{dt} = -k_{\text{OH}} [\cdot\text{OH}][\text{PFAS}]$$

where $k_{\text{OH}}$ is the compound-specific bimolecular rate constant for
hydroxyl-radical attack, loaded from the compound database. Temperature enters
through an Arrhenius correction applied to $k_f$ (the Fenton rate constant)
and $k_r$ (the back reaction), with activation energy $E_a = 50$ kJ/mol
referenced to 25°C [@delaat1999]. At the BOMARC field temperature of 12.9°C,
the Arrhenius factor is approximately 0.43, reducing the Fenton rate to 43%
of its 25°C reference value.

## Parameter transparency

Every kinetic parameter carries an explicit status in `photofenton/data/parameters.json`:

- **Literature:** $k_f$, $k_r$, $k_{\text{scav}}$, $k_{\text{Fe}}$, $E_a$
  [@walling1975; @buxton1988; @delaat1999]
- **Placeholder:** $k_I$ (lumped intermediate scavenging) and $k_s$ (background
  OH scavenging). These are order-of-magnitude estimates and have not been
  independently measured.
- **Uncalibrated:** $k_{h\nu,\text{H}_2\text{O}_2}$ and $k_{h\nu,\text{Fe}}$
  (photolysis rate constants). These scale with lamp photon flux, which is not
  available for this model.

Hydroxyl-radical rate constants for PFOA ($3 \times 10^5$ L mol⁻¹ s⁻¹) and
PFBA ($9.31 \times 10^7$ L mol⁻¹ s⁻¹) are from @qian2016; the PFOS value
($1 \times 10^5$ L mol⁻¹ s⁻¹) is from @vecitis2008.

## Extensible compound database

Compound properties are stored in `photofenton/data/compounds.json` with fields
for molar mass, $k_{\text{OH}}$, its citation, GAC removal fraction and its
citation, and regulatory limits (New York MCL and EPA MCL). Adding a new
compound requires editing only the JSON file; no source code changes are needed.
Five compounds ship with the package: PFOA, PFOS, PFBA, PFNA, and PFHxS. The
last two carry estimated $k_{\text{OH}}$ values and are labeled accordingly.

## Cost model

Reagent cost per litre is calculated at the time of 90% removal ($t_{90}$)
as the sum of H₂O₂ cost, Fe²⁺ cost, and UV lamp operating cost:

$$C = [\text{H}_2\text{O}_2] \cdot c_{\text{H}_2\text{O}_2}
      + [\text{Fe}^{2+}] \cdot c_{\text{Fe}}
      + t_{90} \cdot c_{\text{UV}}$$

where concentrations are in mmol/L and µmol/L respectively and unit costs
are user-supplied. Cost per microgram removed is calculated from mass units
only, so molar mass does not enter the cost calculation. If 90% removal is
not reached within the simulation window, cost functions return `None` rather
than a sentinel value, and the result is labeled "not reached" in all outputs.

## Package structure

```
photofenton/
├── __init__.py          public API
├── core.py              ODE system, simulate(), cost functions,
│                        parameter_sweep(), sensitivity_analysis()
├── compounds.py         JSON loader with validation
├── parameters.py        JSON loader with status metadata
└── data/
    ├── compounds.json   molar masses, k_OH, GAC removal, MCLs
    └── parameters.json  kinetic parameters with source and status
tests/
├── test_photofenton.py  38 unit tests
└── test_validation.py   validation against Tang et al. (2012),
                         including one documented expected failure
examples/
└── bomarc_screening.py  end-to-end worked example, no GUI required
app/
└── app.py               Streamlit web application (optional)
```

# Validation

We compared the model against @tang2012, a homogeneous UV–Fenton study of
PFOA at pH 3.0, 30 mM H₂O₂, 2.0 mM Fe²⁺, and a 9 W, 254 nm UV lamp in a
200 mL quartz reactor. This study reports 87.9% PFOA removal at 1 h and
95.2% at 5 h, and no degradation in dark Fe²⁺–H₂O₂ controls. Temperature
is not reported; we assumed 25°C. UV fluence is not reported; photolysis
constants are left at their uncalibrated defaults.

| Behavior | Tang et al. (2012) | Model |
|---|---|---|
| H₂O₂ depleted within 1 h | Yes | Yes |
| Dark Fenton control, no UV | No degradation | 0.1% |
| PFOA removal at 1 h | 87.9% | 5.7% |
| PFOA removal at 5 h | 95.2% | 5.7% |

The model reproduces H₂O₂ depletion kinetics and the dark-control result.
It underpredicts UV-driven PFOA removal by approximately 15-fold at 1 h.
Sensitivity analysis shows that changing the uncalibrated Fe³⁺ photolysis
constant ($k_{h\nu,\text{Fe}}$) tenfold shifts the 5 h removal by less than
one percentage point, confirming that the gap is not attributable to the
uncalibrated constants.

@tang2012 identify two mechanisms that explain the gap. First, their dark
controls demonstrate that UV activation of the PFOA anion is required for
degradation; the hydroxyl-radical attack pathway is insufficient alone.
Second, a Fe³⁺–PFOA complex drives a second stage of degradation after H₂O₂
is fully consumed; this pathway is absent from the model. We did not tune
parameters to close the gap, because a fit to one dataset would not represent
the mechanism. The mismatch is recorded as an expected failure in
`tests/test_validation.py` using `pytest.mark.xfail(strict=True)`, so the
test suite will flag any future change that closes it.

# Limitations

- **Validated pH range.** The model accepts pH 2.5–3.5 only, where $k_{f,\text{ref}}$
  was measured [@walling1975]. The BOMARC field pH of 5.83 is outside this range.
- **Missing mechanisms.** UV activation of PFOA and the Fe³⁺–PFOA complex
  pathway are not implemented. The model underpredicts UV-driven PFOA removal
  by approximately a factor of fifteen.
- **Placeholder and uncalibrated parameters.** $k_I$, $k_s$,
  $k_{h\nu,\text{H}_2\text{O}_2}$, and $k_{h\nu,\text{Fe}}$ are not measured values.
- **Batch reactor only.** No flow, residence time distribution, reactor geometry,
  or mass transfer.
- **Lumped intermediate.** One species represents all fluorinated degradation
  products. Mineralization is not tracked.
- **GAC not simulated.** Removal values for GAC are fixed literature constants.
- **Concentration range.** The validation experiment used 20 µM PFOA,
  approximately five orders of magnitude above BOMARC site concentrations.
  Pseudo-first-order behavior may not hold across this range.
- **No external validation.** The package has no external users at the time of
  submission. It is currently used only for BOMARC site screening.

# AI usage disclosure

Generative AI (Claude, Anthropic) was used to draft and iteratively refactor
portions of the source code and to draft text in this paper. The author ran
all code, reviewed and approved every function, verified tests against the
physical system, and checked each cited value against its original source.
The scientific interpretation, parameter status assignments, and the decision
not to tune parameters to fit the validation dataset were made by the author.
AI-generated code and text were not submitted unreviewed.

# Acknowledgements

The author thanks the Suffolk County Department of Health Services for making
the BOMARC site investigation reports publicly available, and the USGS for
the Long Island groundwater quality dataset.

# References
