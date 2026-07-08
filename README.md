# Long Island PFAS Remediation Modeling Framework

**Chemical kinetic simulation and economic optimization for Advanced Oxidation Processes (AOPs) in PFAS-contaminated groundwater.**

### Overview
Developed a Python-based computational model using a 6-equation stiff ODE system to simulate photo-Fenton degradation of PFAS in real Suffolk County, Long Island conditions. The framework evaluates treatment feasibility, performs iron catalyst optimization, and calculates cost-per-µg removed.

**Key Results**:
- Photo-Fenton AOPs achieved only ~60% degradation after 2 hours due to radical scavenging.
- Treatment costs reached up to $0.44 per microgram removed, making chemical oxidation impractical at scale for high-concentration plumes.
- Constrained optimization and sensitivity analysis identified optimal conditions while highlighting why public water extensions are the more practical near-term solution.

### Technical Details
- **Language**: Python
- **Solver**: SciPy `solve_ivp` with BDF method for stiff systems
- **Features**: Iron dose optimization, cost analysis (reagent + UV energy), baseline comparison to UV/H₂O₂ only
- **Context**: Modeled on reported Calverton plume data (PFAS concentrations up to 1000× regulatory limits)

### Ongoing Work
- Refining model with real Suffolk County data from collaboration with county hydrogeologist.
- Linking to experimental graphene oxide nanosheet catalyst testing at Medgar Evers College.
