# Contributing to photofenton

Thank you for considering a contribution.

## Reporting bugs
Open an issue at https://github.com/LeoCataldo0701/li-water-aop-feasibility/issues. Include:
- Your Python version and operating system
- The exact command or code that failed
- The full error message

## Requesting features or asking questions
Open an issue with the label you think fits best and describe the research use case.

## Submitting changes
1. Fork the repository and create a branch from `main`.
2. Install the development environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```
3. Make your change. Add or update tests in `tests/`.
4. Run the tests and confirm they pass:
   ```bash
   pytest tests/ -v
   ```
5. Open a pull request describing what changed and why. Changes that alter model results must state the effect on the validation test in `tests/test_validation.py`.

## Scientific changes
Kinetic constants must cite a source in `photofenton/data/parameters.json` or `compounds.json`, and must carry a status of `literature`, `placeholder`, or `uncalibrated`. Do not change a constant to match an experiment without documenting why.
