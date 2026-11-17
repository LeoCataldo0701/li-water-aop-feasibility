"""
Compound loader for photofenton.

Reads compound properties from photofenton/data/compounds.json.
Users can add new PFAS compounds by editing that file without touching source code.
"""
import json
from pathlib import Path

_DATA_FILE = Path(__file__).parent / "data" / "compounds.json"

_REQUIRED_KEYS = {"molar_mass", "k_OH", "gac_removal_fraction"}


def load_compounds() -> dict:
    """Return all compounds from the JSON database as a dict keyed by compound name.

    Returns
    -------
    dict
        Keys are compound names (e.g. 'PFOA'). Values are property dicts.

    Raises
    ------
    FileNotFoundError
        If the compounds.json data file is missing.
    KeyError
        If a compound entry is missing a required field.
    """
    with open(_DATA_FILE) as f:
        raw = json.load(f)

    compounds = {}
    for name, props in raw.items():
        if name.startswith("_"):
            continue  # skip metadata keys
        missing = _REQUIRED_KEYS - set(props.keys())
        if missing:
            raise KeyError(
                f"Compound '{name}' in compounds.json is missing required fields: {missing}"
            )
        compounds[name] = props

    return compounds


def get_compound(name: str) -> dict:
    """Return the property dict for a single compound.

    Parameters
    ----------
    name : str
        Compound name, e.g. 'PFOA'. Case-sensitive.

    Returns
    -------
    dict
        Compound properties.

    Raises
    ------
    KeyError
        If the compound is not in the database.
    """
    compounds = load_compounds()
    if name not in compounds:
        raise KeyError(
            f"'{name}' not found in compound database. "
            f"Known compounds: {list(compounds.keys())}. "
            "To add a new compound, edit photofenton/data/compounds.json."
        )
    return compounds[name]
