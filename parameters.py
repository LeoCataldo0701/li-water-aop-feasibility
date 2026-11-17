"""
Kinetic parameter loader for photofenton.

Reads parameters from photofenton/data/parameters.json.
Each parameter has a status: 'literature', 'placeholder', or 'uncalibrated'.
"""
import json
from pathlib import Path

_DATA_FILE = Path(__file__).parent / "data" / "parameters.json"

PARAM_STATUS = {
    "literature":    "Directly measured value from peer-reviewed source.",
    "placeholder":   "Order-of-magnitude estimate. Needs calibration before use in design.",
    "uncalibrated":  "Structurally correct form but magnitude depends on lamp/reactor spec.",
}


def load_parameters() -> dict:
    """Return kinetic parameters as a flat dict of {name: value}.

    Also exposes parameter metadata (status, source, units) via load_parameter_metadata().

    Returns
    -------
    dict
        Parameter names mapped to their numeric values.
    """
    meta = load_parameter_metadata()
    return {k: v["value"] for k, v in meta.items()}


def load_parameter_metadata() -> dict:
    """Return full parameter metadata including value, units, status, and source.

    Returns
    -------
    dict
        Parameter names mapped to their full metadata dicts.
    """
    with open(_DATA_FILE) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}
