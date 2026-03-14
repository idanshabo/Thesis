"""
Central config loader for the pipeline.

Provides paths for data, families, trees, covariance matrices, etc.
All code should use this instead of hardcoding paths.

Config is loaded from pipeline_files/config.json (next to this file).
Override with KAVERET_CONFIG env var to point to a different config file.
"""

import json
import os

_CONFIG = None
_CONFIG_PATH = os.environ.get(
    "KAVERET_CONFIG",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
)


def load_config():
    """Load and cache the config."""
    global _CONFIG
    if _CONFIG is None:
        with open(_CONFIG_PATH) as f:
            _CONFIG = json.load(f)
    return _CONFIG


def get_data_dir():
    return load_config()["data_dir"]


def get_pfam_bulk_file():
    return load_config()["pfam_bulk_file"]


def get_family_msa_path(family):
    """Path to the Stockholm MSA file for a family."""
    return os.path.join(get_data_dir(), f"{family.upper()}.stockholm")


def get_family_calc_dir(family):
    """Path to the calculations directory for a family."""
    return os.path.join(get_data_dir(), f"{family.upper()}_calculations")


def get_family_tree_path(family):
    """Path to the .tree file for a family."""
    fam = family.upper()
    return os.path.join(get_family_calc_dir(fam), f"{fam}.tree")


def get_family_cov_path(family):
    """Path to the ordered covariance matrix CSV for a family."""
    fam = family.upper()
    return os.path.join(get_family_calc_dir(fam), f"{fam}_cov_mat_tree_ordered.csv")


def get_family_output_dir(family):
    """Path to the outputs directory for a family."""
    return os.path.join(get_data_dir(), f"{family.upper()}_outputs")
