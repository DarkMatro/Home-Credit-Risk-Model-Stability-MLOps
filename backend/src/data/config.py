"""
Read config files.

This file contains the following functions:
    * get_config

Version: 1.0
"""

import yaml

PARAMS_PATH = "../config/params.yml"


def get_config() -> dict:
    """
    Returns configuration data.

    Returns
    -------
    config: dict | None
    """
    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
