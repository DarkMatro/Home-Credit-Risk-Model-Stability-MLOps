"""
Read config files.

This file contains the following functions:
    * get_endpoint
    * get_config

Version: 1.0
"""

import yaml


PARAMS_PATH = "../config/params.yml"
TEXTS_PATH = "../config/texts.yml"
EDA_PLOTS_PATH = "../config/eda_plots.yml"
SHAP_PLOTS_PATH = "../config/shap_plots.yml"


def get_endpoint(url: str) -> str:
    """
    Returns endpoint by key

    Parameters
    ----------
    url: str
        Key

    Returns
    -------
    endpoint: str
    """
    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config["endpoints"][url]


def get_config(cfg_type: str = "params") -> dict | None:
    """
    Returns configuration data.

    Parameters
    ----------
    cfg_type: str
        One of 'params', 'texts', 'shap_plots', 'eda_plots'

    Returns
    -------
    config: dict | None
    """
    if cfg_type == "params":
        path = PARAMS_PATH
    elif cfg_type == "texts":
        path = TEXTS_PATH
    elif cfg_type == "shap_plots":
        path = SHAP_PLOTS_PATH
    elif cfg_type == "eda_plots":
        path = EDA_PLOTS_PATH
    else:
        return None
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
