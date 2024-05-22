"""
Explain model and save shap values.

This file contains the following functions:
    * save_shap

Version: 1.0
"""

import json
from pathlib import Path

import joblib
import shap

from ..data.config import get_config
from ..data.get_data import get_dataset


def save_shap() -> bool:
    """
    Explain model using SHAP and save shap_values.

    Returns
    -------
    out: bool
    """
    config = get_config()
    preproc = config["preprocessing"]
    shap_config = config["shap"]
    shap_values_data_path = shap_config["shap_values_data_path"]
    model = joblib.load(config["train"]["tuned_model_path"])
    x_test = get_dataset(preproc["test_data_path"])
    x_test.pop(preproc["group_col"])
    x_test.pop(preproc["target_col"])
    feature_names = x_test.columns.tolist()

    explainer = shap.TreeExplainer(model, feature_names=feature_names)
    shap_values_legacy = explainer.shap_values(x_test)
    shap_values = explainer(x_test)
    base_value = explainer.expected_value
    shap_data = {"base_value": base_value, "feature_names": feature_names}

    with open(shap_values_data_path, "w", encoding="utf-8") as file:
        json.dump(shap_data, file)

    joblib.dump(shap_values, shap_config["shap_values_path"])
    joblib.dump(shap_values_legacy, shap_config["shap_values_legacy_path"])

    return (
        Path(shap_values_data_path).exists()
        and Path(shap_config["shap_values_path"]).exists()
        and Path(shap_config["shap_values_legacy_path"]).exists()
    )
