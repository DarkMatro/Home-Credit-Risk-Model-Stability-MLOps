"""
Get many types of data for frontend part.

This file contains the following functions:
    * get_dataframe
    * get_metrics
    * get_study
    * is_trained
    * get_unique_values
    * example_dataset
    * prepare_file
    * get_shap_data
    * get_finance_data

Version: 1.0
"""

import io
import json
from pathlib import Path

import joblib
import optuna
import pandas as pd
import requests
from optuna import load_study
from streamlit import cache_data
from streamlit.runtime.uploaded_file_manager import UploadedFile

from .config import get_endpoint, get_config


def get_dataframe(
    file_path: str, columns: list[str] | None = None, n_rows: int | None = None
) -> pd.DataFrame:
    """
    Read parquet file with only n_rows and selected columns.

    Parameters
    ----------
    file_path: str
        String, path object (implementing os. PathLike[str]), or file-like object implementing a
         binary read() function.

    columns: list[str] | None = None
        If not None, only these columns will be read from the file.

    n_rows: int | None = None
        If not None, first n rows will be returned

    Returns
    -------
    df: pd.DataFrame
    """
    df = pd.read_parquet(file_path, columns=columns)
    if n_rows is not None:
        df = df.head(n_rows)
    return df


def get_metrics() -> tuple[dict, dict] | None:
    """
    Read metrics.json with new_metrics values and old_metrics

    Returns
    -------
    out tuple[dict, dict]
        2 dicts with new_metrics and old_metrics
    """
    config = get_config()
    metrics_path = config["train"]["metrics_path"]
    if not Path(metrics_path).exists():
        return None
    with open(metrics_path, "r", encoding="utf-8") as file:
        content = json.load(file)
    return content["new_metrics"], content["old_metrics"]


def get_study() -> optuna.Study:
    """
    Returns optuna study for tuned CatBoost model

    Returns
    -------
    study: optuna.Study
    """
    config = get_config()
    study_path = config["train"]["study_path"]
    storage_name = f"sqlite:///{study_path}"
    study = load_study(study_name=study_path, storage=storage_name)
    return study


def is_trained() -> bool:
    """
    Checks tuned_model_path exists

    Returns
    -------
    out: bool
    """
    config = get_config()
    return Path(config["train"]["tuned_model_path"]).exists()


def get_unique_values(f_type: str = "cat") -> dict:
    """
    Read json with unique values for  input forms.

    Parameters
    ----------
    f_type: str
        'cat' for categorical, 'num' for numerical values

    Returns
    -------
    unique_v: dict
    """
    config = get_config()
    file_type = "uniq_cat_values_path" if f_type == "cat" else "min_max_num_values_path"
    path = config["preprocessing"][file_type]
    with open(path, encoding="utf-8") as file:
        unique_v = json.load(file)
    return unique_v


def example_dataset() -> io.BytesIO:
    """
    Returns dataset from check_data_path as bytes for download button.

    Returns
    -------
    dataset_bytes_obj: io.BytesIO
    """
    config = get_config()
    example_file_path = config["preprocessing"]["check_data_path"]
    df = pd.read_parquet(example_file_path)
    dataset_bytes_obj = io.BytesIO()
    df.to_parquet(dataset_bytes_obj)
    dataset_bytes_obj.seek(0)
    return dataset_bytes_obj


def prepare_file(upload_file: UploadedFile) -> dict:
    """
    Prepare uploaded file for request

    Parameters
    ----------
    upload_file: UploadedFile

    Returns
    -------
    out: dict
    """
    dataset = pd.read_parquet(upload_file)
    dataset_bytes_obj = io.BytesIO()
    dataset.to_parquet(dataset_bytes_obj)
    dataset_bytes_obj.seek(0)
    return {"file": (upload_file.name, dataset_bytes_obj, "multipart/form-data")}


@cache_data()
def get_shap_data() -> dict:
    """
    Read cached shap data.

    Returns
    -------
    out: dict
        base_value, feature_names, shap_values, shap_values_legacy
    """
    config = get_config()
    shap_config = config["shap"]
    with open(shap_config["shap_values_data_path"], encoding="utf-8") as file:
        shap_data = json.load(file)
    shap_values = joblib.load(shap_config["shap_values_path"])
    shap_values_legacy = joblib.load(shap_config["shap_values_legacy_path"])
    return {
        "base_value": shap_data["base_value"],
        "feature_names": shap_data["feature_names"],
        "shap_values": shap_values,
        "shap_values_legacy": shap_values_legacy,
    }


@cache_data()
def get_finance_data() -> dict:
    """
    Returns request result for finance data.

    Returns
    -------
    out: dict
    """
    endpoint = get_endpoint("finance")
    output = requests.get(endpoint, timeout=10_000)
    return output.json()
