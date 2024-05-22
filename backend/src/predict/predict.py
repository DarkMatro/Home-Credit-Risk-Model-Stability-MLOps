"""
Predict

This file contains the following functions:
    * pipeline_predict

Version: 1.0
"""

import hashlib

import joblib
import numpy as np
import pandas as pd

from ..data.get_data import get_config, get_random_row
from ..pipeline.pipeline import pipeline_preprocess


def pipeline_predict(
    client_info: dict | None = None, dataset: pd.DataFrame | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    If client_info is not None - take random row from check dataset and replace data from
    client_info.
    If dataset is not None - preprocess file from file and then predict

    Parameters
    ----------
    client_info: dict, optional
        Data from input fields

    dataset: pd.DataFrame, optional
        name of target feature

    Returns
    -------
    y_pred: predicted values

    y_score: scores
    """
    config = get_config()
    model_path = config["train"]["tuned_model_path"]
    assert (
        client_info is not None or dataset is not None
    ), "Dataset of client_info must be passed."
    if client_info is not None:
        rand = _get_hash_value(client_info)
        dataset = get_random_row(rand, client_info)

    # preprocessing
    test_data = pipeline_preprocess(dataset, is_predict=True, **config["preprocessing"])
    model = joblib.load(model_path)
    y_pred = model.predict(test_data)
    y_score = model.predict_proba(test_data)[:, 1]

    return y_pred, y_score


def _get_hash_value(client_info: dict) -> int:
    """
    Generate hash value on the input data values.

    Parameters
    ----------
    client_info: dict
        input data

    Returns
    -------
    hash_value: int
    """
    hash_str = ""
    for v in client_info.values():
        hash_str += str(v)
    hash_obj = hashlib.shake_128(hash_str.encode())
    hash_value = int(hash_obj.hexdigest(4), base=16)
    return hash_value
