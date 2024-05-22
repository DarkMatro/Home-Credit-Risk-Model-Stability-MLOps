"""
Preprocess transform dataset

This file contains the following functions:
    * pipeline_preprocess

Version: 1.0
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def pipeline_preprocess(
    data: pd.DataFrame, is_predict: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Preprocess dataset. Drop columns, set index, fill None for categorical features.

    Parameters
    ----------
    data: pd.DataFrame
        Dataset

    is_predict: bool, default = False
        name of target feature

    Returns
    -------
    data: pd.DataFrame
        preprocessed dataset
    """
    if is_predict:
        data.drop(kwargs["group_col"], axis=1, inplace=True, errors="ignore")
    else:
        data.set_index(kwargs["index_col"], inplace=True)
    data.drop(kwargs["drop_columns"], axis=1, inplace=True, errors="ignore")
    cat_features = data.select_dtypes(exclude=np.number).columns.tolist()

    if (
        is_predict
        and Path(kwargs["uniq_cat_values_path"]).exists()
        and Path(kwargs["min_max_num_values_path"]).exists()
    ):
        _check_columns(data=data, **kwargs)
    # Fill None as 'None'
    imputer_cat = SimpleImputer(
        missing_values=None, strategy="constant", fill_value="None"
    )
    data[cat_features] = imputer_cat.fit_transform(data[cat_features])
    data[cat_features] = data[cat_features].astype("category")

    for col in kwargs["transform_cols"]:
        data[col] = data[col].apply(lambda x: 0 if x > 0 else -x)
    if not is_predict:
        _save_unique_train_data(data, **kwargs)

    return data


def _check_columns(data: pd.DataFrame, **kwargs) -> None:
    """
    Check columns set is same like for trained model.

    Parameters
    ----------
    data: pd.DataFrame
        Dataset

    Returns
    -------
    None
    """
    cat_features = data.select_dtypes(exclude=np.number).columns.tolist()
    num_features = data.select_dtypes(include=np.number).columns.tolist()
    if kwargs["target_col"] in data.columns:
        num_features.remove(kwargs["target_col"])
    if kwargs["group_col"] in data.columns:
        num_features.remove(kwargs["group_col"])

    with open(kwargs["uniq_cat_values_path"], encoding="utf-8") as json_file:
        uniq_cat_values = json.load(json_file)
    with open(kwargs["min_max_num_values_path"], encoding="utf-8") as json_file:
        min_max_num_values = json.load(json_file)
    assert set(uniq_cat_values.keys()) == set(cat_features) and set(
        min_max_num_values.keys()
    ) == set(num_features), "Different features set"


def _save_unique_train_data(data: pd.DataFrame, **kwargs) -> None:
    """
    Save unique cat values and num values.

    Parameters
    ----------
    data: pd.DataFrame
        Dataset
    """
    cat_features = data.select_dtypes(exclude=np.number).columns.tolist()
    num_features = data.select_dtypes(include=np.number).columns.tolist()
    num_features.remove(kwargs["target_col"])
    num_features.remove(kwargs["group_col"])

    _save_unique_cat_values(data, cat_features, kwargs["uniq_cat_values_path"])
    _save_min_max_num_values(data, num_features, kwargs["min_max_num_values_path"])


def _save_unique_cat_values(data: pd.DataFrame, cols: list[str], path: str) -> None:
    """
    Save .json file into path with unique values per categorical features (cols) in data.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    cols: list[str]
        categorical features names
    path: str
        to save .json file
    """
    dict_unique = {col: data[col].unique().tolist() for col in cols}
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict_unique, file)


def _save_min_max_num_values(data: pd.DataFrame, cols: list[str], path: str) -> None:
    """
    Save .json file into path with (min, max, default values, have_nan flag) of numerical features
     in data.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset
    cols: list[str]
        categorical features names
    path: str
        to save .json file
    """
    dict_vals = {
        col: (
            data[col].min().astype(float),
            data[col].max().astype(float),
            data[col].median().astype(float),
            data[col].isnull().values.any().astype(float),
        )
        for col in cols
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(dict_vals, file)
