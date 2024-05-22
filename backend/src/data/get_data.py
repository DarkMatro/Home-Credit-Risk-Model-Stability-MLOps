"""
Load dataframe from file

This script allows the user to get different data from files

This file contains the following functions:
    * get_dataset
    * get_processed_dataset
    * get_random_row

Version: 1.0
"""

import numpy as np
import pandas as pd
from .config import get_config


def get_dataset(
    path: str, columns: list[str] | None = None, n_rows: int | None = None
) -> pd.DataFrame:
    """
    Read parquet file with only n_rows and selected columns.

    Parameters
    ----------
    path: str
        String, path object (implementing os. PathLike[str]), or file-like object implementing
         a binary read() function.

    columns: list[str] | None = None
        If not None, only these columns will be read from the file.

    n_rows: int | None = None
        If not None, first n rows will be returned

    Returns
    -------
    df: pd.DataFrame
    """
    df = pd.read_parquet(path, columns=columns)
    if n_rows is not None:
        df = df.head(n_rows)
    return df


def get_processed_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read parquet file with preprocessed train and test data.

    Returns
    -------
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    """
    config = get_config()
    preproc_params = config["preprocessing"]
    df_train = pd.read_parquet(preproc_params["train_data_path"])
    df_test = pd.read_parquet(preproc_params["test_data_path"])
    return df_train, df_test


def get_random_row(rand: int, client_info: dict) -> pd.DataFrame:
    """
    Takes random row from check_data_path dataset and replace data by client_info.

    Parameters
    ----------
    rand: int
        random_state

    client_info: dict
        Info to replace data

    Returns
    -------
    df: pd.DataFrame
    """
    path = get_config()["preprocessing"]["check_data_path"]
    df = get_dataset(path)
    df = df.sample(1, random_state=rand)
    for k in client_info:
        if client_info[k] is None:
            client_info[k] = np.nan

    for k, v in client_info.items():
        col_idx = df.columns.get_loc(k)
        df.iloc[0, col_idx] = v

    return df
