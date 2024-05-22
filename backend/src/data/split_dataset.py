"""
Split data

This file contains the following functions:
    * split_train_test
    * get_train_test_data

Version: 1.0
"""

import pandas as pd

from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split


def split_train_test(
    dataset: pd.DataFrame, **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split into train and test data and save.

    Parameters
    ----------
    dataset: pd.DataFrame
        Dataset

    Returns
    -------
    df_train: pd.DataFrame
        preprocessed train dataset
    df_test: pd.DataFrame
        preprocessed test dataset
    """
    # Split in train/test
    df_train, df_test = train_test_split(
        dataset,
        stratify=dataset[kwargs["target_col"]],
        test_size=kwargs["test_size"],
        random_state=kwargs["random_state"],
        shuffle=True,
    )
    df_train.to_parquet(kwargs["train_data_path"])
    df_test.to_parquet(kwargs["test_data_path"])
    return df_train, df_test


def get_train_test_data(
    x_train: pd.DataFrame, x_test: pd.DataFrame, target: str, group: str
) -> tuple[DataFrame, DataFrame, Series, Series, Series, Series]:
    """
    Get train test data split into X, y and weeks

    Parameters
    ----------
    x_train: pd.DataFrame
        train dataset

    x_test: pd.DataFrame
        test dataset

    target: str,
        column name of target feature

    group: str,
        column name of group (week_num) feature

    Returns
    -------
    X_train: DataFrame
        X data train
    X_test: DataFrame
        X data test
    y_train: Series
        y target train
    y_test: Series
        y target test
    weeks_train: Series
        weeks train
    weeks_test: Series
        weeks test
    """
    y_train, y_test = x_train.pop(target), x_test.pop(target)
    weeks_train, weeks_test = x_train.pop(group), x_test.pop(group)
    return x_train, x_test, y_train, y_test, weeks_train, weeks_test
