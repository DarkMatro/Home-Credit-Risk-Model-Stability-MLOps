"""
Script for statistical tests

This file contains the following functions:
    * get_stat_test_result

Version: 1.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu

from ..data.config import get_config
from ..data.get_data import get_dataframe


def get_stat_test_result(test_type: str, columns: list[str]) -> str:
    """Branch function returns get_corr or get_similarity result depending on test_type.
    Dataset is taken from train_dataset path

    Parameters
    ----------
    test_type: str
        'corr' for get_corr / 'similarity' for get_similarity

    columns: list[str]
        Column names.
        ["feature1", "feature2"] for corr
        ["feature1", "target"] for similarity

    Returns
    -------
    res: str
        Message to print
    """
    return _get_corr(columns) if test_type == "corr" else _get_similarity(columns)


def _get_similarity(columns: list[str]) -> str:
    """Returns check_distributions_for_similarity result
    Dataset is taken from train_dataset path

    Parameters
    ----------
    columns: list[str]
        Column names like ["feature1", "target"]

    Returns
    -------
    out: str
        Message to print
    """
    path = get_config()["preprocessing"]["train_data_path"]
    data = get_dataframe(path, columns)
    return _check_distributions_for_similarity(data, columns[0], columns[1])


def _get_corr(columns: list[str]) -> str:
    """Returns message text with correlation coefficient between columns[0] and columns[1]
    Dataset is taken from train_dataset path

    Parameters
    ----------
    columns: list[str]
        Column names like ["feature1", "target"]

    Returns
    -------
    out: str
        Message to print
    """
    path = get_config()["preprocessing"]["train_data_path"]
    data = get_dataframe(path, columns)
    cor = data.corr(method="spearman")
    coef = np.triu(cor)[0][1]
    out = f"Correlation coefficient between {columns[0]} and {columns[1]} = {np.round(coef, 4)}."
    return out


def _check_normal(
    x: pd.DataFrame, p_value_threshold: float = 0.05
) -> tuple[bool, float, float]:
    """
    Perform the Shapiro-Wilks test for normality.

    The Shapiro-Wilks test tests the null hypothesis that the
    data was drawn from a normal distribution.

    Parameters
    ----------
    x : array_like
        Array of sample data.

    p_value_threshold : float
        default = 0.05

    Returns
    -------
    normal : bool
        True if x has a normal distribution (p-value >= p_value_threshold)
    p-value : float
        The p-value for the hypothesis test.
    """
    x = x.dropna()
    n = min(x.shape[0], 5000)
    x = x.sample(n)
    stat, p_value = stats.shapiro(x)
    return p_value >= p_value_threshold, np.round(p_value, 5), np.round(stat, 5)


def _check_distributions_for_similarity(
    df: pd.DataFrame, col_name: str, target_col: str
) -> str:
    """
    Print normality test results for target = 1 and 0
    If both are normal - perform T-test and bootstrap test
    else perform Mann-Whitney U rank test on two independent samples

    Parameters
    ----------
    df: pd.DataFrame

    col_name : str
        feature name

    target_col : str
        target feature name

    Returns
    -------
    final_msg: str
        Message to print
    """
    df_1 = df[df[target_col] == 1]
    df_0 = df[df[target_col] == 0]
    df_0 = df_0.dropna()
    df_1 = df_1.dropna()
    check_normal_1 = _check_normal(df_1[col_name])
    check_normal_0 = _check_normal(df_0[col_name])
    final_msg = ""
    for res, group_name in [(check_normal_1, "1"), (check_normal_0, "0")]:
        msg = (
            f"Распределение для признака {col_name} target = {group_name} "
            f"{'' if res[0] else 'не '}нормальное, "
            f"P-value = {res[1]}, stat = {res[2]}. \n"
        )
        final_msg += msg + "\n"

    if all([check_normal_0[0], check_normal_1[0]]):
        data_0 = df_0[col_name].values
        data_1 = df_1[col_name].values
        final_msg += _t_test(data_0, data_1)
    else:
        statistic, p_value = mannwhitneyu(df_1[col_name], df_0[col_name])
        final_msg += _check_p_value(p_value) + "\n"
        final_msg += (
            f" Mann–Whitney U Test for {col_name}: statistic={np.round(statistic, 5)}"
        )
    return final_msg


def _t_test(data1: np.ndarray, data2: np.ndarray) -> str:
    """
    Calculate the T-test for the means of *two independent* samples of scores.

    Parameters
    ----------
    data1: np.ndarray

    data2: np.ndarray

    Returns
    -------
    str
    """
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    statistic, p_value = stats.ttest_ind(data1, data2)
    msg = _check_p_value(p_value)
    msg += f"T test statistic = {statistic}"
    return msg


def _check_p_value(p_value: float) -> str:
    """
    Returns message of p_value checking

    Parameters
    ----------
    p_value: float

    Returns
    -------
    out: str

    """
    try:
        return (
            f"Средние {'схожи' if p_value >= 0.05 else 'различны'}, "
            f"p-value={np.round(p_value, 5)}"
        )
    except TypeError as ex:
        return f"message: {ex}"
