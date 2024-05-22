"""
Utils for SHAP values

This file contains the following functions:
    * get_n_most_important_features

Version: 1.0
"""

import pandas as pd
import shap


def get_n_most_important_features(
    col_names: list[str], shap_values: shap.Explanation, n: int
) -> list[str]:
    """Sort shap_values by mean value and return names of N most important features.

    Parameters
    ----------
    col_names: list[str]
        List of feature names
    shap_values: shap.Explanation
         matrix of SHAP values (# samples x # features).
         Each row sums to the difference between the model output
    n: int
        Number of features to return

    Returns
    ----------
    out: list[str]
        Most important features
    """
    feature_importance = pd.DataFrame(
        list(zip(col_names, shap_values.abs.mean(0).values)),
        columns=["col_name", "FI_vals"],
    )
    feature_importance.sort_values(by=["FI_vals"], ascending=False, inplace=True)
    return feature_importance.head(n)["col_name"].values.tolist()
