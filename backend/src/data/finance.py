"""
Estimate confusion_matrix for baseline and tuned model. And estimate finance profit of tuned model.

This file contains the following functions:
    * get_finance_data

Version: 1.0
"""

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import get_config
from .get_data import get_dataset


def get_finance_data() -> dict:
    """Estimate confusion_matrix for baseline and tuned model. And estimate finance profit
     of tuned model.

    Returns
    -------
    out: dict
        cm_baseline: confusion_matrix for baseline model
        cm_tuned: confusion_matrix for tuned model
        baseline_data: saved money and loss percentage
        tuned_data: saved money and loss percentage
    """
    config = get_config()
    baseline = joblib.load(config["train"]["baseline_model_path"])
    tuned_model = joblib.load(config["train"]["tuned_model_path"])
    x_test = get_dataset(config["preprocessing"]["test_data_path"])
    x_test.pop(config["preprocessing"]["group_col"])
    y_test = x_test.pop(config["preprocessing"]["target_col"])
    y_pred_baseline = baseline.predict(x_test)
    cm_baseline = confusion_matrix(y_test, y_pred_baseline)

    y_pred_tuned = tuned_model.predict(x_test)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)

    saved_money_b, _, percent_b = _finance(
        baseline, x_test[config["preprocessing"]["finance_amount_col"]], y_test, x_test
    )
    saved_money, _, percent = _finance(
        tuned_model, x_test["credamount_770A"], y_test, x_test
    )
    return {
        "cm_baseline": cm_baseline.tolist(),
        "cm_tuned": cm_tuned.tolist(),
        "baseline_data": (saved_money_b, percent_b),
        "tuned_data": (saved_money, percent),
    }


def _finance(
    model, cred_amount: pd.Series, y_true: pd.Series, x: pd.DataFrame
) -> tuple:
    """Estimate saved and lost money using current model.

    Returns amount of saved_money as tp_sum + tn_sum,
    amount of lost money as fp_sum + fn_sum
    and percent as lost_money / maximal possible saved money amount

    Parameters
    ----------
    model: sklearn like Classifier

    cred_amount: pd.Series
        With values of credit price.

    y_true: array-like of shape (n_samples,)
        True labels.

    x: {array-like, sparse matrix, DataFrame} of shape (n_samples, n_features)
        Predict on these vectors, where `n_samples` is the number of samples and
        `n_features` is the number of features.
    Returns
    -------
    out : tuple
        saved_money, lost_money, percent
    """
    y_pred = model.predict(x)
    tp_sum = 0
    fp_sum = 0
    tn_sum = 0
    fn_sum = 0
    for i in range(cred_amount.shape[0]):
        amount = cred_amount.iloc[i]
        true_label = y_true.iloc[i]
        pred_label = y_pred[i]
        if true_label == pred_label == 1:
            tp_sum += amount
        elif true_label == pred_label == 0:
            tn_sum += amount
        elif pred_label == 1:
            fp_sum += amount
        elif pred_label == 0:
            fn_sum += amount
    saved_money = tp_sum + tn_sum
    lost_money = fp_sum + fn_sum
    max_possible_win = cred_amount.sum()
    percent = lost_money * 100 / max_possible_win
    return saved_money, lost_money, round(percent, 2)
