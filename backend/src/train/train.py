"""
Optuna search and train tuned model.

This file contains the following functions:
    * train_model
    * find_optimal_params

Version: 1.0
"""

import pickle
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, Pool
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import train_test_split

from ..data.config import get_config
from ..data.get_data import get_processed_dataset
from ..data.split_dataset import get_train_test_data
from ..train.metrics import save_metrics


def find_optimal_params() -> dict:
    """
    Find best params with optuna and save study.

    Returns
    -------
    out : dict
        is_last_trial_the_best, best_trial, best_value, n_trials
    """
    data_train, data_test = get_processed_dataset()
    config = get_config()
    preproc_params = config["preprocessing"]
    train_params = config["train"]

    x_train, _, y_train, _, weeks_train, _ = get_train_test_data(
        data_train, data_test, preproc_params["target_col"], preproc_params["group_col"]
    )
    scale_pos_weight = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    cat_features = x_train.select_dtypes(exclude=np.number).columns.tolist()
    train_params["scale_pos_weight"] = scale_pos_weight
    train_params["cat_features"] = cat_features

    study_path = train_params["study_path"]
    storage_name = f"sqlite:///{study_path}"
    pruner = _get_pruner(train_params["pruner_path"])
    study = optuna.create_study(
        storage=storage_name,
        pruner=pruner,
        study_name=study_path,
        direction="maximize",
        load_if_exists=True,
    )
    func = lambda trial: _objective(
        trial, x_train, y_train, weeks_train, **train_params
    )
    study.optimize(
        func,
        train_params["n_trials"],
        show_progress_bar=True,
        n_jobs=train_params["n_jobs"],
        gc_after_trial=True,
    )
    # Save pruner.
    with open(train_params["pruner_path"], "wb") as f:
        pickle.dump(study.pruner, f)

    is_last_trial_the_best = study.trials[-1].number == study.best_trial.number
    return {
        "is_last_trial_the_best": is_last_trial_the_best,
        "best_trial": study.best_trial.number,
        "best_value": np.round(study.best_value, 5),
        "n_trials": study.trials[-1].number,
    }


def _objective(
    trial: optuna.trial.Trial,
    x: pd.DataFrame,
    y: pd.Series,
    weeks: pd.Series,
    n_folds: int = 5,
    **kwargs,
) -> float:
    """
    Optimization objective function for CatBoost model with cross-validation.

    Parameters
    ----------
    trial : optuna.trial.Trial
       instance represents a process of evaluating an objective function
    x: pd.DataFrame
        X_train to split into x_train_cv and x_val_cv
    y: pd.Series
        y_train to split into y_train_cv and y_val_cv
    weeks: pd.Series
        For StratifiedGroupKFold as groups

    Returns
    -------
    out : float
       mean ROC_AUC
    """
    params = {
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [kwargs["learning_rate"]]
        ),
        "n_estimators": trial.suggest_categorical(
            "n_estimators", [kwargs["n_estimators"]]
        ),
        "max_depth": trial.suggest_int("max_depth", 6, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-4, 1e4),
        "random_strength": trial.suggest_float("random_strength", 0.01, 1e4),
        "bootstrap_type": trial.suggest_categorical(
            "bootstrap_type", ["MVS", "Bernoulli", "No"]
        ),
        "leaf_estimation_iterations": trial.suggest_int(
            "leaf_estimation_iterations", 1, 32
        ),
    }
    if params["bootstrap_type"] == "Bernoulli":
        params["subsample"] = trial.suggest_float("subsample", 0.1, 0.5)
    if params["bootstrap_type"] in ["Bernoulli", "MVS"]:
        params["grow_policy"] = trial.suggest_categorical("grow_policy", ["Lossguide"])
    elif params["bootstrap_type"] in ["Depthwise"]:
        params["grow_policy"] = trial.suggest_categorical(
            "grow_policy", ["SymmetricTree", "Lossguide"]
        )
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True)
    cv_predicts = np.empty(n_folds)
    for idx, (train_idx, val_idx) in enumerate(cv.split(x, y, groups=weeks)):
        x_train_cv, x_val_cv = x.iloc[train_idx], x.iloc[val_idx]
        y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

        train_data = Pool(
            data=x_train_cv, label=y_train_cv, cat_features=kwargs["cat_features"]
        )
        eval_data = Pool(
            data=x_val_cv, label=y_val_cv, cat_features=kwargs["cat_features"]
        )

        model = CatBoostClassifier(
            random_state=kwargs["rand"],
            scale_pos_weight=kwargs["scale_pos_weight"],
            cat_features=kwargs["cat_features"],
            use_best_model=True,
            od_wait=10,
            rsm=0.1,
            border_count=254,
            **params,
        )
        model.fit(train_data, eval_set=eval_data, early_stopping_rounds=10, verbose=0)

        y_score = model.predict_proba(x_val_cv)
        current_auc = roc_auc_score(y_val_cv, y_score[:, 1])
        cv_predicts[idx] = current_auc
        trial.report(current_auc, idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    av_auc = np.mean(cv_predicts)
    return av_auc


def _get_pruner(path: str) -> SuccessiveHalvingPruner:
    """
    Load from file or create new pruner if not exists.

    Parameters
    ----------
    path: str
        to pruner file

    Returns
    -------
    out : SuccessiveHalvingPruner
    """
    pruner_file = Path(path)
    if pruner_file.exists():
        return pickle.load(open(path, "rb"))
    return SuccessiveHalvingPruner()


def train_model() -> bool:
    """
    Train model on best params and save it.

    Returns
    -------
    out : bool
    """
    config = get_config()
    preproc_params = config["preprocessing"]
    train_params = config["train"]

    # Get data.
    data_train, data_test = get_processed_dataset()
    x_train, x_test, y_train, y_test, weeks_train, weeks_test = get_train_test_data(
        data_train, data_test, preproc_params["target_col"], preproc_params["group_col"]
    )
    x_train_, x_val, y_train_, y_val, _, _ = train_test_split(
        x_train,
        y_train,
        weeks_train,
        test_size=preproc_params["test_size_val"],
        stratify=y_train,
        shuffle=True,
        random_state=preproc_params["random_state"],
    )
    scale_pos_weight = float(np.sum(y_train_ == 0)) / np.sum(y_train_ == 1)
    cat_features = x_train.select_dtypes(exclude=np.number).columns.tolist()
    study_path = train_params["study_path"]
    storage_name = f"sqlite:///{study_path}"
    study = optuna.load_study(study_name=study_path, storage=storage_name)

    # Training with best params.
    clf = CatBoostClassifier(
        random_state=train_params["rand"],
        scale_pos_weight=scale_pos_weight,
        cat_features=cat_features,
        **study.best_params,
        **train_params["static_params_catboost"],
    )
    eval_set = [(x_val, y_val)]
    clf.fit(x_train_, y_train_, eval_set=eval_set, early_stopping_rounds=10, verbose=0)

    # Save model.
    joblib.dump(clf, train_params["tuned_model_path"])

    # Save metrics.
    kwargs = {
        "model": clf,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "weeks_train": weeks_train,
        "weeks_test": weeks_test,
        "name": "CatBoost",
    }
    save_metrics(train_params["metrics_path"], **kwargs)
    return Path(train_params["tuned_model_path"]).exists()
