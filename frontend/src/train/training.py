"""
Frontend part of training model process

This file contains the following functions:
    * train_model - send requests for train

Version: 1.0
"""

import requests
import streamlit as st
from ..data.config import get_config


def train_model() -> None:
    """With status bar send 3 requests:
    1 - preprocess datasets
    2 - find best params using optuna
    3 - train tuned CatBoost model
    """
    config = get_config()
    endpoint_preproc = config["endpoints"]["train_preprocess"]
    endpoint_optuna = config["endpoints"]["train_optuna"]
    endpoint_train = config["endpoints"]["train"]
    status = st.status("Training CatBoost model", expanded=True)
    with status:
        st.write("Preprocessing...")
        output = requests.post(endpoint_preproc, timeout=10_000)
        is_preprocessed = output.json()["is_preprocessed"]
        if is_preprocessed:
            st.write("Preprocessing finished.")
        else:
            st.write("Preprocessing failed.")
            st.error("Failed on preprocessing")
            return
        st.write("Search for hyperparameters in progress...")
        output = requests.post(endpoint_optuna, timeout=10_000)
        study_result = output.json()
        msg = (
            "Better hyperparameter was found."
            if study_result["is_last_trial_the_best"]
            else "Hyperparameters still the same."
        )
        msg += (
            f" Best trial: {study_result['best_trial']}/{study_result['n_trials']}. Best value:"
            f" {study_result['best_value']}"
        )
        st.write(msg)
        st.write("Training on best params...")
        output = requests.post(endpoint_train, timeout=10_000)
        is_trained = output.json()["is_trained"]
        st.write(f"Model is{'' if is_trained else ' not'} trained.")
