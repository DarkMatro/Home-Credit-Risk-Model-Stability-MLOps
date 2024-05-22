"""
Prediction from file page

This file contains the following functions:
    * predict_from_file

Version: 1.0
"""

import streamlit as st
import pandas as pd
import requests
from ..data.get_data import is_trained, example_dataset, prepare_file
from ..data.config import get_endpoint


def predict_from_file() -> None:
    """Widgets:
    1. Check model is fitted and exists
    2. Upload dataset form
    3. Predict button
    """
    is_model_trained = is_trained()
    endpoint = get_endpoint("predict_from_file")
    # Prevent multiple clicks.
    if (
        "predict_file_button" in st.session_state
        and st.session_state.predict_file_button
    ):
        st.session_state.running = True
    else:
        st.session_state.running = False
    if "df_pred" not in st.session_state:
        st.session_state.df_pred = None
    # Header.
    col1, col2, col3 = st.columns([0.5, 0.3, 0.2])
    with col1:
        st.title("Predict using file")
    with col2:
        dataset_bytes = example_dataset()
        st.download_button(
            "Download example dataset", dataset_bytes, "test_data.parquet"
        )
    with col3:
        if is_model_trained:
            st.success("Модель обучена")
        else:
            st.error("Модель не обучена. Сначала обучите модель.", icon="ℹ️")
            return

    # Upload file.
    upload_file = st.file_uploader("Dataset", ["parquet"], False)
    if upload_file and st.session_state.df_pred is None:
        dataset = pd.read_parquet(upload_file)
        df_upload = st.dataframe(dataset, use_container_width=True)
    else:
        st.session_state.df_pred = None
        return

    # Predict from file.
    if st.button(
        "Predict",
        help="Прогноз для данных из файла",
        disabled=st.session_state.running,
        key="predict_file_button",
        use_container_width=True,
    ):
        with st.spinner():
            files = prepare_file(upload_file)
            output = requests.post(endpoint, timeout=8000, files=files)
            df_pred = output.json()
            st.session_state.df_pred = pd.DataFrame(df_pred)
            df_upload.empty()
        st.dataframe(st.session_state.df_pred, use_container_width=True)
