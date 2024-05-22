"""
Frontend part of the project

This script allows the user to open pages using streamlit.


This file contains the following functions:
    * navigation - Contains sidebar and mapping to page functions.
    * main - the main function of the script

Version: 1.0
"""

import streamlit as st
from st_on_hover_tabs import on_hover_tabs

from src import (
    eda_page,
    home_page,
    train_page,
    prediction_page,
    predict_from_file,
    feature_importance_page,
    finance_page,
    get_config,
)


def navigation() -> None:
    """Starting point with navigation by pages.
    Contains sidebar and mapping to page functions.
    """
    with st.sidebar:
        tabs = on_hover_tabs(
            tabName=[
                "Description",
                "EDA",
                "Train model",
                "Prediction",
                "Prediction from file",
                "Feature importance",
                "Finance",
            ],
            iconName=[
                "description",
                "query_stats",
                "model_training",
                "online_prediction",
                "batch_prediction",
                "waterfall_chart",
                "attach_money",
            ],
            default_choice=0,
        )
    page_names_to_funcs = {
        "Description": home_page,
        "EDA": eda_page,
        "Train model": train_page,
        "Prediction": prediction_page,
        "Prediction from file": predict_from_file,
        "Feature importance": feature_importance_page,
        "Finance": finance_page,
    }
    page_names_to_funcs[tabs]()


def main() -> None:
    """Set configuration, css style."""
    config = get_config()
    st.set_page_config(**config["frontend"]["page_config"])
    with open("./style.css", encoding="utf-8") as f:
        style = f.read()
    st.markdown("<style>" + style + "</style>", unsafe_allow_html=True)
    navigation()


if __name__ == "__main__":
    main()
