"""
Feature importance SHAP page

This file contains the following functions:
    * feature_importance_page

Version: 1.0
"""

import shap
import streamlit as st
import streamlit.components.v1 as components
import requests
from ..data.get_data import is_trained, get_endpoint, get_shap_data
from ..data.caching import check_shap_values
from ..data.config import get_config
from ..plotting.get_figure import get_image
from ..plotting.charts import force_plot
from ..data.shap_utils import get_n_most_important_features


def feature_importance_page():
    """
    1. Check model is fitted.
    2. Show tabs with plots.
    """
    is_model_trained = is_trained()

    # Header.
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
    with col1:
        st.title("Feature Importance")
    with col2:
        if is_model_trained:
            st.success("Модель обучена")
        else:
            st.error("Модель не обучена. Сначала обучите модель.", icon="ℹ️")
            return
    with col3:
        is_shap_values_updated = check_shap_values()
        if not is_shap_values_updated:
            is_shap_values_updated = _update_shap()
        if is_shap_values_updated:
            st.success("SHAP values updated")
        else:
            st.error("SHAP values not updated or doesnt exists.", icon="ℹ️")
            return
    shap_data = get_shap_data()
    # Tabs.
    _show_tabs(shap_data)


def _show_tabs(shap_data: dict) -> None:
    """
    Show tabs with beeswarm plot, scatter plots, force plots

    Parameters
    ----------
    shap_data: dict
        has base_value, feature_names, shap_values, shap_values_legacy
    """
    texty = get_config("texts")
    plot_params = get_config("shap_plots")
    shap_values_legacy = shap_data["shap_values_legacy"]
    shap_values_legacy_sample = shap.sample(shap_values_legacy, 1000)

    tab1, tab2, tab3 = st.tabs(texty["shap_tabs_titles"])
    with tab1:
        img = get_image(
            _shap_values=shap_data["shap_values"], **plot_params["beeswarm_plot"]
        )
        st.image(img, "Most important features", 1900, True)
    with tab2:
        img = get_image(
            _shap_values=shap_data["shap_values"], **plot_params["scatter_plots"]
        )
        capture = "Scatter plots for age, payments number, tax amount and employment experience"
        st.image(img, capture, 1900, True)
        with st.expander('Описание', True):
            st.markdown(texty['shap_scatters_conclusion'])
    with tab3:
        cols = get_n_most_important_features(
            shap_data["feature_names"], shap_data["shap_values"], 20
        )
        plot_html = force_plot(
            shap_data["base_value"], shap_values_legacy_sample, cols, dark_mode=True
        )
        components.html(plot_html, height=400, width=1450)

        sample_id = st.slider("Sample ID", 0, shap_values_legacy.shape[0], 0, 1)
        plot_html = force_plot(
            shap_data["base_value"],
            shap_values_legacy[sample_id],
            feature_names=shap_data["feature_names"],
            dark_mode=True,
        )
        components.html(plot_html, height=200, width=1450)


def _update_shap() -> bool:
    """
    Request check SHAP values exists and updated

    Returns
    -------
    is_shap_updated: bool
    """
    endpoint = get_endpoint("update_shap")
    with st.spinner("Updating SHAP in progress..."):
        output = requests.get(endpoint, timeout=10_000)
        is_shap_updated = output.json()["is_shap_updated"]
    return is_shap_updated
