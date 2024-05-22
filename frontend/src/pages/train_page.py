"""
Train page streamlit widgets

This file contains the following functions:
    * train_page

Version: 1.0
"""

import streamlit as st
import numpy as np
import optuna
from ..train.training import train_model
from ..data.get_data import get_metrics, get_study, is_trained


def train_page() -> None:
    """Train catboost model page.
    If model is trained - show metrics and study results, best params + 'train more' button
    If not trained - show only 'train' button
    """
    is_model_trained = is_trained()
    # Prevent multiple clicks.
    if "train_button" in st.session_state and st.session_state.train_button:
        st.session_state.running = True
    else:
        st.session_state.running = False

    # Header.
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.title("Training CatBoost model")
    with col2:
        if is_model_trained:
            info_text = st.success("Модель обучена")
        else:
            info_text = st.info("Модель не обучена", icon="ℹ️")

    # Train button.
    if st.button(
        "Train",
        help="Подобрать гиперпараметры и обучить CatBoost модель",
        disabled=st.session_state.running,
        key="train_button",
    ):
        train_model()
        info_text.empty()
        is_model_trained = True

    if not is_model_trained:
        return

    with st.container():
        # Metrics.
        _show_metrics()

        # Study best params, plots.
        _show_study()


def _show_metrics() -> None:
    """Read and show metrics for tuned model."""
    metrics = get_metrics()
    if metrics is None:
        return
    new_metrics, old_metrics = metrics
    n_cols = len(new_metrics.keys())
    cols = st.columns(n_cols)
    for i, (metric_name, metric_value) in enumerate(new_metrics.items()):
        if not old_metrics:
            delta = None
        else:
            delta = np.round(metric_value - old_metrics[metric_name], 5)
        delta_color = (
            "inverse" if metric_name in ["Logloss", "overfitting, %"] else "normal"
        )
        if delta is None or delta == 0.0:
            delta_color = "off"
        cols[i].metric(metric_name, np.round(metric_value, 5), delta, delta_color)


def _show_study() -> None:
    """Show optuna plots with tabs"""
    study = get_study()
    hi, ohp, bm, leaf, bp = st.tabs(
        [
            "Hyperparameter importances",
            "Optimization history plot",
            "Bootstrap type - max depth",
            "L2 leaf reg - leaf estimation iterations",
            "Best params",
        ]
    )
    if len(study.trials) > 1:
        with hi:
            fig = optuna.visualization.plot_param_importances(study)
            st.plotly_chart(fig, use_container_width=True)

    with ohp:
        fig = optuna.visualization.plot_optimization_history(
            study, target_name="ROC AUC"
        )
        st.plotly_chart(fig, use_container_width=True)

    with bm:
        fig = optuna.visualization.plot_contour(
            study, params=["bootstrap_type", "max_depth"], target_name="ROC AUC"
        )
        st.plotly_chart(fig, use_container_width=True)

    with leaf:
        fig = optuna.visualization.plot_contour(
            study,
            params=["l2_leaf_reg", "leaf_estimation_iterations"],
            target_name="ROC AUC",
        )
        st.plotly_chart(fig, use_container_width=True)

    with bp:
        st.markdown(f"\tBest value (AUC): {study.best_value:.5f}")
        st.markdown("Best params:")
        for key, value in study.best_params.items():
            st.markdown(f"- {key}: {value}")
