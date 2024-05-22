"""
Finance analysis page

This file contains the following functions:
    * finance_page

Version: 1.0
"""

import numpy as np
import streamlit as st

from ..plotting.charts import confusion_plot
from ..data.get_data import is_trained, get_finance_data


def finance_page() -> None:
    """
    1. Check model is fitted
    2. Show finance result
    """
    is_model_trained = is_trained()

    # Header.
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.title("Finance")
    with col2:
        if is_model_trained:
            st.success("Модель обучена")
        else:
            st.error("Модель не обучена. Сначала обучите модель.", icon="ℹ️")
            return

    _show_finance()


def _show_finance() -> None:
    """
    1. Request finance data.
    2. Show metrics with saved money and loss %
    3. Show confusion plot
    """
    finance_data = get_finance_data()
    cm_baseline = np.array(finance_data["cm_baseline"])
    cm_tuned = np.array(finance_data["cm_tuned"])
    saved_money_b = np.round(finance_data["baseline_data"][0])
    percent_lost_b = finance_data["baseline_data"][1]
    saved_money_t = np.round(finance_data["tuned_data"][0])
    percent_lost_t = finance_data["tuned_data"][1]

    # Metrics
    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        delta_money = saved_money_t - saved_money_b
        st.metric("Saved money total, $", saved_money_t, delta_money)
    with col2:
        delta = np.round(percent_lost_t - percent_lost_b, 3)
        st.metric("Loss in %", percent_lost_t, delta, "inverse")

    # Conclusion
    msg = (
        f"Видим что количество ложно-положительных предсказаний FP (отказали платежеспособному"
        f" клиенту) намного больше количества ложно-отрицательных FN (дали кредит тому кто не"
        f" вернет). Tuned Catboost модель уменьшает количество ошибок 1 и 2 рода и позволяет"
        f" сохранить {np.round(delta_money * 1e-6, 1)} млн, что на {delta} % больше чем для"
        f" Baseline модели."
    )
    st.write(msg)

    # Plot
    fig = confusion_plot(cm_baseline, cm_tuned, True)
    st.pyplot(fig)
