"""
Home page with description

This file contains the following functions:
    * home_page

Version: 1.0
"""

import streamlit as st
import streamlit_antd_components as sac

from ..data.config import get_config


def home_page() -> None:
    """Home description page."""
    texty = get_config("texts")

    # header
    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.title("Описание проекта")
        lnk = "https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview"
        st.header(f"[Home Credit - Credit Risk Model Stability]({lnk})")
    with col2:
        st.image("../media/images/header.png", use_column_width="auto")

    # body
    st.write(
        "Задача бинарной классификации для предсказания вероятности дефолта по кредиту"
        " (target = 1)."
    )
    st.write(texty["desc"])
    with st.expander("Дополнительное описание"):
        st.write(texty["add_desc"])
    st.markdown("### Описание полей")
    st.markdown(texty["fields_desc"])

    # Contacts
    sac.divider("Made by Oleg Frolov", align="center", color="dark")
    sac.buttons(
        [
            sac.ButtonsItem(icon="google", href=texty["mail"], color="#25C3B0"),
            sac.ButtonsItem(icon="telegram", href=texty["tg"], color="blue"),
        ],
        align="center",
        index=-1,
    )
