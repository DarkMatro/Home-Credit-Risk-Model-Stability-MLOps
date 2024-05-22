"""
Prediction from form page streamlit widgets

This file contains the following functions:
    * prediction_page

Version: 1.0
"""

from typing import Any

import numpy as np
import requests
import streamlit as st

from ..data.config import get_endpoint, get_config
from ..data.get_data import is_trained, get_unique_values


def prediction_page() -> None:
    """
    1. Check model is trained
    2. Show form with sliders, input fields
    3. Predict button
    """
    is_model_trained = is_trained()
    endpoint = get_endpoint("predict")
    if "predict_result" not in st.session_state:
        st.session_state.predict_result = None

    # Header.
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.title("Prediction")
    with col2:
        if is_model_trained:
            st.success("Модель обучена")
        else:
            st.error("Модель не обучена. Сначала обучите модель.", icon="ℹ️")
            return

    # Inputs form.
    input_data = _prediction_input_data()

    # Predict.
    if st.button(
        "Predict",
        help="Предсказать вероятность дефолта по кредиту",
        use_container_width=True,
    ):
        with st.spinner():
            output = requests.post(endpoint, timeout=8000, json=input_data)
        st.session_state.predict_result = output.json()

    if st.session_state.predict_result is not None:
        _show_result(st.session_state.predict_result)


def _show_result(result: dict) -> None:
    """
    Show prediction result with target and score for inputted data.

    Parameters
    ----------
    result: dict
        prediction result from request with target and score
    """
    target = result["target"]
    score = result["score"] * 100.0
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        st.metric("Вероятность дефолта, %", np.round(score, 1))
    with col2:
        if target:
            st.warning(
                "Высокая вероятность дефолта. У клиента могут быть проблемы с выплатой",
                icon="🚨",
            )
        elif not target and score > 25.0:
            st.warning(
                "Подозрительная вероятность дефолта. Возможны проблемы с выплатами."
            )
        else:
            st.success(
                "Низкая вероятность дефолта. Клиент скорее всего полностью вернет кредит"
            )


def _prediction_input_data() -> dict:
    """
    Show input fields and return data in it.

    Returns
    -------
    input_data: dict
        data from input fields
    """
    transform_cols = get_config()["preprocessing"]["transform_cols"]
    unique_cat_values = get_unique_values("cat")
    num_values = get_unique_values("num")

    col1, col2, col3, col4, col5 = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
    with col1:
        gender = st.selectbox("Пол", unique_cat_values["gender"])
    with col2:
        age_years = _number_input(
            num_values["age_years"], int, "Возраст", bound_shift=(-2, 30)
        )
    with col3:
        employed_from = _number_input(
            num_values["employedfrom"],
            label="Стаж, лет",
            scale_coef=1 / 365.25,
            bound_shift=(None, 20.0),
        )
    with col4:
        income_type = st.selectbox(
            "Источник основного дохода", unique_cat_values["incometype_1044T"], 2
        )
    with col5:
        is_bid = st.checkbox("Перекрестная продажа", int(num_values["isbidproduct"][2]))
    col1, col2, col3 = st.columns([0.33, 0.33, 0.33])
    with col1:
        main_inc = _number_input(
            num_values["maininc_215A"], label="Размер основного дохода"
        )
    with col2:
        tax_amount_max = _number_input(
            num_values["tax_amount_max"], label="Максимальный размер налога"
        )
    with col3:
        pmt_num = _number_input(
            num_values["pmtnum_254L"],
            int,
            "Общее количество платежей по кредиту, осуществленных клиентом",
            overwrite_values=(0, None),
        )
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        num_cred_active = _number_input(
            num_values["num_cred_active"], int, "Количество активных кредитов"
        )
    with col2:
        cred_amount = _slider_input(
            num_values["credamount_770A"], label="Cумма активных кредитов"
        )
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        num_cred_closed = _number_input(
            num_values["num_cred_closed"], int, "Количество закрытых кредитов"
        )
    with col2:
        mobile_phn_cnt = _slider_input(
            num_values["mobilephncnt_593L"],
            int,
            "Количество клиентов с таким же номером мобильного телефона",
        )
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        total_amount_closed_contracts = _number_input(
            num_values["total_amount_closed_contracts"],
            label="Общая сумма закрытых контрактов",
        )
    with col2:
        annuity = _slider_input(
            num_values["annuity_780A"], label="Размер ежемесячного платежа"
        )
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        number_overdue_inst_days = _number_input(
            num_values["numberofoverdueinstlmaxdat_148D"],
            int,
            "Сколько дней прошло с последней просрочки платежа"
            " по закрытому договору",
        )
    with col2:
        debt_outstanding_total = _slider_input(
            num_values["debt_outstand_total"], label="Общая сумма непогашенного долга"
        )
    input_data = {
        "gender": gender,
        "age_years": age_years,
        "employedfrom": employed_from,
        "incometype_1044T": income_type,
        "isbidproduct": int(is_bid),
        "maininc_215A": main_inc,
        "tax_amount_max": tax_amount_max,
        "pmtnum_254L": pmt_num,
        "num_cred_active": num_cred_active,
        "credamount_770A": cred_amount,
        "num_cred_closed": num_cred_closed,
        "mobilephncnt_593L": mobile_phn_cnt,
        "total_amount_closed_contracts": total_amount_closed_contracts,
        "annuity_780A": annuity,
        "numberofoverdueinstlmaxdat_148D": number_overdue_inst_days,
        "debt_outstand_total": debt_outstanding_total,
    }
    for col in transform_cols:
        if input_data[col] is not None:
            input_data[col] *= -1
    return input_data


def _slider_input(values: list, number_type: type = float, label: str = "") -> Any:
    """
    streamlit slider

    Parameters
    ----------
    values: list
        0 - min value, 1 - max value, 2 - median value

    number_type: type
        int or float

    label: str
        label of slider

    Returns
    -------
    out: Any
        chosen number in slider
    """
    min_v = number_type(values[0])
    max_v = number_type(values[1])
    median_value = number_type(values[2])
    if number_type == float:
        min_v = np.round(min_v)
        max_v = np.round(max_v)
        median_value = np.round(median_value)
    step = 1 if number_type == int else 1.0
    return st.slider(label, min_v, max_v, median_value, step)


def _number_input(
    values: list,
    number_type: type = float,
    label: str = "",
    overwrite_values: tuple | None = None,
    bound_shift: tuple | None = None,
    scale_coef: float | None = None,
) -> int | float | None:
    """
    streamlit number input

    Parameters
    ----------
    values: list
        0 - min value, 1 - max value, 2 - median value, 3 - 0 if may be None, 1 - can not be None

    number_type: type
        int or float

    label: str
        label of slider

    overwrite_values: tuple, optional
        tuple with 2 values: 0 - overwrite min value, 1 - max value

    bound_shift: tuple, optional
        tuple with 2 values: 0 - shift min value, 1 - max value

    scale_coef: float, optional
        multiply all value by this coefficient

    Returns
    -------
    out: int | float | None
        chosen number
    """
    min_v = number_type(values[0])
    max_v = number_type(values[1])
    median_value = number_type(values[2])

    if scale_coef is not None:
        min_v *= scale_coef
        max_v *= scale_coef
        median_value *= scale_coef

    if bound_shift is not None and bound_shift[0] is not None:
        min_v += bound_shift[0]
    if bound_shift is not None and bound_shift[1] is not None:
        max_v += bound_shift[1]

    if overwrite_values is not None:
        min_v = overwrite_values[0] if overwrite_values[0] is not None else min_v
        max_v = overwrite_values[1] if overwrite_values[1] is not None else max_v

    default_v = None if values[3] else median_value
    step = 1 if number_type == int else 1.0
    help_text = (
        f"Оставить пустым если информации нет. Медиана {median_value}"
        if values[3]
        else ""
    )
    return st.number_input(label, min_v, max_v, default_v, step, help=help_text)
