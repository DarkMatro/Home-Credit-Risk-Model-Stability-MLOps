"""
Exploratory data analysis page

This file contains the following functions:
    * eda_page

Version: 1.0
"""

import streamlit as st
from ..data.get_data import get_dataframe
from ..data.config import get_config
from ..stat_test.stat_test import get_stat_test_result
from ..plotting.get_figure import get_image


def eda_page() -> None:
    """EDA page.
    with dataframe, hypotheses and plots.
    """
    st.title("Exploratory data analysis")
    config = get_config()
    texty = get_config("texts")
    plot_params = get_config("eda_plots")

    # dataframe
    cols = config["preprocessing"]["relevant_columns"]
    df = get_dataframe(
        file_path=config["preprocessing"]["train_data_path"], columns=cols, n_rows=5
    )
    st.dataframe(df, use_container_width=True)

    # tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(texty["hyp_titles"])

    with tab1:
        st.markdown(texty["hypotheses"][0])
        img = get_image(**plot_params["plot_1_1"])
        st.image(
            img, "Распределение возраста клиентов по категориям дохода", 1900, True
        )
        img_2: bytes = get_image(**plot_params["plot_1_2"])
        st.image(img_2, use_column_width=True)
        with st.expander("Вывод"):
            st.write(texty["hypotheses_conclusion"][0])
    with tab2:
        st.markdown(texty["hypotheses"][1])
        img = get_image(**plot_params["plot_2"])
        st.image(
            img,
            "Размер ежемесячного платежа в зависимости от размера налога",
            use_column_width=True,
        )
        stat_text = get_stat_test_result(**plot_params["stat_2"])
        st.markdown(stat_text)
        with st.expander("Вывод"):
            st.write(texty["hypotheses_conclusion"][1])
    with tab3:
        st.markdown(texty["hypotheses"][2])
        img = get_image(**plot_params["plot_3"])
        st.image(img, "Стаж в разрезе целевого признака", 1900, True)
        stat_text = get_stat_test_result(**plot_params["stat_3"])
        st.markdown(stat_text)
        with st.expander("Вывод"):
            st.write(texty["hypotheses_conclusion"][2])
    with tab4:
        st.markdown(texty["hypotheses"][3])
        img = get_image(**plot_params["plot_4"])
        st.image(
            img,
            "Зависимость размера кредита от непогашенного долга",
            use_column_width=True,
        )
        stat_text = get_stat_test_result(**plot_params["stat_4"])
        st.markdown(stat_text)
        with st.expander("Вывод"):
            st.write(texty["hypotheses_conclusion"][3])
    with tab5:
        st.markdown(texty["hypotheses"][4])
        img = get_image(**plot_params["plot_5"])
        st.image(
            img,
            "Дата последней просрочки в разрезе целевого признака",
            use_column_width=True,
        )
        stat_text = get_stat_test_result(**plot_params["stat_5"])
        st.markdown(stat_text)
        with st.expander("Вывод"):
            st.write(texty["hypotheses_conclusion"][4])
    with tab6:
        st.markdown(texty["hypotheses"][5])
        img = get_image(**plot_params["plot_6_1"])
        st.image(
            img,
            "Зависимость общей суммы закрытых кредитов от количества закрытых кредитов",
            use_column_width=True,
        )
        stat_text = get_stat_test_result(**plot_params["stat_6_1"])
        st.markdown(stat_text)
        img = get_image(**plot_params["plot_6_2"])
        st.image(
            img,
            "Общая сумма всех закрытых кредитов в разрезе target",
            use_column_width=True,
        )
        stat_text = get_stat_test_result(**plot_params["stat_6_2"])
        st.markdown(stat_text)
        with st.expander("Вывод"):
            st.write(texty["hypotheses_conclusion"][5])
