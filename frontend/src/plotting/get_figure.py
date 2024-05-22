"""
Script for getting images

This file contains the following functions:
    * get_image

Version: 1.0
"""

import shap

from matplotlib.pyplot import Figure
from matplotlib.image import imread
from streamlit import cache_data

from .charts import (
    overlapping_densities,
    boxplot_with_stripplot,
    joint_plot,
    displot,
    relplot,
    boxplot,
    cat_reg_plot,
    beeswarm_plot,
    scatter_plots,
    decision_plot,
)
from ..data.get_data import get_dataframe
from ..data.config import get_config
from ..data.caching import find_cached_image

IMG_TYPE_TO_FUNC = {
    "overlapping_densities": overlapping_densities,
    "cat_reg_plot": cat_reg_plot,
    "boxplot_with_stripplot": boxplot_with_stripplot,
    "joint_plot": joint_plot,
    "boxplot": boxplot,
    "displot": displot,
    "relplot": relplot,
    "beeswarm_plot": beeswarm_plot,
    "scatter_plots": scatter_plots,
    "decision_plot": decision_plot,
}


def _build_figure(img_type: str, **kwargs) -> Figure:
    """
    Master function to choose function (charts) for plotting.

    Parameters
    ----------
    img_type: str
        One of 'overlapping_densities', 'cat_reg_plot', 'boxplot_with_stripplot', 'joint_plot',
         'joint_plot', 'boxplot', 'displot', 'relplot', 'beeswarm_plot', 'scatter_plots',
          decision_plot'

    Returns
    -------
    str
    """
    assert img_type in IMG_TYPE_TO_FUNC
    config = get_config()
    if "shap_values" not in kwargs and "shap_values_legacy" not in kwargs:
        kwargs["data"] = get_dataframe(config["preprocessing"]["train_data_path"])
    func = IMG_TYPE_TO_FUNC[img_type]
    fig = func(**kwargs)
    return fig


@cache_data()
def get_image(
    img_type: str,
    img_format: str,
    params: dict,
    _shap_values: shap.Explanation | None = None,
    _shap_values_legacy: shap.Explanation | None = None,
) -> str | bytes:
    """
    Requests dataframe from backend.

    Parameters
    ----------
    img_type: str
        One of 'overlapping_densities', 'cat_reg_plot', 'boxplot_with_stripplot', 'joint_plot',
         'joint_plot', 'boxplot', 'displot', 'relplot', 'beeswarm_plot', 'scatter_plots',
          decision_plot'

    img_format: str
        'svg', 'jpeg', 'tiff' and etc

    params: dict
        static params for plots

    _shap_values: shap.Explanation
         matrix of SHAP values (# samples x # features). Each row sums to the difference between
          the model output

    _shap_values_legacy: shap.Explanation
         matrix of SHAP values (# samples x # features). Each row sums to the difference between
          the model output
         Taken from explainer.shap_values(X)
    Returns
    -------
    imt: str | bytes
        str if svg, bytes - image from file
    """
    config = get_config()

    if _shap_values is not None or _shap_values_legacy is not None:
        ref_file_path = config["preprocessing"]["test_data_path"]
    else:
        ref_file_path = config["preprocessing"]["train_data_path"]

    # Get cached image name if exists
    img_exists, file_name = find_cached_image(
        img_type, img_format, ref_file_path=ref_file_path, **params
    )

    if _shap_values is not None:
        params.update({"shap_values": _shap_values})
    if _shap_values_legacy is not None:
        params.update({"_shap_values_legacy": _shap_values_legacy})

    if not img_exists:
        # Create and save new image.
        fig = _build_figure(img_type, **params)
        fig.savefig(file_name, format=img_format, dpi=600)

    if img_format.lower() == "svg":
        with open(file_name, encoding="utf-8") as f:
            img = f.read()
    else:
        img = imread(file_name)
    return img
