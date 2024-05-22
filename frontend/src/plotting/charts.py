"""
Build plots functions

This file contains the following functions:
    -EDA
    * boxplot
    * boxplot_with_stripplot
    * cat_reg_plot
    * displot
    * joint_plot
    * overlapping_densities
    * relplot
    -SHAP
    * beeswarm_plot
    * scatter_plots
    * decision_plot
    * force_plot
    -Finance
    * confusion_plot

Version: 1.0
"""

import re
from itertools import product

import matplotlib.pyplot as plt
import mplcyberpunk
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib.pyplot import Figure
from sklearn.metrics import ConfusionMatrixDisplay


def boxplot(
    x_label: str = "", title: str = "", dark_mode: bool = True, **kwargs
) -> Figure:
    """
    Build boxplot.

    Parameters
    ----------
    x_label: str
        label for x-axis

    title: str
        title of figure

    dark_mode: bool, Default = True
        Use dark theme or not

    Returns
    -------
    fig: figure
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
    sns.boxplot(fill=0, ax=ax, **kwargs)
    if dark_mode:
        mplcyberpunk.make_lines_glow(ax)
    if x_label:
        ax.set_xlabel(x_label)
    if title:
        ax.set_title(title)
    return fig


def boxplot_with_stripplot(
    x_label: str = "",
    title: str = "",
    dark_mode: bool = True,
    use_glow_effect: bool = False,
    **kwargs,
) -> Figure:
    """
    Build boxplot with stripplot.

    Parameters
    ----------
    x_label: str
        label for x-axis

    title: str
        title of the figure

    dark_mode: bool, Default = True
        Use dark theme or not

    use_glow_effect: bool, Default = False

    Returns
    -------
    f: figure
    """
    sns.set_theme(style="ticks", font_scale=1)
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
    f, ax = plt.subplots(figsize=(20, 6))
    sns.boxplot(fill=0, width=0.6, ax=ax, **kwargs)
    if dark_mode and use_glow_effect:
        mplcyberpunk.make_lines_glow(ax, n_glow_lines=10)
    color = "w" if dark_mode else "b"
    # kwargs['data']: pd.DataFrame = kwargs['data'].sample(frac=.1)
    sns.stripplot(color=color, size=1, legend=0, dodge=True, **kwargs)
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    if x_label:
        ax.set(xlabel=x_label)
    sns.despine(trim=True, left=True)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return f


def cat_reg_plot(
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    dark_mode: bool = True,
    **kwargs,
) -> Figure:
    """
    Build catplot with regplot.

    Parameters
    ----------
    x_label: str
        label for x-axis

    y_label: str
        label for y-axis

    title: str
        title of figure

    dark_mode: bool, Default = True
        Use dark theme or not

    Returns
    -------
    fig: figure
    """
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
    fg = sns.catplot(native_scale=True, zorder=1, **kwargs)
    kwargs.pop("hue")
    kwargs.pop("palette")
    kwargs.pop("aspect")
    color = "w" if dark_mode else "black"
    sns.regplot(scatter=False, truncate=True, order=1, color=color, **kwargs)
    fg.set(xscale="log", yscale="log")
    fg.despine(left=True, bottom=True)
    fg.ax.xaxis.grid(True, "minor", linewidth=0.25)
    fg.ax.yaxis.grid(True, "minor", linewidth=0.25)
    if x_label and y_label:
        fg.set_axis_labels(x_label, y_label)
    if title:
        fg.ax.set_title(title)
    # noinspection PyTestUnpassedFixture
    return fg.figure


def displot(
    x_label: str = "", title: str = "", dark_mode: bool = True, **kwargs
) -> Figure:
    """
    Build JointGrid with scatterplot and marginals.

    Parameters
    ----------
    x_label: str
        label for x-axis

    title: str
        title of figure

    dark_mode: bool, Default = True
        Use dark theme or not

    Returns
    -------
    f: figure
    """
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
    fg = sns.displot(kind="kde", common_norm=False, height=6, aspect=3, **kwargs)
    if dark_mode:
        mplcyberpunk.add_gradient_fill(fg.ax)
    fg.ax.set_title(title)
    fg.ax.set_xlabel(x_label)
    plt.tight_layout()
    # noinspection PyTestUnpassedFixture
    return fg.figure


# noinspection PyTestUnpassedFixture
def joint_plot(
    data: pd.DataFrame,
    target_col: str,
    x_label: str = "",
    y_label: str = "",
    dark_mode: bool = True,
    **kwargs,
) -> Figure:
    """
    Build JointGrid with scatterplot and marginals.

    Parameters
    ----------
    data: pd.DataFrame
        Dataset

    target_col: str
        name of target feature

    x_label: str
        label for x-axis

    y_label: str
        label for y-axis

    dark_mode: bool, Default = True
        Use dark theme or not

    Returns
    -------
    f: figure
    """
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
        sns.set_theme(style="white", color_codes=True)

    pj_kwargs = {}
    for par in ["size_order", "sizes"]:
        if par in kwargs:
            pj_kwargs[par] = kwargs.pop(par)
    if "palette" in kwargs:
        pj_kwargs["palette"] = kwargs["palette"]

    g = sns.JointGrid(data, space=0, ratio=17, **kwargs)
    g.plot_joint(
        sns.scatterplot, data=data, size=data[target_col], alpha=0.5, **pj_kwargs
    )

    if x_label and y_label:
        g.set_axis_labels(x_label, y_label)
    g.plot_marginals(sns.rugplot, height=1, alpha=0.1)
    g.ax_joint.set(xscale="log", yscale="log")
    g.figure.set_figwidth(15)
    g.figure.set_figheight(6)
    return g.figure


# noinspection PyTestUnpassedFixture
def overlapping_densities(
    target_col: str,
    category_col: str,
    dark_mode: bool = True,
    title: str = "",
    x_label: str = "",
    **kwargs,
) -> Figure:
    """
    Plot Overlapping densities (‘ridge plot’)

    Parameters
    ----------
    target_col: str
        column name in the data

    category_col: str
        for hue

    dark_mode: bool, Default = True
        Use dark theme or not

    title: str,
        Figure title

    x_label: str,
        Name of x-axis

    Returns
    -------
    fig: figure
    """
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
        sns.set_theme(style="white")
    plt.rcParams["axes.facecolor"] = (0, 0, 0, 0)
    if "label_x_pos" in kwargs:
        label_x_pos = kwargs.pop("label_x_pos")
    else:
        label_x_pos = 0.8
    x_lim = kwargs.pop("x_lim") if "x_lim" in kwargs else None

    # Initialize the FacetGrid object
    g = sns.FacetGrid(row=category_col, hue=category_col, **kwargs)

    # Draw the densities in a few steps
    g.map(
        sns.kdeplot,
        target_col,
        bw_adjust=0.5,
        clip_on=False,
        alpha=1,
        linewidth=1.5,
        fill=not dark_mode,
    )
    g.map(sns.kdeplot, target_col, clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    g.map(_label, target_col, **{"label_x_pos": label_x_pos})

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    if dark_mode:
        for ax in g.axes:
            mplcyberpunk.make_lines_glow(ax=ax[0], n_glow_lines=5)
            mplcyberpunk.add_underglow(ax=ax[0], alpha_underglow=0.1)
    if x_lim is not None:
        g.axes[-1][0].set_xlim(x_lim)
    if title:
        g.axes[0][0].set_title(title)
    if x_label:
        g.axes[-1][0].set_xlabel(x_label)
    return g.figure


def _label(_: pd.Series, color: tuple, label: str, label_x_pos: int = 0.0) -> None:
    """
    Define a simple function to label the plot in axes coordinates

    Parameters
    ----------
    _: pd.Series

    color: tuple
        RGB

    label : str
        Category

    Returns
    -------
    None
    """
    ax = plt.gca()
    ax.text(
        label_x_pos,
        0.2,
        label,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )


def relplot(
    x_label: str = "",
    y_label: str = "",
    title: str = "",
    dark_mode: bool = True,
    **kwargs,
) -> Figure:
    """
    Build relplot.

    Parameters
    ----------
    x_label: str
        label for x-axis

    y_label: str
        label for y-axis

    title: str
        title of figure

    dark_mode: bool, Default = True
        Use dark theme or not

    Returns
    -------
    f: figure
    """
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
    rot = np.random.uniform(-1.0, 1.0)
    cmap = sns.cubehelix_palette(rot=rot, light=0.8, dark=0.2, as_cmap=True)
    sizes = None
    if "sizes" in kwargs:
        sizes = kwargs.pop("sizes")
        if isinstance(sizes, list):
            sizes = (sizes[0], sizes[-1])
    color = "w" if dark_mode else "black"
    fg = sns.relplot(palette=cmap, color=color, sizes=sizes, **kwargs)
    fg.ax.xaxis.grid(True, "minor", linewidth=0.25)
    fg.ax.yaxis.grid(True, "minor", linewidth=0.25)
    fg.set(xscale="log", yscale="log")
    fg.despine(left=True, bottom=True)
    if x_label and y_label:
        fg.set_axis_labels(x_label, y_label)
    if title:
        fg.ax.set_title(title)
    # noinspection PyTestUnpassedFixture
    return fg.figure


def beeswarm_plot(shap_values: shap.Explanation, dark_mode: bool = True) -> Figure:
    """Beeswarm plot

    Parameters
    ----------
    shap_values: np.ndarray
         matrix of SHAP values (# samples x # features).
         Each row sums to the difference between the model output

    dark_mode: bool, default = True
        Use dark theme

    Returns
    ----------
    fig: Figure
    """
    plt.clf()
    plt.ioff()
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
    axis_color = "w" if dark_mode else "black"
    ax = shap.plots.beeswarm(
        shap_values,
        axis_color=axis_color,
        color=plt.get_cmap("cool"),
        show=False,
        max_display=21,
        plot_size=(20, 9),
    )
    plt.tight_layout()
    return ax.get_figure()


def scatter_plots(
    shap_values: shap.Explanation,
    features: list[str],
    color_features: list[str],
    dark_mode: bool = True,
) -> Figure:
    """Plot 4 scatter plots for shap_values by 4 features

    Parameters
    ----------
    shap_values: np.ndarray
         matrix of SHAP values (# samples x # features).
         Each row sums to the difference between the model output
    features: list[str]
        4 feature names

    color_features: list[str]
        4 feature names for color map

    dark_mode: bool, default = True
        Use dark theme

    Returns
    ----------
    fig: Figure
    """
    assert len(features) == 4, "4 feature names must be passed"
    assert len(color_features) == 4, "4 color_features names must be passed"
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axis_color = "w" if dark_mode else "black"
    for i, ax_id in enumerate(list(product([0, 1], repeat=2))):
        shap.plots.scatter(
            shap_values[:, features[i]],
            ax=axes[ax_id],
            show=False,
            color=shap_values[:, color_features[i]],
            cmap=plt.get_cmap("cool"),
            axis_color=axis_color,
        )
    plt.tight_layout()
    return fig


def decision_plot(
    base_value: float,
    shap_values_legacy: np.ndarray,
    feature_names: list[str],
    dark_mode: bool = True,
) -> Figure:
    """Decision plot

    Parameters
    ----------
    base_value: float
        This is the reference value that the feature contributions start from. Usually, this is
         `explainer.expected_value`.
    shap_values_legacy: np.ndarray
         matrix of SHAP values (# samples x # features).
         Each row sums to the difference between the model output
         Taken from explainer.shap_values(X)
    feature_names: list[str]
        List of feature names (# features). If ``None``, names may be derived from the
        ``features`` argument if a Pandas object is provided. Otherwise, numeric feature
        names will be generated.
    dark_mode: bool, default = True
        Use dark theme

    Returns
    ----------
    fig: Figure
    """
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
    plt.close()
    plt.ioff()
    axis_color = "w" if dark_mode else "black"
    shap.plots.decision(
        base_value,
        shap_values_legacy,
        feature_names=feature_names,
        axis_color=axis_color,
        y_demarc_color=axis_color,
        show=False,
    )
    plt.tight_layout()
    fig = plt.figure(plt.gcf().number, dpi=600, figsize=(2, 1))
    return fig


def force_plot(
    base_value: float,
    shap_values_legacy: np.ndarray,
    features: list[str] | None = None,
    feature_names: list[str] | None = None,
    dark_mode: bool = True,
) -> str:
    """Force plot as html code

    Parameters
    ----------
    base_value: float
        This is the reference value that the feature contributions start from. Usually, this is
         `explainer.expected_value`.
    shap_values_legacy: np.ndarray
         matrix of SHAP values (# samples x # features).
         Each row sums to the difference between the model output
         Taken from explainer.shap_values(X)
    features: list[str]
         Matrix of feature values (# features) or (# samples x # features). This provides the
          values of all the
         features, and should be the same shape as the ``shap_values`` argument.
    feature_names: list[str]
         Matrix of feature names
    dark_mode: bool, default = True
        Use dark theme

    Returns
    ----------
    shap_html: str
    """
    force = shap.plots.force(
        base_value,
        shap_values_legacy,
        features=features,
        feature_names=feature_names,
        plot_cmap="CyPU",
    )
    shap_html = f"<head>{shap.getjs()}</head><body>{force.html()}</body>"
    if dark_mode:
        shap_html = re.sub(r"#000", "#FFFFFF", shap_html)
        shap_html = re.sub(r"#fff", "#000000", shap_html)
        shap_html = re.sub(r"#ccc", "#FFFFFF", shap_html)
        shap_html = re.sub(
            r"font-family: arial;", "font-family: arial; color: white;", shap_html
        )
        shap_html = re.sub(r"background: none;", "background: #212946;", shap_html)
        shap_html = "<div style='background-color:#212946;'>" + shap_html + "</div>"
    return shap_html


def confusion_plot(
    cm_baseline: np.ndarray, cm_tuned: np.ndarray, dark_mode: bool = False
) -> Figure:
    """ConfusionMatrixDisplay plot

    Parameters
    ----------
    cm_baseline: np.ndarray
        confusion matrix for baseline model

    cm_tuned: np.ndarray
        confusion matrix for tuned model

    dark_mode: bool, default = True
        Use dark theme

    Returns
    ----------
    fig: Figure
    """
    if dark_mode:
        plt.style.use("cyberpunk")
    else:
        plt.rcdefaults()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    disp_b = ConfusionMatrixDisplay(cm_baseline)
    disp_t = ConfusionMatrixDisplay(cm_tuned)
    disp_b.plot(ax=ax[0])
    disp_t.plot(ax=ax[1])
    ax[0].title.set_text("Confusion Matrix CatBoost Baseline")
    ax[1].title.set_text("Confusion Matrix CatBoost Tuned")
    ax[0].grid(False)
    ax[1].grid(False)
    plt.tight_layout()
    return fig
