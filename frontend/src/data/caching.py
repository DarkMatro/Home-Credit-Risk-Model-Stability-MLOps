"""
Functions for caching images

This file contains the following functions:
    * find_cached_image
    * check_shap_values

Version: 1.0
"""

import pathlib
import datetime
import hashlib
import pandas as pd

from .config import get_config


def _file_changed_date(path: str) -> tuple[str, float]:
    """
    Returns file's changed date in iso format like '2024-05-03T23:10:06.861165+00:00 and as msec'

    Parameters
    ----------
    path: str
        String, path object (implementing os. PathLike[str]).

    Returns
    -------
    iso: str
    """
    f_name = pathlib.Path(path)
    msec = f_name.stat().st_mtime
    m_time = datetime.datetime.fromtimestamp(msec)
    iso = m_time.isoformat()
    return iso, msec


def find_cached_image(
    img_type: str, img_format: str, ref_file_path: str, **kwargs
) -> tuple[bool, str]:
    """
    Check that image already created in cached folder.
    Image is looking for hash. Hash is result of img_type + img_format + kwargs that means requests
     of image with
    same parameters will return same image.

    Parameters
    ----------
    img_type: str
        One of 'overlapping_densities', 'cat_reg_plot', 'boxplot_with_stripplot', 'joint_plot',
         'joint_plot', 'boxplot', 'displot', 'relplot', 'beeswarm_plot', 'scatter_plots',
          decision_plot'

    img_format: str
        'svg', 'jpeg', 'tiff' and etc

    ref_file_path: str
        path to file to check its changed datetime

    Returns
    -------
    out: tuple[bool, str]
        has_cached_image, path to cached image
    """
    img_format = img_format.lower()
    v = str(img_type) + str(kwargs) + img_format
    hex_hash = hash_value(v)
    config = get_config()
    cache_path = config["caching"]["cached_images_path"]
    cache_info_path = config["caching"]["cached_images_info_path"]

    iso, msec = _file_changed_date(ref_file_path)
    new_record = {"hash": [hex_hash], "msec": [msec], "date": [iso]}

    if not pathlib.Path(cache_info_path).exists():
        _create_info_dataframe(cache_info_path, new_record)
        return False, cache_path + "0." + img_format

    df = pd.read_parquet(cache_info_path)
    df_q = df.query(f"msec >= {msec} and hash == {hex_hash}")

    if not df_q.empty:
        f_name = cache_path + f"{df_q.index[-1]}.{img_format}"
        if pathlib.Path(f_name).exists():
            return True, f_name

    _add_info_record(cache_info_path, new_record)
    return False, cache_path + f"{df.index[-1] + 1}.{img_format}"


def hash_value(value: str) -> int:
    """
    Return hash value.

    Parameters
    ----------
    value: str
        Hash it

    Returns
    -------
    hex_hash: int
        hashed int
    """
    hash_obj = hashlib.shake_128(value.encode())
    hex_hash = int(hash_obj.hexdigest(8), base=16)
    return hex_hash


def _create_info_dataframe(path: str, first_record: dict) -> None:
    """
    Create master table to control cached images.
    Tables has index - hash, columns: msec and date

    Parameters
    ----------
    path: str
        cached_images_info_path

    first_record: dict
        first row of new dataset
    """
    df = pd.DataFrame(first_record)
    df.to_parquet(path)


def _add_info_record(path: str, new_record: dict) -> None:
    """
    Appends row to master table.
    Tables has index - hash, columns: msec and date

    Parameters
    ----------
    path: str
        cached_images_info_path

    new_record: dict
        new row of new dataset
    """
    df = pd.read_parquet(path)
    df = pd.concat([df, pd.DataFrame(new_record)])
    df.reset_index(inplace=True, drop=True)
    df.to_parquet(path)


def check_shap_values() -> bool:
    """
    Checking that shap values need to be updated. If they are not there or they are outdated.

    Returns
    -------
    out: bool
        True - shap values exists and updated
        False - shap values needs to be recalculated
    """
    config = get_config()
    shap_values_data_path = config["shap"]["shap_values_data_path"]
    if not pathlib.Path(shap_values_data_path).exists():
        return False
    _, msec_shap = _file_changed_date(shap_values_data_path)
    _, msec_model = _file_changed_date(config["train"]["tuned_model_path"])
    if msec_shap < msec_model:
        return False
    return True
