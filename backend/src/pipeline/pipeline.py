"""
Training pipeline

This file contains the following functions:
    * pipeline_training

Version: 1.0
"""

from ..data.config import get_config
from ..data.get_data import get_dataset
from ..transform.transform import pipeline_preprocess
from ..data.split_dataset import split_train_test


def pipeline_training() -> bool:
    """
    Preprocess dataset, split and save it.

    Returns
    -------
    out: bool
    """
    config = get_config()
    preproc = config["preprocessing"]

    # get data
    train_data = get_dataset(preproc["raw_data_path"])

    # preprocessing
    train_data = pipeline_preprocess(train_data, is_predict=False, **preproc)

    # split data
    df_train, df_test = split_train_test(dataset=train_data, **preproc)

    return not (df_train.empty and df_test.empty)
