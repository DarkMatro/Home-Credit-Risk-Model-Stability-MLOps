"""
Backend part of the project

This script allows the user to get data, train models and predict.


This file contains the following functions:
    * train_preprocess
    * train_optuna
    * train
    * predict
    * predict_from_file
    * update_shap
    * finance

Version: 1.0
"""

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.responses import Response

from src import (
    pipeline_training,
    find_optimal_params,
    get_finance_data,
    train_model,
    pipeline_predict,
    save_shap,
)

app = FastAPI()


@app.post("/train_preprocess")
async def train_preprocess() -> dict:
    """
    Preprocess datasets before optuna search.

    Returns
    -------
    out: dict
    """
    is_preprocessed = pipeline_training()
    return {"is_preprocessed": is_preprocessed}


@app.post("/train_optuna")
async def train_optuna() -> dict:
    """
    Find best params using optuna.

    Returns
    -------
    out: dict
    """
    study_result: dict = find_optimal_params()
    return study_result


@app.post("/train")
async def train() -> dict:
    """
    Train CatBoost model with best params found by optuna.

    Returns
    -------
    out: dict
    """
    is_trained: bool = train_model()
    return {"is_trained": is_trained}


@app.post("/predict")
def predict(client_info: dict) -> dict:
    """
    Predict using input data from form by user.

    Parameters
    ----------
    client_info: dict
        input data

    Returns
    -------
    out: dict
    """
    y_pred, y_score = pipeline_predict(client_info=client_info)
    result = {"target": int(y_pred[0]), "score": float(np.round(y_score[0], 3))}
    return result


@app.post("/predict_from_file")
def predict_from_file(file: UploadFile = File(...)) -> Response:
    """
    Predict using uploaded file.

    Parameters
    ----------
    file: UploadFile

    Returns
    -------
    out: Response
    """
    df = pd.read_parquet(file.file)
    file.file.close()
    y_pred, y_score = pipeline_predict(dataset=df)
    cols_order = ["target", "score"] + df.columns.tolist()
    df["target"] = y_pred
    df["score"] = y_score
    df = df.loc[:, cols_order]
    return Response(df.to_json(), media_type="application/json")


@app.get("/update_shap")
async def update_shap() -> dict:
    """
    Calculate SHAP values for tuned CatBoost model and save it.

    Returns
    -------
    out: dict
    """
    is_shap_updated: bool = save_shap()
    return {"is_shap_updated": is_shap_updated}


@app.get("/finance")
async def finance() -> dict:
    """
    Evaluate finance data for baseline and tuned model.

    Returns
    -------
    out: dict
    """
    finance_data = get_finance_data()
    return finance_data


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=2080)
