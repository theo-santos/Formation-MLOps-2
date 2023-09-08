import os
import time

import joblib
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

import xgboost as xgb


def train_model_with_io(features_path: str, model_registry_folder: str, xgb=False) -> None:
    features = pd.read_parquet(features_path)
    if xgb:
        eval_df = train_xgb_model(features, model_registry_folder)
    else:
        eval_df = train_model(features, model_registry_folder)
    return eval_df


def train_model(features: pd.DataFrame, model_registry_folder: str) -> None:
    target = 'Ba_avg'
    X = features.drop(columns=[target])
    y = features[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    model = RandomForestRegressor(n_estimators=25, max_depth=10, n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    time_str = time.strftime('%Y%m%d-%H%M%S')
    eval_df = pd.DataFrame({
        "timestamp": [time_str],
        "rmse": [rmse],
        "mse": [mse]
    })
    eval_df["timestamp"] = pd.to_datetime(eval_df["timestamp"], format='%Y%m%d-%H%M%S')
    joblib.dump(model, os.path.join(model_registry_folder, time_str + ".joblib"))
    return eval_df



def train_xgb_model(features: pd.DataFrame, model_registry_folder: str) -> None:
    target = 'Ba_avg'
    X = features.drop(columns=[target])
    y = features[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    with mlflow.start_run():
        mlflow.xgboost.autolog(log_models=True)
        # Create regression matrices
        dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)
        # Define hyperparameters
        params = {"objective": "reg:squarederror"}
        n = 100

        evals = [(dtest_reg, "validation"), (dtrain_reg, "train")]


        model = xgb.train(
        params=params,
        dtrain=dtrain_reg,
        num_boost_round=n,
        evals=evals,
        verbose_eval=10 # Every ten rounds
        )
        y_pred = model.predict(dtest_reg)
        predictions = [round(value) for value in y_pred]
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        time_str = time.strftime('%Y%m%d-%H%M%S')
        eval_df = pd.DataFrame({
            "timestamp": [time_str],
            "rmse": [rmse],
            "mse": [mse]
        })
        eval_df["timestamp"] =  pd.to_datetime(eval_df["timestamp"], format='%Y%m%d-%H%M%S')
    
    joblib.dump(model, os.path.join(model_registry_folder, time_str + 'xgb.joblib'))
    return eval_df


def predict_with_io(features_path: str, model_path: str, predictions_folder: str) -> None:
    features = pd.read_parquet(features_path)
    features = predict(features, model_path)
    time_str = time.strftime('%Y%m%d-%H%M%S')
    features['predictions_time'] = time_str
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, time_str + '.csv'),
                                                         index=False)
    features[['predictions', 'predictions_time']].to_csv(os.path.join(predictions_folder, 'latest.csv'), index=False)


def predict(features: pd.DataFrame, model_path: str) -> pd.DataFrame:
    model = joblib.load(model_path)
    if "xgb" in model_path:
        X = xgb.DMatrix(features, enable_categorical=True)
        features['predictions'] = model.predict(X)
    else: 
        features['predictions'] = model.predict(features)
    return features
