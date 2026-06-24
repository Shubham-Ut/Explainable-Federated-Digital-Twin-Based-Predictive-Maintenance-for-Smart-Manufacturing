import os
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_xgboost_model(df):
    drop_columns = [
        "engine_id",
        "RUL",
        "machine_status",
        "health_score"
    ]

    X = df.drop(columns=drop_columns)
    y = df["RUL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results = {
        "Model": "XGBoost",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_fd001.pkl")

    return model, results