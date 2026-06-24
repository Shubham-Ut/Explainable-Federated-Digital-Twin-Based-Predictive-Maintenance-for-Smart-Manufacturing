import os
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_random_forest(train_df):
    drop_columns = ["engine_id", "RUL"]

    X = train_df.drop(columns=drop_columns)
    y = train_df["RUL"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/random_forest_fd001.pkl")

    return model, results