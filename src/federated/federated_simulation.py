import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DROP_COLUMNS = [
    "engine_id",
    "RUL",
    "machine_status",
    "health_score"
]


def train_local_model(client_path, client_id):
    df = pd.read_csv(client_path)

    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    y = df["RUL"]

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

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    os.makedirs("models/federated", exist_ok=True)

    model_path = f"models/federated/client_{client_id}_model.pkl"
    joblib.dump(model, model_path)

    results = {
        "Client": f"Client {client_id}",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Model_Path": model_path
    }

    return results


def run_federated_simulation(client_paths):
    all_results = []

    for i, path in enumerate(client_paths):
        print(f"\nTraining Client {i + 1} model...")

        result = train_local_model(path, i + 1)

        all_results.append(result)

        print(result)

    results_df = pd.DataFrame(all_results)

    os.makedirs("reports/results", exist_ok=True)

    results_df.to_csv(
        "reports/results/federated_client_results.csv",
        index=False
    )

    avg_results = {
        "Average MAE": results_df["MAE"].mean(),
        "Average RMSE": results_df["RMSE"].mean(),
        "Average R2": results_df["R2"].mean()
    }

    return results_df, avg_results