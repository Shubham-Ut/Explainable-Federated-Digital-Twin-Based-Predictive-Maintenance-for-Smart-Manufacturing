import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


MODEL_PATH = "models/anomaly_detector.pkl"
SCALER_PATH = "models/anomaly_scaler.pkl"
RESULTS_DIR = "reports/results"


DROP_COLUMNS = [
    "engine_id",
    "RUL",
    "machine_status",
    "health_score"
]


def prepare_anomaly_features(df):
    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    return X


def train_anomaly_detector(df):
    os.makedirs("models", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X = prepare_anomaly_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    model.fit(X_scaled)

    anomaly_labels = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)

    df_result = df.copy()
    df_result["anomaly_label"] = anomaly_labels
    df_result["anomaly_score"] = anomaly_scores

    df_result["anomaly_status"] = df_result["anomaly_label"].apply(
        lambda x: "Anomaly" if x == -1 else "Normal"
    )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    df_result.to_csv(
        f"{RESULTS_DIR}/anomaly_detection_results.csv",
        index=False
    )

    return model, scaler, df_result


def predict_single_anomaly(row_df):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    X = prepare_anomaly_features(row_df)
    X_scaled = scaler.transform(X)

    label = model.predict(X_scaled)[0]
    score = model.decision_function(X_scaled)[0]

    status = "Anomaly" if label == -1 else "Normal"

    return {
        "Anomaly Label": int(label),
        "Anomaly Score": float(score),
        "Anomaly Status": status
    }