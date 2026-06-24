import os
import pandas as pd

from src.models.anomaly_detection import (
    train_anomaly_detector,
    predict_single_anomaly
)


DATA_PATH = "data/processed/train_engineered.csv"


def main():
    print("Starting Phase 5: Anomaly Detection")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "train_engineered.csv not found. Please run Phase 3 first."
        )

    df = pd.read_csv(DATA_PATH)

    print("Data loaded successfully.")
    print("Data Shape:", df.shape)

    model, scaler, result_df = train_anomaly_detector(df)

    print("\nAnomaly detector trained successfully.")

    total_records = len(result_df)
    anomaly_count = (result_df["anomaly_status"] == "Anomaly").sum()
    normal_count = (result_df["anomaly_status"] == "Normal").sum()

    print("\n========== ANOMALY SUMMARY ==========")
    print("Total Records:", total_records)
    print("Normal Records:", normal_count)
    print("Anomaly Records:", anomaly_count)
    print("Anomaly Percentage:", round((anomaly_count / total_records) * 100, 2), "%")

    sample_row = df.iloc[[100]]

    prediction = predict_single_anomaly(sample_row)

    print("\n========== SAMPLE ANOMALY PREDICTION ==========")
    for key, value in prediction.items():
        print(f"{key}: {value}")

    print("\nPhase 5 completed successfully!")
    print("Saved files:")
    print("models/anomaly_detector.pkl")
    print("models/anomaly_scaler.pkl")
    print("reports/results/anomaly_detection_results.csv")


if __name__ == "__main__":
    main()