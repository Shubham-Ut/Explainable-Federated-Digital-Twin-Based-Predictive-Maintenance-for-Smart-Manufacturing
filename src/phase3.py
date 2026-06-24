import os
import pandas as pd

from src.data.feature_engineering import (
    create_engineered_features,
    save_engineered_data
)

from src.models.xgboost_model import train_xgboost_model


def main():
    print("Starting Phase 3: Feature Engineering + XGBoost")

    input_path = "data/processed/train_processed.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError("train_processed.csv not found. Run Phase 1 first.")

    df = pd.read_csv(input_path)

    print("Original Data Shape:", df.shape)

    df = create_engineered_features(df)

    print("Engineered Data Shape:", df.shape)

    save_engineered_data(df)

    print("Engineered data saved at data/processed/train_engineered.csv")

    model, xgb_results = train_xgboost_model(df)

    print("\nXGBoost Results:")
    print("MAE:", xgb_results["MAE"])
    print("RMSE:", xgb_results["RMSE"])
    print("R2 Score:", xgb_results["R2"])

    rf_results = {
        "Model": "Random Forest",
        "MAE": 25.450433729101043,
        "RMSE": 35.93435947447518,
        "R2": 0.7173700821322665
    }

    comparison_df = pd.DataFrame([rf_results, xgb_results])

    os.makedirs("reports/results", exist_ok=True)

    comparison_df.to_csv(
        "reports/results/model_comparison.csv",
        index=False
    )

    print("\nModel Comparison:")
    print(comparison_df)

    print("\nPhase 3 completed successfully!")


if __name__ == "__main__":
    main()