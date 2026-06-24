import os
import pandas as pd

from src.digital_twin.twin_engine import create_digital_twin
from src.explainability.shap_analysis import (
    generate_shap_summary,
    get_single_prediction_explanation
)


ENGINEERED_DATA_PATH = "data/processed/train_engineered.csv"


def main():
    print("Starting Phase 4: Digital Twin + SHAP Explainability")

    if not os.path.exists(ENGINEERED_DATA_PATH):
        raise FileNotFoundError(
            "train_engineered.csv not found. Please run Phase 3 first."
        )

    df = pd.read_csv(ENGINEERED_DATA_PATH)

    print("Engineered data loaded successfully.")
    print("Data Shape:", df.shape)

    row_index = 100

    engine_id = df.iloc[row_index]["engine_id"]
    max_rul = df["RUL"].max()

    predicted_rul, explanation_df = get_single_prediction_explanation(
        df,
        row_index=row_index,
        top_n=10
    )

    digital_twin_output = create_digital_twin(
        engine_id=engine_id,
        predicted_rul=predicted_rul,
        max_rul=max_rul
    )

    print("\n========== DIGITAL TWIN OUTPUT ==========")

    for key, value in digital_twin_output.items():
        print(f"{key}: {value}")

    print("\n========== TOP SHAP EXPLANATION ==========")
    print(explanation_df)

    print("\nGenerating SHAP summary plot...")
    generate_shap_summary(df, sample_size=500)

    print("\nPhase 4 completed successfully!")
    print("Saved files:")
    print("reports/figures/shap_summary.png")
    print("reports/results/single_prediction_shap_explanation.csv")


if __name__ == "__main__":
    main()