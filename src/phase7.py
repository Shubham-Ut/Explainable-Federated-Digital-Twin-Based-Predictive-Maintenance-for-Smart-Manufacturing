import os
import pandas as pd

from src.explainability.shap_analysis import get_single_prediction_explanation
from src.digital_twin.twin_engine import create_digital_twin
from src.rag.maintenance_assistant import generate_maintenance_recommendation


DATA_PATH = "data/processed/train_engineered.csv"
RESULTS_DIR = "reports/results"


def extract_shap_reasons(explanation_df, top_n=5):
    reasons = []

    for _, row in explanation_df.head(top_n).iterrows():
        reasons.append(row["Feature"])

    return reasons


def main():
    print("Starting Phase 7: RAG Maintenance Assistant")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "train_engineered.csv not found. Please run Phase 3 first."
        )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    row_index = 100

    engine_id = df.iloc[row_index]["engine_id"]
    max_rul = 125

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

    shap_reasons = extract_shap_reasons(
        explanation_df,
        top_n=5
    )

    recommendation, retrieved_docs = generate_maintenance_recommendation(
        failure_risk=digital_twin_output["Failure Risk"],
        machine_status=digital_twin_output["Machine Status"],
        predicted_rul=digital_twin_output["Predicted RUL"],
        shap_reasons=shap_reasons
    )

    print("\n========== DIGITAL TWIN OUTPUT ==========")
    for key, value in digital_twin_output.items():
        print(f"{key}: {value}")

    print("\n========== RAG MAINTENANCE RECOMMENDATION ==========")
    print(recommendation)

    with open(
        f"{RESULTS_DIR}/rag_maintenance_recommendation.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(recommendation)

    print("\nPhase 7 completed successfully!")
    print("Saved file:")
    print("reports/results/rag_maintenance_recommendation.txt")


if __name__ == "__main__":
    main()