import os
import pandas as pd

from src.explainability.shap_analysis import get_single_prediction_explanation
from src.digital_twin.twin_engine import create_digital_twin
from src.rag.maintenance_assistant import generate_maintenance_recommendation
from src.llm.maintenance_report_generator import generate_ai_maintenance_report


DATA_PATH = "data/processed/train_engineered.csv"
RESULTS_DIR = "reports/results"


def extract_shap_reasons(explanation_df, top_n=5):
    reasons = []

    for _, row in explanation_df.head(top_n).iterrows():
        reasons.append(row["Feature"])

    return reasons


def main():
    print("Starting Phase 8: AI Maintenance Engineer")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("train_engineered.csv not found. Run Phase 3 first.")

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

    _, retrieved_docs = generate_maintenance_recommendation(
        failure_risk=digital_twin_output["Failure Risk"],
        machine_status=digital_twin_output["Machine Status"],
        predicted_rul=digital_twin_output["Predicted RUL"],
        shap_reasons=shap_reasons
    )

    print("\nGenerating AI Maintenance Report...")

    ai_report = generate_ai_maintenance_report(
        digital_twin_output=digital_twin_output,
        shap_reasons=shap_reasons,
        retrieved_docs=retrieved_docs
    )

    print("\n========== AI MAINTENANCE REPORT ==========\n")
    print(ai_report)

    with open(
        f"{RESULTS_DIR}/ai_maintenance_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(ai_report)

    print("\nPhase 8 completed successfully!")
    print("Saved file:")
    print("reports/results/ai_maintenance_report.txt")


if __name__ == "__main__":
    main()