import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap


MODEL_PATH = "models/xgboost_fd001.pkl"
FIGURES_DIR = "reports/figures"
RESULTS_DIR = "reports/results"


DROP_COLUMNS = [
    "engine_id",
    "RUL",
    "machine_status",
    "health_score"
]


def prepare_features(df):
    """
    Prepare features exactly like XGBoost training.
    """

    X = df.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    return X


def generate_shap_summary(df, sample_size=500):
    """
    Generate SHAP summary plot for XGBoost model.
    """

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run Phase 3 first."
        )

    model = joblib.load(MODEL_PATH)

    X = prepare_features(df)

    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X.copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        show=False
    )

    plt.savefig(
        f"{FIGURES_DIR}/shap_summary.png",
        bbox_inches="tight"
    )
    plt.close()

    print("SHAP summary plot saved.")

    return shap_values, X_sample


def get_single_prediction_explanation(df, row_index=0, top_n=10):
    """
    Get top SHAP contributing features for one prediction.
    """

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run Phase 3 first."
        )

    model = joblib.load(MODEL_PATH)

    X = prepare_features(df)

    if row_index >= len(X):
        raise IndexError("row_index is out of range.")

    single_row = X.iloc[[row_index]]

    prediction = model.predict(single_row)[0]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(single_row)

    explanation_df = pd.DataFrame({
        "Feature": single_row.columns,
        "Feature_Value": single_row.iloc[0].values,
        "SHAP_Contribution": shap_values[0]
    })

    explanation_df["Absolute_Contribution"] = explanation_df[
        "SHAP_Contribution"
    ].abs()

    explanation_df = explanation_df.sort_values(
        by="Absolute_Contribution",
        ascending=False
    )

    top_explanation = explanation_df.head(top_n)

    top_explanation.to_csv(
        f"{RESULTS_DIR}/single_prediction_shap_explanation.csv",
        index=False
    )

    return prediction, top_explanation