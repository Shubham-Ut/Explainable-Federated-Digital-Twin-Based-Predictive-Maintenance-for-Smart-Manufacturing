import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


PROCESSED_DATA_PATH = "data/processed/train_processed.csv"
MODEL_PATH = "models/random_forest_fd001.pkl"

FIGURES_DIR = "reports/figures"
RESULTS_DIR = "reports/results"


def create_directories():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load_processed_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Processed data not found at {PROCESSED_DATA_PATH}. "
            "Please run Phase 1 first."
        )

    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df


def dataset_overview(df):
    print("\n========== DATASET OVERVIEW ==========")
    print("Dataset Shape:", df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nBasic Statistics:")
    print(df.describe())

    df.describe().to_csv(f"{RESULTS_DIR}/dataset_statistics.csv")


def correlation_with_rul(df):
    print("\n========== CORRELATION WITH RUL ==========")

    corr = df.corr(numeric_only=True)

    rul_corr = corr["RUL"].sort_values(ascending=False)

    print(rul_corr)

    rul_corr.to_csv(f"{RESULTS_DIR}/rul_correlation.csv")

    plt.figure(figsize=(10, 8))
    rul_corr.drop("RUL").plot(kind="bar")
    plt.title("Feature Correlation with RUL")
    plt.xlabel("Features")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rul_correlation.png")
    plt.close()

    plt.figure(figsize=(18, 12))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/correlation_matrix.png")
    plt.close()

    return rul_corr


def plot_rul_distribution(df):
    print("\nSaving RUL distribution graph...")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["RUL"], bins=50, kde=True)
    plt.title("RUL Distribution")
    plt.xlabel("Remaining Useful Life")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/rul_distribution.png")
    plt.close()


def plot_sensor_trends(df):
    print("\nSaving sensor trend graphs...")

    important_sensors = [
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_11",
        "sensor_12",
        "sensor_15",
        "sensor_17",
        "sensor_20",
        "sensor_21"
    ]

    available_sensors = [
        sensor for sensor in important_sensors if sensor in df.columns
    ]

    engine_ids = df["engine_id"].unique()[:3]

    for engine_id in engine_ids:
        engine_data = df[df["engine_id"] == engine_id]

        for sensor in available_sensors:
            plt.figure(figsize=(10, 5))
            plt.plot(engine_data["cycle"], engine_data[sensor])
            plt.title(f"{sensor} Trend for Engine {engine_id}")
            plt.xlabel("Cycle")
            plt.ylabel(sensor)
            plt.tight_layout()
            plt.savefig(
                f"{FIGURES_DIR}/{sensor}_engine_{engine_id}_trend.png"
            )
            plt.close()


def feature_importance_analysis(df):
    print("\n========== FEATURE IMPORTANCE ==========")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please run Phase 1 first."
        )

    model = joblib.load(MODEL_PATH)

    drop_columns = ["engine_id", "RUL"]

    X = df.drop(columns=drop_columns)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 15 Important Features:")
    print(importance_df.head(15))

    importance_df.to_csv(
        f"{RESULTS_DIR}/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=importance_df.head(15),
        x="Importance",
        y="Feature"
    )
    plt.title("Top 15 Feature Importance - Random Forest")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/feature_importance.png")
    plt.close()

    return importance_df


def plot_top_feature_vs_rul(df, importance_df):
    print("\nSaving top feature vs RUL graphs...")

    top_features = importance_df["Feature"].head(6).tolist()

    for feature in top_features:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x=df[feature],
            y=df["RUL"],
            alpha=0.4
        )
        plt.title(f"{feature} vs RUL")
        plt.xlabel(feature)
        plt.ylabel("RUL")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/{feature}_vs_rul.png")
        plt.close()


def main():
    create_directories()

    print("Starting Phase 2: EDA and Sensor Analysis...")

    df = load_processed_data()

    dataset_overview(df)

    correlation_with_rul(df)

    plot_rul_distribution(df)

    plot_sensor_trends(df)

    importance_df = feature_importance_analysis(df)

    plot_top_feature_vs_rul(df, importance_df)

    print("\nPhase 2 completed successfully!")
    print(f"Graphs saved in: {FIGURES_DIR}")
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()