import pandas as pd


IMPORTANT_SENSORS = [
    "sensor_11",
    "sensor_9",
    "sensor_4",
    "sensor_12",
    "sensor_14",
    "sensor_7",
    "sensor_15",
    "sensor_21",
    "sensor_2",
    "sensor_3",
    "sensor_20",
    "sensor_8",
    "sensor_13",
]


def add_health_score(df):
    max_rul = df["RUL"].max()
    df["health_score"] = (df["RUL"] / max_rul) * 100
    return df


def add_machine_status(df):
    def status(rul):
        if rul > 80:
            return "Healthy"
        elif rul > 30:
            return "Warning"
        else:
            return "Critical"

    df["machine_status"] = df["RUL"].apply(status)
    return df


def add_rolling_features(df):
    df = df.sort_values(["engine_id", "cycle"])

    for sensor in IMPORTANT_SENSORS:
        if sensor in df.columns:
            df[f"{sensor}_rolling_mean"] = (
                df.groupby("engine_id")[sensor]
                .rolling(window=5, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"{sensor}_change"] = (
                df.groupby("engine_id")[sensor]
                .diff()
                .fillna(0)
            )

    return df


def create_engineered_features(df):
    df = add_health_score(df)
    df = add_machine_status(df)
    df = add_rolling_features(df)
    return df


def save_engineered_data(df, path="data/processed/train_engineered.csv"):
    df.to_csv(path, index=False)