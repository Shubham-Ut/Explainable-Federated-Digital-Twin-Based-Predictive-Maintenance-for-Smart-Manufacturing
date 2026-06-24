import pandas as pd


def add_rul_column(train_df):
    max_cycle = train_df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]

    train_df = train_df.merge(max_cycle, on="engine_id", how="left")
    train_df["RUL"] = train_df["max_cycle"] - train_df["cycle"]

    train_df.drop(columns=["max_cycle"], inplace=True)

    return train_df


def remove_constant_columns(df):
    constant_columns = []

    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_columns.append(col)

    df = df.drop(columns=constant_columns)

    return df, constant_columns


def save_processed_data(train_df, path="data/processed/train_processed.csv"):
    train_df.to_csv(path, index=False)