import pandas as pd


def get_column_names():
    columns = ["engine_id", "cycle"]
    columns += [f"op_setting_{i}" for i in range(1, 4)]
    columns += [f"sensor_{i}" for i in range(1, 22)]
    return columns


def load_train_data(path="data/raw/train_FD001.txt"):
    columns = get_column_names()

    train_df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    train_df.columns = columns
    return train_df


def load_test_data(path="data/raw/test_FD001.txt"):
    columns = get_column_names()

    test_df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )

    test_df.columns = columns
    return test_df


def load_rul_data(path="data/raw/RUL_FD001.txt"):
    rul_df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["true_RUL"]
    )

    return rul_df