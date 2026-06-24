import os
import pandas as pd


DATA_PATH = "data/processed/train_engineered.csv"
CLIENTS_DIR = "data/processed/federated_clients"


def split_data_into_clients(num_clients=3):
    os.makedirs(CLIENTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    engine_ids = df["engine_id"].unique()

    client_engine_ids = [
        engine_ids[i::num_clients]
        for i in range(num_clients)
    ]

    client_paths = []

    for i, ids in enumerate(client_engine_ids):
        client_df = df[df["engine_id"].isin(ids)]

        path = f"{CLIENTS_DIR}/client_{i + 1}.csv"
        client_df.to_csv(path, index=False)

        client_paths.append(path)

        print(f"Client {i + 1} data saved: {client_df.shape}")

    return client_paths