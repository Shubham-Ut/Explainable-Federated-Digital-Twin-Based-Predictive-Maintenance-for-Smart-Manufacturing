from src.federated.split_clients import split_data_into_clients
from src.federated.federated_simulation import run_federated_simulation


def main():
    print("Starting Phase 6: Federated Learning Simulation")

    print("\nSplitting dataset into virtual factories...")
    client_paths = split_data_into_clients(num_clients=3)

    print("\nTraining local models for each client...")
    results_df, avg_results = run_federated_simulation(client_paths)

    print("\n========== FEDERATED CLIENT RESULTS ==========")
    print(results_df)

    print("\n========== AVERAGE FEDERATED RESULTS ==========")
    for key, value in avg_results.items():
        print(f"{key}: {value}")

    print("\nPhase 6 completed successfully!")
    print("Saved files:")
    print("data/processed/federated_clients/")
    print("models/federated/")
    print("reports/results/federated_client_results.csv")


if __name__ == "__main__":
    main()