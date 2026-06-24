from src.data.load_data import load_train_data
from src.data.preprocess import add_rul_column, remove_constant_columns, save_processed_data
from src.models.random_forest import train_random_forest


def main():
    print("Loading NASA FD001 dataset...")

    train_df = load_train_data()

    print("Raw training data shape:", train_df.shape)

    train_df = add_rul_column(train_df)

    print("RUL column added successfully.")

    train_df, removed_columns = remove_constant_columns(train_df)

    print("Removed constant columns:", removed_columns)

    save_processed_data(train_df)

    print("Processed data saved successfully.")

    model, results = train_random_forest(train_df)

    print("\nRandom Forest Model Results:")
    print("MAE:", results["MAE"])
    print("RMSE:", results["RMSE"])
    print("R2 Score:", results["R2"])

    print("\nPhase 1 completed successfully!")


if __name__ == "__main__":
    main()