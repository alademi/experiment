import os
import pandas as pd


def rank_models_in_csv(directory):
    # Create output directory if it doesn't exist
    output_directory = os.path.join(directory, "ranking")
    os.makedirs(output_directory, exist_ok=True)

    # Get all CSV files in the directory
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]

    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)

        # Ensure numerical columns (ignoring the first column which contains cluster names)
        model_columns = df.columns[1:]

        # Convert non-numeric values to NaN
        df[model_columns] = df[model_columns].apply(pd.to_numeric, errors='coerce')

        # Rank models in each row based on RMSE (lower is better, 1 is best rank)
        ranked_df = df.copy()
        ranked_df[model_columns] = df[model_columns].rank(axis=1, method='min', ascending=True)

        # Save the ranked dataframe with modified filename
        output_file = f"{os.path.splitext(file)[0]}-ranking.csv"
        output_path = os.path.join(output_directory, output_file)
        ranked_df.to_csv(output_path, index=False)

        print(f"Processed and saved: {output_path}")


# Example usage
directory = "/Users/aalademi/PycharmProjects/first_experiment/Model/first_experiment/results/evaluation"
rank_models_in_csv(directory)
