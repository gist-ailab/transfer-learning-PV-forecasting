import os
import pandas as pd
from datetime import timedelta

def filter_csv_by_date(input_dir, base_output_dir, durations):
    """
    Filters CSV files in a directory to retain data for the specified durations.

    Parameters:
        input_dir (str): Directory containing the input CSV files.
        base_output_dir (str): Base directory to save the filtered CSV files.
        durations (dict): Dictionary of durations (e.g., {'2weeks': 14, '1month': 30, '3months': 90, '6months': 180}).
    """
    # Process each file in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            
            # Load the CSV file
            df = pd.read_csv(file_path)

            # Ensure 'timestamp' column is in datetime format
            if 'timestamp' not in df.columns:
                print(f"Skipping {file_name}: 'timestamp' column not found.")
                continue

            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

            # Drop rows with invalid timestamps
            df = df.dropna(subset=['timestamp'])

            # Find the most recent date in the dataset
            max_date = df['timestamp'].max()

            # Filter for each duration
            for duration_name, days in durations.items():
                # Create output directory for the duration if it doesn't exist
                output_dir = os.path.join(base_output_dir, f"{os.path.basename(input_dir)}_{duration_name}")
                os.makedirs(output_dir, exist_ok=True)

                filtered_df = df[df['timestamp'] >= max_date - timedelta(days=days)]

                # Save the filtered DataFrame
                output_file_name = f"{os.path.splitext(file_name)[0]}.csv"
                output_file_path = os.path.join(output_dir, output_file_name)
                filtered_df.to_csv(output_file_path, index=False)

                print(f"Filtered file saved: {output_file_path}")

# Example usage
dataset_name = 'OEDI_Georgia'  # Name of the dataset
input_directory = f"/ailab_mat/dataset/PV/{dataset_name}/processed_data_all"  # Directory containing the original CSV files
base_output_directory = f"/ailab_mat/dataset/PV/{dataset_name}"  # Base directory to save the filtered CSV files
filter_durations = {
    # '2weeks': 14,
    # '1month': 30,
    # '3months': 90,
    # '4months': 120,
    # '6months': 180,
    # '8months': 240,
    # '9months': 270,
    # '12months': 365,
    '16months': 480,

}

filter_csv_by_date(input_directory, base_output_directory, filter_durations)
