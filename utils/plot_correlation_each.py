import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_feature_vs_active_power(data_dir, save_dir, dataset_name):
    """
    Plots the relationship between Active Power and various features for CSV files in a directory.
    Also plots a combined correlation plot for all data.

    Parameters:
    - data_dir (str): The directory containing the CSV files.
    - save_dir (str): The directory to save the plots.
    - dataset_name (str): The name of the dataset to adjust features and titles accordingly.
    """
    print("Plotting")
    if dataset_name == 'OEDI_California' or dataset_name == 'OEDI_Georgia':
        features = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Wind_Speed']
        colors = ['blue', 'green', 'purple']
        titles = ['Active Power [kW] vs Global Horizontal Radiation [w/m²]',
                  'Active Power [kW] vs Weather Temperature [℃]',
                  'Active Power [kW] vs Wind Speed [m/s]']
    else:
        features = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity', 'Wind_Speed']
        colors = ['blue', 'green', 'red', 'purple']
        titles = ['Active Power [kW] vs Global Horizontal Radiation [w/m²]',
                  'Active Power [kW] vs Weather Temperature [℃]',
                  'Active Power [kW] vs Weather Relative Humidity [%]',
                  'Active Power [kW] vs Wind Speed [m/s]']

    data_list = [file for file in os.listdir(data_dir) if file.endswith('.csv')]
    os.makedirs(save_dir, exist_ok=True)

    combined_df = pd.DataFrame()

    for file in tqdm(data_list, desc='Processing data files'):
        data_path = os.path.join(data_dir, file)
        df = pd.read_csv(data_path)
        
        # Drop 'timestamp' column if present
        if 'timestamp' in df.columns:
            df.drop(columns=['timestamp'], inplace=True)

        # Normalize Active_Power by dividing by the maximum value for each site
        df['Normalized_Active_Power'] = df['Active_Power'] / df['Active_Power'].max()

        # Append the processed data to the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

        # Calculate the correlation between Active_Power and other features for the individual file
        correlations = df.corr()['Active_Power']

        # Plot the relationship between Active Power and other features for the individual file
        plt.figure(figsize=(18, 12))
        for i, feature in enumerate(features):
            plt.subplot(len(features), 1, i + 1)
            corr_value = correlations.get(feature, 0)  # Get correlation, default to 0 if not present
            plt.scatter(df[feature], df['Normalized_Active_Power'], color=colors[i], marker='x', alpha=0.6)
            plt.xlabel(f'{feature}')
            plt.ylabel('Active Power (kW)')
            plt.title(f'{titles[i]} (Correlation: {corr_value:.2f})')
            plt.grid(True, alpha=0.3)

        plt.subplots_adjust(hspace=0.5, top=0.93)
        plt.suptitle(f'Feature Analysis vs Normalized Active Power for {file}', fontsize=20, y=1)
        plot_filename = f'{file[:-4]}_feature_vs_power_plot.png'
        plot_path = os.path.join(save_dir, plot_filename)
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()  # Close the plot to free up memory

        print(f"Plot saved for {file} at: {plot_path}")

    # Plot combined data correlation if there is any data
        # Plot combined data correlation if there is any data
    if not combined_df.empty:
        combined_correlations = combined_df.corr()['Normalized_Active_Power']

        plt.figure(figsize=(18, 12))
        for i, feature in enumerate(features):
            plt.subplot(len(features), 1, i + 1)
            corr_value = combined_correlations.get(feature, 0)  # Get correlation, default to 0 if not present
            plt.scatter(combined_df[feature], combined_df['Normalized_Active_Power'], color=colors[i], marker='x', alpha=0.6)
            plt.xlabel(f'{feature}')
            plt.ylabel('Normalized Active Power')
            plt.title(f'{titles[i]} (Correlation: {corr_value:.2f})')
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        combined_plot_path = os.path.join(save_dir, f'{dataset_name}_combined_feature_vs_power_plot.png')
        plt.savefig(combined_plot_path, bbox_inches='tight')
        plt.close()

        print(f"Combined plot saved at: {combined_plot_path}")
