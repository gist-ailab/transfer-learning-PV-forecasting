import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Get the absolute path of the current file
current_file_path = os.path.abspath(__file__)

# Get the root directory (assuming the root is two levels up from the current file)
project_root = os.path.dirname(os.path.dirname(current_file_path))

# Each sites' data directory
data_dir = os.path.join(project_root, 'data/GIST_dataset/converted')
#data_dir = os.path.join(project_root, 'data/DKASC_AliceSprings/converted')
# data_dir = os.path.join(project_root, 'data/DKASC_Yulara/converted')
# data_dir = os.path.join(project_root, 'data/Miryang/PV_merged')
# data_dir = os.path.join(project_root, 'data/Germany_Household_Data/preprocessed')
# data_dir = os.path.join(project_root, 'data/UK_data/preprocessed')
# data_dir = os.path.join(project_root, 'data/OEDI/9069(Georgia)/preprocessed')
# data_dir = os.path.join(project_root, 'data/OEDI/9069(Georgia)/preprocessed')
data_list = os.listdir(data_dir)
data_list = [file for file in data_list if file.endswith('.csv')]


combined_df = pd.DataFrame()
for file in tqdm(data_list, desc='Loading data'):
    data_path = os.path.join(data_dir, file)
    df = pd.read_csv(data_path)
    df.drop(columns=['timestamp'], inplace=True)
    # df = df[df['Global_Horizontal_Radiation'] > 0].dropna()
    # Filter out rows where Active_Power is negative
    df = df[df['Active_Power'] >= 0]
    # Normalize Active_Power by dividing by the maximum value for each site
    df['Normalized_Active_Power'] = df['Active_Power'] / df['Active_Power'].max()
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# Calculate the correlation between Active_Power and other features
correlations = combined_df.corr()['Active_Power']

# Plot the relationship between Active_Power and other features
plt.figure(figsize=(18, 12))

# List of features to plot
features = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity', 'Wind_Speed']
# features = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity']
# features = ['Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Wind_Speed']

colors = ['blue', 'green', 'red', 'purple']
titles = ['Active Power [kW] vs Global Horizontal Radiation [w/m²]',
          'Active Power [kW] vs Weather Temperature [℃]',
          'Active Power [kW] vs Weather Relative Humidity [%]',
          'Active Power [kW] vs Wind Speed [m/s]']

# Loop through each feature and create a scatter plot
for i, feature in enumerate(features):
    plt.subplot(len(features), 1, i+1)
    corr_value = correlations[feature]  # Get the correlation value for each feature
    plt.scatter(combined_df[feature], combined_df['Normalized_Active_Power'], color=colors[i], marker='x', alpha=0.6)
    plt.xlabel(f'{feature}')
    plt.ylabel('Active Power (kW)')
    plt.title(f'{titles[i]} (Correlation: {corr_value:.2f})')  # Add correlation to title
    # plt.title(titles[i])
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Save the plot as an image file (e.g., PNG format)
data_name = "GIST"
plot_path = os.path.join(project_root, f'visualizations/{data_name}_feature_vs_power_plot.png')
plt.savefig(plot_path)

print(f"Plot saved at: {plot_path}")