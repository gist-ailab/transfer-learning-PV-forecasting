import os
import pandas as pd
import sys
import os
# 현재 파일에서 두 단계 상위 디렉토리를 sys.path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))

# 이제 상위 폴더의 상위 폴더 내부의 utils 폴더의 파일 import 가능
from utils import plot_correlation_each, check_data

import shutil
import os

# 디렉토리 삭제 함수
def remove_directory_if_exists(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Deleted directory: {dir_path}")
    else:
        print(f"Directory not found or not a directory: {dir_path}")

# Detect 2 consecutive NaN values in any column
def detect_consecutive_nans(df, max_consecutive=2):
    """
    This function detects rows where any column has max_consecutive or more NaN values.
    It will return a boolean mask.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        # Get a boolean mask for NaN values
        is_nan = df[col].isna()
        # Rolling window to find consecutive NaNs
        nan_consecutive = is_nan.rolling(window=max_consecutive, min_periods=max_consecutive).sum() == max_consecutive
        mask[col] = nan_consecutive
    return mask.any(axis=1)

# Detect 10 or more consecutive identical non-zero values
def detect_consecutive_identical_values(df, column, min_consecutive=10):
    """
    This function detects rows where a column has min_consecutive or more consecutive identical non-zero values.
    """
    mask = (df[column] != 0) & (df[column] == df[column].shift(1))
    count_series = mask.groupby(mask.ne(mask.shift()).cumsum()).cumsum()
    return count_series >= min_consecutive



import logging
import pandas as pd

# Set up logging to file
logging.basicConfig(filename='missing_timestamps.log', level=logging.INFO, format='%(message)s')

# Function to ensure all hours are present for each day
def ensure_full_day_timestamps(df, timestamp_col='timestamp'):
    """
    This function ensures that each day has all 24 hours (00:00 to 23:00).
    If any hour is missing, it will log the missing timestamps and return the modified DataFrame.
    """
    # Create a full range of timestamps for each unique date in the DataFrame
    min_date = df[timestamp_col].min().floor('D')
    max_date = df[timestamp_col].max().ceil('D') - pd.Timedelta(hours=1)
    full_timestamps = pd.date_range(start=min_date, end=max_date, freq='H')
    
    # Find missing timestamps
    missing_timestamps = full_timestamps.difference(df[timestamp_col])
    if not missing_timestamps.empty:
        logging.info("Missing timestamps detected:")
        for ts in missing_timestamps:
            logging.info(ts)
            
    # Reindex DataFrame to include all timestamps
    df = df.set_index(timestamp_col).reindex(full_timestamps).reset_index()
    df.rename(columns={'index': timestamp_col}, inplace=True)
    
    return df

# Drop days where Active_Power is 0 for 24 or more hours
def drop_days_with_excessive_zeros(df, timestamp_col='timestamp', power_col='Active_Power', zero_count_threshold=24):
    """
    Drops days where the number of zero values in the Active_Power column is greater than or equal to the threshold.
    """
    zero_count_per_day = df.groupby(df[timestamp_col].dt.date)[power_col].apply(lambda x: (x == 0).sum())
    days_to_drop = zero_count_per_day[zero_count_per_day >= zero_count_threshold].index
    return df[~df[timestamp_col].dt.date.isin(days_to_drop)]


if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)
    
    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    
    dataset_name = 'Miryang'
    
    save_dir = os.path.join(project_root, f'data/{dataset_name}/processed_data_night')
    log_save_dir = os.path.join(project_root, f'data_preprocessing_night/{dataset_name}/processed_info')

    # 디렉토리 삭제
    remove_directory_if_exists(save_dir)
    remove_directory_if_exists(log_save_dir)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)
    
    hourly_csv_data_dir = os.path.join(project_root, f'data/{dataset_name}/uniform_format_data')  # for local
    hourly_csv_list = [os.path.join(hourly_csv_data_dir, _) for _ in os.listdir(hourly_csv_data_dir) if _.endswith('.csv')]
    
    for file_path in hourly_csv_list:
        df_hourly = pd.read_csv(file_path)

        max_active_power = df_hourly['Active_Power'].max(skipna=True)
        print("max active power: "+ str(max_active_power))

        df_hourly['Normalized_Active_Power'] = df_hourly['Active_Power']/ max_active_power
        
        file_name= file_path.split("/")[-1]
        print(file_name)

        if file_name in ['C_99kW.csv']:
            print(file_name+ " 사이트 active power 이상치 제거")
            df_hourly.loc[df_hourly['Normalized_Active_Power'] > 0.5, 'Active_Power'] = pd.NA
            df_hourly['Normalized_Active_Power'] = df_hourly['Active_Power']/ max(df_hourly['Active_Power'])
        if file_name in ['G_50kW.csv']:
            print(file_name+ " 사이트 active power 이상치 제거")
            df_hourly.loc[df_hourly['Normalized_Active_Power'] > 0.2, 'Active_Power'] = pd.NA
            df_hourly['Normalized_Active_Power'] = df_hourly['Active_Power']/ max(df_hourly['Active_Power'])
    
        # Modify Active_Power based on the condition of Normalized_Active_Power
        df_hourly.loc[(df_hourly['Normalized_Active_Power'] >= -0.05) & (df_hourly['Normalized_Active_Power'] < 0), 'Active_Power'] = 0
        df_hourly.loc[(df_hourly['Normalized_Active_Power'] < -0.05), 'Active_Power'] = pd.NA

        df_hourly['Normalized_Active_Power'] = df_hourly['Active_Power']/ max(df_hourly['Active_Power'])
        
        # Ensure 'timestamp' is in datetime format
        df_hourly['timestamp'] = pd.to_datetime(df_hourly['timestamp'], errors='coerce')

        # Ensure full day timestamps
        df_hourly = ensure_full_day_timestamps(df_hourly, 'timestamp')
        
        # Apply the conditions to replace values with NaN
        df_hourly.loc[df_hourly['Global_Horizontal_Radiation'] > 2000, 'Global_Horizontal_Radiation'] = pd.NA
        df_hourly.loc[df_hourly['Weather_Temperature_Celsius'] < -10, 'Weather_Temperature_Celsius'] = pd.NA
        df_hourly.loc[df_hourly['Wind_Speed'] < 0, 'Wind_Speed'] = pd.NA
        df_hourly.loc[(df_hourly['Weather_Relative_Humidity'] < 0) | (df_hourly['Weather_Relative_Humidity'] > 100), 'Weather_Relative_Humidity'] = pd.NA
        df_hourly.loc[(df_hourly['Normalized_Active_Power'] <= 0.05) & (df_hourly['Global_Horizontal_Radiation'] > 200), 'Active_Power'] = pd.NA
        df_hourly.loc[(df_hourly['Normalized_Active_Power'] > 0.1) & (df_hourly['Global_Horizontal_Radiation'] < 10), 'Active_Power'] = pd.NA

        # Detect and replace with NaN if there are 10 or more consecutive identical non-zero values in 'Active_Power'
        identical_values_mask = detect_consecutive_identical_values(df_hourly, 'Active_Power', min_consecutive=10)
        df_hourly.loc[identical_values_mask, 'Active_Power'] = pd.NA
        
        # Detect 2 consecutive NaN values in any column
        consecutive_nan_mask = detect_consecutive_nans(df_hourly, max_consecutive=2)
        
        # Remove entire days where 2 consecutive NaNs were found
        days_with_2_nan = df_hourly[consecutive_nan_mask]['timestamp'].dt.date.unique()
        df_hourly = df_hourly[~df_hourly['timestamp'].dt.date.isin(days_with_2_nan)]
        
        # Interpolate NaN values (1 or fewer consecutive NaNs)
        df_hourly.interpolate(method='linear', limit=1, inplace=True)

        # Apply the function after ensuring full day timestamps
        df_hourly = drop_days_with_excessive_zeros(df_hourly, timestamp_col='timestamp', power_col='Active_Power', zero_count_threshold=20)
        
        # Save the processed DataFrame
        max_active_power = df_hourly['Active_Power'].max(skipna=True)
        output_file_path = os.path.join(save_dir, str(max_active_power)+"_"+os.path.basename(file_path))
        df_hourly.to_csv(output_file_path, index=False)
        
        print(f"Processed and saved: {output_file_path}")
    
    check_data.process_data_and_log(
    folder_path=os.path.join(project_root, save_dir),
    log_file_path=os.path.join(log_save_dir, 'processed_data_info.txt')
    )
    plot_correlation_each.plot_feature_vs_active_power(
            data_dir=save_dir, 
            save_dir=log_save_dir, 
            dataset_name=dataset_name
            )

