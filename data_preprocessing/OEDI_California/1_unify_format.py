import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

from tqdm import tqdm
from copy import deepcopy
import pvlib
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

def convert_poa2ghi(combined_data):
    combined_data = combined_data.copy()
    # Calculate Global Horizontal Radiation (GHI) from POA Irradiance using pvlib
    # Set arbitrary values for latitude, longitude, timezone, tilt, and azimuth
    lat, lon = 38.996306, -122.134111  # Example coordinates
    tz = 'America/Los_Angeles'  # Pacific Time Zone
    tilt = 25  # degrees (arbitrary value)
    azimuth = 180  # degrees (south-facing)

    # Create a pvlib Location object
    site = pvlib.location.Location(lat, lon, tz=tz)

    # Get solar position data
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
    combined_data = combined_data.set_index('timestamp')

    # Localize the index to make it tz-aware
    combined_data.index = combined_data.index.tz_localize(tz, nonexistent='shift_forward', ambiguous='NaT')
    times = combined_data.index  # Now both combined_data and times are tz-aware

    solar_position = site.get_solarposition(times)

    # Calculate Angle of Incidence (AOI)
    aoi = pvlib.irradiance.aoi(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )

    # Estimate DNI and DHI from POA and AOI
    poa_global = combined_data['POA_Irradiance']
    # Use the isotropic sky model to estimate the diffuse component
    poa_diffuse = pvlib.irradiance.isotropic(tilt, poa_global)
    dhi = poa_diffuse
    # Avoid division by zero by replacing zeros in cos(aoi) with a small number
    cos_aoi = np.cos(np.radians(aoi))
    cos_aoi = cos_aoi.replace(0, 1e-6)
    dni = (poa_global - dhi) / cos_aoi

    # Compute GHI
    cos_zenith = np.cos(np.radians(solar_position['apparent_zenith']))
    cos_zenith = cos_zenith.replace(0, 1e-6)  # Avoid division by zero
    ghi = dni * cos_zenith + dhi

    # Replace 'Global_Horizontal_Radiation' with the calculated GHI
    combined_data['Global_Horizontal_Radiation'] = ghi
    return combined_data

def merge_raw_data(active_power_path, env_path, irrad_path, meter_path):

    active_power = pd.read_csv(active_power_path)
    env = pd.read_csv(env_path)
    irrad = pd.read_csv(irrad_path)
    meter = pd.read_csv(meter_path)

    df_list = [active_power, env, irrad, meter]
    df_merged = df_list[0]

    for df in df_list[1:]:
        df_merged = pd.merge(df_merged, df, on='measured_on', how='outer')

    columns_to_keep = [
        'measured_on',
        'ambient_temperature_o_149575',   # Weather_Temperature_Fahrenheit
        'wind_speed_o_149576',
        'poa_irradiance_o_149574',        # POA_Irradiance
    ]
    # Add inverter columns (inv1 ~ inv24)
    for i in range(1, 25):
        if i != 15:
            inv_col = f'inv_{i:02d}_ac_power_inv_{149583 + (i-1)*5}'
        else:
            # Special case ('inv_15_ac_power_iinv_149653')
            inv_col = 'inv_15_ac_power_iinv_149653'
        columns_to_keep.append(inv_col)

    # Keep only the relevant columns
    df_filtered = df_merged[columns_to_keep]
    # Rename columns
    rename_dict = {
        'measured_on': 'timestamp',
        'ambient_temperature_o_149575': 'Weather_Temperature_Fahrenheit',
        'wind_speed_o_149576': 'Wind_Speed',
        'poa_irradiance_o_149574': 'POA_Irradiance',  # Renamed for clarity
    }

    # Rename inverter columns
    for i in range(1, 25):
        if i != 15:
            old_name = f'inv_{i:02d}_ac_power_inv_{149583 + (i-1)*5}'
        else:
            old_name = 'inv_15_ac_power_iinv_149653'
        rename_dict[old_name] = f'inv{i}'

    df_filtered = df_filtered.rename(columns=rename_dict)

    # Data preprocessing and per-inverter processing
    # Convert timestamp to datetime format
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

    # Resample data at 1-hour intervals
    df_filtered.set_index('timestamp', inplace=True)
    df_resampled = df_filtered.resample('1H').mean()
    df_resampled.reset_index(inplace=True)
    combined_data = df_resampled
    combined_data = convert_poa2ghi(combined_data)
    # Reset index
    combined_data.reset_index(inplace=True)
    combined_data['timestamp'] = combined_data['timestamp'].dt.tz_convert(None)

    return combined_data

def change_unit(combined_data, invertor_list):
    combined_data = combined_data.copy()
    combined_data['Weather_Temperature_Celsius'] = (combined_data['Weather_Temperature_Fahrenheit'] - 32) * 5 / 9 # Fahrenheit to Celsius
    combined_data.drop('Weather_Temperature_Fahrenheit', axis=1, inplace=True)
    combined_data['Wind_Speed'] *= 0.44704 # mph to m/s
    return combined_data

def make_unifrom_csv_files(merged_data, save_dir, invertor_list):
    print("Start making uniform format files")
    os.makedirs(save_dir, exist_ok=True)

    for invertor_name in tqdm(invertor_list, desc="Saving CSV files"):
        # df = merged_data[['timestamp', invertor_name, 'Global_Horizontal_Radiation', 'Weather_Relative_Humidity', 'Weather_Temperature_Celsius', 'Wind_Speed']]
        df = merged_data[['timestamp',invertor_name, 'Global_Horizontal_Radiation','Weather_Temperature_Celsius', 'Wind_Speed', 'POA_Irradiance']]
        df = df.rename(columns={invertor_name: 'Active_Power'})
        df.to_csv(os.path.join(save_dir, invertor_name+".csv"), index=False)
    print("Finished!")

if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    dataset_name = 'OEDI_California'

    save_dir=os.path.join(project_root,  f'data/{dataset_name}/uniform_format_data')
    log_save_dir = os.path.join(project_root, f'data_preprocessing_night/{dataset_name}/raw_info')

    # 디렉토리 삭제
    remove_directory_if_exists(save_dir)
    remove_directory_if_exists(log_save_dir)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)

    active_power_path = os.path.join(project_root, f'data/{dataset_name}/2107_electrical_data.csv') 
    env_path = os.path.join(project_root, f'data/{dataset_name}/2107_environment_data.csv')
    irrad_path = os.path.join(project_root, f'data/{dataset_name}/2107_irradiance_data.csv')
    meter_path = os.path.join(project_root, f'data/{dataset_name}/2107_meter_15m_data.csv')

    merged_data = merge_raw_data(active_power_path, env_path, irrad_path, meter_path)
    invertor_list = [f'inv{i}' for i in range(1, 25)]

    unit_changed_data = change_unit(merged_data, invertor_list)
    make_unifrom_csv_files(unit_changed_data, save_dir, invertor_list)
    
    check_data.process_data_and_log(
    folder_path=os.path.join(project_root, save_dir),
    log_file_path=os.path.join(log_save_dir, 'raw_data_info.txt')
    )
    plot_correlation_each.plot_feature_vs_active_power(
            data_dir=save_dir, 
            save_dir=log_save_dir, 
            dataset_name=dataset_name
            )

    
