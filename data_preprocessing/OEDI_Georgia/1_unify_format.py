import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta

from tqdm import tqdm
from copy import deepcopy
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

def combine_into_each_invertor(invertor_name, index_of_invertor,
                           save_dir, raw_df):
    os.makedirs(save_dir, exist_ok=True)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    # print(raw_df.head())

    '''1. Extract only necessary columns'''
    df = raw_df[['timestamp',invertor_name, 'Global_Horizontal_Radiation','Weather_Temperature_Celsius', 'Wind_Speed']]
    df = df.rename(columns={invertor_name: 'Active_Power'})

    df.to_csv(os.path.join(save_dir, f'{invertor_name}.csv'), index=False)


def merge_raw_data(active_power_path, env_path, irrad_path, meter_path):
    
    active_power = pd.read_csv(active_power_path)
    env = pd.read_csv(env_path)
    irrad = pd.read_csv(irrad_path)
    meter = pd.read_csv(meter_path)

    df_list = [active_power, env, irrad, meter]
    df_merged = df_list[0]

    for df in df_list[1:]:
        df_merged = pd.merge(df_merged, df, on='measured_on', how='left')
    columns_to_keep = [
        'measured_on',
        'weather_station_01_ambient_temperature_(sensor_1)_(c)_o_150245',   # Weather_Temperature_Celsius
        'pyranometer_(class_a)_12_ghi_irradiance_(w/m2)_o_150231',  # Global_Horizontal_Radiation
        'wind_sensor_12b_wind_speed_(m/s)_o_150271'
    ]
    # 인버터 열 추가 (inv1 ~ inv40)
    for i in range(1, 41):
        inv_col = f'inverter_{i:02d}_ac_power_(kw)_inv_{150952 + i}'
        columns_to_keep.append(inv_col)

    # df_merged에서 해당 열들만 남기기
    df_filtered = df_merged[columns_to_keep]
    # 열 이름 변경
    mydic = {
        'measured_on': 'timestamp',
        'weather_station_01_ambient_temperature_(sensor_1)_(c)_o_150245': 'Weather_Temperature_Celsius',
        'pyranometer_(class_a)_12_ghi_irradiance_(w/m2)_o_150231': 'Global_Horizontal_Radiation',
        'wind_sensor_12b_wind_speed_(m/s)_o_150271': 'Wind_Speed'
    }

    # 인버터 열 이름 변경 추가
    for i in range(1, 41):
        old_name = f'inverter_{i:02d}_ac_power_(kw)_inv_{150952 + i}'
        new_name = f'inv{i}'
        mydic[old_name] = new_name

    df_filtered.rename(columns=mydic, inplace=True)

    # 데이터 전처리 및 인버터별 처리
    # timestamp를 datetime 형식으로 변환
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

    # 데이터를 1시간 간격으로 리샘플링
    df_filtered.set_index('timestamp', inplace=True)
    df_resampled = df_filtered.resample('1H').mean()
    df_resampled.reset_index(inplace=True)
    # df_resampled['Active_Power'] = df_resampled['Active_Power']
    combined_data = df_resampled

    return combined_data


if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    dataset_name = 'OEDI_Georgia'

    save_dir=os.path.join(project_root,  f'data/{dataset_name}/uniform_format_data')
    log_save_dir = os.path.join(project_root, f'data_preprocessing_night/{dataset_name}/raw_info')

    # 디렉토리 삭제
    remove_directory_if_exists(save_dir)
    remove_directory_if_exists(log_save_dir)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)

    # active_power_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/9069(Georgia)/9069_electrical_ac.csv')
    # env_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/9069(Georgia)/9069_environment_data.csv')
    # irrad_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/9069(Georgia)/9069_irradiance_data.csv')
    # meter_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/9069(Georgia)/9069_meter_data.csv')
    active_power_path = os.path.join(project_root, f'data/{dataset_name}/9069_electrical_ac.csv')
    env_path = os.path.join(project_root, f'data/{dataset_name}/9069_environment_data.csv')
    irrad_path = os.path.join(project_root, f'data/{dataset_name}/9069_irradiance_data.csv')
    meter_path = os.path.join(project_root, f'data/{dataset_name}/9069_meter_data.csv')
    merged_data = merge_raw_data(active_power_path, env_path, irrad_path, meter_path)

    # site_list = ['YMCA', 'Maple Drive East', 'Forest Road', 'Elm Crescent','Easthill Road']
    invertor_list = [f'inv{i}' for i in range(1,41)]

    # log_file_path = os.path.join(project_root, '/ailab_mat/dataset/PV/OEDI/9069(Georgia)/log.txt')
    for i, invertor_name in enumerate(invertor_list):
        combine_into_each_invertor(
            invertor_name, 
            i, 
            save_dir=os.path.join(project_root, save_dir),
            raw_df= merged_data
        )
    
    check_data.process_data_and_log(
    folder_path=os.path.join(project_root, save_dir),
    log_file_path=os.path.join(log_save_dir, 'raw_data_info.txt')
    )
    plot_correlation_each.plot_feature_vs_active_power(
            data_dir=save_dir, 
            save_dir=log_save_dir, 
            dataset_name=dataset_name
            )

