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

def combine_humidity_data(raw_weather_list, save_path):
    # 모든 데이터프레임을 저장할 리스트 초기화
    humidity_dfs = []
    
    # wind speed 파일 리스트를 순회하면서 데이터프레임으로 읽어옵니다.
    for weather_file in raw_weather_list:
        df_weather = pd.read_csv(weather_file)
        df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
        humidity_dfs.append(df_weather)
        
    # 데이터프레임들을 하나로 합칩니다.
    combined_weather_df = pd.concat(humidity_dfs, ignore_index=True)
    combined_weather_df.sort_values('timestamp', inplace=True)
    combined_weather_df = combined_weather_df[['timestamp', 'Weather_Relative_Humidity']]
    combined_weather_df['Weather_Relative_Humidity'] = pd.to_numeric(combined_weather_df['Weather_Relative_Humidity'], errors='coerce')
    combined_weather_df['Weather_Relative_Humidity'] = combined_weather_df['Weather_Relative_Humidity']
    
    # 1시간 단위로 리샘플링하여 평균을 계산합니다.
    combined_weather_df.set_index('timestamp', inplace=True)
    combined_weather_hourly = combined_weather_df.resample('h').mean().reset_index()
    
    # 결과를 CSV 파일로 저장합니다.
    combined_weather_hourly.to_csv(save_path, index=False)

def combine_into_each_site(file_path, i,save_dir,
                            combined_weather_hourly):
    os.makedirs(save_dir, exist_ok=True)
    file_name = file_path.split('/')[-1]
    print(file_name)
    raw_df = pd.read_csv(file_path, encoding='unicode_escape')
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

    '''1. 데이터의 맨 처음과 끝을 일 단위로 끊고 불필요한 열 제거'''
    # 데이터 시작 시간을 기준으로 다음날 00:00 이후 데이터 사용
    start_date = raw_df['timestamp'].dt.normalize().iloc[0] + pd.Timedelta(days=1)
    # 데이터 종료 시간을 기준으로 당일 23:59:59 이전 데이터 사용
    end_date = raw_df['timestamp'].dt.normalize().iloc[-1]
    raw_df = raw_df[(raw_df['timestamp'] >= start_date) & (raw_df['timestamp'] < end_date)]

    necessary_columns = ['Active_Power',
                         'Global_Horizontal_Radiation',
                         'Weather_Temperature_Celsius',
                         'Wind_Speed']
                        #  'Weather_Relative_Humidity'] 1시간 단위로 변환후 추가
    df = raw_df.loc[:, ['timestamp'] + necessary_columns]
    
    '''6. 1시간 단위로 데이터를 sampling. Margin은 1시간으로 유지'''
    # 1. 1시간 단위로 평균 계산
    df_hourly = df.resample('h', on='timestamp').mean().reset_index()
    df_hourly = df_hourly.dropna(how='all', subset=df.columns[1:])

    # combined_weather_hourly와 병합
    df_hourly = pd.merge(df_hourly, combined_weather_hourly, on='timestamp', how='left')

    df_hourly.to_csv(os.path.join(save_dir, f'{file_name}'), index=False)



if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    dataset_name = 'DKASC_Yulara'

    save_dir=os.path.join(project_root,  f'data/{dataset_name}/uniform_format_data')
    log_save_dir = os.path.join(project_root, f'data_preprocessing_all/{dataset_name}/raw_info')

    # 디렉토리 삭제
    remove_directory_if_exists(save_dir)
    remove_directory_if_exists(log_save_dir)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)

    raw_csv_data_dir = os.path.join(project_root, f'data/{dataset_name}/raw')  # for local
    raw_file_list = [os.path.join(raw_csv_data_dir, _) for _ in os.listdir(raw_csv_data_dir)]

    raw_weather_data_dir = os.path.join(project_root, f'data/{dataset_name}/weather_data')  # for local
    raw_weather_list = [os.path.join(raw_weather_data_dir, _) for _ in os.listdir(raw_weather_data_dir)]
    raw_weather_list.sort()

    # combined_weather.csv를 저장할 경로 설정
    combined_weather_path = os.path.join(project_root, f'data/{dataset_name}/combined_weather.csv')
    # wind speed 데이터를 합치고 저장합니다.
    combine_humidity_data(raw_weather_list, combined_weather_path)

    # combined_weather.csv를 데이터프레임으로 읽어옵니다.
    combined_weather_hourly = pd.read_csv(combined_weather_path)
    combined_weather_hourly['timestamp'] = pd.to_datetime(combined_weather_hourly['timestamp'])

    for i, file_path in enumerate(raw_file_list):
        combine_into_each_site(file_path, i,save_dir,
                            combined_weather_hourly)
        
    check_data.process_data_and_log(
    folder_path=os.path.join(project_root, save_dir),
    log_file_path=os.path.join(log_save_dir, 'raw_data_info.txt')
    )
    plot_correlation_each.plot_feature_vs_active_power(
            data_dir=save_dir, 
            save_dir=log_save_dir, 
            dataset_name=dataset_name
            )
