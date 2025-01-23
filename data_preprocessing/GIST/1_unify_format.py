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

def combine_into_each_site(file_list, index_of_site,
                           kor_name, eng_name,
                           weather_data,
                           save_dir):
    os.makedirs(save_dir, exist_ok=True)
    preprocessed_df = pd.DataFrame(columns=['date', 'time', 'Active_Power', 'Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity', 'Weend_Speed'])
    
    weather_info = pd.read_csv(weather_data, encoding='unicode_escape')
    weather_info.columns = ['datetime', 'temperature', 'wind_speed', 'precipitation', 'humidity']
    weather_info['datetime'] = pd.to_datetime(weather_info['datetime'])
    # print(weather_info)

    # Define file paths for storing outliers
    env_columns = ['datetime', 'Global_Horizontal_Radiation', 'Weather_Temperature_Celsius', 'Direct_Normal_Irradiance', 'Module_Temperature_Celsius', ]

    empty_rows = pd.concat([pd.DataFrame(preprocessed_df.columns)]*24, axis=1).T
    empty_rows.columns = preprocessed_df.columns

    for i, file in tqdm(enumerate(file_list), total=len(file_list), desc=f'Processing {kor_name}. Out of {index_of_site+1}/16'):
        # read pv info
        daily_pv_data = pd.read_csv(file)
        daily_pv_data.columns = daily_pv_data.iloc[0]
        daily_pv_data.columns.values[:len(env_columns)] = env_columns
        daily_pv_data = daily_pv_data.drop([0, 1, 2])
        daily_pv_data = daily_pv_data.reset_index(drop=True)

        if kor_name not in daily_pv_data.columns:
            continue

        columns_to_keep = daily_pv_data.columns[:5].tolist()  # 첫 5개의 영문 컬럼 유지
        columns_to_keep.append(kor_name)  # kor_name 컬럼 유지
        # 나머지 컬럼 삭제
        daily_pv_data = daily_pv_data[columns_to_keep]

        # 결측치 처리:'-' 또는 빈 값을 NaN으로 변환
        daily_pv_data = daily_pv_data.map(lambda x: np.nan if x in ['-', '', ' '] else x)

        # get date
        pv_date = file.split('_')[-2]
        pv_date = pd.to_datetime(pv_date).date()
        daily_weather_data = weather_info[weather_info['datetime'].dt.date == pv_date]
        daily_weather_data = daily_weather_data.reset_index(drop=True)

        # Simply copy the datetime from daily_weather_data to daily_pv_data
        if len(daily_pv_data) != len(daily_weather_data):
            raise ValueError("The number of rows in daily_pv_data and daily_weather_data do not match.")
        daily_pv_data['datetime'] = daily_weather_data['datetime']

        filtered_df = create_combined_data(preprocessed_df, daily_pv_data, daily_weather_data)

        # DataFrame 결합 (concat)
        if preprocessed_df.empty:
            preprocessed_df = deepcopy(filtered_df)
        else:
            preprocessed_df = pd.concat([preprocessed_df, filtered_df], ignore_index=True)

    save_path = os.path.join(save_dir, f'{eng_name}.csv')
    with open(save_path, 'w') as f:
        preprocessed_df.to_csv(f, index=False)


def create_combined_data(preprocessed_df, daily_pv_data, daily_weather_data):
    # Step 1: 'date'와 'time' 데이터를 추출하여 새로운 DataFrame 생성
    df = pd.DataFrame(
        columns=['timestamp', 'Active_Power', 'Global_Horizontal_Radiation', 'Weather_Temperature_Celsius',
                 'Weather_Relative_Humidity', 'Wind_Speed'])

    # Step 2: temp_df에 데이터 채워넣기
    df['timestamp'] = daily_weather_data['datetime']
    df['Active_Power'] = daily_pv_data[kor_name].astype(float)
    df['Global_Horizontal_Radiation'] = daily_pv_data['Global_Horizontal_Radiation'].astype(float)
    df['Weather_Temperature_Celsius'] = daily_weather_data['temperature']
    df['Weather_Relative_Humidity'] = daily_weather_data['humidity']
    df['Wind_Speed'] = daily_weather_data['wind_speed']

    return df


def create_combined_weather_csv(create_path, project_root):
    weather_data_dir = os.path.join(project_root, 'data/GIST_dataset/weather')
    weather_csv_files = [f for f in os.listdir(weather_data_dir) if f.endswith('.csv')]
    weather_csv_files.sort()

    data_frames = []
    for file in weather_csv_files:
        if 'GIST_AWS' not in file:
            continue
        file_path = os.path.join(weather_data_dir, file)
        try:
            df = pd.read_csv(file_path, encoding='utf-8', skiprows=1, header=None)
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin1', skiprows=1, header=None)

        data_frames.append(df)
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    combined_df.drop(combined_df.columns[:2], axis=1, inplace=True)

    # Define the column names the add
    column_names = ['datetime', 'temperature', 'wind_speed', 'precipitation', 'humidity']
    combined_df.columns = column_names
    combined_df['datetime'] = pd.to_datetime(combined_df.iloc[:, 0])    # 3번째 컬럼을 datetime 형식으로 변환 (시간 관련 처리를 위해)
   
    # Step 1: datetime을 인덱스로 설정하고 1시간 단위로 리샘플링
    combined_df.set_index('datetime', inplace=True)
    # Step 2: 1시간 단위로 리샘플링하여 결측값을 확인
    df_resampled = combined_df.resample('1h').mean()

    # # Check for missing dates
    # unique_dates = pd.to_datetime(df_cleaned.index.date).unique()
    # missing_dates = pd.date_range(start=unique_dates[0], end=unique_dates[-1]).difference(unique_dates)
    # print(f'Missing dates: {missing_dates}') if missing_dates else print('No missing dates')
    # # Check for missing hours
    # full_time_range = pd.date_range(start=df_cleaned.index.min(), end=df_cleaned.index.max(), freq='h')
    # actual_times = df_cleaned.index
    # missing_times = full_time_range.difference(actual_times)
    # print(f'Missing times: {missing_times}') if missing_times else print('No missing times')

    # Save the combined DataFrame to a new CSV file
    df_resampled.to_csv(create_path, index=True)

def convert_excel_to_hourly_csv(file_list):
    for i, xls_file in tqdm(enumerate(file_list), total=len(file_list), desc='Converting Excel to CSV'):
        df = pd.read_excel(xls_file, engine='xlrd')
        df = df.drop([0, 1])

        start_column = 5  # 6번째 열의 인덱스는 5 (0부터 시작하므로)
        row_index = 0  # 1번째 행의 인덱스는 0
        for col in range(df.shape[1] - 1, start_column - 1, -1):
            if col + 1 < df.shape[1]:
                df.iloc[row_index, col + 1] = df.iloc[row_index, col]

        # 6번째 열부터 짝수 인덱스를 가진 열들을 삭제 -> 시간당발전량만 남김
        columns_to_drop = [i for i in range(start_column, df.shape[1]) if (i - start_column) % 2 == 0]
        df.drop(df.columns[columns_to_drop], axis=1, inplace=True)

        last_valid_row = df[df.iloc[:, 0] == '23 시'].index  # '23 시'가 있는 행의 인덱스를 찾음
        df = df.iloc[:last_valid_row[-1]-1]  # '23 시'가 있는 행까지만 유지, 그 이후는 제거

        save_name = xls_file.split('/')[-1].replace('xls', 'csv')
        save_name = '.'.join(save_name.split('.')[1:])
        save_dir = xls_file.split('/')[:-2]
        save_dir.append('daily_PV_csv')
        save_dir = '/'.join(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        df.to_csv(save_path, index=False)
    print('Conversion completed!')


if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    dataset_name = 'GIST'

    save_dir=os.path.join(project_root,  'data/{dataset_name}/uniform_format_data')
    log_save_dir = os.path.join(project_root, f'data_preprocessing_night/{dataset_name}/raw_info')

    # 디렉토리 삭제
    remove_directory_if_exists(save_dir)
    remove_directory_if_exists(log_save_dir)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)

    # # Get the path to the daily PV xls data
    pv_xls_data_dir = os.path.join(project_root, f'data/{dataset_name}/daily_PV_xls')
    pv_file_list = [os.path.join(pv_xls_data_dir, _) for _ in os.listdir(pv_xls_data_dir)]
    pv_file_list.sort()

    # Define the path to save the combined CSV file
    # weather_data = os.path.join(project_root, 'data/GIST_dataset/GIST_weather_data.csv')
    weather_data = os.path.join(project_root, f'data/{dataset_name}/GIST_weather_data.csv')

    
    if not os.path.exists(weather_data):
        create_combined_weather_csv(weather_data, project_root)

    # Check for new columns in the PV data
    # check_new_columns(pv_file_list)

    if not os.path.exists(os.path.join(project_root, f'data/{dataset_name}/daily_PV_csv')):
        convert_excel_to_hourly_csv(pv_file_list)
    else:
        print('Skip converting xls to csv since csv files already exists')

    # raw_csv_data_dir = os.path.join(project_root, 'data/GIST_dataset/daily_PV_csv')
    raw_csv_data_dir = os.path.join(project_root, f'data/{dataset_name}/daily_PV_csv')

    raw_file_list = [os.path.join(raw_csv_data_dir, _) for _ in os.listdir(raw_csv_data_dir)]
    raw_file_list.sort()

    site_dict = {
        '축구장': 'Soccer-Field',
        '학생회관': 'W06_Student-Union',
        '중앙창고': 'W13_Centeral-Storage',
        '학사과정': 'E11_DormA',
        '다산빌딩': 'C09_Dasan',
        '시설관리동': 'W11_Facility-Maintenance-Bldg',
        # '대학C동': 'N06_College-Bldg',
        # '동물실험동': 'E02_Animal-Recource-Center',
        '중앙도서관': 'N01_Central-Library',
        'LG도서관': 'N02_LG-Library',
        '신재생에너지동': 'C10_Renewable-E-Bldg',
        '삼성환경동': 'C07_Samsung-Env-Bldg',
        '중앙연구기기센터': 'C11_GAIA',
        '산업협력관': 'E03_GTI',
        '기숙사 B동': 'E12_DormB',
        '자연과학동': 'E8_Natural-Science-Bldg'
    }

    for i, (kor_name, eng_name) in enumerate(site_dict.items()):
        combine_into_each_site(file_list=raw_file_list,
                               index_of_site=i,
                               kor_name=kor_name, eng_name=eng_name,
                               weather_data=weather_data,
                               save_dir=save_dir)
    check_data.process_data_and_log(
    folder_path=os.path.join(project_root, save_dir),
    log_file_path=os.path.join(log_save_dir, 'raw_data_info.txt')
    )
    plot_correlation_each.plot_feature_vs_active_power(
            data_dir=save_dir, 
            save_dir=log_save_dir, 
            dataset_name=dataset_name
            )