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

### 기상데이터 column 정보
# result_Code 결과코드
# result_Msg  결과메시지
# rcdcnt      한 페이지 결과 수
# page_No     페이지 번호
# total_Count 전체 결과 수
# no          데이터 검색결과 정렬 번호
# stn_Code    관측지점코드
# stn_Name    관측지점명
# date        관측시각
# temp        기온(℃)
# max_Temp    최고기온(℃)
# min_Temp    최저기온(℃)
# hum         습도(%)
# widdir      풍향
# wind        풍속(m/s)
# max_Wind    풍속(m/s)
# rain        강수량(mm)
# sun_Time    일조시간(MM)
# sun_Qy      일사량(MJ/m²)
# condens_Time결로시간(MM)
# gr_Temp     초상온도(℃)
# soil_Temp   지중온도(℃)
# soil_Wt     토양수분보정값(%)

import shutil
import os

# 디렉토리 삭제 함수
def remove_directory_if_exists(dir_path):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
        print(f"Deleted directory: {dir_path}")
    else:
        print(f"Directory not found or not a directory: {dir_path}")

def convert_excel_to_csv(file_list):
    csv_save_path = file_list[0].split('/')[:-2]
    csv_save_path.append('PV_csv')
    csv_save_path = '/'.join(csv_save_path)
    os.makedirs(csv_save_path, exist_ok=True)
    
    for i, file in tqdm(enumerate(file_list), total=len(file_list), desc='Converting Excel to CSV'):
        df = pd.read_excel(file,
                           header=None,
                           engine='openpyxl') # calaminefor reading xlsx files
       
        # 첫 번째 열은 'timestamp'로, 두 번째 열은 'Active_Power'로 지정하고 나머지 컬럼 이름 생성
        df.columns = ['timestamp', 'Active_Power'] + [f'Inverter_{i}' for i in range(1, df.shape[1] - 1)]
        df['timestamp'] = pd.to_datetime(df['timestamp'])   # 'timestamp' 컬럼을 datetime 타입으로 변환

        # 결측치 처리:'-' 또는 빈 값을 NaN으로 변환
        # df = df.map(lambda x: np.nan if x in ['-', '', ' '] else x)
        df = df.replace(['-', '', ' '], np.nan)#, inplace=True)
   
        # 첫 번째 열을 제외한 나머지 열에 대해 결측치 처리 적용
        for column in df.columns[1:]:
            # 처리 단계
            filtered_indices = []
            i = 0
            while i < len(df):
                if pd.isna(df.loc[i, column]):
                    start = i
                    # 연속된 결측치의 범위를 찾기
                    while i < len(df) and pd.isna(df.loc[i, column]):
                        i += 1
                    end = i - 1
                    
                    num_missing = end - start + 1

                    # 앞뒤가 모두 0이면 0으로 채우기
                    if (start > 0 and end < len(df) - 1 and 
                            df.loc[start - 1, column] == 0 and df.loc[end + 1, column] == 0):
                        df.loc[start:end, column] = 0
                else:
                    i += 1

        base_filename = os.path.basename(file)
        output_filename = os.path.splitext(base_filename)[0] + '.csv'
        output_path = os.path.join(csv_save_path, output_filename)

        df.to_csv(output_path, index=False)

    print('Conversion completed!')

    return csv_save_path


def combine_csv_files(csv_file_dir, weather_file_dir):
    # Get the list of CSV files
    csv_file_list = [os.path.join(csv_file_dir, _) for _ in os.listdir(csv_file_dir)]
    csv_file_list.sort()

    # Get the list of weather files
    weather_file_list = [os.path.join(weather_file_dir, _) for _ in os.listdir(weather_file_dir)]
    for i in weather_file_list:
        weather_df = pd.read_csv(i, parse_dates=['date'])
        weather_df = weather_df.drop(weather_df.columns[0:4], axis=1)
        weather_df = weather_df.rename(columns={'date': 'timestamp'})
        # timestamp, temp, hum, sun_Qy 열만 남기고 나머지 열 삭제
        columns_to_keep = ['timestamp', 'temp', 'hum', 'sun_Qy', 'wind']
        weather_df = weather_df[columns_to_keep]
        # 'sun_Qy' 값을 시간 단위의 일사량으로 변환 (현재 값에서 이전 값을 뺌)
        weather_df['sun_Qy']= weather_df['sun_Qy'].diff()
        weather_df['sun_Qy'] = weather_df['sun_Qy'].clip(lower=0)   # 음수 값은 0으로 변환
        weather_df['sun_Qy'] = (weather_df['sun_Qy'] * 1_000_000) / 3600    # 누적 일사량 MJ/m²를 순간 일사량 W/m²로 변환
        weather_df['sun_Qy'] = weather_df['sun_Qy'].round(4)   # 소수점 8자리까지 표시

        # 열 이름 변경
        weather_df = weather_df.rename(columns={
            'temp': 'Weather_Temperature_Celsius',
            'hum': 'Weather_Relative_Humidity',
            'sun_Qy': 'Global_Horizontal_Radiation',
            'wind': 'Wind_Speed'
        })

        # 파일 이름에 따라 데이터프레임 저장
        if '산내면' in i:
            산내면_기상 = weather_df
        elif '상남면' in i:
            상남면_기상 = weather_df

    # Combine the CSV files
    for file in csv_file_list:
        power_df = pd.read_csv(file, parse_dates=['timestamp'])
        file = os.path.basename(file)
        
        # 기존의 'Active_Power' 열을 그대로 사용
        # 불필요한 인버터별 합산 부분 제거
        # 발전량 데이터프레임에서 'Active_Power' 열만 유지하고 사용
        power_df = power_df[['timestamp', 'Active_Power']]
        
        # 발전량 데이터와 기상 데이터를 'timestamp'를 기준으로 병합 (left join)
        if 'A' in file:
            weather_df = deepcopy(산내면_기상)
        elif 'A' not in file:
            weather_df = deepcopy(상남면_기상)
        merged_df = pd.merge(power_df, weather_df, on='timestamp', how='left')

        # 새 파일 이름 만들기
        base_filename = os.path.basename(file)
        new_filename = "_".join(base_filename.split('_')[:2]) + '.csv'  # 첫 번째와 두 번째 요소만 사용

        save_dir = os.path.join(csv_file_dir.split('PV_csv')[0], 'uniform_format_data')
        os.makedirs(save_dir, exist_ok=True)

        output_path = os.path.join(save_dir, new_filename)
        merged_df.to_csv(output_path, index=False)
        print(f'Saved: {output_path}')

    print('Combination and merging completed!')



if __name__ == '__main__':
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the root directory (assuming the root is two levels up from the current file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

    dataset_name = 'Miryang'

    save_dir=os.path.join(project_root,  f'data/{dataset_name}/uniform_format_data')
    log_save_dir = os.path.join(project_root, f'data_preprocessing_night/{dataset_name}/raw_info')

    # 디렉토리 삭제
    remove_directory_if_exists(save_dir)
    remove_directory_if_exists(log_save_dir)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_save_dir, exist_ok=True)

    pv_xls_data_dir = os.path.join(project_root, f'data/{dataset_name}/PV_xls')
    pv_file_list = [os.path.join(pv_xls_data_dir, _) for _ in os.listdir(pv_xls_data_dir)]
    pv_file_list.sort()

      
    csv_file_dir = os.path.join(project_root, f'data/{dataset_name}/PV_csv')
    if not os.path.exists(csv_file_dir):
        # Convert Excel files to CSV files
        csv_file_dir = convert_excel_to_csv(pv_file_list) 
    weather_file_dir = os.path.join(project_root, f'data/{dataset_name}/weather')

    combine_csv_files(csv_file_dir, weather_file_dir)

    check_data.process_data_and_log(
    folder_path=save_dir,
    log_file_path= os.path.join(log_save_dir, 'raw_data_info.txt')
    )
    plot_correlation_each.plot_feature_vs_active_power(
            data_dir=save_dir, 
            save_dir=log_save_dir, 
            dataset_name=dataset_name
            )

