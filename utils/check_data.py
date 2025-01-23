import pandas as pd
import os

def process_data_and_log(folder_path, log_file_path):
    """
    Processes CSV files in a folder, logs row counts and column stats to a log file.

    Parameters:
    - folder_path (str): The folder path containing CSV files.
    - log_file_path (str): The file path to save the log.
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # 해당 폴더 내 모든 CSV 파일 가져오기
    file_names = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

    # 파일별 row 개수를 저장할 딕셔너리
    row_counts = {}

    # 파일별 열별 결측치, 최소, 최대값을 저장할 딕셔너리
    column_stats = {}

    # 각 파일을 불러와 row 개수와 열별 결측치 개수 및 최소, 최대값 저장
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)

        # timestamp 열을 날짜 형식으로 변환
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')  # 변환 실패 시 NaT로 처리

        # row 수 계산
        row_counts[file_name] = len(df)

        # 열별 결측치, 최소, 최대값 계산
        stats = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            min_value = df[column].min() if df[column].dtype in ['float64', 'int64', 'datetime64[ns]'] else 'N/A'
            max_value = df[column].max() if df[column].dtype in ['float64', 'int64', 'datetime64[ns]'] else 'N/A'
            stats[column] = {
                'missing': missing_count,
                'min': min_value,
                'max': max_value
            }
        column_stats[file_name] = stats

    # 파일명을 기준으로 정렬하여 row 수, 열별 결측치, 최소, 최대값을 로그 파일에 기록
    with open(log_file_path, 'w') as log_file:
        total_rows = 0
        log_file.write("파일별 row 수 및 열별 결측치, 최소, 최대값:\n")
        for file_name in sorted(row_counts.keys()):
            log_file.write(f"\n{file_name}: {row_counts[file_name]}개\n")
            total_rows += row_counts[file_name]

            # 열별 결측치, 최소, 최대값 출력
            log_file.write("열별 결측치, 최소, 최대값:\n")
            for column, stat in column_stats[file_name].items():
                log_file.write(
                    f" - {column}: 결측치={stat['missing']}개, 최소값={stat['min']}, 최대값={stat['max']}\n"
                )

        log_file.write('\n총 데이터 수: ' + str(total_rows) + '개\n')

if __name__ == '__main__':
    # 독립 실행 시 기본 경로 사용 예시
    folder_path = '../../data/DKASC_AliceSprings/processed_data'
    log_file_path = './processed_info/processed_data_info.txt'
    process_data_and_log(folder_path, log_file_path)
