import os
import pandas as pd
import plotly.graph_objects as go

# 파일 경로 설정
directory = '/home/bak/Projects/PatchTST/data/processed_data_day_Alice_Springs'

# 데이터 파일 불러오기
# 모든 파일을 읽어 결합
all_data = []
for file in os.listdir(directory):
    if file.endswith('.csv'):  # Assuming the data files are in CSV format
        filepath = os.path.join(directory, file)
        df = pd.read_csv(filepath)
        all_data.append(df)

# 데이터프레임 합치기
combined_df = pd.concat(all_data, ignore_index=True)

# 관심 있는 컬럼 선택
columns_of_interest = ['Normalized_Active_Power', 'Global_Horizontal_Radiation',
                       'Weather_Temperature_Celsius', 'Weather_Relative_Humidity', 'Wind_Speed']

# 선택한 컬럼들이 데이터에 모두 있는지 확인 후 사용
existing_columns = [col for col in columns_of_interest if col in combined_df.columns]
data = combined_df[existing_columns]

# 개별 상관 관계 그래프 그리기 (Scattergl 사용)
for feature in existing_columns[1:]:  # 첫 번째 컬럼은 'Normalized_Active_Power'이므로 제외하고 반복
    correlation_value = data['Normalized_Active_Power'].corr(data[feature])
    
    # 개별 상관관계 그래프 그리기 (Plotly Scattergl 사용)
    fig = go.Figure(data=go.Scattergl(
        x=data[feature],
        y=data['Normalized_Active_Power'],
        mode='markers',
        marker=dict(size=5, opacity=0.6),
    ))
    
    # 그래프 레이아웃 설정
    fig.update_layout(
        title=f"Correlation between Normalized Active Power and {feature}: {correlation_value:.2f}",
        xaxis_title=feature,
        yaxis_title="Normalized Active Power",
        template="plotly_white"
    )
    
    # 그래프 출력
    fig.show()