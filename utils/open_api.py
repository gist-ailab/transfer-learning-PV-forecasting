import requests
import pandas as pd
from urllib.parse import urlencode, quote_plus, unquote
from datetime import datetime, timedelta
import xmltodict
from tqdm import tqdm

# 시작일과 종료일 설정
start_date = datetime(2022, 8, 1)
end_date = datetime(2024, 9, 30)
# end_date = datetime(2024, 9, 30)

# 빈 데이터프레임 생성
산내면_all_data = []
상남면_all_data = []

# 날짜 범위 생성 및 반복
for i in tqdm(range((end_date - start_date).days + 1), desc='Processing'):
    frm_date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')  # 날짜 형식 설정

    # 서비스 요청 주소, 서비스키, 요청 내용을 입력
    # API 요청 주소
    api = 'http://apis.data.go.kr/1390802/AgriWeather/WeatherObsrInfo/V2/GnrlWeather/getWeatherTimeList'
    # 서비스키. 괄호 안에는 본인의 서비스키를 입력
    key = unquote('ShIBr9QjkCLieGfgf5yQOaaf6gmdehCRoO2%2BtLEc6UF3FTo3NNQW%2F3r4hA5GC%2BBxgMRfX6G0idGyeZBd%2BP6OYQ%3D%3D')
    queryParams = '?' + urlencode({
        quote_plus('serviceKey'): key,
        quote_plus('Page_No'): '1',
        quote_plus('Page_Size'): '48',  # 밀양시 산내면, 상남면 데이터를 모두 가져오기 위해 48로 설정
        quote_plus('date_Time'): frm_date,
        quote_plus('obsr_Spot_Nm'): '밀양시'
    })
    url = api + queryParams

    # 데이터 요청 및 변환
    res = requests.get(url).content
    xml = xmltodict.parse(res)

    for val in xml['response']['body']['items']['item']:
        if val['stn_Name'] == '밀양시 산내면':
            산내면_data = val
            산내면_all_data.append(산내면_data)
        elif val['stn_Name'] == '밀양시 상남면':
            상남면_data = val
            상남면_all_data.append(상남면_data)

# DataFrame으로 변환
산내면_df = pd.DataFrame(산내면_all_data)
상남면_df = pd.DataFrame(상남면_all_data)

# 결과 저장
산내면_df.to_csv("산내면.csv")
상남면_df.to_csv("상남면.csv")