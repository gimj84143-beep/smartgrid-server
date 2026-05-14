import os
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import platform
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings('ignore')

# 기본 경로 및 데이터 로드 (기존과 동일)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEN_CSV_PATH = os.path.join(BASE_DIR, "한국전력거래소_지역별 시간별 태양광 및 풍력 발전량_20241231.csv")
WEATHER_XLSX_PATH = os.path.join(BASE_DIR, "2024년 제주도 날씨 정보.xlsx")
CONSUME_CSV_PATH = os.path.join(BASE_DIR, "가구당_시간별_전력소비량.csv")

app = FastAPI()

# 앱과의 통신을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI 모델 로드 (앱이 켜질 때 한 번만 로드)
def load_models():
    # (기존 load_models() 코드 내용과 완전히 동일하게 작성 - 생략 방지를 위해 핵심만 요약)
    df_consume = pd.read_csv(CONSUME_CSV_PATH, encoding='cp949')
    df_consume['날짜'] = pd.to_datetime(df_consume['날짜'])
    X_con = pd.DataFrame({'월': df_consume['날짜'].dt.month, '일': df_consume['날짜'].dt.day, '요일': df_consume['날짜'].dt.dayofweek})
    hours_cols = [f'{i}시' for i in range(1, 25)]
    y_con = df_consume[hours_cols] * 0.174
    consume_model = RandomForestRegressor(n_estimators=100, random_state=42)
    consume_model.fit(X_con, y_con)

    gen = pd.read_csv(GEN_CSV_PATH, encoding="cp949")
    gen = gen[gen['지역'].str.contains("제주")].copy()
    gen['datetime'] = pd.to_datetime(gen['거래일자']) + pd.to_timedelta(gen['거래시간'] - 1, unit='h')
    solar = gen[gen['연료원'] == '태양광'][['datetime', '전력거래량(MWh)']].rename(columns={'전력거래량(MWh)': 'solar_gen'})

    weather = pd.read_excel(WEATHER_XLSX_PATH, engine="openpyxl")
    weather = weather[["일시", "기온(°C)", "풍속(m/s)", "전운량(10분위)"]]
    weather["일시"] = pd.to_datetime(weather["일시"])
    weather = weather.set_index("일시").sort_index()
    weather.columns = ["temp", "wind_speed", "cloud"]
    
    df_gen = solar.set_index('datetime').join(weather, how="inner").interpolate(method='linear').fillna(0)
    df_gen['month'] = df_gen.index.month
    df_gen['hour'] = df_gen.index.hour
    
    X_gen = df_gen[['temp', 'wind_speed', 'cloud', 'month', 'hour']]
    y_gen = df_gen['solar_gen']
    gen_model = RandomForestRegressor(n_estimators=100, random_state=42)
    gen_model.fit(X_gen, y_gen)
    
    return consume_model, gen_model, df_gen['solar_gen'].max()

consume_model, gen_model, max_gen_hist = load_models()

def fetch_weather_data(api_key):
    lat, lon = 33.4996, 126.5312
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200: return None
    data = response.json()
    weather_list = []
    for item in data['list']:
        kst_time = pd.to_datetime(item['dt_txt']) + pd.Timedelta(hours=9)
        weather_list.append({
            'datetime': kst_time, 'temp': item['main']['temp'],
            'wind_speed': item['wind']['speed'], 'cloud': item['clouds']['all'] / 10.0 
        })
    df = pd.DataFrame(weather_list).set_index('datetime')
    return df.resample('1h').interpolate(method='linear')

def get_dynamic_price(hour):
    if 13 <= hour <= 17: return 250
    elif 18 <= hour <= 22: return 200
    elif hour >= 23 or hour <= 8: return 80
    else: return 130

USER_API_KEY = "c836bff1b19e7105c684199643d71474"

# ==========================================
# 여기가 Flutter 앱과 통신하는 핵심 API 엔드포인트입니다.
# ==========================================
@app.get("/api/simulation")
def run_simulation():
    try:
        future_weather = fetch_weather_data(USER_API_KEY)
        if future_weather is None:
            return {"error": "날씨 데이터를 가져올 수 없습니다."}
        
        future_weather['month'] = future_weather.index.month
        future_weather['hour'] = future_weather.index.hour
        
        X_future_gen = future_weather[['temp', 'wind_speed', 'cloud', 'month', 'hour']]
        pred_solar = gen_model.predict(X_future_gen)
        max_gen = max_gen_hist if max_gen_hist > 0 else 1
        future_weather['home_gen_kwh'] = (np.maximum(pred_solar, 0) / max_gen) * 3.0
        
        unique_dates = pd.Series(future_weather.index.date).unique()
        home_use_list = []
        for d in unique_dates:
            target_date = pd.to_datetime(d)
            X_consume = pd.DataFrame({'월': [target_date.month], '일': [target_date.day], '요일': [target_date.dayofweek]})
            home_use_list.extend(consume_model.predict(X_consume).flatten())
        future_weather['home_use_kwh'] = home_use_list[:len(future_weather)]
        
        future_weather['net_power_kwh'] = future_weather['home_gen_kwh'] - future_weather['home_use_kwh']
        prices = [get_dynamic_price(h) for h in future_weather['hour']]
        
        future_weather['normal_cost'] = future_weather['home_use_kwh'] * prices 
        future_weather['trade_profit_krw'] = future_weather['net_power_kwh'] * prices 
        future_weather['cumulative_profit'] = future_weather['trade_profit_krw'].cumsum() 
        
        total_normal_pay = float(future_weather['normal_cost'].sum())
        total_solar_result = float(future_weather['trade_profit_krw'].sum())
        total_savings = float(total_normal_pay + total_solar_result)
        
        # 날짜 인덱스를 문자열로 변환 (JSON 직렬화를 위해)
        future_weather.index = future_weather.index.strftime('%Y-%m-%d %H:%M')
        
        return {
            "status": "success",
            "summary": {
                "total_normal_pay": total_normal_pay,
                "total_solar_result": total_solar_result,
                "total_savings": total_savings
            },
            # 플러터에서 그래프를 그릴 수 있도록 리스트 형태로 변환하여 전송
            "chart_data": future_weather[['home_gen_kwh', 'home_use_kwh', 'net_power_kwh', 'cumulative_profit']].reset_index().to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" 이 부분이 핸드폰 접속을 허용하는 핵심입니다.
    uvicorn.run(app, host="0.0.0.0", port=8080)