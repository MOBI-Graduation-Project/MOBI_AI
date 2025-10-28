# =================================================================
# ★★★ v8 모델 일일 자동 예측 스크립트 (predict.py) ★★★
# ★★★ (플랜 B: CSV 폴백 기능 탑재) ★★★
# =================================================================
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
import datetime
import os
import joblib 
from sqlalchemy import create_engine

import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import TFBertModel, BertTokenizer

print(f"[{datetime.datetime.now()}] 예측 스크립트 시작...")

# --- 2. 기본 경로 및 상수 설정 ---
BASE_PATH = "C:/MOBI_AI_TEST/" # (서버 환경에 맞게 수정)

# ★★★★★ 수정점 1: v8 파일 이름으로 변경 ★★★★★
MODEL_PATH = os.path.join(BASE_PATH, "best_kospi_model_v8.keras")
SCALER_PATH = os.path.join(BASE_PATH, "kospi_scaler_v8.pkl")
# ★★★★★

CSV_NEWS_PATH = os.path.join(BASE_PATH, "news_headlines_security.csv") 
BERT_MODEL_NAME = "klue/bert-base" # v8은 klue/bert-base로 학습됨 (v5와 동일)
DB_CONNECTION_STRING = "mysql+pymysql://USER:PASSWORD@HOST/DATABASE" # (서버 환경에 맞게 수정)


# --- 3. v8 모델 전처리 함수 ---
# (v5와 완벽하게 동일합니다. 수정할 필요가 없습니다.)
def feature_engineer_live(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['V_MA5'] = df['Volume'].rolling(window=5).mean()
    df['V_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Change_Ratio'] = (df['Close'] - df['Open']) / df['Open'] * 100
    return df

def get_latest_numeric_data():
    print("  [데이터 수집] KOSPI 및 거시 경제 지표 수집 중...")
    start_date = (datetime.date.today() - datetime.timedelta(days=60)).strftime('%Y-%m-%d')
    
    # 1. KOSPI 데이터 로드 (v8 학습 데이터와 동일)
    kospi = fdr.DataReader('KS11', start_date)
    kospi_features = feature_engineer_live(kospi.copy())
    
    # 2. 거시 경제 지표 로드 (v8 학습 데이터와 동일)
    data_symbols = {
        'USD/KRW': 'USD/KRW', 'FRED:DGS10': 'US_10Y_BOND', 'GC=F': 'GOLD', 
        'CL=F': 'WTI_OIL', 'US500': 'SP500'
    }
    macro_features = []
    for symbol, name in data_symbols.items():
        df = fdr.DataReader(symbol, start_date)
        if 'Close' in df.columns: feature = df['Close'].rename(name)
        elif 'DGS10' in df.columns: feature = df['DGS10'].rename(name)
        else: feature = df.iloc[:, 0].rename(name)
        macro_features.append(feature)
    macro_df = pd.concat(macro_features, axis=1).ffill()
    
    # 3. KOSPI와 거시 지표 병합
    final_numeric_df = pd.merge(kospi_features, macro_df, left_index=True, right_index=True, how='left').ffill()
    
    # 4. v8 모델이 학습한 피처만 정확히 선택 (v8 학습 스크립트의 16개 피처 리스트)
    v8_feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'Change', 
        'MA5', 'MA20', 'V_MA5', 'V_MA20', 'Change_Ratio',
        'USD/KRW', 'US_10Y_BOND', 'GOLD', 'WTI_OIL', 'SP500'
    ]
    # (이 리스트는 이제 v8 스케일러와 완벽히 호환됩니다)
    final_numeric_df = final_numeric_df[v8_feature_columns].dropna()
    return final_numeric_df.iloc[[-1]]

def crawl_today_news():
    # (v5와 완벽하게 동일합니다. v8은 v5와 동일한 뉴스를 사용했습니다.)
    print("  [데이터 수집] 오늘 날짜 '증권' 뉴스 크롤링 중... (플랜 A)")
    # ... (내부 코드 v5와 동일) ...
    base_url = "https://news.einfomax.co.kr/news/articleList.html"
    params = {'sc_section_code': 'S1N2', 'sc_order_by': 'E', 'page': 1}
    headers = {'User-Agent': 'Mozilla/5.0 ...'}
    today_str = datetime.date.today().strftime('%Y.%m.%d')
    today_headlines = []
    
    for page_num in range(1, 11): 
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=5) 
            response.raise_for_status() 
            soup = BeautifulSoup(response.text, 'html.parser')
            article_blocks = soup.select("ul.type1 > li")
            if not article_blocks: break 
            
            page_done = False
            for article in article_blocks:
                date_tag = article.select_one("em.info.dated")
                if date_tag:
                    date_text = date_tag.get_text(strip=True).split(' ')[0]
                    if date_text == today_str:
                        title = article.select_one("h4.titles > a").get_text(strip=True)
                        today_headlines.append(title)
                    elif date_text < today_str:
                        page_done = True 
                        break
            if page_done: break
            time.sleep(0.2)
        except Exception as e:
            print(f"  [플랜 A 에러] 뉴스 크롤링 중단: {e}")
            return "" 
            
    return ' '.join(today_headlines)

def get_latest_news_from_csv():
    # (v5와 완벽하게 동일합니다.)
    print("  [Plan B] 실시간 크롤링 실패. CSV에서 최신 뉴스 로드 중...")
    # ... (내부 코드 v5와 동일) ...
    try:
        if not os.path.exists(CSV_NEWS_PATH):
            print(f"  [Plan B 에러] {CSV_NEWS_PATH} 파일을 찾을 수 없습니다.")
            return ""
            
        df_security = pd.read_csv(CSV_NEWS_PATH)
        df_security['Date'] = pd.to_datetime(df_security['Date'], errors='coerce')
        df_security.dropna(subset=['Date', 'Title'], inplace=True)
        df_security.sort_values(by='Date', inplace=True)
        news_grouped = df_security.groupby('Date')['Title'].apply(lambda x: ' '.join(x)).reset_index()
        
        if not news_grouped.empty:
            latest_headlines = news_grouped.iloc[-1]['Title']
            latest_date = news_grouped.iloc[-1]['Date'].strftime('%Y-%m-%d')
            print(f"  [Plan B] CSV의 가장 최신 날짜({latest_date}) 뉴스를 사용합니다.")
            return latest_headlines
        else:
            print("  [Plan B 에러] CSV 파일에 유효한 뉴스가 없습니다.")
            return ""
    except Exception as e:
        print(f"  [Plan B 에러] CSV 로드 중 문제 발생: {e}")
        return ""

def save_to_db(prediction_result, model_accuracy):
    # (로컬 테스트 시에는 이 함수 내부를 주석 처리)
    pass 


# --- 4. 메인 예측 로직 실행 (v8용) ---
def main():
    try:
        # 1. 모델 로드 (v8 파일 로드)
        print("[1/5] v8 모델 및 스케일러 로딩...")
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        # 2. 전문가(BERT) 로드 (v8이 사용한 klue/bert-base 로드)
        print("[2/5] KLUE-BERT 모델 로딩...")
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model = TFBertModel.from_pretrained(BERT_MODEL_NAME, from_pt=True)
        
        # 3. 실시간 데이터 수집 (v8 데이터와 100% 호환)
        print("[3/5] 실시간 데이터 수집...")
        today_numeric_df = get_latest_numeric_data()
        today_headlines_str = crawl_today_news() 
        
        is_plan_b = False
        if not today_headlines_str:
            print("  [경고] 실시간 뉴스 크롤링 실패. Plan B (CSV)를 가동합니다.")
            today_headlines_str = get_latest_news_from_csv()
            is_plan_b = True
            
        if today_numeric_df.empty or not today_headlines_str:
            print("  [최종 경고] 숫자 데이터 또는 Plan B 뉴스 데이터가 수집되지 않아 예측을 중단합니다.")
            return

        # 4. 데이터 전처리
        print("[4/5] 데이터 전처리...")
        # (v8 스케일러는 이제 이 16개 피처와 완벽히 호환됩니다)
        numeric_scaled = scaler.transform(today_numeric_df)
        bert_inputs = tokenizer(
            [today_headlines_str], max_length=128, truncation=True, 
            padding='max_length', return_tensors='tf'
        )
        bert_outputs = bert_model(bert_inputs)
        news_features = bert_outputs.last_hidden_state[:, 0, :].numpy()
        
        # 5. 예측 수행 및 저장
        print("[5/5] 예측 수행 및 DB 저장...")
        pred_prob = model.predict({'numeric_input': numeric_scaled, 'news_input': news_features})
        prediction_result = "상승📈" if pred_prob[0][0] > 0.5 else "하락📉"
        
        # ★★★★★ 수정점 2: v8의 정확도로 변경 ★★★★★
        # (Colab에서 v8 학습 후 나온 최종 정확도(예: 53.40)를 여기에 입력해야 합니다)
        model_accuracy = 53.40 # 예시입니다. Colab 결과값으로 꼭 수정하세요!
        # ★★★★★
        
        if is_plan_b:
             print(f"  [예측 완료 - Plan B] 내일 KOSPI 예측: {prediction_result} (Prob: {pred_prob[0][0]:.4f})")
             print("  [주의] 이 예측은 실시간 뉴스가 아닌, 저장된 최신 뉴스를 기반으로 합니다.")
        else:
             print(f"  [예측 완료 - Plan A] 내일 KOSPI 예측: {prediction_result} (Prob: {pred_prob[0][0]:.4f})")
        
        # DB에 저장
        # save_to_db(prediction_result, model_accuracy)
        print("\n★★★ 로컬 테스트 성공! DB 저장을 건너뛰었습니다. ★★★") # 로컬 테스트용

    except Exception as e:
        print(f"[{datetime.datetime.now()}] 스크립트 실행 중 치명적인 에러 발생: {e}")

if __name__ == "__main__":
    main()
    print(f"[{datetime.datetime.now()}] 예측 스크립트 종료.")