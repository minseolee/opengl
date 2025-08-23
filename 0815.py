#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
곤지암 리조트 식음업장 수요예측 고도화 ML 모델 v3.1
- 데이터 형식 문제 해결
- Rolling features 안정화
- 프로젝트 지식 기반 체계적 Feature Engineering
- 실전용 장시간 학습 코드
"""

# ========================================
# 1. 라이브러리 임포트
# ========================================
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import glob
import os
import re
from tqdm import tqdm
import pickle
import time
import gc
from collections import defaultdict

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'AppleGothic'  # macOS 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# 머신러닝 모델
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# 부스팅 모델
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 시계열 및 통계
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr, spearmanr
import holidays


# ========================================
# 2. 평가 메트릭 및 유틸리티
# ========================================

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error - 대회 평가지표"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)


def weighted_smape(y_true, y_pred, weights):
    """업장별 가중치가 적용된 sMAPE"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weights = np.array(weights)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0

    weighted_diff = diff * weights
    return 100 * np.sum(weighted_diff) / np.sum(weights)


def print_section(title, emoji="🔍"):
    """섹션 제목 출력"""
    print("\n" + "=" * 70)
    print(f"{emoji} {title}")
    print("=" * 70)


# ========================================
# 3. 도메인 지식 기반 특성 엔지니어링
# ========================================

class DomainFeatureEngineer:
    """프로젝트 지식을 활용한 도메인 특화 특성 엔지니어링"""

    def __init__(self):
        self.kr_holidays = holidays.KR(years=range(2023, 2026))
        self.menu_categories = self._init_menu_categories()
        self.restaurant_info = self._init_restaurant_info()
        self.menu_associations = self._init_menu_associations()

    def _init_menu_categories(self):
        """메뉴 카테고리 정의 - 프로젝트 지식 기반"""
        return {
            '메인메뉴': ['본삼겹', '불고기', '갈비탕', '국밥', '해장국', '냉면', '비빔밥',
                     '파스타', '피자', '리조또', '플래터', '브런치', '정식', '돈까스',
                     '볶음밥', '우동', '짜장면', '짬뽕', '떡볶이', '순대', '파전', '꼬치어묵'],
            '사이드메뉴': ['공깃밥', '라면사리', '메밀면사리', '쌈야채세트', '쌈장',
                      '야채추가', '고기추가', '면추가', '빵추가', '햇반'],
            '주류': ['막걸리', '소주', '맥주', '와인', '칵테일', '하이볼', '참이슬',
                   '처음처럼', '카스', '테라', '버드와이저', '스텔라', '하이네켄'],
            '음료': ['아메리카노', '라떼', '스프라이트', '콜라', '제로콜라', '에이드',
                   '아이스티', '식혜', '생수'],
            '아이스크림': ['아이스크림', '뻥스크림'],
            '장소': ['대여료', '룸', '이용료', 'Conference', 'Convention', 'Grand', 'OPUS']
        }

    def _init_restaurant_info(self):
        """업장별 특성 정의 - 프로젝트 지식 기반"""
        return {
            '느티나무 셀프BBQ': {
                'type': 'outdoor_bbq',
                'season_preference': 'summer',
                'time_preference': 'evening',
                'customer_type': 'group',
                'weight': 1.0
            },
            '담하': {
                'type': 'traditional_korean',
                'season_preference': 'all',
                'time_preference': 'lunch_dinner',
                'customer_type': 'family',
                'weight': 2.0  # 높은 가중치
            },
            '라그로타': {
                'type': 'italian_wine',
                'season_preference': 'all',
                'time_preference': 'dinner',
                'customer_type': 'couple',
                'weight': 1.0
            },
            '미라시아': {
                'type': 'brunch_buffet',
                'season_preference': 'all',
                'time_preference': 'brunch',
                'customer_type': 'family_with_kids',
                'weight': 2.0  # 높은 가중치
            },
            '연회장': {
                'type': 'conference',
                'season_preference': 'all',
                'time_preference': 'business_hours',
                'customer_type': 'business',
                'weight': 1.0
            },
            '카페테리아': {
                'type': 'casual_dining',
                'season_preference': 'winter',
                'time_preference': 'all_day',
                'customer_type': 'ski_guests',
                'weight': 1.0
            },
            '포레스트릿': {
                'type': 'snack',
                'season_preference': 'winter',
                'time_preference': 'all_day',
                'customer_type': 'ski_guests',
                'weight': 1.0
            },
            '화담숲주막': {
                'type': 'korean_pub',
                'season_preference': 'spring_fall',
                'time_preference': 'evening',
                'customer_type': 'forest_visitors',
                'weight': 1.0
            },
            '화담숲카페': {
                'type': 'cafe',
                'season_preference': 'spring_fall',
                'time_preference': 'afternoon',
                'customer_type': 'forest_visitors',
                'weight': 1.0
            }
        }

    def _init_menu_associations(self):
        """메뉴 연관관계 정의 - 프로젝트 지식 기반"""
        return {
            # 업장 내 종속관계 (A → B)
            'internal_dependencies': {
                '참이슬': ['일회용 소주컵'],
                '스프라이트': ['일회용 종이컵'],
                '카스': ['일회용 종이컵'],
                '콜라': ['일회용 종이컵'],
                '생목살 김치전골': ['라면사리'],
                '생목살 김치찌개': ['라면사리'],
                'BBQ Platter': ['BBQ 고기추가'],
                '모둠 돈육구이': ['삼겹살추가'],
                '파스타': ['파스타면 추가']
            },
            # 업장 간 상관관계 (동시 발생 가능성 높음)
            'cross_restaurant_correlations': {
                '주류': ['주류'],  # 맥주, 소주 등 주류끼리 상관관계
                '음료': ['음료'],  # 음료끼리 상관관계
                '메인메뉴': ['사이드메뉴']  # 메인메뉴 주문 시 사이드메뉴 가능성
            }
        }

    def get_menu_category(self, menu_name):
        """메뉴명으로부터 카테고리 추출"""
        for category, keywords in self.menu_categories.items():
            for keyword in keywords:
                if keyword in menu_name:
                    return category
        return '기타'

    def get_restaurant_weight(self, restaurant_name):
        """업장별 가중치 반환"""
        for rest_key, info in self.restaurant_info.items():
            if rest_key in restaurant_name:
                return info['weight']
        return 1.0


# ========================================
# 4. 고도화된 데이터 처리기
# ========================================

class AdvancedDataProcessor:
    """도메인 지식을 활용한 고도화 데이터 처리"""

    def __init__(self):
        self.domain_engineer = DomainFeatureEngineer()
        self.kr_holidays = holidays.KR(years=range(2023, 2026))
        self.label_encoders = {}
        self.scalers = {}
        self.menu_stats = {}
        self.correlation_features = {}

    def load_data(self, train_path='./train/train.csv', test_dir='./test/'):
        """데이터 로드 및 기본 전처리"""
        print_section("데이터 로드", "📊")

        # Train 데이터
        self.train = pd.read_csv(train_path)
        print(f"✓ Train 데이터: {self.train.shape}")
        print(f"✓ Train 컬럼: {list(self.train.columns)}")

        # 영업장명_메뉴명 분리
        self.train[['업장명', '메뉴명']] = self.train['영업장명_메뉴명'].str.split('_', n=1, expand=True)

        # 날짜 처리
        if self.train['영업일자'].dtype == 'object':
            self.train['영업일자'] = pd.to_datetime(self.train['영업일자'])

        # Test 데이터들
        test_files = sorted(glob.glob(os.path.join(test_dir, 'TEST_*.csv')))
        self.test_data = {}

        for file in test_files:
            test_name = os.path.basename(file).replace('.csv', '')
            df = pd.read_csv(file)
            print(f"✓ {test_name} 컬럼: {list(df.columns)}")
            df[['업장명', '메뉴명']] = df['영업장명_메뉴명'].str.split('_', n=1, expand=True)

            if df['영업일자'].dtype == 'object':
                df['영업일자'] = pd.to_datetime(df['영업일자'])

            self.test_data[test_name] = df
            print(f"✓ {test_name} 데이터: {df.shape}")

        # Sample submission
        try:
            self.sample_submission = pd.read_csv('./sample_submission.csv')
            print(f"✓ Submission 형식: {self.sample_submission.shape}")
            print(f"✓ Submission 컬럼 수: {len(self.sample_submission.columns)}")
        except FileNotFoundError:
            print("⚠️ Sample submission 파일을 찾을 수 없습니다.")
            self.sample_submission = None
        except Exception as e:
            print(f"⚠️ Sample submission 로드 중 오류: {e}")
            self.sample_submission = None

        # 기본 컬럼 순서 저장 (sample_submission이 없는 경우 대비)
        if self.sample_submission is None:
            # Train 데이터에서 메뉴 순서 가져오기
            menu_columns = ['영업일자'] + sorted(self.train['영업장명_메뉴명'].unique().tolist())
            print(f"✓ 기본 컬럼 순서 생성: {len(menu_columns)}개 컬럼")
            # 임시 sample_submission 생성
            temp_data = {col: [0] if col == '영업일자' else [0] for col in menu_columns}
            temp_data['영업일자'] = ['TEST_00+1일']
            self.sample_submission = pd.DataFrame(temp_data)

        return self.train, self.test_data

    def extract_datetime_features(self, df):
        """고도화된 시간 특성 추출"""
        df = df.copy()

        # 기본 날짜 특성
        df['date'] = pd.to_datetime(df['영업일자'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter

        # 주말/평일
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_friday'] = (df['dayofweek'] == 4).astype(int)
        df['is_saturday'] = (df['dayofweek'] == 5).astype(int)
        df['is_sunday'] = (df['dayofweek'] == 6).astype(int)

        # 공휴일
        df['is_holiday'] = df['date'].apply(lambda x: 1 if x in self.kr_holidays else 0)

        # 계절성 - 프로젝트 지식 기반
        def get_season_features(month):
            # 겨울: 스키 고객, 봄가을: 화담숲 방문객, 여름: 가족단위 체류형 관광객
            if month in [12, 1, 2]:
                return 'winter', 1, 0, 0  # winter, is_ski_season, is_forest_season, is_family_season
            elif month in [3, 4, 5, 9, 10, 11]:
                return 'spring_fall', 0, 1, 0
            else:  # 6, 7, 8
                return 'summer', 0, 0, 1

        season_info = df['month'].apply(get_season_features)
        df['season'] = [x[0] for x in season_info]
        df['is_ski_season'] = [x[1] for x in season_info]
        df['is_forest_season'] = [x[2] for x in season_info]
        df['is_family_season'] = [x[3] for x in season_info]

        # 연말연시 특별 기간
        df['is_year_end'] = ((df['month'] == 12) & (df['day'] >= 20)).astype(int)
        df['is_new_year'] = ((df['month'] == 1) & (df['day'] <= 10)).astype(int)

        # 주기적 특성 (sin/cos 변환)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        return df

    def extract_menu_features(self, df):
        """메뉴 특성 추출 - 도메인 지식 기반"""
        df = df.copy()

        # 메뉴 카테고리
        df['menu_category'] = df['메뉴명'].apply(self.domain_engineer.get_menu_category)

        # 온도 기반 특성 (HOT/ICE)
        df['is_hot_menu'] = df['메뉴명'].str.contains('HOT', case=False, na=False).astype(int)
        df['is_ice_menu'] = df['메뉴명'].str.contains('ICE', case=False, na=False).astype(int)

        # 업장별 특성
        df['restaurant_type'] = df['업장명'].apply(lambda x: self._get_restaurant_type(x))
        df['restaurant_weight'] = df['업장명'].apply(self.domain_engineer.get_restaurant_weight)

        # 가격대 추정 (메뉴명에서 숫자 추출)
        df['estimated_price'] = df['메뉴명'].str.extract(r'(\d+)').astype(float).fillna(0)

        # 단체 메뉴 여부
        df['is_group_menu'] = df['메뉴명'].str.contains('단체', na=False).astype(int)

        # 정식/패키지 메뉴 여부
        df['is_set_menu'] = df['메뉴명'].str.contains('정식|패키지', na=False).astype(int)

        return df

    def _get_restaurant_type(self, restaurant_name):
        """업장 타입 반환"""
        for rest_key, info in self.domain_engineer.restaurant_info.items():
            if rest_key in restaurant_name:
                return info['type']
        return 'other'

    def extract_lag_features_safe(self, df, is_train=True):
        """안전한 시계열 지연 특성 추출"""
        if not is_train or '매출수량' not in df.columns:
            return df

        df = df.copy()
        df = df.sort_values(['업장명', '메뉴명', 'date'])

        print("지연 특성 추출 중...")

        # 메뉴별로 따로 처리
        all_dfs = []
        for menu in tqdm(df['영업장명_메뉴명'].unique(), desc="메뉴별 지연 특성"):
            menu_df = df[df['영업장명_메뉴명'] == menu].copy()
            menu_df = menu_df.sort_values('date')

            # 기본 지연 특성
            for lag in [1, 2, 3, 7, 14, 21, 28]:
                menu_df[f'sales_lag_{lag}'] = menu_df['매출수량'].shift(lag)

            # 이동평균 - 더 안전한 방법
            for window in [3, 7, 14, 28]:
                menu_df[f'sales_ma_{window}'] = menu_df['매출수량'].rolling(
                    window=window, min_periods=1
                ).mean()

            # 이동표준편차
            for window in [7, 14, 28]:
                menu_df[f'sales_std_{window}'] = menu_df['매출수량'].rolling(
                    window=window, min_periods=1
                ).std().fillna(0)

            all_dfs.append(menu_df)

        # 모든 메뉴 데이터 합치기
        result_df = pd.concat(all_dfs, ignore_index=True)

        # 요일별/월별/계절별 평균 추가
        print("통계적 특성 추가 중...")

        # 요일별 평균
        dayofweek_stats = result_df.groupby(['영업장명_메뉴명', 'dayofweek'])['매출수량'].mean().reset_index()
        dayofweek_stats.columns = ['영업장명_메뉴명', 'dayofweek', 'sales_dayofweek_mean']
        result_df = result_df.merge(dayofweek_stats, on=['영업장명_메뉴명', 'dayofweek'], how='left')

        # 월별 평균
        month_stats = result_df.groupby(['영업장명_메뉴명', 'month'])['매출수량'].mean().reset_index()
        month_stats.columns = ['영업장명_메뉴명', 'month', 'sales_month_mean']
        result_df = result_df.merge(month_stats, on=['영업장명_메뉴명', 'month'], how='left')

        # 계절별 평균
        season_stats = result_df.groupby(['영업장명_메뉴명', 'season'])['매출수량'].mean().reset_index()
        season_stats.columns = ['영업장명_메뉴명', 'season', 'sales_season_mean']
        result_df = result_df.merge(season_stats, on=['영업장명_메뉴명', 'season'], how='left')

        return result_df

    def extract_correlation_features(self, df, is_train=True):
        """메뉴 간 상관관계 특성 추출"""
        if not is_train or '매출수량' not in df.columns:
            return df

        df = df.copy()

        print("상관관계 특성 추출 중...")

        # 같은 업장 내 다른 메뉴들의 매출 합계
        restaurant_daily_sales = df.groupby(['업장명', 'date'])['매출수량'].sum().reset_index()
        restaurant_daily_sales.columns = ['업장명', 'date', 'restaurant_total_sales']
        df = df.merge(restaurant_daily_sales, on=['업장명', 'date'], how='left')

        # 같은 카테고리 메뉴들의 매출 합계
        category_daily_sales = df.groupby(['menu_category', 'date'])['매출수량'].sum().reset_index()
        category_daily_sales.columns = ['menu_category', 'date', 'category_total_sales']
        df = df.merge(category_daily_sales, on=['menu_category', 'date'], how='left')

        # 주류 총 매출 (업장 간 상관관계)
        alcohol_sales = df[df['menu_category'] == '주류'].groupby('date')['매출수량'].sum().reset_index()
        alcohol_sales.columns = ['date', 'total_alcohol_sales']
        df = df.merge(alcohol_sales, on='date', how='left')
        df['total_alcohol_sales'] = df['total_alcohol_sales'].fillna(0)

        # 메인메뉴 총 매출
        main_sales = df[df['menu_category'] == '메인메뉴'].groupby('date')['매출수량'].sum().reset_index()
        main_sales.columns = ['date', 'total_main_sales']
        df = df.merge(main_sales, on='date', how='left')
        df['total_main_sales'] = df['total_main_sales'].fillna(0)

        return df

    def calculate_menu_statistics(self, df):
        """메뉴별 통계 계산 및 저장"""
        print_section("메뉴별 통계 계산", "📈")

        for menu in df['영업장명_메뉴명'].unique():
            menu_data = df[df['영업장명_메뉴명'] == menu]['매출수량']

            weekend_data = df[(df['영업장명_메뉴명'] == menu) & (df['is_weekend'] == 1)]['매출수량']
            weekday_data = df[(df['영업장명_메뉴명'] == menu) & (df['is_weekend'] == 0)]['매출수량']

            self.menu_stats[menu] = {
                'mean': menu_data.mean(),
                'std': menu_data.std(),
                'min': menu_data.min(),
                'max': menu_data.max(),
                'q25': menu_data.quantile(0.25),
                'q75': menu_data.quantile(0.75),
                'zero_ratio': (menu_data == 0).mean(),
                'weekend_mean': weekend_data.mean() if len(weekend_data) > 0 else menu_data.mean(),
                'weekday_mean': weekday_data.mean() if len(weekday_data) > 0 else menu_data.mean(),
            }

        print(f"✓ {len(self.menu_stats)}개 메뉴의 통계 정보 계산 완료")

    def prepare_features(self, df, is_train=True):
        """전체 특성 엔지니어링 파이프라인"""
        print_section(f"{'Train' if is_train else 'Test'} 데이터 특성 엔지니어링", "🔧")

        # 1. 시간 특성
        df = self.extract_datetime_features(df)
        print("✓ 시간 특성 추출 완료")

        # 2. 메뉴 특성
        df = self.extract_menu_features(df)
        print("✓ 메뉴 특성 추출 완료")

        # 3. 지연 특성 (Train만) - 안전한 방법 사용
        if is_train:
            df = self.extract_lag_features_safe(df, is_train=True)
            print("✓ 시계열 지연 특성 추출 완료")

        # 4. 상관관계 특성 (Train만)
        if is_train:
            df = self.extract_correlation_features(df, is_train=True)
            print("✓ 메뉴 간 상관관계 특성 추출 완료")

        # 5. 메뉴 통계 (Train만)
        if is_train:
            self.calculate_menu_statistics(df)

        # 6. 카테고리 인코딩
        cat_columns = ['업장명', '메뉴명', '영업장명_메뉴명', 'menu_category', 'season', 'restaurant_type']
        for col in cat_columns:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    unique_values = df[col].unique().tolist()
                    self.label_encoders[col].fit(unique_values + ['UNKNOWN'])

                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: self.label_encoders[col].transform([x])[0]
                    if x in self.label_encoders[col].classes_
                    else self.label_encoders[col].transform(['UNKNOWN'])[0]
                )

        print("✓ 카테고리 인코딩 완료")
        print(f"✓ 최종 특성 개수: {df.shape[1]}")

        return df


# ========================================
# 5. 고도화된 ML 모델
# ========================================

class AdvancedMLModel:
    """고도화된 앙상블 ML 모델"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_features = {}

    def create_model_ensemble(self):
        """강화된 앙상블 모델 생성"""
        models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=2000,
                max_depth=8,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror',
                verbosity=0
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=2000,
                max_depth=8,
                learning_rate=0.02,
                num_leaves=64,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=2000,
                depth=6,
                learning_rate=0.02,
                l2_leaf_reg=5,
                random_seed=42,
                verbose=False,
                early_stopping_rounds=100
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        }
        return models

    def train_models(self, X_train, y_train, X_val, y_val, model_name="default", weights=None):
        """모델 학습"""
        print_section(f"{model_name} 모델 학습", "🚀")

        start_time = time.time()
        models = self.create_model_ensemble()
        results = {}

        # 데이터 타입 검증 및 변환
        print(f"학습 데이터 형태: {X_train.shape}")
        print(f"데이터 타입: {X_train.dtypes.value_counts()}")

        # 문자열 컬럼이 있는지 확인
        string_cols = X_train.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            print(f"⚠️ 문자열 컬럼 발견: {list(string_cols)}")
            # 문자열 컬럼 제거
            X_train = X_train.select_dtypes(exclude=['object'])
            X_val = X_val.select_dtypes(exclude=['object'])
            print(f"문자열 컬럼 제거 후 형태: {X_train.shape}")

        # 모든 컬럼을 숫자형으로 변환
        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')

        # 무한대 값 처리
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 데이터 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        self.scalers[model_name] = scaler

        for name, model in models.items():
            print(f"\n→ {name} 학습 중...", end=" ")
            model_start = time.time()

            try:
                # 가중치 적용 학습
                if weights is not None and hasattr(model, 'fit'):
                    try:
                        model.fit(X_train_scaled, y_train, sample_weight=weights)
                    except:
                        model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train_scaled, y_train)

                # 예측
                pred = model.predict(X_val_scaled)
                pred = np.maximum(pred, 0)  # 음수 제거

                # 성능 계산
                score = smape(y_val, pred)

                results[name] = {
                    'model': model,
                    'pred': pred,
                    'smape': score
                }

                print(f"완료 ({time.time() - model_start:.1f}초, sMAPE: {score:.3f})")

            except Exception as e:
                print(f"실패 ({str(e)})")
                continue

        print(f"\n✓ 총 학습 시간: {time.time() - start_time:.1f}초")
        print(f"✓ 성공한 모델: {len(results)}개")

        return results

    def weighted_ensemble_predict(self, models, X_test, scaler, weights=None):
        """가중 앙상블 예측"""
        # 데이터 타입 검증 및 변환
        if hasattr(X_test, 'select_dtypes'):
            # 문자열 컬럼 제거
            string_cols = X_test.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                X_test = X_test.select_dtypes(exclude=['object'])

            # 숫자형으로 변환
            X_test = X_test.astype('float32')

            # 무한대 값 처리
            X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

        X_test_scaled = scaler.transform(X_test)
        predictions = []

        if weights is None:
            # sMAPE 기반 자동 가중치
            scores = [m['smape'] for m in models.values() if 'smape' in m]
            if scores:
                inv_scores = [1 / (s + 1) for s in scores]
                total = sum(inv_scores)
                weights = {name: inv_scores[i] / total for i, name in enumerate(models.keys())}
            else:
                weights = {name: 1 / len(models) for name in models.keys()}

        for name, model_info in models.items():
            if 'model' in model_info:
                pred = model_info['model'].predict(X_test_scaled)
                pred = np.maximum(pred, 0)
                predictions.append(pred * weights.get(name, 0))

        if predictions:
            return np.sum(predictions, axis=0)
        else:
            return np.zeros(len(X_test))


# ========================================
# 6. 메인 파이프라인
# ========================================

class AdvancedPipeline:
    """고도화된 전체 파이프라인"""

    def __init__(self):
        self.processor = AdvancedDataProcessor()
        self.model = AdvancedMLModel()
        self.menu_models = {}
        self.global_models = {}

    def analyze_data(self, df):
        """데이터 분석 및 시각화"""
        print_section("데이터 분석", "📊")

        # 기본 통계
        print("📈 매출수량 기본 통계:")
        print(df['매출수량'].describe())

        # 업장별 매출 분석
        restaurant_sales = df.groupby('업장명')['매출수량'].agg(['sum', 'mean', 'count'])
        print(f"\n🏪 업장별 매출 현황:")
        print(restaurant_sales.sort_values('sum', ascending=False))

        # 카테고리별 매출 분석
        if 'menu_category' in df.columns:
            category_sales = df.groupby('menu_category')['매출수량'].agg(['sum', 'mean', 'count'])
            print(f"\n🍽️ 메뉴 카테고리별 매출 현황:")
            print(category_sales.sort_values('sum', ascending=False))

        # 계절성 분석
        if 'season' in df.columns:
            seasonal_sales = df.groupby('season')['매출수량'].agg(['sum', 'mean'])
            print(f"\n🌸 계절별 매출 현황:")
            print(seasonal_sales)

        # 요일별 분석
        if 'dayofweek' in df.columns:
            dow_sales = df.groupby('dayofweek')['매출수량'].agg(['sum', 'mean'])
            dow_sales.index = ['월', '화', '수', '목', '금', '토', '일']
            print(f"\n📅 요일별 매출 현황:")
            print(dow_sales)

    def create_future_features(self, last_row, pred_date, feature_cols):
        """미래 예측을 위한 특성 생성"""
        future_data = last_row.copy()

        # 날짜 관련 특성 업데이트
        future_data['date'] = pred_date
        future_data['year'] = pred_date.year
        future_data['month'] = pred_date.month
        future_data['day'] = pred_date.day
        future_data['dayofweek'] = pred_date.dayofweek
        future_data['dayofyear'] = pred_date.dayofyear
        future_data['week'] = pred_date.isocalendar().week
        future_data['quarter'] = pred_date.quarter

        # 주말/평일
        future_data['is_weekend'] = 1 if pred_date.dayofweek >= 5 else 0
        future_data['is_friday'] = 1 if pred_date.dayofweek == 4 else 0
        future_data['is_saturday'] = 1 if pred_date.dayofweek == 5 else 0
        future_data['is_sunday'] = 1 if pred_date.dayofweek == 6 else 0

        # 공휴일
        future_data['is_holiday'] = 1 if pred_date in self.processor.kr_holidays else 0

        # 계절성
        month = pred_date.month
        if month in [12, 1, 2]:
            future_data['season'] = 'winter'
            future_data['is_ski_season'] = 1
            future_data['is_forest_season'] = 0
            future_data['is_family_season'] = 0
        elif month in [3, 4, 5, 9, 10, 11]:
            future_data['season'] = 'spring_fall'
            future_data['is_ski_season'] = 0
            future_data['is_forest_season'] = 1
            future_data['is_family_season'] = 0
        else:
            future_data['season'] = 'summer'
            future_data['is_ski_season'] = 0
            future_data['is_forest_season'] = 0
            future_data['is_family_season'] = 1

        # 연말연시
        future_data['is_year_end'] = 1 if (month == 12 and pred_date.day >= 20) else 0
        future_data['is_new_year'] = 1 if (month == 1 and pred_date.day <= 10) else 0

        # 주기적 특성
        future_data['month_sin'] = np.sin(2 * np.pi * month / 12)
        future_data['month_cos'] = np.cos(2 * np.pi * month / 12)
        future_data['day_sin'] = np.sin(2 * np.pi * pred_date.day / 31)
        future_data['day_cos'] = np.cos(2 * np.pi * pred_date.day / 31)
        future_data['dayofweek_sin'] = np.sin(2 * np.pi * pred_date.dayofweek / 7)
        future_data['dayofweek_cos'] = np.cos(2 * np.pi * pred_date.dayofweek / 7)

        # 카테고리 인코딩
        for col in ['season']:
            if f'{col}_encoded' in feature_cols and col in self.processor.label_encoders:
                future_data[f'{col}_encoded'] = self.processor.label_encoders[col].transform([future_data[col]])[0]

        return future_data

    def create_submission(self, predictions):
        """제출 파일 생성"""
        print_section("제출 파일 생성", "📝")

        pred_df = pd.DataFrame(predictions)
        print(f"원본 예측 데이터 형태: {pred_df.shape}")
        print(f"컬럼: {list(pred_df.columns)}")

        # 음수 제거 및 정수 변환
        pred_df['매출수량'] = pred_df['매출수량'].clip(lower=0).round().astype(int)

        # Long format을 Wide format으로 변환
        print("Long format을 Wide format으로 변환 중...")

        # pivot_table 사용하여 변환
        wide_df = pred_df.pivot_table(
            index='영업일자',
            columns='영업장명_메뉴명',
            values='매출수량',
            fill_value=0
        ).reset_index()

        print(f"Wide format 형태: {wide_df.shape}")

        # sample_submission이 있는 경우 컬럼 순서 맞추기
        if hasattr(self.processor, 'sample_submission') and self.processor.sample_submission is not None:
            try:
                # sample_submission의 컬럼 순서대로 정렬
                missing_cols = set(self.processor.sample_submission.columns) - set(wide_df.columns)
                if missing_cols:
                    print(f"⚠️ 누락된 컬럼 {len(missing_cols)}개를 0으로 추가")
                    for col in missing_cols:
                        wide_df[col] = 0

                # 컬럼 순서 맞추기
                wide_df = wide_df[self.processor.sample_submission.columns]
                print("✓ Sample submission 형식에 맞춰 컬럼 순서 조정 완료")

            except Exception as e:
                print(f"⚠️ Sample submission 형식 맞추기 실패: {e}")
                print("기본 형식으로 저장합니다.")
        else:
            print("⚠️ Sample submission 파일이 없어 기본 형식으로 저장")

        # 저장
        output_file = f'advanced_submission_{datetime.now().strftime("%m%d_%H%M")}.csv'
        wide_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"✓ 제출 파일 저장: {output_file}")
        print(f"✓ 예측 날짜 수: {len(wide_df)}")
        print(f"✓ 메뉴 수: {len(wide_df.columns) - 1}")  # 영업일자 제외
        print(f"✓ 총 예측값 합계: {wide_df.select_dtypes(include=[np.number]).sum().sum()}")

        return wide_df

    def calculate_test_smape(self, predictions, test_data):
        """테스트 데이터에 대한 sMAPE 계산 (검증용)"""
        print_section("테스트 성능 검증", "🎯")

        # 실제 값이 있는 경우에만 계산
        total_smape = 0
        valid_predictions = 0

        for test_name, test_df in test_data.items():
            if '매출수량' in test_df.columns:
                # 해당 테스트의 예측값 필터링
                test_preds = [p for p in predictions if test_name in p['영업일자']]

                if test_preds:
                    pred_df = pd.DataFrame(test_preds)

                    # 실제값과 예측값 매칭
                    merged = test_df.merge(pred_df, on='영업장명_메뉴명', how='inner')

                    if len(merged) > 0:
                        smape_score = smape(merged['매출수량_x'], merged['매출수량_y'])
                        print(f"✓ {test_name} sMAPE: {smape_score:.3f}")
                        total_smape += smape_score
                        valid_predictions += 1

        if valid_predictions > 0:
            avg_smape = total_smape / valid_predictions
            print(f"\n🎯 평균 sMAPE: {avg_smape:.3f}")
            return avg_smape
        else:
            print("⚠️ 실제값을 포함한 테스트 데이터가 없어 sMAPE 계산 불가")
            return None

    def run(self):
        """전체 파이프라인 실행"""
        print_section("고도화된 곤지암 리조트 수요예측 파이프라인 시작", "🚀")

        total_start = time.time()

        # 1. 데이터 로드
        train, test_data = self.processor.load_data()

        # 2. 특성 엔지니어링
        train_fe = self.processor.prepare_features(train, is_train=True)

        # 3. 데이터 분석
        self.analyze_data(train_fe)

        # 4. 특성 선택 - 문자열 컬럼 제외
        exclude_cols = [
            '영업일자', '영업장명_메뉴명', '매출수량', 'date', '업장명', '메뉴명',
            'season', 'menu_category', 'restaurant_type'  # 문자열 컬럼들 제외
        ]

        # 숫자형 컬럼만 선택
        feature_cols = []
        for col in train_fe.columns:
            if col not in exclude_cols:
                # 숫자형 데이터인지 확인
                if train_fe[col].dtype in ['int64', 'float64', 'int32', 'float32', 'bool']:
                    feature_cols.append(col)
                elif col.endswith('_encoded'):  # 인코딩된 컬럼은 포함
                    feature_cols.append(col)

        print(f"\n📊 사용할 특성 개수: {len(feature_cols)}")
        print(f"📊 특성 목록 (처음 20개): {feature_cols[:20]}")

        target_col = '매출수량'

        # 5. 시계열 분할
        train_fe = train_fe.sort_values('date')
        split_date = train_fe['date'].max() - pd.Timedelta(days=45)

        train_set = train_fe[train_fe['date'] < split_date]
        val_set = train_fe[train_fe['date'] >= split_date]

        X_train = train_set[feature_cols].fillna(0)
        y_train = train_set[target_col]
        X_val = val_set[feature_cols].fillna(0)
        y_val = val_set[target_col]

        # 업장별 가중치 계산
        train_weights = train_set['restaurant_weight'].values

        print(f"학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

        # 6. 전체 모델 학습
        print_section("전체 통합 모델 학습", "🎯")
        self.global_models = self.model.train_models(
            X_train, y_train, X_val, y_val, "global", weights=train_weights
        )

        # 7. 주요 메뉴별 개별 모델 학습
        print_section("주요 메뉴별 개별 모델 학습", "🎯")

        # 담하와 미라시아 메뉴 우선 + 매출 상위 메뉴
        high_weight_menus = train_fe[
            (train_fe['업장명'].str.contains('담하|미라시아', na=False)) |
            (train_fe.groupby('영업장명_메뉴명')['매출수량'].transform('sum') > train_fe['매출수량'].sum() * 0.005)
            ]['영업장명_메뉴명'].unique()

        print(f"개별 모델 학습 대상: {len(high_weight_menus)}개 메뉴")

        for menu in tqdm(high_weight_menus, desc="개별 모델 학습"):
            menu_data = train_fe[train_fe['영업장명_메뉴명'] == menu]

            if len(menu_data) < 100:  # 데이터가 부족한 경우 스킵
                continue

            menu_train = menu_data[menu_data['date'] < split_date]
            menu_val = menu_data[menu_data['date'] >= split_date]

            if len(menu_val) < 5:
                continue

            X_train_menu = menu_train[feature_cols].fillna(0)
            y_train_menu = menu_train[target_col]
            X_val_menu = menu_val[feature_cols].fillna(0)
            y_val_menu = menu_val[target_col]

            menu_weights = menu_train['restaurant_weight'].values

            # 메뉴별 모델 학습
            menu_models = self.model.train_models(
                X_train_menu, y_train_menu,
                X_val_menu, y_val_menu,
                menu, weights=menu_weights
            )

            self.menu_models[menu] = menu_models

        # 8. 최종 모델 재학습
        print_section("최종 모델 재학습", "🎯")

        X_full = train_fe[feature_cols].fillna(0)
        y_full = train_fe[target_col]
        full_weights = train_fe['restaurant_weight'].values

        # 마지막 10% 검증용
        val_size = max(int(len(X_full) * 0.1), 1000)

        final_models = self.model.train_models(
            X_full[:-val_size], y_full.iloc[:-val_size],
            X_full[-val_size:], y_full.iloc[-val_size:],
            "final", weights=full_weights[:-val_size]
        )

        # 9. 테스트 데이터 예측
        print_section("테스트 데이터 예측", "📝")

        all_predictions = []

        for test_name, test_df in test_data.items():
            print(f"\n→ {test_name} 예측 중...")

            # 테스트 데이터 특성 엔지니어링 (간단 버전)
            test_fe = test_df.copy()
            test_fe = self.processor.extract_datetime_features(test_fe)
            test_fe = self.processor.extract_menu_features(test_fe)

            # 카테고리 인코딩
            cat_columns = ['업장명', '메뉴명', '영업장명_메뉴명', 'menu_category', 'season', 'restaurant_type']
            for col in cat_columns:
                if col in test_fe.columns and col in self.processor.label_encoders:
                    test_fe[f'{col}_encoded'] = test_fe[col].apply(
                        lambda x: self.processor.label_encoders[col].transform([x])[0]
                        if x in self.processor.label_encoders[col].classes_
                        else self.processor.label_encoders[col].transform(['UNKNOWN'])[0]
                    )

            # 마지막 날짜
            last_date = test_df['영업일자'].max()
            if not isinstance(last_date, pd.Timestamp):
                last_date = pd.to_datetime(last_date)

            # 7일간 예측
            for day_ahead in range(1, 8):
                pred_date = last_date + timedelta(days=day_ahead)

                for menu in test_df['영업장명_메뉴명'].unique():
                    # 해당 메뉴의 마지막 데이터 행
                    menu_last_row = test_fe[test_fe['영업장명_메뉴명'] == menu].iloc[-1].copy()

                    # 미래 특성 생성
                    future_features = self.create_future_features(
                        menu_last_row, pred_date, feature_cols
                    )

                    # 누락된 특성 처리
                    for col in feature_cols:
                        if col not in future_features.index:
                            future_features[col] = 0

                    # DataFrame으로 변환하고 문자열 컬럼 제거
                    future_df = pd.DataFrame([future_features[feature_cols]]).fillna(0)

                    # 문자열 컬럼 제거 및 숫자형 변환
                    string_cols = future_df.select_dtypes(include=['object']).columns
                    if len(string_cols) > 0:
                        future_df = future_df.select_dtypes(exclude=['object'])

                    future_df = future_df.astype('float32')
                    future_df = future_df.replace([np.inf, -np.inf], np.nan).fillna(0)

                    # 예측
                    if menu in self.menu_models and len(self.menu_models[menu]) > 0:
                        # 메뉴별 모델 사용
                        pred_value = self.model.weighted_ensemble_predict(
                            self.menu_models[menu],
                            future_df,
                            self.model.scalers[menu]
                        )[0]
                    else:
                        # 전체 모델 사용
                        pred_value = self.model.weighted_ensemble_predict(
                            final_models,
                            future_df,
                            self.model.scalers["final"]
                        )[0]

                    # 후처리
                    if menu in self.processor.menu_stats:
                        stats = self.processor.menu_stats[menu]

                        # 극단값 제한
                        pred_value = np.clip(pred_value, 0, stats['max'] * 2.0)

                        # 요일별 조정
                        if pred_date.dayofweek >= 5:  # 주말
                            weekend_ratio = stats.get('weekend_mean', stats['mean']) / (stats['mean'] + 1e-6)
                            pred_value *= max(weekend_ratio, 0.5)

                        # 공휴일 조정
                        if pred_date in self.processor.kr_holidays:
                            pred_value *= 1.3

                        # 계절별 조정
                        month = pred_date.month
                        restaurant = menu.split('_')[0]

                        if '느티나무' in restaurant and month in [6, 7, 8]:  # 여름 BBQ
                            pred_value *= 1.2
                        elif '화담숲' in restaurant and month in [3, 4, 5, 9, 10, 11]:  # 봄가을 화담숲
                            pred_value *= 1.2
                        elif ('카페테리아' in restaurant or '포레스트릿' in restaurant) and month in [12, 1, 2]:  # 겨울 스키
                            pred_value *= 1.2

                        # HOT/ICE 메뉴 계절별 조정
                        if 'HOT' in menu and month in [12, 1, 2]:
                            pred_value *= 1.3
                        elif 'ICE' in menu and month in [6, 7, 8]:
                            pred_value *= 1.3
                        elif 'ICE' in menu and month in [12, 1, 2]:
                            pred_value *= 0.7

                    all_predictions.append({
                        '영업일자': f"{test_name}+{day_ahead}일",
                        '영업장명_메뉴명': menu,
                        '매출수량': max(0, int(round(pred_value)))
                    })

        # 10. 제출 파일 생성
        submission_df = self.create_submission(all_predictions)

        # 11. 테스트 성능 계산 (가능한 경우)
        test_smape = self.calculate_test_smape(all_predictions, test_data)

        # 12. 최종 결과 요약
        total_time = time.time() - total_start
        print_section("파이프라인 완료", "✅")
        print(f"✅ 총 실행 시간: {total_time / 60:.1f}분")
        print(f"✅ 학습된 전체 모델: {len(self.global_models)}개")
        print(f"✅ 학습된 개별 모델: {len(self.menu_models)}개")
        print(f"✅ 총 예측 건수: {len(all_predictions)}개")
        if test_smape:
            print(f"✅ 예상 sMAPE: {test_smape:.3f}")

        return submission_df


# ========================================
# 7. 실행
# ========================================

if __name__ == "__main__":
    # 파이프라인 실행
    pipeline = AdvancedPipeline()
    result = pipeline.run()

    print("\n🎯 고도화된 곤지암 리조트 수요예측 모델 실행 완료!")
    print("📊 결과 파일이 생성되었습니다.")