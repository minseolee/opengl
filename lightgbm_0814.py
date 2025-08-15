#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
곤지암 리조트 식음업장 수요예측 고급 ML 모델 v2.0
- 실제 학습 시간 증가 (5-10분)
- 메뉴별 개별 모델링
- 적절한 데이터 스케일링
- Test에서도 사용 가능한 특성만 활용
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

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'AppleGothic'  # macOS 한글 폰트
plt.rcParams['axes.unicode_minus'] = False

# 머신러닝 모델
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet

# 부스팅 모델
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 시계열 모델
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import holidays


# ========================================
# 2. 평가 메트릭 정의
# ========================================
def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    경진대회 평가 지표
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)


# ========================================
# 3. 데이터 로드 및 전처리
# ========================================
class DataProcessor:
    def __init__(self):
        self.kr_holidays = holidays.KR(years=range(2023, 2026))  # 2023-2025년 한국 공휴일
        self.label_encoders = {}
        self.scalers = {}
        self.menu_stats = {}  # 메뉴별 통계 저장

    def load_data(self, train_path='./train/train.csv', test_dir='./test/'):
        """데이터 로드"""
        print("=" * 50)
        print("📊 데이터 로드 중...")

        # Train 데이터
        self.train = pd.read_csv(train_path)
        print(f"✓ Train 데이터: {self.train.shape}")

        # 데이터 타입 변환
        if self.train['영업일자'].dtype == 'object':
            self.train['영업일자'] = pd.to_datetime(self.train['영업일자'])

        # Test 데이터들
        test_files = sorted(glob.glob(os.path.join(test_dir, 'TEST_*.csv')))
        self.test_data = {}
        for file in test_files:
            test_name = os.path.basename(file).replace('.csv', '')
            df = pd.read_csv(file)
            if df['영업일자'].dtype == 'object':
                df['영업일자'] = pd.to_datetime(df['영업일자'])
            self.test_data[test_name] = df
            print(f"✓ {test_name} 데이터: {df.shape}")

        # Sample submission
        self.sample_submission = pd.read_csv('./sample_submission.csv')
        print(f"✓ Submission 형식: {self.sample_submission.shape}")

        return self.train, self.test_data

    def extract_datetime_features(self, df):
        """시간 관련 특성 추출 - Test에서도 사용 가능한 특성만"""
        df = df.copy()

        # 날짜 파싱
        if 'date' not in df.columns:
            if df['영업일자'].dtype != 'datetime64[ns]':
                df['date'] = pd.to_datetime(df['영업일자'])
            else:
                df['date'] = df['영업일자']

        # 기본 시간 특성
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek  # 0=월요일, 6=일요일
        df['quarter'] = df['date'].dt.quarter
        df['weekofyear'] = df['date'].dt.isocalendar().week
        df['dayofyear'] = df['date'].dt.dayofyear

        # 주말/평일
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # 공휴일
        df['is_holiday'] = df['date'].apply(lambda x: x in self.kr_holidays).astype(int)

        # 연휴 (공휴일 + 전후)
        df['day_before_holiday'] = df['date'].apply(
            lambda x: (x + timedelta(days=1)) in self.kr_holidays
        ).astype(int)
        df['day_after_holiday'] = df['date'].apply(
            lambda x: (x - timedelta(days=1)) in self.kr_holidays
        ).astype(int)
        df['is_long_weekend'] = ((df['is_holiday'] == 1) |
                                 (df['day_before_holiday'] == 1) |
                                 (df['day_after_holiday'] == 1)).astype(int)

        # 계절 (한국 기준)
        df['season'] = df['month'].apply(lambda x:
                                         1 if x in [3, 4, 5] else  # 봄
                                         2 if x in [6, 7, 8] else  # 여름
                                         3 if x in [9, 10, 11] else  # 가을
                                         4  # 겨울
                                         )

        # 월초/월중/월말
        df['month_period'] = df['day'].apply(lambda x:
                                             1 if x <= 10 else
                                             2 if x <= 20 else
                                             3
                                             )

        # 주차
        df['week_of_month'] = (df['day'] - 1) // 7 + 1

        # Sin/Cos 변환 (주기성 반영)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['weekofyear_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
        df['weekofyear_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)

        return df

    def extract_menu_features(self, df):
        """메뉴 관련 특성 추출"""
        df = df.copy()

        # 업장명과 메뉴명 분리
        df['업장명'] = df['영업장명_메뉴명'].apply(lambda x: x.split('_')[0])
        df['메뉴명'] = df['영업장명_메뉴명'].apply(lambda x: '_'.join(x.split('_')[1:]))

        # 메뉴 카테고리 추출
        df['is_단체'] = df['메뉴명'].str.contains('단체', na=False).astype(int)
        df['is_정식'] = df['메뉴명'].str.contains('정식', na=False).astype(int)
        df['is_후식'] = df['메뉴명'].str.contains('후식', na=False).astype(int)
        df['is_브런치'] = df['메뉴명'].str.contains('브런치', na=False).astype(int)
        df['is_주류'] = df['메뉴명'].apply(lambda x:
                                      any(word in str(x) for word in
                                          ['맥주', '소주', '막걸리', '와인', '주류', '카스', '테라', '참이슬', '처음처럼'])
                                      ).astype(int)
        df['is_음료'] = df['메뉴명'].apply(lambda x:
                                      any(word in str(x) for word in
                                          ['콜라', '사이다', '스프라이트', '커피', '아메리카노', '라떼', '에이드', '차'])
                                      ).astype(int)
        df['is_면류'] = df['메뉴명'].apply(lambda x:
                                      any(word in str(x) for word in ['면', '우동', '파스타', '스파게티', '짜장', '짬뽕'])
                                      ).astype(int)
        df['is_고기'] = df['메뉴명'].apply(lambda x:
                                      any(word in str(x) for word in ['고기', '삼겹', '갈비', '스테이크', '불고기', '목살'])
                                      ).astype(int)

        # 메뉴명 길이 (복잡도)
        df['메뉴명_길이'] = df['메뉴명'].str.len()

        return df

    def calculate_menu_statistics(self, train_df):
        """메뉴별 통계 계산 (Train 데이터 기반)"""
        print("📊 메뉴별 통계 계산 중...")

        for menu in tqdm(train_df['영업장명_메뉴명'].unique(), desc="메뉴 통계"):
            menu_data = train_df[train_df['영업장명_메뉴명'] == menu]['매출수량']

            self.menu_stats[menu] = {
                'mean': menu_data.mean(),
                'std': menu_data.std(),
                'median': menu_data.median(),
                'min': menu_data.min(),
                'max': menu_data.max(),
                'q25': menu_data.quantile(0.25),
                'q75': menu_data.quantile(0.75),
                'zero_ratio': (menu_data == 0).mean(),
                'positive_ratio': (menu_data > 0).mean()
            }

    def add_menu_statistics(self, df):
        """메뉴 통계 특성 추가"""
        df = df.copy()

        # 메뉴별 통계 추가
        for stat in ['mean', 'std', 'median', 'zero_ratio', 'positive_ratio']:
            df[f'menu_{stat}'] = df['영업장명_메뉴명'].apply(
                lambda x: self.menu_stats.get(x, {}).get(stat, 0)
            )

        # 업장별 통계
        store_stats = df.groupby('업장명')['매출수량'].agg(['mean', 'std']).reset_index()
        store_stats.columns = ['업장명', 'store_mean', 'store_std']
        df = df.merge(store_stats, on='업장명', how='left')

        return df


# ========================================
# 4. 고급 모델 학습 클래스
# ========================================
class AdvancedDemandForecastModel:
    def __init__(self):
        self.models = {}
        self.processor = DataProcessor()
        self.best_params = {}
        self.scalers = {}

    def prepare_features(self, df, is_train=True):
        """특성 엔지니어링 통합"""
        # 1. 시간 특성
        df = self.processor.extract_datetime_features(df)

        # 2. 메뉴 특성
        df = self.processor.extract_menu_features(df)

        # 3. 메뉴 통계 특성 추가
        if is_train:
            self.processor.calculate_menu_statistics(df)
        df = self.processor.add_menu_statistics(df)

        # 4. 카테고리 인코딩
        cat_columns = ['업장명', '메뉴명', '영업장명_메뉴명']
        for col in cat_columns:
            if col not in self.processor.label_encoders:
                self.processor.label_encoders[col] = LabelEncoder()
                # Unknown 카테고리 처리를 위해 fit 시 모든 값 포함
                all_values = df[col].unique().tolist()
                self.processor.label_encoders[col].fit(all_values + ['UNKNOWN'])

            # 변환 시 unknown 처리
            df[f'{col}_encoded'] = df[col].apply(
                lambda x: self.processor.label_encoders[col].transform([x])[0]
                if x in self.processor.label_encoders[col].classes_
                else self.processor.label_encoders[col].transform(['UNKNOWN'])[0]
            )

        return df

    def create_model_ensemble(self):
        """앙상블 모델 생성 - 더 강력한 설정"""
        models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=2000,  # 증가
                max_depth=12,
                learning_rate=0.005,  # 감소
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror',
                early_stopping_rounds=100
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=2000,  # 증가
                max_depth=12,
                learning_rate=0.005,  # 감소
                num_leaves=50,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                force_col_wise=True
            ),
            'catboost': CatBoostRegressor(
                iterations=2000,  # 증가
                depth=10,
                learning_rate=0.005,  # 감소
                l2_leaf_reg=3,
                border_count=128,
                random_seed=42,
                verbose=False,
                early_stopping_rounds=100,
                task_type='CPU'
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=500,  # 증가
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=500,  # 증가
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=False,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.85,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        }
        return models

    def train_models(self, X_train, y_train, X_val, y_val, menu_name="전체"):
        """여러 모델 학습 - 개선된 버전"""
        print(f"\n🤖 [{menu_name}] 모델 학습 중...")

        # 데이터 스케일링
        scaler = RobustScaler()  # 이상치에 강한 스케일러
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 스케일러 저장
        self.scalers[menu_name] = scaler

        models = self.create_model_ensemble()
        results = {}

        start_time = time.time()

        for name, model in models.items():
            print(f"  → {name} 학습 중...", end=" ")
            model_start = time.time()

            try:
                if name in ['xgboost', 'lightgbm', 'catboost']:
                    # Early stopping을 위한 eval_set
                    if name == 'xgboost':
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=[(X_val_scaled, y_val)],
                            verbose=False
                        )
                    elif name == 'lightgbm':
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=[(X_val_scaled, y_val)],
                            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
                        )
                    elif name == 'catboost':
                        model.fit(
                            X_train_scaled, y_train,
                            eval_set=(X_val_scaled, y_val),
                            verbose=False
                        )
                else:
                    # 일반 모델
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

                print(f"완료 ({time.time() - model_start:.1f}초, sMAPE: {score:.2f})")

            except Exception as e:
                print(f"실패 ({str(e)})")
                continue

        print(f"  총 학습 시간: {time.time() - start_time:.1f}초")

        return results

    def weighted_ensemble_predict(self, models, X_test, scaler, weights=None):
        """가중 앙상블 예측"""
        X_test_scaled = scaler.transform(X_test)
        predictions = []

        if weights is None:
            # sMAPE 기반 자동 가중치
            scores = [m['smape'] for m in models.values() if 'smape' in m]
            if scores:
                inv_scores = [1 / (s + 1) for s in scores]
                total = sum(inv_scores)
                weights = {name: inv_scores[i] / total
                           for i, name in enumerate(models.keys())}
            else:
                weights = {name: 1 / len(models) for name in models.keys()}

        for name, model_info in models.items():
            if 'model' in model_info:
                pred = model_info['model'].predict(X_test_scaled)
                pred = np.maximum(pred, 0)  # 음수 제거
                predictions.append(pred * weights.get(name, 0))

        if predictions:
            return np.sum(predictions, axis=0)
        else:
            return np.zeros(len(X_test))


# ========================================
# 5. 메인 파이프라인
# ========================================
class Pipeline:
    def __init__(self):
        self.model = AdvancedDemandForecastModel()
        self.menu_models = {}  # 메뉴별 모델 저장

    def run(self):
        """전체 파이프라인 실행"""
        print("=" * 70)
        print("🚀 곤지암 리조트 식음업장 수요예측 모델 파이프라인 시작")
        print("=" * 70)

        total_start = time.time()

        # 1. 데이터 로드
        train, test_data = self.model.processor.load_data()

        # 2. 특성 엔지니어링
        print("\n📈 Train 데이터 특성 엔지니어링 진행 중...")
        train_fe = self.model.prepare_features(train, is_train=True)

        # 3. 데이터 분석
        self.analyze_data(train_fe)

        # 4. 특성 선택 (Test에서도 사용 가능한 특성만)
        exclude_cols = ['영업일자', '영업장명_메뉴명', '매출수량', 'date', '업장명', '메뉴명']
        feature_cols = [col for col in train_fe.columns if col not in exclude_cols]

        print(f"\n📊 사용할 특성 개수: {len(feature_cols)}")

        target_col = '매출수량'

        # 5. 전체 모델 학습 (빠른 예측을 위한 기본 모델)
        print("\n" + "=" * 50)
        print("🎯 전체 데이터 통합 모델 학습")
        print("=" * 50)

        # 시계열 분할
        train_fe = train_fe.sort_values('date')
        split_date = train_fe['date'].max() - pd.Timedelta(days=60)

        train_set = train_fe[train_fe['date'] < split_date]
        val_set = train_fe[train_fe['date'] >= split_date]

        if len(val_set) < 100:
            # 검증 데이터가 너무 적으면 비율로 분할
            split_idx = int(len(train_fe) * 0.85)
            train_set = train_fe.iloc[:split_idx]
            val_set = train_fe.iloc[split_idx:]

        X_train = train_set[feature_cols].fillna(0)
        y_train = train_set[target_col]
        X_val = val_set[feature_cols].fillna(0)
        y_val = val_set[target_col]

        print(f"학습 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

        # 전체 모델 학습
        global_models = self.model.train_models(X_train, y_train, X_val, y_val, "전체")

        # 성능 출력
        self.print_model_performance(global_models)

        # 6. 주요 메뉴별 개별 모델 학습 (선택적)
        print("\n" + "=" * 50)
        print("🎯 주요 메뉴별 개별 모델 학습")
        print("=" * 50)

        # 매출이 많은 상위 메뉴 선택
        menu_sales = train_fe.groupby('영업장명_메뉴명')['매출수량'].agg(['sum', 'count'])
        top_menus = menu_sales[menu_sales['count'] > 100].sort_values('sum', ascending=False).head(20).index

        print(f"상위 {len(top_menus)}개 메뉴에 대해 개별 모델 학습")

        for menu in tqdm(top_menus, desc="메뉴별 모델"):
            menu_data = train_fe[train_fe['영업장명_메뉴명'] == menu]

            if len(menu_data) < 200:
                continue

            # 시계열 분할
            menu_train = menu_data[menu_data['date'] < split_date]
            menu_val = menu_data[menu_data['date'] >= split_date]

            if len(menu_val) < 10:
                continue

            X_train_menu = menu_train[feature_cols].fillna(0)
            y_train_menu = menu_train[target_col]
            X_val_menu = menu_val[feature_cols].fillna(0)
            y_val_menu = menu_val[target_col]

            # 모델 학습
            menu_models = self.model.train_models(
                X_train_menu, y_train_menu,
                X_val_menu, y_val_menu,
                menu
            )

            self.menu_models[menu] = menu_models

        # 7. 최종 모델로 전체 데이터 재학습
        print("\n" + "=" * 50)
        print("🎯 최종 모델 학습 (전체 데이터)")
        print("=" * 50)

        X_full = train_fe[feature_cols].fillna(0)
        y_full = train_fe[target_col]

        # 검증용 마지막 10%
        val_size = max(int(len(X_full) * 0.1), 1000)
        final_models = self.model.train_models(
            X_full[:-val_size], y_full.iloc[:-val_size],
            X_full[-val_size:], y_full.iloc[-val_size:],
            "최종"
        )

        # 8. 테스트 데이터 예측
        print("\n" + "=" * 50)
        print("📝 테스트 데이터 예측")
        print("=" * 50)

        all_predictions = []

        for test_name, test_df in test_data.items():
            print(f"\n→ {test_name} 예측 중...")

            # 특성 엔지니어링
            test_fe = self.model.prepare_features(test_df, is_train=False)

            # 마지막 날짜
            last_date = test_df['영업일자'].max()
            if not isinstance(last_date, pd.Timestamp):
                last_date = pd.to_datetime(last_date)

            # 7일간 예측
            for day_ahead in range(1, 8):
                pred_date = last_date + timedelta(days=day_ahead)

                # 각 메뉴에 대해 예측
                for menu in test_df['영업장명_메뉴명'].unique():
                    # 미래 특성 생성
                    future_features = self.create_future_features(
                        test_fe[test_fe['영업장명_메뉴명'] == menu].iloc[-1:],
                        pred_date,
                        feature_cols
                    )

                    # 예측 (메뉴별 모델 또는 전체 모델 사용)
                    if menu in self.menu_models:
                        # 메뉴별 모델 사용
                        pred_value = self.model.weighted_ensemble_predict(
                            self.menu_models[menu],
                            future_features[feature_cols].fillna(0),
                            self.model.scalers[menu]
                        )[0]
                    else:
                        # 전체 모델 사용
                        pred_value = self.model.weighted_ensemble_predict(
                            final_models,
                            future_features[feature_cols].fillna(0),
                            self.model.scalers["최종"]
                        )[0]

                    # 후처리: 메뉴 통계 기반 조정
                    if menu in self.model.processor.menu_stats:
                        stats = self.model.processor.menu_stats[menu]
                        # 극단값 제한
                        pred_value = np.clip(pred_value, 0, stats['max'] * 1.5)

                        # 주말/평일 조정
                        if pred_date.dayofweek >= 5:  # 주말
                            pred_value *= 1.2

                        # 공휴일 조정
                        if pred_date in self.model.processor.kr_holidays:
                            pred_value *= 1.3

                    all_predictions.append({
                        '영업일자': f"{test_name}+{day_ahead}일",
                        '영업장명_메뉴명': menu,
                        '매출수량': max(0, int(round(pred_value)))
                    })

        # 9. 제출 파일 생성
        self.create_submission(all_predictions)

        total_time = time.time() - total_start
        print("\n" + "=" * 70)
        print(f"✅ 파이프라인 완료! (총 소요시간: {total_time / 60:.1f}분)")
        print("=" * 70)

        return final_models

    def analyze_data(self, df):
        """데이터 분석"""
        print("\n📊 데이터 분석 중...")

        print("\n[매출수량 기본 통계]")
        print(df['매출수량'].describe())

        print("\n[업장별 매출 TOP 5]")
        store_sales = df.groupby('업장명')['매출수량'].agg(['sum', 'mean'])
        print(store_sales.sort_values('sum', ascending=False).head())

        print("\n[요일별 평균 매출]")
        weekday_sales = df.groupby('dayofweek')['매출수량'].mean()
        weekdays = ['월', '화', '수', '목', '금', '토', '일']
        for day, sales in weekday_sales.items():
            print(f"  {weekdays[day]}: {sales:.2f}")

    def print_model_performance(self, results):
        """모델 성능 출력"""
        print("\n🏆 모델 성능 비교 (sMAPE)")
        print("-" * 40)

        sorted_results = sorted(results.items(), key=lambda x: x[1].get('smape', 999))
        for name, info in sorted_results:
            if 'smape' in info:
                print(f"  {name:20s}: {info['smape']:.4f}")

        if sorted_results and 'smape' in sorted_results[0][1]:
            print("-" * 40)
            print(f"  🥇 최고 성능: {sorted_results[0][0]} ({sorted_results[0][1]['smape']:.4f})")

    def create_future_features(self, last_row, future_date, feature_cols):
        """미래 날짜의 특성 생성"""
        future_row = last_row.copy()

        # 날짜 업데이트
        future_row['date'] = future_date
        future_row['year'] = future_date.year
        future_row['month'] = future_date.month
        future_row['day'] = future_date.day
        future_row['dayofweek'] = future_date.dayofweek
        future_row['quarter'] = future_date.quarter
        future_row['weekofyear'] = future_date.isocalendar()[1]
        future_row['dayofyear'] = future_date.dayofyear
        future_row['is_weekend'] = 1 if future_date.dayofweek >= 5 else 0
        future_row['is_holiday'] = 1 if future_date in self.model.processor.kr_holidays else 0

        # 계절
        future_row['season'] = (
            1 if future_date.month in [3, 4, 5] else
            2 if future_date.month in [6, 7, 8] else
            3 if future_date.month in [9, 10, 11] else
            4
        )

        # Sin/Cos 업데이트
        future_row['month_sin'] = np.sin(2 * np.pi * future_date.month / 12)
        future_row['month_cos'] = np.cos(2 * np.pi * future_date.month / 12)
        future_row['day_sin'] = np.sin(2 * np.pi * future_date.day / 31)
        future_row['day_cos'] = np.cos(2 * np.pi * future_date.day / 31)
        future_row['dayofweek_sin'] = np.sin(2 * np.pi * future_date.dayofweek / 7)
        future_row['dayofweek_cos'] = np.cos(2 * np.pi * future_date.dayofweek / 7)

        return future_row

    def create_submission(self, predictions):
        """제출 파일 생성"""
        print("\n📋 제출 파일 생성 중...")

        pred_df = pd.DataFrame(predictions)
        sample = pd.read_csv('./sample_submission.csv')

        # 제출 형식에 맞게 변환
        submission = sample.copy()
        submission.iloc[:, 1:] = 0

        # 예측값 채우기
        for _, row in pred_df.iterrows():
            date = row['영업일자']
            menu = row['영업장명_메뉴명']
            value = row['매출수량']

            if date in submission['영업일자'].values and menu in submission.columns:
                submission.loc[submission['영업일자'] == date, menu] = value

        # 저장
        submission.to_csv('advanced_submission_v2.csv', index=False, encoding='utf-8-sig')
        print(f"✓ 제출 파일 저장 완료: advanced_submission_v2.csv")


# ========================================
# 6. 실행
# ========================================
if __name__ == "__main__":
    pipeline = Pipeline()
    final_models = pipeline.run()

    print("\n🎉 모든 작업이 완료되었습니다!")
    print("📁 생성된 파일:")
    print("  - advanced_submission_v2.csv: 제출용 예측 파일")