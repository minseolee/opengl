#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
곤지암 리조트 식음업장 수요예측 고급 ML 모델 v2.1
- Train 기반 통계만 Test에 적용 (데이터 누설 방지)
- 카테고리 안정 인코딩(OrdinalEncoder, unknown 처리)
- 미래특성 생성 로직과 학습 특성 로직 완전 일치
- sMAPE 기반 앙상블 가중치 계산 안정화
- feature_cols 스냅샷 고정으로 컬럼 정합성 보장
"""

# ========================================
# 1. 라이브러리 임포트
# ========================================
import os
import re
import glob
import time
import pickle
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 시각화 (필요 시 사용)
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 머신러닝 도구
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.validation import check_is_fitted

# 부스팅
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# 공휴일
try:
    import holidays
    KR_HOLIDAYS = holidays.KR(years=range(2023, 2026))
except Exception:
    KR_HOLIDAYS = set()  # 폴백: 비어있는 공휴일 집합


# ========================================
# 2. 평가 메트릭
# ========================================
def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)
    diff[denom == 0] = 0.0
    return 100 * np.mean(diff)


# ========================================
# 3. 설정
# ========================================
@dataclass
class Config:
    train_path: str = './train/train.csv'
    test_dir: str = './test/'
    sample_path: str = './sample_submission.csv'
    out_csv: str = 'advanced_submission_v2_1.csv'

    # 분할/학습
    val_hold_days: int = 60      # 마지막 N일을 검증으로 홀드
    min_val_rows: int = 100      # 검증 최소 샘플 수
    global_val_ratio: float = 0.85

    # 상위 메뉴 개별 모델
    top_menu_min_count: int = 100
    top_menu_n: int = 20
    per_menu_min_rows: int = 200
    per_menu_min_val_rows: int = 10

    # 모델 공통
    random_state: int = 42
    n_jobs: int = -1

    # 부스팅 라운드
    n_estimators_big: int = 2000

CFG = Config()


# ========================================
# 4. 데이터 처리 클래스
# ========================================
class DataProcessor:
    def __init__(self):
        self.kr_holidays = KR_HOLIDAYS
        self.menu_stats: Dict[str, Dict[str, float]] = {}
        self.store_stats: Optional[pd.DataFrame] = None
        self.ord_enc: Optional[OrdinalEncoder] = None

    def load_data(self, train_path=CFG.train_path, test_dir=CFG.test_dir):
        print("=" * 50)
        print("📊 데이터 로드 중...")

        train = pd.read_csv(train_path)
        if train['영업일자'].dtype == 'object':
            train['영업일자'] = pd.to_datetime(train['영업일자'])
        print(f"✓ Train: {train.shape}")

        test_files = sorted(glob.glob(os.path.join(test_dir, 'TEST_*.csv')))
        test_data = {}
        for fp in test_files:
            name = os.path.basename(fp).replace('.csv', '')
            df = pd.read_csv(fp)
            if df['영업일자'].dtype == 'object':
                df['영업일자'] = pd.to_datetime(df['영업일자'])
            test_data[name] = df
            print(f"✓ {name}: {df.shape}")

        sample = pd.read_csv(CFG.sample_path)
        print(f"✓ Sample: {sample.shape}")
        self.sample = sample

        self.train = train
        self.test_data = test_data
        return train, test_data

    # ---------- 공통 특성 ----------
    def extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # date 파생
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['영업일자'])
        else:
            df['date'] = pd.to_datetime(df['date'])

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
        df['dayofyear'] = df['date'].dt.dayofyear

        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_holiday'] = df['date'].apply(lambda x: x in self.kr_holidays).astype(int)

        df['day_before_holiday'] = df['date'].apply(lambda x: (x + timedelta(days=1)) in self.kr_holidays).astype(int)
        df['day_after_holiday'] = df['date'].apply(lambda x: (x - timedelta(days=1)) in self.kr_holidays).astype(int)
        df['is_long_weekend'] = ((df['is_holiday'] == 1) |
                                 (df['day_before_holiday'] == 1) |
                                 (df['day_after_holiday'] == 1)).astype(int)

        # 계절/월초중말/주차
        df['season'] = df['month'].apply(lambda m: 1 if m in [3,4,5] else 2 if m in [6,7,8] else 3 if m in [9,10,11] else 4)
        df['month_period'] = df['day'].apply(lambda d: 1 if d<=10 else 2 if d<=20 else 3)
        df['week_of_month'] = ((df['day'] - 1) // 7 + 1).astype(int)

        # 주기성
        df['month_sin'] = np.sin(2*np.pi*df['month']/12)
        df['month_cos'] = np.cos(2*np.pi*df['month']/12)
        df['day_sin'] = np.sin(2*np.pi*df['day']/31)
        df['day_cos'] = np.cos(2*np.pi*df['day']/31)
        df['dayofweek_sin'] = np.sin(2*np.pi*df['dayofweek']/7)
        df['dayofweek_cos'] = np.cos(2*np.pi*df['dayofweek']/7)
        df['weekofyear_sin'] = np.sin(2*np.pi*df['weekofyear']/52)
        df['weekofyear_cos'] = np.cos(2*np.pi*df['weekofyear']/52)

        return df

    def extract_menu_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['업장명'] = df['영업장명_메뉴명'].apply(lambda x: str(x).split('_')[0])
        df['메뉴명'] = df['영업장명_메뉴명'].apply(lambda x: '_'.join(str(x).split('_')[1:]))

        def contains_any(x, words): return int(any(w in str(x) for w in words))
        df['is_단체'] = df['메뉴명'].str.contains('단체', na=False).astype(int)
        df['is_정식'] = df['메뉴명'].str.contains('정식', na=False).astype(int)
        df['is_후식'] = df['메뉴명'].str.contains('후식', na=False).astype(int)
        df['is_브런치'] = df['메뉴명'].str.contains('브런치', na=False).astype(int)
        df['is_주류'] = df['메뉴명'].apply(lambda x: contains_any(x, ['맥주','소주','막걸리','와인','주류','카스','테라','참이슬','처음처럼']))
        df['is_음료'] = df['메뉴명'].apply(lambda x: contains_any(x, ['콜라','사이다','스프라이트','커피','아메리카노','라떼','에이드','차']))
        df['is_면류'] = df['메뉴명'].apply(lambda x: contains_any(x, ['면','우동','파스타','스파게티','짜장','짬뽕']))
        df['is_고기'] = df['메뉴명'].apply(lambda x: contains_any(x, ['고기','삼겹','갈비','스테이크','불고기','목살']))
        df['메뉴명_길이'] = df['메뉴명'].astype(str).str.len()
        return df

    # ---------- Train 기반 통계 ----------
    def calculate_train_statistics(self, train_df: pd.DataFrame):
        print("📊 Train 기반 메뉴/업장 통계 계산 중...")
        self.menu_stats = {}
        for menu, grp in tqdm(train_df.groupby('영업장명_메뉴명'), desc="메뉴 통계"):
            q = grp['매출수량']
            self.menu_stats[menu] = {
                'mean': q.mean(),
                'std': q.std(),
                'median': q.median(),
                'min': q.min(),
                'max': q.max(),
                'q25': q.quantile(0.25),
                'q75': q.quantile(0.75),
                'zero_ratio': (q == 0).mean(),
                'positive_ratio': (q > 0).mean(),
            }

        store_stats = train_df.groupby('업장명')['매출수량'].agg(['mean','std']).reset_index()
        store_stats.columns = ['업장명','store_mean','store_std']
        self.store_stats = store_stats

    def add_train_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # 메뉴 통계
        for stat in ['mean','std','median','zero_ratio','positive_ratio']:
            df[f'menu_{stat}'] = df['영업장명_메뉴명'].map(
                lambda x: self.menu_stats.get(x, {}).get(stat, 0.0)
            )
        # 업장 통계 (Train에서 계산된 프레임을 머지)
        if self.store_stats is not None:
            df = df.merge(self.store_stats, on='업장명', how='left')
        else:
            df['store_mean'] = 0.0
            df['store_std'] = 0.0
        return df

    # ---------- 카테고리 인코딩 ----------
    def fit_encoders(self, df: pd.DataFrame):
        # 안정적 unknown 처리용 OrdinalEncoder
        self.ord_enc = OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            dtype=np.int64,
        )
        cat_cols = ['업장명','메뉴명','영업장명_메뉴명']
        self.ord_enc.fit(df[cat_cols])

    def transform_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cat_cols = ['업장명','메뉴명','영업장명_메뉴명']
        enc = self.ord_enc.transform(df[cat_cols])
        df[[f'{c}_encoded' for c in cat_cols]] = enc
        return df


# ========================================
# 5. 모델 앙상블
# ========================================
class AdvancedDemandForecastModel:
    def __init__(self):
        self.models = {}
        self.scalers: Dict[str, RobustScaler] = {}
        self.processor = DataProcessor()
        self.best_params = {}
        self.feature_cols: List[str] = []

    def _create_model_ensemble(self):
        return {
            'xgboost': xgb.XGBRegressor(
                n_estimators=CFG.n_estimators_big,
                max_depth=12,
                learning_rate=0.005,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=CFG.random_state,
                n_jobs=CFG.n_jobs,
                objective='reg:squarederror',
                early_stopping_rounds=100
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=CFG.n_estimators_big,
                max_depth=12,
                learning_rate=0.005,
                num_leaves=50,
                min_child_samples=20,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=CFG.random_state,
                n_jobs=CFG.n_jobs,
                verbose=-1,
                force_col_wise=True
            ),
            'catboost': CatBoostRegressor(
                iterations=CFG.n_estimators_big,
                depth=10,
                learning_rate=0.005,
                l2_leaf_reg=3,
                border_count=128,
                random_seed=CFG.random_state,
                verbose=False,
                early_stopping_rounds=100,
                task_type='CPU'
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=True,
                oob_score=False,
                random_state=CFG.random_state,
                n_jobs=CFG.n_jobs
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                bootstrap=False,
                random_state=CFG.random_state,
                n_jobs=CFG.n_jobs
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.01,
                subsample=0.85,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=CFG.random_state
            ),
        }

    # ---------- 파생/인코딩 일괄 ----------
    def _prepare_features(self, df: pd.DataFrame, is_train=True) -> pd.DataFrame:
        df = self.processor.extract_datetime_features(df)
        df = self.processor.extract_menu_features(df)

        if is_train:
            self.processor.calculate_train_statistics(df)
            self.processor.fit_encoders(df)

        df = self.processor.add_train_statistics(df)
        df = self.processor.transform_encoders(df)
        return df

    def _fit_models(self, X_train, y_train, X_val, y_val, key_for_scaler: str):
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X_train)
        X_va = scaler.transform(X_val)
        self.scalers[key_for_scaler] = scaler

        models = self._create_model_ensemble()
        results = {}
        start = time.time()

        for name, model in models.items():
            t0 = time.time()
            print(f"  → {name} 학습 중...", end=" ")

            try:
                if name in ['xgboost','lightgbm','catboost']:
                    if name == 'xgboost':
                        model.fit(X_tr, y_train, eval_set=[(X_va, y_val)], verbose=False)
                    elif name == 'lightgbm':
                        model.fit(X_tr, y_train,
                                  eval_set=[(X_va, y_val)],
                                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
                    else:  # catboost
                        model.fit(X_tr, y_train, eval_set=(X_va, y_val), verbose=False)
                else:
                    model.fit(X_tr, y_train)

                pred = model.predict(X_va)
                pred = np.maximum(pred, 0)
                score = smape(y_val, pred)

                results[name] = {'model': model, 'smape': float(score)}
                print(f"완료 ({time.time()-t0:.1f}초, sMAPE: {score:.2f})")
            except Exception as e:
                print(f"실패 ({e})")

        print(f"  총 학습 시간: {time.time()-start:.1f}초")
        return results

    @staticmethod
    def _calc_weights_by_smape(results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        # 낮을수록 좋은 sMAPE → 1/(s+1) 가중치
        smapes = {k: v['smape'] for k, v in results.items() if 'smape' in v and np.isfinite(v['smape'])}
        if not smapes:
            return {k: 1.0/len(results) for k in results.keys()}

        inv = {k: 1.0/(s+1.0) for k, s in smapes.items()}
        ssum = sum(inv.values())
        if ssum == 0:
            return {k: 1.0/len(results) for k in results.keys()}
        return {k: v/ssum for k, v in inv.items()}

    def _predict_ensemble(self, models_dict, X, scaler, weights: Optional[Dict[str, float]]=None):
        Xs = scaler.transform(X)
        if weights is None:
            weights = self._calc_weights_by_smape(models_dict)

        preds = []
        for name, info in models_dict.items():
            model = info.get('model')
            if model is None:
                continue
            try:
                p = model.predict(Xs)
                p = np.maximum(p, 0)
                preds.append(p * weights.get(name, 0.0))
            except Exception:
                continue

        if len(preds) == 0:
            return np.zeros(X.shape[0])
        return np.sum(preds, axis=0)

    # ---------- 공개 인터페이스 ----------
    def fit_global_and_permenu(self, train_df: pd.DataFrame, feature_cols: List[str]):
        self.feature_cols = feature_cols[:]  # 스냅샷 고정

        # 시계열 분할
        train_df = train_df.sort_values('date')
        split_date = train_df['date'].max() - pd.Timedelta(days=CFG.val_hold_days)

        tr = train_df[train_df['date'] < split_date]
        va = train_df[train_df['date'] >= split_date]
        if len(va) < CFG.min_val_rows:
            split_idx = int(len(train_df) * CFG.global_val_ratio)
            tr = train_df.iloc[:split_idx]
            va = train_df.iloc[split_idx:]

        X_tr = tr[self.feature_cols].fillna(0)
        y_tr = tr['매출수량']
        X_va = va[self.feature_cols].fillna(0)
        y_va = va['매출수량']

        print("\n" + "="*50)
        print("🎯 전체 데이터 통합 모델 학습")
        print("="*50)
        global_models = self._fit_models(X_tr, y_tr, X_va, y_va, key_for_scaler="__GLOBAL__")
        self._print_model_performance(global_models)

        # 상위 메뉴 선별
        print("\n" + "="*50)
        print("🎯 주요 메뉴별 개별 모델 학습")
        print("="*50)
        sale_stat = train_df.groupby('영업장명_메뉴명')['매출수량'].agg(['sum','count'])
        top = sale_stat[sale_stat['count'] > CFG.top_menu_min_count].sort_values('sum', ascending=False).head(CFG.top_menu_n).index
        print(f"상위 {len(top)}개 메뉴에 대해 개별 모델 학습")

        self.menu_models: Dict[str, Dict] = {}
        self.menu_scalers: Dict[str, RobustScaler] = {}

        for menu in tqdm(top, desc="메뉴별 모델"):
            md = train_df[train_df['영업장명_메뉴명'] == menu].sort_values('date')
            if len(md) < CFG.per_menu_min_rows:
                continue
            trm = md[md['date'] < split_date]
            vam = md[md['date'] >= split_date]
            if len(vam) < CFG.per_menu_min_val_rows:
                continue

            X_trm = trm[self.feature_cols].fillna(0)
            y_trm = trm['매출수량']
            X_vam = vam[self.feature_cols].fillna(0)
            y_vam = vam['매출수량']

            res = self._fit_models(X_trm, y_trm, X_vam, y_vam, key_for_scaler=f"MENU::{menu}")
            self.menu_models[menu] = res
            self.menu_scalers[menu] = self.scalers[f"MENU::{menu}"]

        # 최종 (홀드아웃 10%)
        print("\n" + "="*50)
        print("🎯 최종 모델 학습 (전체 데이터)")
        print("="*50)
        X_full = train_df[self.feature_cols].fillna(0)
        y_full = train_df['매출수량']
        val_size = max(int(len(X_full)*0.1), 1000) if len(X_full) > 1000 else max(int(len(X_full)*0.1), 50)

        final = self._fit_models(
            X_full[:-val_size], y_full.iloc[:-val_size],
            X_full[-val_size:],  y_full.iloc[-val_size:],
            key_for_scaler="__FINAL__"
        )
        self.final_models = final
        return global_models, final

    def predict_for_tests(self, test_dict: Dict[str, pd.DataFrame], feature_cols: List[str]) -> List[Dict]:
        preds_out = []
        for test_name, test_df in test_dict.items():
            print(f"\n→ {test_name} 예측 중...")
            test_fe = self._prepare_features(test_df, is_train=False)

            last_date = pd.to_datetime(test_df['영업일자'].max())
            menus = test_df['영업장명_메뉴명'].unique().tolist()

            for day_ahead in range(1, 8):
                pred_date = last_date + timedelta(days=day_ahead)

                for menu in menus:
                    last_row = test_fe[test_fe['영업장명_메뉴명'] == menu].tail(1).copy()
                    future_row = self._create_future_row(last_row, pred_date)
                    # 컬럼 정합성
                    for c in feature_cols:
                        if c not in future_row.columns:
                            future_row[c] = 0
                    future_X = future_row[feature_cols].fillna(0)

                    # 메뉴 전용 또는 최종 사용
                    if hasattr(self, 'menu_models') and (menu in self.menu_models):
                        pred = self._predict_ensemble(self.menu_models[menu], future_X, self.menu_scalers[menu])
                    else:
                        pred = self._predict_ensemble(self.final_models, future_X, self.scalers["__FINAL__"])

                    pred_value = float(pred[0]) if np.ndim(pred) > 0 else float(pred)

                    # 후처리: Train 통계 기반
                    ms = self.processor.menu_stats.get(menu, {})
                    mmax = ms.get('max', np.inf)
                    pred_value = np.clip(pred_value, 0, (mmax if np.isfinite(mmax) else pred_value) * 1.5)

                    if pred_date.dayofweek >= 5:
                        pred_value *= 1.2
                    if pred_date in self.processor.kr_holidays:
                        pred_value *= 1.3

                    preds_out.append({
                        '영업일자': f"{test_name}+{day_ahead}일",
                        '영업장명_메뉴명': menu,
                        '매출수량': int(round(max(0.0, pred_value)))
                    })
        return preds_out

    def _create_future_row(self, last_row: pd.DataFrame, future_date: pd.Timestamp) -> pd.DataFrame:
        # last_row의 카테고리/메뉴 특성은 유지, 날짜계열 특성만 재계산
        row = last_row.copy()
        row['영업일자'] = future_date
        row['date'] = future_date

        # 날짜 특성 재계산: 동일 함수 사용
        row = self.processor.extract_datetime_features(row)

        # 주기적/구간 특성 등은 extract_datetime_features가 모두 계산
        # 메뉴 관련 통계는 그대로 유지 (Train 기반)
        # 인코딩된 카테고리 컬럼은 last_row에서 유지됨

        return row

    @staticmethod
    def _print_model_performance(results: Dict[str, Dict[str, float]]):
        print("\n🏆 모델 성능 비교 (sMAPE)")
        print("-"*40)
        sorted_items = sorted(
            [(k, v) for k, v in results.items() if 'smape' in v],
            key=lambda x: x[1]['smape']
        )
        for name, info in sorted_items:
            print(f"  {name:20s}: {info['smape']:.4f}")
        if sorted_items:
            best = sorted_items[0]
            print("-"*40)
            print(f"  🥇 최고 성능: {best[0]} ({best[1]['smape']:.4f})")


# ========================================
# 6. 파이프라인
# ========================================
class Pipeline:
    def __init__(self):
        self.model = AdvancedDemandForecastModel()

    def run(self):
        print("=" * 70)
        print("🚀 곤지암 리조트 식음업장 수요예측 모델 파이프라인 시작 (v2.1)")
        print("=" * 70)
        t0 = time.time()

        # 1) 로드
        train, test_dict = self.model.processor.load_data()

        # 2) 특성 엔지니어링(Train)
        print("\n📈 Train 특성 엔지니어링...")
        train_fe = self.model._prepare_features(train, is_train=True)

        # 3) 분석(간단 로그)
        self._simple_eda(train_fe)

        # 4) 학습에 사용할 특성 고정
        exclude = ['영업일자','영업장명_메뉴명','매출수량','date','업장명','메뉴명']
        feature_cols = [c for c in train_fe.columns if c not in exclude]
        self.model.feature_cols = feature_cols[:]
        print(f"\n📊 사용할 특성 수: {len(feature_cols)}")

        # 5) 글로벌 & 메뉴별 학습
        global_models, final_models = self.model.fit_global_and_permenu(train_fe, feature_cols)

        # 6) 테스트 예측
        print("\n" + "="*50)
        print("📝 테스트 데이터 예측")
        print("="*50)
        preds = self.model.predict_for_tests(test_dict, feature_cols)

        # 7) 제출 파일 생성
        self._create_submission(preds)

        print("\n" + "="*70)
        print(f"✅ 파이프라인 완료! (총 소요: {(time.time()-t0)/60:.1f}분)")
        print("="*70)
        return final_models

    def _simple_eda(self, df: pd.DataFrame):
        print("\n📊 데이터 요약")
        print("[매출수량 통계]")
        print(df['매출수량'].describe())

        print("\n[업장별 매출 TOP 5]")
        st = df.groupby('업장명')['매출수량'].agg(['sum','mean']).sort_values('sum', ascending=False).head(5)
        print(st)

        print("\n[요일별 평균 매출]")
        wk = df.groupby('dayofweek')['매출수량'].mean()
        weekdays = ['월','화','수','목','금','토','일']
        for d, v in wk.items():
            print(f"  {weekdays[d]}: {v:.2f}")

    def _create_submission(self, pred_list: List[Dict]):
        print("\n📋 제출 파일 생성 중...")
        pred_df = pd.DataFrame(pred_list)
        sample = self.model.processor.sample.copy()
        sample.iloc[:, 1:] = 0

        # 채우기
        # 성능: 행 인덱스 맵/열 맵을 만들어 벡터화할 수도 있으나, 안전하게 루프 유지
        idx_map = {d: i for i, d in enumerate(sample['영업일자'].values)}
        col_set = set(sample.columns[1:])

        for _, r in pred_df.iterrows():
            d = r['영업일자']
            m = r['영업장명_메뉴명']
            v = r['매출수량']
            if (d in idx_map) and (m in col_set):
                sample.at[idx_map[d], m] = v

        sample.to_csv(CFG.out_csv, index=False, encoding='utf-8-sig')
        print(f"✓ 제출 파일 저장 완료: {CFG.out_csv}")


# ========================================
# 7. 실행
# ========================================
if __name__ == "__main__":
    pipeline = Pipeline()
    final_models = pipeline.run()

    print("\n🎉 모든 작업이 완료되었습니다!")
    print("📁 생성된 파일:")
    print(f"  - {CFG.out_csv}: 제출용 예측 파일")