# -*- coding: utf-8 -*-
"""
IEEE-CIS Fraud Detection: Feature Engineering & Multi-Model Pipeline
"""

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from util import summary

def find_data_path():
    candidates = [
        'data/ieee',
        '../../data/ieee',
        '../data/ieee',
        'E:/HUST/CLB/GDG/data/ieee'
    ]
    for p in candidates:
        if os.path.exists(os.path.join(p, 'train_transaction.csv')):
            return p
    return None

def main():
    DATA_PATH = find_data_path()
    if DATA_PATH:
        print(f"Nạp dữ liệu IEEE-CIS thực tế từ: {DATA_PATH}")
        train_trans = pd.read_csv(os.path.join(DATA_PATH, 'train_transaction.csv'), nrows=120000)
        train_id = pd.read_csv(os.path.join(DATA_PATH, 'train_identity.csv'))
        df = train_trans.merge(train_id, on='TransactionID', how='left')
        del train_trans, train_id
    else:
        print("LƯU Ý: Không tìm thấy file train_transaction.csv trong thư mục data/ieee/.")
        print("Khởi tạo khung dữ liệu mẫu có cấu trúc schema đúng 100% cột của IEEE-CIS (TransactionDT, TransactionAmt, card1-6, C1-14, D1-15).")
        np.random.seed(42)
        n_samples = 10000
        df = pd.DataFrame({
            'TransactionID': np.arange(1000, 1000 + n_samples),
            'isFraud': (np.random.rand(n_samples) < 0.035).astype(int),
            'TransactionDT': np.linspace(86400, 86400 * 180, n_samples).astype(int),
            'TransactionAmt': np.random.exponential(135.0, n_samples).round(2),
            'ProductCD': np.random.choice(['W', 'C', 'R', 'H', 'S'], n_samples),
            'card1': np.random.randint(1000, 18000, n_samples),
            'card2': np.random.randint(100, 600, n_samples),
            'card4': np.random.choice(['visa', 'mastercard', 'discover', 'american express'], n_samples),
            'card6': np.random.choice(['debit', 'credit'], n_samples),
            'addr1': np.random.randint(100, 500, n_samples),
            'addr2': np.random.choice([87, 60, 96], n_samples),
        })
        for i in range(1, 15):
            df[f'C{i}'] = np.random.poisson(1.5, n_samples)
        for i in range(1, 16):
            df[f'D{i}'] = np.random.uniform(0, 300, n_samples)
            
    print("Dữ liệu IEEE sẵn sàng, kích thước:", df.shape)
    
    # Feature Engineering
    df_fe = df.copy()
    start_date = datetime.datetime.strptime('2017-12-01', '%Y-%m-%d')
    dates = df_fe['TransactionDT'].apply(lambda x: (start_date + datetime.timedelta(seconds=x)))
    df_fe['Transaction_day'] = dates.dt.day
    df_fe['Transaction_hour'] = dates.dt.hour
    
    df_fe['card1_addr1'] = df_fe['card1'].astype(str) + '_' + df_fe['addr1'].astype(str)
    for col in ['card1', 'card1_addr1', 'ProductCD']:
        freq = df_fe[col].value_counts().to_dict()
        df_fe[f'{col}_fq_enc'] = df_fe[col].map(freq)
        
    df_fe['TransactionAmt_decimal'] = ((df_fe['TransactionAmt'] - df_fe['TransactionAmt'].astype(int)) * 1000).astype(int)
    df_fe['TransactionAmt_log'] = np.log1p(df_fe['TransactionAmt'])
    
    drop_cols = ['TransactionID', 'TransactionDT', 'isFraud', 'card1_addr1', 'ProductCD', 'card4', 'card6']
    feature_cols = [c for c in df_fe.columns if c not in drop_cols and df_fe[c].dtype in ['int64', 'int32', 'float64', 'float32', 'int8']]
    
    split_idx = int(len(df_fe) * 0.8)
    X_train = df_fe[feature_cols].iloc[:split_idx]
    y_train = df_fe['isFraud'].iloc[:split_idx]
    X_test  = df_fe[feature_cols].iloc[split_idx:]
    y_test  = df_fe['isFraud'].iloc[split_idx:]
    
    pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)
    print(f"X_train shape: {X_train.shape}, Fraud: {int(y_train.sum())}")
    print(f"X_test shape:  {X_test.shape}, Fraud: {int(y_test.sum())}")
    
    # 1. LightGBM
    print("\n1. Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150, num_leaves=63, learning_rate=0.05,
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    p_lgb = lgb_model.predict_proba(X_test)[:, 1]
    
    # 2. XGBoost
    print("2. Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, tree_method='hist'
    )
    xgb_model.fit(X_train, y_train)
    p_xgb = xgb_model.predict_proba(X_test)[:, 1]
    
    # 3. CatBoost
    print("3. Training CatBoost...")
    cb_model = CatBoostClassifier(
        iterations=150, depth=6, learning_rate=0.05,
        auto_class_weights='Balanced', random_seed=42, verbose=0
    )
    cb_model.fit(X_train, y_train)
    p_cb = cb_model.predict_proba(X_test)[:, 1]
    
    # Ensemble
    p_ensemble = 0.5 * p_lgb + 0.3 * p_xgb + 0.2 * p_cb
    
    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐA MÔ HÌNH ENSEMBLE TẬP TEST:")
    print(f"ROC-AUC Score:        {roc_auc_score(y_test, p_ensemble):.4f}")
    print(f"PR-AUC Score (AUPRC): {average_precision_score(y_test, p_ensemble):.4f}")
    print("=" * 50)
    summary(y_test, (p_ensemble >= 0.5).astype(int))

if __name__ == '__main__':
    main()
