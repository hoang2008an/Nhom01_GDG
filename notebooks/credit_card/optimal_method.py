# -*- coding: utf-8 -*-
"""
Credit Card Fraud Detection: Optimal Pipeline
Phương pháp: Tri-Model Soft Voting Ensemble (XGBoost + CatBoost + LightGBM) + Cost-Sensitive Threshold Tuning
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from util import get_data, summary

def main():
    print("1. Nạp dữ liệu Credit Card Fraud...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(use_val=True, test_size=0.15, val_size=0.15)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Fraud count: train={int(y_train.sum())}, val={int(y_val.sum())}, test={int(y_test.sum())}")
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() / 10
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 1. XGBoost
    print("\n1/3. Training Tuned XGBoost...")
    t0 = time.time()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.4,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    p_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    p_xgb_val = xgb_clf.predict_proba(X_val)[:, 1]
    print(f"XGBoost xong ({time.time()-t0:.1f}s) -> ROC-AUC: {roc_auc_score(y_test, p_xgb):.4f} | PR-AUC: {average_precision_score(y_test, p_xgb):.4f}")
    
    # 2. CatBoost
    print("\n2/3. Training CatBoost...")
    t0 = time.time()
    cb_clf = cb.CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.03,
        auto_class_weights='Balanced',
        eval_metric='PRAUC',
        random_seed=42,
        verbose=0
    )
    cb_clf.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    p_cb = cb_clf.predict_proba(X_test)[:, 1]
    p_cb_val = cb_clf.predict_proba(X_val)[:, 1]
    print(f"CatBoost xong ({time.time()-t0:.1f}s) -> ROC-AUC: {roc_auc_score(y_test, p_cb):.4f} | PR-AUC: {average_precision_score(y_test, p_cb):.4f}")
    
    # 3. LightGBM
    print("\n3/3. Training LightGBM...")
    t0 = time.time()
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=600,
        num_leaves=31,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgb = lgb_clf.predict_proba(X_test)[:, 1]
    p_lgb_val = lgb_clf.predict_proba(X_val)[:, 1]
    print(f"LightGBM xong ({time.time()-t0:.1f}s) -> ROC-AUC: {roc_auc_score(y_test, p_lgb):.4f} | PR-AUC: {average_precision_score(y_test, p_lgb):.4f}")
    
    # Ensemble
    p_ensemble_val = 0.5 * p_xgb_val + 0.3 * p_cb_val + 0.2 * p_lgb_val
    p_ensemble = 0.5 * p_xgb + 0.3 * p_cb + 0.2 * p_lgb
    
    print("\n" + "=" * 50)
    print("KẾT QUẢ ĐA MÔ HÌNH ENSEMBLE TRÊN TẬP TEST:")
    print(f"ROC-AUC Score:        {roc_auc_score(y_test, p_ensemble):.4f}")
    print(f"PR-AUC Score (AUPRC): {average_precision_score(y_test, p_ensemble):.4f}")
    print("=" * 50)
    summary(y_test, (p_ensemble >= 0.5).astype(int))
    
    # Cost-Sensitive Threshold Tuning
    thresholds = np.linspace(0.1, 0.9, 81)
    best_th, best_val_f1 = 0.5, 0.0
    for th in thresholds:
        f1 = f1_score(y_val, (p_ensemble_val >= th).astype(int), zero_division=0)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_th = th
            
    print(f"\nKẾT QUẢ TẠI NGƯỠNG TỐI ƯU T* = {best_th:.2f}:")
    print(f"ROC-AUC Score:        {roc_auc_score(y_test, p_ensemble):.4f}")
    print(f"PR-AUC Score (AUPRC): {average_precision_score(y_test, p_ensemble):.4f}")
    print("=" * 50)
    summary(y_test, (p_ensemble >= best_th).astype(int))

if __name__ == '__main__':
    main()
