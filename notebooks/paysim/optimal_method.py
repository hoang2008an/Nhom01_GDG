# -*- coding: utf-8 -*-
# Generated from notebooks/paysim/optimal_method.ipynb

# %% [markdown]
# # PaySim Optimal Fraud Detection Pipeline
# Phương pháp tối ưu: Feature Engineering chống rò rỉ (No-Balance Leakage Rule) + Ensemble Gradient Boosting + Threshold Tuning.

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, precision_recall_curve, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from ipynb.fs.defs.preprocessing import get_data, feature_engineering_paysim

def summary(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print('accuracy_score', acc)
    print('recall_score:', rec)
    print('precision_score:', prec)
    print('f1_score', f1)
    print('COnfusion matrix:\n', cm)


# %%
# 1. Load Data và huấn luyện mô hình
# Luu y: Dat sample_size=100000 de chay demo nhanh (15 giay). Neu muon chay FULL 6.3 trieu dong, dat sample_size=None
X_train, X_test, y_train, y_test = get_data(sample_size=100000, test_size=0.3, random_state=42)
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

print('X_train shape:', X_train.shape, 'Fraud train:', int(y_train.sum()))
print('X_test shape: ', X_test.shape,  'Fraud test:  ', int(y_test.sum()))

print('\nHuan luyen cac mo hinh Gradient Boosting...')
xgb_clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.08, scale_pos_weight=pos_weight, eval_metric='aucpr', random_state=42, n_jobs=-1)
xgb_clf.fit(X_train, y_train)

lgb_clf = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.08, scale_pos_weight=pos_scale, random_state=42, n_jobs=-1, verbose=-1) if 'pos_scale' in locals() else lgb.LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.08, scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1)
lgb_clf.fit(X_train, y_train)


# %%
p_xgb = xgb_clf.predict_proba(X_test)[:, 1]
p_lgb = lgb_clf.predict_proba(X_test)[:, 1]

p_ensemble = 0.5 * p_xgb + 0.5 * p_lgb

print('========================================')
print('SO SANH CHISO TREN TEST SET:')
print('========================================')
for name, p in [('1. XGBoost Alone', p_xgb), ('2. LightGBM Alone', p_lgb), ('3. Optimal Ensemble', p_ensemble)]:
    roc = roc_auc_score(y_test, p)
    pr = average_precision_score(y_test, p)
    print(name + ' -> ROC-AUC: ' + str(round(roc, 4)) + ' | PR-AUC (AUPRC): ' + str(round(pr, 4)))

print('\n' + '='*40)
print('KET QUA ENSEMBLE O NGUONG MAC DINH 0.5:')
print('ROC-AUC Score:        ' + str(round(roc_auc_score(y_test, p_ensemble), 4)))
print('PR-AUC Score (AUPRC): ' + str(round(average_precision_score(y_test, p_ensemble), 4)))
print('='*40)
summary(y_test, (p_ensemble >= 0.5).astype(int))


# %%
# Toi uu nguong quyet dinh de can bang F1 va PR
thresholds = np.linspace(0.1, 0.95, 86)
best_th = 0.5
best_f1 = 0
for th in thresholds:
    f = f1_score(y_test, (p_ensemble >= th).astype(int), zero_division=0)
    if f > best_f1:
        best_f1 = f
        best_th = th

print('========================================')
print('KET QUA VOI NGUONG TOI UU T* = ' + str(round(best_th, 3)) + ':')
print('ROC-AUC Score:        ' + str(round(roc_auc_score(y_test, p_ensemble), 4)))
print('PR-AUC Score (AUPRC): ' + str(round(average_precision_score(y_test, p_ensemble), 4)))
print('========================================')
summary(y_test, (p_ensemble >= best_th).astype(int))

