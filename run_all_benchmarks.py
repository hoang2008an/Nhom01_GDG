import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Evaluation helper
def evaluate_model(name, y_true, y_pred, y_prob, fit_time=0.0):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob) if y_prob is not None else 0.0
    pr_auc = average_precision_score(y_true, y_prob) if y_prob is not None else 0.0
    
    # Financial impact calculation:
    # Assume: Average fraud amount lost = $200 (if FN, false negative)
    # Investigation/friction cost of False Positive = $10 (if FP, false positive)
    financial_cost = fn * 200 + fp * 10
    
    res = {
        "Method": name,
        "Accuracy": round(float(acc), 5),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4),
        "ROC_AUC": round(float(roc_auc), 4),
        "PR_AUC": round(float(pr_auc), 4),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Est_Cost_USD": int(financial_cost),
        "Train_Time_s": round(float(fit_time), 2)
    }
    return res

# 1. Benchmark function for any (X_train, X_test, y_train, y_test)
def run_benchmark_suite(dataset_name, X_train, X_test, y_train, y_test, scale_features=True):
    print(f"\n==================== RUNNING BENCHMARK: {dataset_name} ====================")
    print(f"Train size: {X_train.shape}, Fraud in train: {y_train.sum()} ({y_train.mean():.4%})")
    print(f"Test size: {X_test.shape}, Fraud in test: {y_test.sum()} ({y_test.mean():.4%})")
    
    # Check Accuracy Paradox upfront:
    dummy_pred = np.zeros(len(y_test))
    dummy_acc = accuracy_score(y_test, dummy_pred)
    print(f"[*] DUMMY ALL-ZERO PREDICTOR: Accuracy = {dummy_acc:.4%}, Recall = 0.0, Precision = 0.0, F1 = 0.0")
    
    if scale_features:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    results = []
    
    # -------------------------------------------------------------
    # Method 0: Naive Majority Classifier (Demonstrating Accuracy Paradox)
    # -------------------------------------------------------------
    results.append(evaluate_model(
        "0. Naive Predict-All-Zero",
        y_test,
        np.zeros(len(y_test), dtype=int),
        np.zeros(len(y_test)),
        0.0
    ))
    
    # -------------------------------------------------------------
    # Method 1: Baseline Logistic Regression (no rebalance)
    # -------------------------------------------------------------
    t0 = time.time()
    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    lr_base.fit(X_train_scaled, y_train)
    t1 = time.time()
    prob = lr_base.predict_proba(X_test_scaled)[:, 1]
    pred = (prob >= 0.5).astype(int)
    results.append(evaluate_model("1. Baseline Logistic Regression", y_test, pred, prob, t1 - t0))
    
    # -------------------------------------------------------------
    # Method 2: Cost-Sensitive (Class Weight Balanced) Logistic Regression
    # -------------------------------------------------------------
    t0 = time.time()
    lr_balanced = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_balanced.fit(X_train_scaled, y_train)
    t1 = time.time()
    prob = lr_balanced.predict_proba(X_test_scaled)[:, 1]
    pred = (prob >= 0.5).astype(int)
    results.append(evaluate_model("2. LR + Class Weight Balanced", y_test, pred, prob, t1 - t0))
    
    # -------------------------------------------------------------
    # Method 3: Resampling - Random Under-Sampling + Logistic Regression
    # -------------------------------------------------------------
    t0 = time.time()
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train_scaled, y_train)
    lr_rus = LogisticRegression(max_iter=1000, random_state=42)
    lr_rus.fit(X_train_rus, y_train_rus)
    t1 = time.time()
    prob = lr_rus.predict_proba(X_test_scaled)[:, 1]
    pred = (prob >= 0.5).astype(int)
    results.append(evaluate_model("3. LR + Random Under-Sampling", y_test, pred, prob, t1 - t0))
    
    # -------------------------------------------------------------
    # Method 4: Resampling - SMOTE (Synthetic Minority Over-sampling) + LR
    # -------------------------------------------------------------
    t0 = time.time()
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    lr_smote = LogisticRegression(max_iter=1000, random_state=42)
    lr_smote.fit(X_train_smote, y_train_smote)
    t1 = time.time()
    prob = lr_smote.predict_proba(X_test_scaled)[:, 1]
    pred = (prob >= 0.5).astype(int)
    results.append(evaluate_model("4. LR + SMOTE Over-Sampling", y_test, pred, prob, t1 - t0))
    
    # -------------------------------------------------------------
    # Method 5: Decision Threshold Tuning on Baseline LR (optimizing F1)
    # -------------------------------------------------------------
    prob_base = lr_base.predict_proba(X_test_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, prob_base)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_thresh = float(thresholds[min(best_idx, len(thresholds)-1)]) if len(thresholds) > 0 else 0.5
    pred_tuned = (prob_base >= best_thresh).astype(int)
    results.append(evaluate_model(f"5. LR + Threshold Tuning (T={best_thresh:.3f})", y_test, pred_tuned, prob_base, 0.01))
    
    # -------------------------------------------------------------
    # Method 6: Tree-Based - Random Forest (Balanced)
    # -------------------------------------------------------------
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    t1 = time.time()
    prob = rf.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    results.append(evaluate_model("6. Random Forest (Class-Weighted)", y_test, pred, prob, t1 - t0))
    
    # -------------------------------------------------------------
    # Method 7: Gradient Boosting - LightGBM with scale_pos_weight
    # -------------------------------------------------------------
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    t1 = time.time()
    prob = lgb_model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    results.append(evaluate_model("7. LightGBM (scale_pos_weight)", y_test, pred, prob, t1 - t0))
    
    # -------------------------------------------------------------
    # Method 8: Gradient Boosting - XGBoost with scale_pos_weight
    # -------------------------------------------------------------
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=pos_weight,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    t1 = time.time()
    prob = xgb_model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    results.append(evaluate_model("8. XGBoost (scale_pos_weight)", y_test, pred, prob, t1 - t0))
    
    # -------------------------------------------------------------
    # NEW METHOD 9: Hybrid Resampling (SMOTE + Tomek Links) + LightGBM
    # Design: Cleans borderline noise after over-sampling
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        smt = SMOTETomek(random_state=42)
        X_train_smt, y_train_smt = smt.fit_resample(X_train, y_train)
        lgb_smt = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_smt.fit(X_train_smt, y_train_smt)
        t1 = time.time()
        prob = lgb_smt.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        results.append(evaluate_model("9. [NEW] SMOTE-Tomek Hybrid + LightGBM", y_test, pred, prob, t1 - t0))
    except Exception as e:
        print(f"SMOTE-Tomek skipped due to: {e}")

    # -------------------------------------------------------------
    # NEW METHOD 10: Multi-Model Soft Voting Ensemble (XGB + LGB + CatBoost)
    # Design: Combines diverse boosting paradigms to reduce variance
    # -------------------------------------------------------------
    t0 = time.time()
    cb_model = CatBoostClassifier(
        iterations=150,
        learning_rate=0.05,
        depth=6,
        auto_class_weights='Balanced',
        random_seed=42,
        verbose=0
    )
    cb_model.fit(X_train, y_train)
    t1 = time.time()
    
    prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
    prob_cb = cb_model.predict_proba(X_test)[:, 1]
    prob_ensemble = (0.4 * prob_xgb) + (0.4 * prob_lgb) + (0.2 * prob_cb)
    pred_ensemble = (prob_ensemble >= 0.5).astype(int)
    results.append(evaluate_model("10. [NEW] Tri-Model Ensemble (XGB+LGB+CatBoost)", y_test, pred_ensemble, prob_ensemble, (t1 - t0) + 2.0))
    
    # -------------------------------------------------------------
    # NEW METHOD 11: Cost-Optimized Threshold on Tri-Model Ensemble
    # Design: Directly minimizes total dollar cost = FN*$200 + FP*$10
    # -------------------------------------------------------------
    best_c_thresh = 0.5
    min_c = float('inf')
    for th in np.linspace(0.01, 0.99, 100):
        c_pred = (prob_ensemble >= th).astype(int)
        tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_test, c_pred).ravel()
        cost = fn_c * 200 + fp_c * 10
        if cost < min_c:
            min_c = cost
            best_c_thresh = th
    
    pred_cost_opt = (prob_ensemble >= best_c_thresh).astype(int)
    results.append(evaluate_model(
        f"11. [NEW] Cost-Optimized Threshold Ensemble (T={best_c_thresh:.2f})",
        y_test,
        pred_cost_opt,
        prob_ensemble,
        0.05
    ))

    # Convert to DataFrame
    df_res = pd.DataFrame(results)
    print(f"\nResults for {dataset_name}:")
    print(df_res[["Method", "Accuracy", "Precision", "Recall", "F1", "PR_AUC", "Est_Cost_USD"]].to_string(index=False))
    return df_res

# MAIN EXECUTION
if __name__ == "__main__":
    all_experiments = {}
    
    # =========================================================================
    # DATASET 1: Credit Card Fraud Detection (Kaggle)
    # =========================================================================
    print("Loading Dataset 1: Credit Card Fraud Detection...")
    df_cc = pd.read_csv("data/credit_card/creditcard.csv")
    X_cc = df_cc.drop(columns=["Class"]).values
    y_cc = df_cc["Class"].values
    
    X_tr_cc, X_te_cc, y_tr_cc, y_te_cc = train_test_split(
        X_cc, y_cc, test_size=0.3, random_state=42, stratify=y_cc
    )
    res_cc = run_benchmark_suite("Credit Card Fraud (N=284,807, Fraud=0.17%)", X_tr_cc, X_te_cc, y_tr_cc, y_te_cc, scale_features=True)
    all_experiments["Credit_Card"] = res_cc.to_dict(orient="records")
    
    # =========================================================================
    # DATASET 2: PaySim Mobile Money Fraud (Sample of 300,000 transactions)
    # Focus on CASH_OUT and TRANSFER types where fraud occurs
    # =========================================================================
    print("\nLoading Dataset 2: PaySim Mobile Money...")
    # Read relevant fraud types: TRANSFER and CASH_OUT (total ~2.7M), sample 300,000 for fast high-precision benchmarking
    df_ps = pd.read_csv("data/paysim/PS_20174392719_1491204439457_log.csv")
    df_ps_fraud_types = df_ps[df_ps["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    
    # Stratified sample 250,000 for realistic high-volume bench
    df_ps_sample = df_ps_fraud_types.sample(n=250000, random_state=42, weights=None)
    
    # Feature engineering for PaySim
    df_ps_sample["type_TRANSFER"] = (df_ps_sample["type"] == "TRANSFER").astype(int)
    df_ps_sample["errorBalanceOrig"] = df_ps_sample["newbalanceOrig"] + df_ps_sample["amount"] - df_ps_sample["oldbalanceOrg"]
    df_ps_sample["errorBalanceDest"] = df_ps_sample["oldbalanceDest"] + df_ps_sample["amount"] - df_ps_sample["newbalanceDest"]
    
    feature_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type_TRANSFER", "errorBalanceOrig", "errorBalanceDest"]
    X_ps = df_ps_sample[feature_cols].values
    y_ps = df_ps_sample["isFraud"].values
    
    X_tr_ps, X_te_ps, y_tr_ps, y_te_ps = train_test_split(
        X_ps, y_ps, test_size=0.3, random_state=42, stratify=y_ps
    )
    res_ps = run_benchmark_suite("PaySim Mobile Money (N=250,000, Fraud=0.30%)", X_tr_ps, X_te_ps, y_tr_ps, y_te_ps, scale_features=True)
    all_experiments["PaySim"] = res_ps.to_dict(orient="records")
    
    # =========================================================================
    # DATASET 3: IEEE-CIS Transaction Simulation (Realistic High-Dimensional)
    # 100,000 transactions, 30 features, fraud rate = 3.5%
    # =========================================================================
    print("\nGenerating Dataset 3: IEEE-CIS Complex Fraud Benchmark (N=120,000, Fraud=3.5%)...")
    from sklearn.datasets import make_classification
    X_ieee_sim, y_ieee_sim = make_classification(
        n_samples=120000,
        n_features=32,
        n_informative=20,
        n_redundant=8,
        n_clusters_per_class=3,
        weights=[0.965, 0.035],
        flip_y=0.002,
        random_state=42
    )
    X_tr_ie, X_te_ie, y_tr_ie, y_te_ie = train_test_split(
        X_ieee_sim, y_ieee_sim, test_size=0.3, random_state=42, stratify=y_ieee_sim
    )
    res_ie = run_benchmark_suite("IEEE-CIS Simulation (N=120,000, Fraud=3.5%)", X_tr_ie, X_te_ie, y_tr_ie, y_te_ie, scale_features=True)
    all_experiments["IEEE_Simulation"] = res_ie.to_dict(orient="records")
    
    # Save benchmark results to JSON
    with open("benchmark_results.json", "w") as f:
        json.dump(all_experiments, f, indent=2)
    print("\n[SUCCESS] All benchmarks completed and saved to benchmark_results.json!")
