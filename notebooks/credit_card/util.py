import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def summary(y_test, y_pred):
    print("accuracy_score:", round(accuracy_score(y_test, y_pred), 4))
    print("recall_score:   ", round(recall_score(y_test, y_pred), 4))
    print("precision_score:", round(precision_score(y_test, y_pred), 4))
    print("f1_score:       ", round(f1_score(y_test, y_pred), 4))
    print("COnfusion matrix:\n", confusion_matrix(y_test, y_pred))


def get_data(use_val=True, seed=42, test_size=0.15, val_size=0.15):
    """
    Nap du lieu Credit Card chuan chi, khong loi Unicode, ho tro split train/val/test
    """
    candidates = [
        "../../data/credit_card/creditcard.csv",
        "data/credit_card/creditcard.csv",
        "../data/credit_card/creditcard.csv",
        "E:/HUST/CLB/GDG/Nhom01_GDG-main/data/credit_card/creditcard.csv",
    ]
    df = None
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p)
            break
    if df is None:
        raise FileNotFoundError("Khong tim thay file creditcard.csv")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    if use_val:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size + val_size, random_state=seed, stratify=y
        )
        rel_test = test_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=rel_test, random_state=seed, stratify=y_temp
        )
        return X_train, X_val, X_test, y_train, y_val, y_test
    else:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
