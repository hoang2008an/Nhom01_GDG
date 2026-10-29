from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)


def summary(y_test, y_pred, threshold=0.5):
    print("accuracy_score", accuracy_score(y_test, y_pred > threshold))
    print("recall_score:", recall_score(y_test, y_pred > threshold))
    print("precision_score:", precision_score(y_test, y_pred > threshold))
    print("f1_score", f1_score(y_test, y_pred > threshold))
    print("ROC-AUC score:", roc_auc_score(y_test, y_pred))
    print("PRAUC score:", average_precision_score(y_test, y_pred))
    print("COnfusion matrix:", confusion_matrix(y_test, y_pred > threshold))
