from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def summary(y_test, y_pred):
    print("accuracy_score", accuracy_score(y_test, y_pred))
    print("recall_score:", recall_score(y_test, y_pred))
    print("precision_score:", precision_score(y_test, y_pred))
    print("f1_score", f1_score(y_test, y_pred))
    print("COnfusion matrix:", confusion_matrix(y_test, y_pred))
