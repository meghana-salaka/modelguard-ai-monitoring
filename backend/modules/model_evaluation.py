import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(y_true, y_pred):

    # Convert inputs into arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Validate inputs
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Labels cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Actual and predicted labels must have equal lengths."
        )

    if np.any(y_true == None) or np.any(y_pred == None):
        raise ValueError("Labels cannot contain missing values.")

    # Identify all classes
    labels = list(
        dict.fromkeys(
            list(y_true) + list(y_pred)
        )
    )

    # Calculate classification metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    # Detailed classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True
    )

    # Return results
    return {

        "total_samples": int(len(y_true)),

        "accuracy": round(float(accuracy), 4),

        "precision_macro": round(float(precision), 4),

        "recall_macro": round(float(recall), 4),

        "f1_score_macro": round(float(f1), 4),

        "precision_weighted": round(
            float(weighted_precision), 4
        ),

        "recall_weighted": round(
            float(weighted_recall), 4
        ),

        "f1_score_weighted": round(
            float(weighted_f1), 4
        ),

        "confusion_matrix": cm.tolist(),

        "class_labels": [str(label) for label in labels],

        "classification_report": report

    }