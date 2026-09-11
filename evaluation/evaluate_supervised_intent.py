import json

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


GOLDEN_FILE = "data/golden/golden_set.jsonl"


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def main():

    print("Loading golden set...")

    data = load_jsonl(GOLDEN_FILE)

    print(f"Total examples: {len(data)}")

    X = [
        item["customer_text"]
        for item in data
    ]

    y = [
        item["intent"]
        for item in data
    ]

    # -----------------------------
    # Model
    # -----------------------------

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    # -----------------------------
    # Stratified cross-validation
    # -----------------------------

    print("\nRunning 5-fold stratified cross-validation...")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    predictions = cross_val_predict(
        model,
        X,
        y,
        cv=cv
    )

    # -----------------------------
    # Metrics
    # -----------------------------

    accuracy = accuracy_score(
        y,
        predictions
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted"
    )

    print("\n" + "=" * 60)
    print("SUPERVISED INTENT CLASSIFIER")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"\nMacro-F1: {macro_f1:.4f}"
    )

    print(
        f"Weighted-F1: {weighted_f1:.4f}"
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    # -----------------------------
    # Confusion matrix
    # -----------------------------

    labels = sorted(set(y))

    matrix = confusion_matrix(
        y,
        predictions,
        labels=labels
    )

    print("\nConfusion Matrix")

    print(
        "\t".join(labels)
    )

    for label, row in zip(
        labels,
        matrix
    ):
        print(
            label,
            "\t",
            "\t".join(
                str(value)
                for value in row
            )
        )


if __name__ == "__main__":
    main()