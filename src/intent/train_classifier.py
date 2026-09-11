import json
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


TRAIN_FILE = "data/processed/intent_training.jsonl"
GOLDEN_FILE = "data/golden/golden_set.jsonl"

MODEL_DIR = "data/processed/intent_model"
MODEL_FILE = f"{MODEL_DIR}/model.joblib"


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def main():

    print("Loading training data...")

    train_data = load_jsonl(TRAIN_FILE)

    print(f"Training examples: {len(train_data)}")

    X_train = [
        item["customer_text"]
        for item in train_data
    ]

    y_train = [
        item["intent"]
        for item in train_data
    ]

    print("\nTraining TF-IDF + Logistic Regression...")

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

    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")

    # -----------------------------
    # Evaluate on golden set
    # -----------------------------

    print("\nEvaluating on golden set...")

    golden_data = load_jsonl(GOLDEN_FILE)

    X_test = [
        item["customer_text"]
        for item in golden_data
    ]

    y_test = [
        item["intent"]
        for item in golden_data
    ]

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("INTENT CLASSIFIER RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()