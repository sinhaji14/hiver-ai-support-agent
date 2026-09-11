import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


TAXONOMY_FILE = "src/intent/taxonomy.json"
GOLDEN_FILE = "data/golden/golden_set.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def load_taxonomy():
    with open(
        TAXONOMY_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def classify(message, taxonomy, model, intent_embeddings):

    message_embedding = model.encode(
        message,
        normalize_embeddings=True
    )

    scores = intent_embeddings @ message_embedding

    best_index = int(np.argmax(scores))

    return taxonomy[best_index]["intent_id"]


def main():

    print("Loading taxonomy...")

    taxonomy = load_taxonomy()

    print("Loading golden set...")

    golden = load_jsonl(GOLDEN_FILE)

    print(
        f"Golden examples: {len(golden)}"
    )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    descriptions = [
        item["description"]
        for item in taxonomy
    ]

    intent_embeddings = model.encode(
        descriptions,
        normalize_embeddings=True
    )

    y_true = []
    y_pred = []

    print("\nRunning intent evaluation...")

    for i, item in enumerate(golden):

        prediction = classify(
            item["customer_text"],
            taxonomy,
            model,
            intent_embeddings
        )

        y_true.append(
            item["intent"]
        )

        y_pred.append(
            prediction
        )

        if (i + 1) % 50 == 0:
            print(
                f"Processed {i + 1}/{len(golden)}"
            )

    # -----------------------------
    # Metrics
    # -----------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\n" + "=" * 60)
    print("INTENT EVALUATION")
    print("=" * 60)

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    # -----------------------------
    # Confusion matrix
    # -----------------------------

    labels = [
        item["intent_id"]
        for item in taxonomy
    ]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    print("\nConfusion Matrix:")

    print(
        "Labels:"
    )

    for i, label in enumerate(labels):
        print(
            f"{i}: {label}"
        )

    print()

    print(matrix)


if __name__ == "__main__":
    main()