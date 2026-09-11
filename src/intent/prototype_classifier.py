import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


GOLDEN_FILE = "data/golden/golden_set.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"

# Use part of the golden set as labelled prototypes.
# The remaining examples are used only for evaluation.
TRAIN_RATIO = 0.75

RANDOM_SEED = 42


def load_jsonl(path):
    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            data.append(
                json.loads(line)
            )

    return data


def classify_with_prototypes(
    message,
    prototype_embeddings,
    prototype_labels,
    model,
):
    """
    Classify a message by finding its nearest
    labelled golden-set example.
    """

    message_embedding = model.encode(
        message,
        normalize_embeddings=True
    )

    scores = (
        prototype_embeddings
        @ message_embedding
    )

    best_index = int(
        np.argmax(scores)
    )

    predicted_intent = (
        prototype_labels[best_index]
    )

    confidence = float(
        scores[best_index]
    )

    return predicted_intent, confidence


def main():

    print("Loading golden set...")

    data = load_jsonl(
        GOLDEN_FILE
    )

    print(
        f"Total examples: {len(data)}"
    )

    # -----------------------------
    # Shuffle reproducibly
    # -----------------------------

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    indices = np.arange(
        len(data)
    )

    rng.shuffle(indices)

    split_index = int(
        len(data) * TRAIN_RATIO
    )

    train_indices = indices[
        :split_index
    ]

    test_indices = indices[
        split_index:
    ]

    train_data = [
        data[i]
        for i in train_indices
    ]

    test_data = [
        data[i]
        for i in test_indices
    ]

    print(
        f"Prototype examples: "
        f"{len(train_data)}"
    )

    print(
        f"Evaluation examples: "
        f"{len(test_data)}"
    )

    # -----------------------------
    # Load embedding model
    # -----------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # -----------------------------
    # Create prototypes
    # -----------------------------

    print(
        "\nEncoding labelled prototypes..."
    )

    prototype_texts = [
        item["customer_text"]
        for item in train_data
    ]

    prototype_labels = [
        item["intent"]
        for item in train_data
    ]

    prototype_embeddings = model.encode(
        prototype_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # -----------------------------
    # Evaluate
    # -----------------------------

    print(
        "\nEvaluating prototype classifier..."
    )

    y_true = []
    y_pred = []

    for i, item in enumerate(
        test_data
    ):

        prediction, confidence = (
            classify_with_prototypes(
                item["customer_text"],
                prototype_embeddings,
                prototype_labels,
                model,
            )
        )

        y_true.append(
            item["intent"]
        )

        y_pred.append(
            prediction
        )

        if (i + 1) % 25 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(test_data)}"
            )

    # -----------------------------
    # Metrics
    # -----------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PROTOTYPE INTENT CLASSIFIER"
    )

    print(
        "=" * 60
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        "\nClassification Report:\n"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()