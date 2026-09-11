import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold


GOLDEN_FILE = "data/golden/golden_set.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"

N_SPLITS = 5

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


def create_class_prototypes(
    train_data,
    model
):
    """
    Create one embedding prototype for each intent.

    The prototype is the mean embedding of all
    training examples belonging to that intent.
    """

    prototypes = {}
    labels = {}

    for item in train_data:

        intent = item["intent"]

        if intent not in labels:
            labels[intent] = []

        labels[intent].append(
            item["customer_text"]
        )

    for intent, texts in labels.items():

        embeddings = model.encode(
            texts,
            normalize_embeddings=True
        )

        # Mean embedding for the class
        prototype = embeddings.mean(
            axis=0
        )

        # Normalize prototype
        prototype = prototype / np.linalg.norm(
            prototype
        )

        prototypes[intent] = prototype

    return prototypes


def predict(
    message,
    prototypes,
    model
):
    """
    Predict the intent by comparing the message
    against the class prototypes.
    """

    message_embedding = model.encode(
        message,
        normalize_embeddings=True
    )

    scores = {}

    for intent, prototype in prototypes.items():

        scores[intent] = float(
            prototype @ message_embedding
        )

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_intent = sorted_scores[0][0]

    best_score = sorted_scores[0][1]

    if len(sorted_scores) > 1:
        second_score = sorted_scores[1][1]
    else:
        second_score = 0.0

    margin = (
        best_score -
        second_score
    )

    return (
        best_intent,
        best_score,
        margin
    )


def main():

    print("Loading golden set...")

    data = load_jsonl(
        GOLDEN_FILE
    )

    print(
        f"Total examples: {len(data)}"
    )

    X = [
        item["customer_text"]
        for item in data
    ]

    y = [
        item["intent"]
        for item in data
    ]

    # -----------------------------
    # Load model
    # -----------------------------

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

    # -----------------------------
    # Stratified CV
    # -----------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    all_predictions = [
        None
        for _ in data
    ]

    all_confidences = [
        None
        for _ in data
    ]

    print(
        "\nRunning "
        f"{N_SPLITS}-fold stratified "
        "prototype evaluation..."
    )

    for fold, (
        train_indices,
        test_indices
    ) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print(
            f"\nFold {fold}/{N_SPLITS}"
        )

        train_data = [
            data[i]
            for i in train_indices
        ]

        test_data = [
            data[i]
            for i in test_indices
        ]

        print(
            f"Training examples: "
            f"{len(train_data)}"
        )

        print(
            f"Test examples: "
            f"{len(test_data)}"
        )

        # Create prototypes using ONLY
        # training examples in this fold.
        prototypes = create_class_prototypes(
            train_data,
            model
        )

        # Predict held-out examples
        for index in test_indices:

            prediction, score, margin = (
                predict(
                    data[index]["customer_text"],
                    prototypes,
                    model
                )
            )

            all_predictions[index] = prediction

            # This is a heuristic confidence signal,
            # not a calibrated probability.
            all_confidences[index] = (
                0.5 * score +
                0.5 * margin
            )

    # -----------------------------
    # Metrics
    # -----------------------------

    y_pred = all_predictions

    accuracy = accuracy_score(
        y,
        y_pred
    )

    macro_f1 = f1_score(
        y,
        y_pred,
        average="macro"
    )

    weighted_f1 = f1_score(
        y,
        y_pred,
        average="weighted"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "CLASS PROTOTYPE INTENT CLASSIFIER"
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
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()