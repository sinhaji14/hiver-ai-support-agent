import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold


GOLDEN_FILE = "data/golden/golden_set.jsonl"

MODEL_NAME = "all-MiniLM-L6-v2"

N_SPLITS = 5
N_PROTOTYPES = 3
RANDOM_SEED = 42


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def create_prototypes(train_data, model):
    """
    Create multiple semantic prototypes for each intent.

    Instead of representing an intent with one mean embedding,
    cluster the training examples within each intent.
    """

    grouped = {}

    for item in train_data:
        intent = item["intent"]

        if intent not in grouped:
            grouped[intent] = []

        grouped[intent].append(item["customer_text"])

    prototypes = {}

    for intent, texts in grouped.items():

        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        # Number of clusters cannot exceed the number
        # of examples available for this intent.
        k = min(
            N_PROTOTYPES,
            len(embeddings)
        )

        if k == 1:
            prototype = embeddings.mean(axis=0)
            prototype = prototype / np.linalg.norm(prototype)

            prototypes[intent] = [prototype]
            continue

        clustering = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=10
        )

        clustering.fit(embeddings)

        intent_prototypes = []

        for cluster_id in range(k):

            cluster_embeddings = embeddings[
                clustering.labels_ == cluster_id
            ]

            prototype = cluster_embeddings.mean(axis=0)

            prototype = prototype / np.linalg.norm(
                prototype
            )

            intent_prototypes.append(
                prototype
            )

        prototypes[intent] = intent_prototypes

    return prototypes


def predict(message, prototypes, model):

    message_embedding = model.encode(
        message,
        normalize_embeddings=True
    )

    intent_scores = {}

    for intent, intent_prototypes in prototypes.items():

        prototype_scores = [
            float(prototype @ message_embedding)
            for prototype in intent_prototypes
        ]

        # Best matching semantic sub-cluster
        intent_scores[intent] = max(
            prototype_scores
        )

    sorted_scores = sorted(
        intent_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_intent = sorted_scores[0][0]
    best_score = sorted_scores[0][1]

    if len(sorted_scores) > 1:
        second_score = sorted_scores[1][1]
    else:
        second_score = 0.0

    margin = best_score - second_score

    confidence = (
        0.5 * best_score +
        0.5 * margin
    )

    return (
        best_intent,
        best_score,
        margin,
        confidence
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

    print(
        "\nLoading embedding model..."
    )

    model = SentenceTransformer(
        MODEL_NAME
    )

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
        f"\nRunning {N_SPLITS}-fold "
        "multi-prototype evaluation..."
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

        prototypes = create_prototypes(
            train_data,
            model
        )

        for index in test_indices:

            (
                prediction,
                score,
                margin,
                confidence
            ) = predict(
                data[index]["customer_text"],
                prototypes,
                model
            )

            all_predictions[index] = prediction

            all_confidences[index] = confidence

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
        "MULTI-PROTOTYPE INTENT CLASSIFIER"
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