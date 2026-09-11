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
TAXONOMY_FILE = "src/intent/taxonomy.json"

MODEL_NAME = "all-MiniLM-L6-v2"

N_SPLITS = 5
N_PROTOTYPES = 3
RANDOM_SEED = 42

# Weight given to real labelled examples vs taxonomy descriptions.
EXAMPLE_WEIGHT = 0.7
DESCRIPTION_WEIGHT = 0.3


def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def load_taxonomy(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_multi_prototypes(train_data, model):
    """
    Create up to N_PROTOTYPES semantic prototypes
    for each intent using ONLY the training fold.
    """

    grouped = {}

    for item in train_data:
        intent = item["intent"]

        if intent not in grouped:
            grouped[intent] = []

        grouped[intent].append(
            item["customer_text"]
        )

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


def create_description_embeddings(taxonomy, model):
    """
    Encode the taxonomy descriptions.
    """

    intent_ids = [
        item["intent_id"]
        for item in taxonomy
    ]

    descriptions = [
        item["description"]
        for item in taxonomy
    ]

    embeddings = model.encode(
        descriptions,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    return intent_ids, embeddings


def predict(
    message,
    prototypes,
    intent_ids,
    description_embeddings,
    model
):
    """
    Hybrid prediction.

    Score =
        70% similarity to real labelled examples
        +
        30% similarity to taxonomy description
    """

    message_embedding = model.encode(
        message,
        normalize_embeddings=True
    )

    message_embedding = np.asarray(
        message_embedding,
        dtype=np.float32
    )

    example_scores = {}

    for intent, intent_prototypes in prototypes.items():

        scores = [
            float(
                prototype @ message_embedding
            )
            for prototype in intent_prototypes
        ]

        example_scores[intent] = max(scores)

    description_scores = (
        description_embeddings @ message_embedding
    )

    description_score_map = {
        intent_ids[i]: float(
            description_scores[i]
        )
        for i in range(len(intent_ids))
    }

    combined_scores = {}

    for intent in intent_ids:

        example_score = example_scores.get(
            intent,
            0.0
        )

        description_score = (
            description_score_map[intent]
        )

        combined_scores[intent] = (
            EXAMPLE_WEIGHT * example_score
            +
            DESCRIPTION_WEIGHT * description_score
        )

    sorted_scores = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_intent = sorted_scores[0][0]
    best_score = sorted_scores[0][1]

    second_score = (
        sorted_scores[1][1]
        if len(sorted_scores) > 1
        else 0.0
    )

    margin = best_score - second_score

    confidence = (
        0.5 * best_score
        +
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

    print("\nLoading taxonomy...")

    taxonomy = load_taxonomy(
        TAXONOMY_FILE
    )

    print(
        f"Taxonomy intents: {len(taxonomy)}"
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

    print(
        "\nEncoding taxonomy descriptions..."
    )

    (
        intent_ids,
        description_embeddings
    ) = create_description_embeddings(
        taxonomy,
        model
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
        "hybrid evaluation..."
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

        prototypes = create_multi_prototypes(
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
                intent_ids,
                description_embeddings,
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
        "HYBRID INTENT CLASSIFIER"
    )

    print(
        "=" * 60
    )

    print(
        f"\nExample weight: "
        f"{EXAMPLE_WEIGHT}"
    )

    print(
        f"Description weight: "
        f"{DESCRIPTION_WEIGHT}"
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