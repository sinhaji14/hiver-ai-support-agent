import json

import numpy as np
from sentence_transformers import SentenceTransformer


GOLDEN_FILE = "data/golden/golden_set.jsonl"

INDEX_DIR = "data/processed/retrieval_index"

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


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


def main():

    print("Loading golden set...")

    golden = load_jsonl(
        GOLDEN_FILE
    )

    print(
        f"Golden examples: {len(golden)}"
    )

    # -----------------------------
    # Load retrieval index
    # -----------------------------

    print(
        "\nLoading retrieval index..."
    )

    embeddings = np.load(
        f"{INDEX_DIR}/embeddings.npy"
    )

    print(
        f"Historical embeddings: "
        f"{len(embeddings)}"
    )

    # -----------------------------
    # Load metadata
    # -----------------------------

    metadata = []

    with open(
        f"{INDEX_DIR}/metadata.jsonl",
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            metadata.append(
                json.loads(line)
            )

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
    # Evaluate retrieval
    # -----------------------------

    top1_scores = []
    top3_scores = []
    top5_scores = []

    print(
        "\nEvaluating retrieval..."
    )

    for i, item in enumerate(
        golden
    ):

        query_embedding = model.encode(
            item["customer_text"],
            normalize_embeddings=True
        )

        scores = (
            embeddings @ query_embedding
        )

        top_indices = np.argsort(
            scores
        )[::-1][:TOP_K]

        retrieved_scores = [
            float(scores[index])
            for index in top_indices
        ]

        top1_scores.append(
            retrieved_scores[0]
        )

        top3_scores.append(
            np.mean(
                retrieved_scores[:3]
            )
        )

        top5_scores.append(
            np.mean(
                retrieved_scores[:5]
            )
        )

        if (i + 1) % 50 == 0:

            print(
                f"Processed "
                f"{i + 1}/{len(golden)}"
            )

    # -----------------------------
    # Results
    # -----------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "RETRIEVAL EVALUATION"
    )

    print(
        "=" * 60
    )

    print(
        f"\nTop-1 similarity: "
        f"{np.mean(top1_scores):.4f}"
    )

    print(
        f"Top-3 average similarity: "
        f"{np.mean(top3_scores):.4f}"
    )

    print(
        f"Top-5 average similarity: "
        f"{np.mean(top5_scores):.4f}"
    )

    print(
        f"\nMinimum Top-1 similarity: "
        f"{np.min(top1_scores):.4f}"
    )

    print(
        f"Maximum Top-1 similarity: "
        f"{np.max(top1_scores):.4f}"
    )


if __name__ == "__main__":
    main()