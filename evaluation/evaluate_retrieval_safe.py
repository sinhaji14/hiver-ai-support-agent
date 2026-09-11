import json
import numpy as np
from sentence_transformers import SentenceTransformer


GOLDEN_PATH = "data/golden/golden_set.jsonl"
INDEX_DIR = "data/processed/retrieval_index_safe"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_jsonl(path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    return items


def main():
    golden = load_jsonl(GOLDEN_PATH)
    metadata = load_jsonl(f"{INDEX_DIR}/metadata.jsonl")
    embeddings = np.load(f"{INDEX_DIR}/embeddings.npy")

    print("=" * 60)
    print("LEAKAGE-SAFE RETRIEVAL EVALUATION")
    print("=" * 60)

    print(f"Golden examples: {len(golden)}")
    print(f"Historical embeddings: {len(metadata)}")

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    queries = [
        item["customer_text"]
        for item in golden
    ]

    print("Encoding golden queries...")
    query_embeddings = model.encode(
        queries,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    top1_scores = []
    top3_scores = []
    top5_scores = []

    for query_embedding in query_embeddings:
        scores = embeddings @ query_embedding
        sorted_scores = np.sort(scores)[::-1]

        top1_scores.append(float(sorted_scores[0]))
        top3_scores.append(float(np.mean(sorted_scores[:3])))
        top5_scores.append(float(np.mean(sorted_scores[:5])))

    print("\nRetrieval Similarity Results")
    print("-" * 60)

    print(f"Top-1 similarity:       {np.mean(top1_scores):.4f}")
    print(f"Top-3 average similarity: {np.mean(top3_scores):.4f}")
    print(f"Top-5 average similarity: {np.mean(top5_scores):.4f}")

    print(f"\nMinimum Top-1 similarity: {np.min(top1_scores):.4f}")
    print(f"Maximum Top-1 similarity: {np.max(top1_scores):.4f}")

    print("\nSimilarity Percentiles")
    print("-" * 60)

    for percentile in [10, 25, 50, 75, 90]:
        value = np.percentile(top1_scores, percentile)
        print(f"P{percentile}: {value:.4f}")

    print("\nNote:")
    print(
        "These are semantic similarity diagnostics, not retrieval accuracy."
    )


if __name__ == "__main__":
    main()