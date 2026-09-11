import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


INDEX_DIR = Path("data/processed/retrieval_index")
GOLDEN_PATH = Path("data/golden/golden_set.jsonl")

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    print("Loading retrieval index...")

    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = load_metadata()

    print(f"Indexed interactions: {len(metadata)}")
    print(f"Embedding shape: {embeddings.shape}")

    print("\nLoading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load first 10 golden examples for a quick sanity check
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = [json.loads(line) for line in f if line.strip()]

    golden = golden[:10]

    for number, item in enumerate(golden, start=1):
        query = item["customer_text"]

        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )

        scores = cosine_similarity(
            query_embedding,
            embeddings,
        )[0]

        # Top 3 results
        top_indices = np.argsort(scores)[-3:][::-1]

        print("\n" + "=" * 80)
        print(f"GOLDEN EXAMPLE {number}")
        print("=" * 80)

        print("\nCUSTOMER:")
        print(query)

        print("\nEXPECTED INTENT:")
        print(item["intent"])

        print("\nTOP RETRIEVED CASES:")

        for rank, idx in enumerate(top_indices, start=1):
            result = metadata[idx]

            print(f"\n--- Result {rank} ---")
            print(f"Similarity: {scores[idx]:.4f}")

            print("Customer:")
            print(result["customer_text"])

            print("Historical agent:")
            print(result["agent_text"])


if __name__ == "__main__":
    main()