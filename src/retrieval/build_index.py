import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


TRAINING_PATH = Path("data/processed/intent_training.jsonl")
INDEX_DIR = Path("data/processed/retrieval_index")

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.jsonl"


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load historical interactions
    with open(TRAINING_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    data = [
        item for item in data
        if item.get("customer_text", "").strip()
    ]

    print(f"Historical interactions: {len(data)}")

    # Load model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings
    texts = [item["customer_text"] for item in data]

    print("Creating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    # Save embeddings
    np.save(EMBEDDINGS_PATH, embeddings)

    # Save metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        for item in data:
            metadata = {
                "interaction_id": item.get("interaction_id"),
                "customer_text": item.get("customer_text", ""),
                "agent_text": item.get("agent_text", ""),
            }

            f.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\nRetrieval index created")
    print("=======================")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings:       {EMBEDDINGS_PATH}")
    print(f"Metadata:         {METADATA_PATH}")


if __name__ == "__main__":
    main()