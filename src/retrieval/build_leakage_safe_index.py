import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer


CONVERSATIONS_PATH = "data/processed/amazon_conversations.jsonl"
GOLDEN_PATH = "data/golden/golden_set.jsonl"

OUTPUT_DIR = "data/processed/retrieval_index_safe"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_golden_texts():
    """Load customer messages used in the golden evaluation set."""
    golden_texts = set()

    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            golden_texts.add(item["customer_text"].strip())

    return golden_texts


def find_excluded_conversations():
    """
    Find every conversation containing a golden-set customer message.

    We exclude the ENTIRE conversation rather than only the exact
    golden interaction to prevent same-conversation leakage.
    """
    golden_texts = load_golden_texts()

    excluded_ids = set()
    total_conversations = 0

    with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            conversation = json.loads(line)
            total_conversations += 1

            conversation_id = conversation["conversation_id"]
            messages = conversation["messages"]

            for message in messages:
                if (
                    message.get("inbound") is True
                    and message.get("text", "").strip() in golden_texts
                ):
                    excluded_ids.add(conversation_id)
                    break

    print(f"Total conversations: {total_conversations}")
    print(f"Excluded conversations: {len(excluded_ids)}")

    return excluded_ids


def build_corpus(excluded_ids):
    """
    Build historical customer -> support interactions while
    excluding conversations that contain golden examples.
    """
    corpus = []

    total_interactions = 0
    skipped_interactions = 0

    with open(CONVERSATIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            conversation = json.loads(line)

            conversation_id = conversation["conversation_id"]

            if conversation_id in excluded_ids:
                continue

            messages = conversation["messages"]

            # Map tweet IDs to messages so parent references can be followed.
            message_by_id = {
                message["tweet_id"]: message
                for message in messages
            }

            for message in messages:
                # We only want AmazonHelp support replies.
                if message.get("inbound") is not False:
                    continue

                parent_id = message.get("parent")

                if parent_id is None:
                    continue

                parent = message_by_id.get(parent_id)

                # Parent must be a customer message.
                if parent is None or parent.get("inbound") is not True:
                    continue

                customer_text = parent.get("text", "").strip()
                agent_text = message.get("text", "").strip()

                if not customer_text or not agent_text:
                    continue

                total_interactions += 1

                corpus.append(
                    {
                        "conversation_id": conversation_id,
                        "customer_tweet_id": parent["tweet_id"],
                        "agent_tweet_id": message["tweet_id"],
                        "customer_text": customer_text,
                        "agent_text": agent_text,
                    }
                )

    print(f"Leakage-safe historical interactions: {len(corpus)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    corpus_path = os.path.join(
        OUTPUT_DIR,
        "metadata.jsonl"
    )

    with open(corpus_path, "w", encoding="utf-8") as f:
        for item in corpus:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved metadata to: {corpus_path}")

    return corpus


def build_embeddings(corpus):
    """Create normalized MiniLM embeddings for the customer messages."""
    print("\nLoading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    customer_texts = [
        item["customer_text"]
        for item in corpus
    ]

    print(f"Encoding {len(customer_texts)} historical messages...")

    embeddings = model.encode(
        customer_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    embeddings_path = os.path.join(
        OUTPUT_DIR,
        "embeddings.npy"
    )

    np.save(embeddings_path, embeddings)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Saved embeddings to: {embeddings_path}")


def main():
    print("=" * 60)
    print("BUILDING LEAKAGE-SAFE RETRIEVAL INDEX")
    print("=" * 60)

    excluded_ids = find_excluded_conversations()

    corpus = build_corpus(excluded_ids)

    build_embeddings(corpus)

    print("\nDone.")


if __name__ == "__main__":
    main()