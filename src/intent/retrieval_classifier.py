import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity


TRAINING_PATH = Path("data/processed/intent_training.jsonl")
GOLDEN_PATH = Path("data/golden/golden_set.jsonl")
TAXONOMY_PATH = Path("src/intent/taxonomy.json")


# --------------------------------------------------
# Load data
# --------------------------------------------------

with open(TRAINING_PATH, "r", encoding="utf-8") as f:
    training = [json.loads(line) for line in f if line.strip()]

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    golden = [json.loads(line) for line in f if line.strip()]

with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    taxonomy = json.load(f)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# Encode historical customer messages
# --------------------------------------------------

print(f"Encoding {len(training)} historical messages...")

training_texts = [
    item["customer_text"]
    for item in training
    if item.get("customer_text")
]

training_embeddings = model.encode(
    training_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
)


# --------------------------------------------------
# Create intent prototypes
# --------------------------------------------------

intent_ids = [
    item["intent_id"]
    for item in taxonomy
]

intent_texts = [
    f"{item['intent_name']}: {item['description']}"
    for item in taxonomy
]

intent_embeddings = model.encode(
    intent_texts,
    normalize_embeddings=True,
)


# --------------------------------------------------
# Classify golden examples
# --------------------------------------------------

predictions = []

print("\nClassifying golden examples...")

for i, item in enumerate(golden):

    text = item["customer_text"]

    query_embedding = model.encode(
        [text],
        normalize_embeddings=True,
    )

    # Similarity to historical examples
    historical_scores = cosine_similarity(
        query_embedding,
        training_embeddings,
    )[0]

    # Take the strongest historical examples
    top_k = min(20, len(historical_scores))

    top_indices = np.argsort(historical_scores)[-top_k:][::-1]

    # Use the intent prototypes to assign the retrieved examples
    # to the closest intent.
    top_historical_embeddings = training_embeddings[top_indices]

    prototype_scores = cosine_similarity(
        top_historical_embeddings,
        intent_embeddings,
    )

    # Combine historical similarity and prototype similarity
    combined_scores = (
        prototype_scores.max(axis=0)
        + historical_scores[top_indices].max()
    )

    predicted_index = combined_scores.argmax()

    predictions.append(intent_ids[predicted_index])

    if (i + 1) % 25 == 0:
        print(f"Processed {i + 1}/{len(golden)}")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

true_labels = [
    item["intent"]
    for item in golden
]

accuracy = accuracy_score(
    true_labels,
    predictions,
)

print("\nRetrieval-Based Intent Classifier")
print("=================================")
print(f"Golden examples: {len(golden)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        labels=intent_ids,
        zero_division=0,
    )
)