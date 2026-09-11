import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity


TAXONOMY_PATH = Path("src/intent/taxonomy.json")
GOLDEN_PATH = Path("data/golden/golden_set.jsonl")


# Load taxonomy
with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

intent_ids = [item["intent_id"] for item in taxonomy]

intent_texts = [
    f"{item['intent_name']}: {item['description']}"
    for item in taxonomy
]


# Load golden set
with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    golden = [json.loads(line) for line in f if line.strip()]


texts = [item["customer_text"] for item in golden]
true_labels = [item["intent"] for item in golden]


print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

intent_embeddings = model.encode(
    intent_texts,
    normalize_embeddings=True
)

text_embeddings = model.encode(
    texts,
    normalize_embeddings=True
)


# Calculate similarity between every customer message
# and every intent description.
similarities = cosine_similarity(
    text_embeddings,
    intent_embeddings
)


# Pick the most similar intent
predictions = [
    intent_ids[row.argmax()]
    for row in similarities
]


# Evaluation
accuracy = accuracy_score(true_labels, predictions)

print("\nEmbedding Classifier")
print("====================")
print(f"Total examples: {len(golden)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        true_labels,
        predictions,
        labels=intent_ids,
        zero_division=0
    )
)