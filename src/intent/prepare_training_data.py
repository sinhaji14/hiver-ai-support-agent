import json
from pathlib import Path


INTERACTIONS_PATH = Path("data/processed/amazon_interactions.jsonl")
GOLDEN_PATH = Path("data/golden/golden_set.jsonl")
OUTPUT_PATH = Path("data/processed/intent_training.jsonl")


# Load golden examples
with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    golden = [json.loads(line) for line in f if line.strip()]

# Use customer text to identify golden examples.
# This avoids depending on inconsistent ID fields.
golden_texts = {
    item.get("customer_text", "").strip()
    for item in golden
    if item.get("customer_text")
}


# Load historical interactions
with open(INTERACTIONS_PATH, "r", encoding="utf-8") as f:
    interactions = [json.loads(line) for line in f if line.strip()]


# Remove golden examples from training data
training = []

for item in interactions:
    customer_text = item.get("customer_text", "").strip()

    if customer_text not in golden_texts:
        training.append(item)


# Save training data
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in training:
        record = {
            "interaction_id": item.get("tweet_id"),
            "customer_text": item.get("customer_text", ""),
            "agent_text": item.get("agent_text", ""),
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")


print("Training dataset created")
print("=========================")
print(f"Original interactions: {len(interactions)}")
print(f"Golden examples:       {len(golden)}")
print(f"Removed from training: {len(interactions) - len(training)}")
print(f"Training examples:     {len(training)}")
print(f"Saved to:              {OUTPUT_PATH}")