import json
import random

INPUT_PATH = "data/processed/amazon_interactions.jsonl"
OUTPUT_PATH = "data/golden/golden_set.jsonl"

SEED = 42
SAMPLE_SIZE = 200

random.seed(SEED)

# Load interactions
interactions = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        interactions.append(json.loads(line))

print(f"Loaded {len(interactions):,} interactions.")

# Random sample
sample = random.sample(
    interactions,
    min(SAMPLE_SIZE, len(interactions))
)

# Save examples with an empty human label
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in sample:
        record = {
            "interaction_id": item.get("tweet_id"),
            "customer_text": item["customer_text"],
            "agent_text": item.get("agent_text", ""),
            "intent": "",
            "notes": ""
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Created {len(sample)} examples.")
print(f"Saved to: {OUTPUT_PATH}")