import json
import random

INPUT_PATH = "data/processed/amazon_interactions.jsonl"
OUTPUT_PATH = "data/processed/intent_sample.jsonl"

SAMPLE_SIZE = 2000
SEED = 42

random.seed(SEED)

print("Loading interactions...")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    interactions = [json.loads(line) for line in f]

print(f"Total interactions: {len(interactions):,}")

sample = random.sample(
    interactions,
    min(SAMPLE_SIZE, len(interactions))
)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in sample:
        f.write(
            json.dumps(item, ensure_ascii=False) + "\n"
        )

print(f"Created sample: {len(sample):,}")
print(f"Saved to: {OUTPUT_PATH}")