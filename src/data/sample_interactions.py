import json
import random

INPUT_PATH = "data/processed/amazon_interactions.jsonl"
OUTPUT_PATH = "data/processed/sample_interactions.txt"

SAMPLE_SIZE = 100
SEED = 42

random.seed(SEED)

# Load interactions
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    interactions = [json.loads(line) for line in f]

print(f"Total interactions: {len(interactions):,}")

# Random sample
sample = random.sample(
    interactions,
    min(SAMPLE_SIZE, len(interactions))
)

# Save readable version
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for i, item in enumerate(sample, 1):

        f.write("=" * 80 + "\n")
        f.write(f"EXAMPLE {i}\n")
        f.write("=" * 80 + "\n\n")

        f.write("CUSTOMER:\n")
        f.write(item["customer_text"])
        f.write("\n\n")

        f.write("AMAZONHELP:\n")
        f.write(item["agent_text"])
        f.write("\n\n")

print(f"Saved sample to: {OUTPUT_PATH}")