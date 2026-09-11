import json
from collections import Counter

PATH = "data/golden/golden_set.jsonl"

with open(PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

labels = [x["intent"] for x in data]

counts = Counter(labels)
majority_intent, majority_count = counts.most_common(1)[0]

accuracy = majority_count / len(labels)

print("Majority Class Baseline")
print("=======================")
print(f"Total examples: {len(labels)}")
print(f"Majority intent: {majority_intent}")
print(f"Majority count: {majority_count}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")