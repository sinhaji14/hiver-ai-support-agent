import json
from statistics import mean

PATH = "data/processed/llm_judge_results.json"

with open(PATH, encoding="utf-8") as f:
    data = json.load(f)

ratings = data["ratings"]

metrics = ["helpfulness", "groundedness", "correctness", "safety"]

print("=" * 60)
print("LLM-AS-A-JUDGE SUMMARY")
print("=" * 60)

print(f"Examples evaluated: {len(ratings)}")
print()

for metric in metrics:
    scores = [r[metric] for r in ratings]
    avg = mean(scores)
    perfect = sum(s == 3 for s in scores)

    print(
        f"{metric.capitalize():15s}: "
        f"{avg:.2f}/3 "
        f"({perfect}/{len(scores)} scored 3)"
    )

overall_scores = [
    mean([r[m] for m in metrics])
    for r in ratings
]

print()
print(f"Overall average: {mean(overall_scores):.2f}/3")
print(f"Overall percentage: {mean(overall_scores) / 3 * 100:.1f}%")

print()
print("Lowest-scoring examples:")

for r in ratings:
    avg = mean([r[m] for m in metrics])
    if avg < 3:
        print(
            f"Sample {r['sample_id']}: "
            f"{avg:.2f}/3 | {r['reason']}"
        )
        