import json
import numpy as np


SCORES_FILE = "data/processed/escalation_scores.jsonl"


def load_scores():
    rows = []

    with open(SCORES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    return rows


def percentiles(values):
    values = np.array(values)

    return {
        "min": np.min(values),
        "p10": np.percentile(values, 10),
        "p25": np.percentile(values, 25),
        "median": np.percentile(values, 50),
        "p75": np.percentile(values, 75),
        "p90": np.percentile(values, 90),
        "max": np.max(values),
        "mean": np.mean(values),
    }


def print_stats(name, values):
    stats = percentiles(values)

    print(f"\n{name}")
    print("-" * 50)

    for key, value in stats.items():
        print(f"{key:>8}: {value:.4f}")


def main():
    rows = load_scores()

    intent_confidence = [
        row["intent_confidence"]
        for row in rows
    ]

    intent_score = [
        row["best_intent_score"]
        for row in rows
    ]

    intent_margin = [
        row["intent_margin"]
        for row in rows
    ]

    top_similarity = [
        row["top_similarity"]
        for row in rows
    ]

    print("=" * 60)
    print("ESCALATION SCORE DISTRIBUTION")
    print("=" * 60)

    print_stats(
        "Intent confidence",
        intent_confidence
    )

    print_stats(
        "Best intent similarity",
        intent_score
    )

    print_stats(
        "Intent margin",
        intent_margin
    )

    print_stats(
        "Top retrieval similarity",
        top_similarity
    )


if __name__ == "__main__":
    main()