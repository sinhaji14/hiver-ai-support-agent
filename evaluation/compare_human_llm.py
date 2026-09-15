import json
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


HUMAN_PATH = Path(
    "evaluation/human_generation_ratings_independent.json"
)
LLM_PATH = Path(
    "data/processed/llm_judge_results.json"
)


DIMENSIONS = [
    "helpfulness",
    "groundedness",
    "correctness",
    "safety",
]


def load_ratings(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return {
        item["sample_id"]: item
        for item in data["ratings"]
    }


def main():
    human = load_ratings(HUMAN_PATH)
    llm = load_ratings(LLM_PATH)

    sample_ids = sorted(set(human) & set(llm))

    print("=" * 60)
    print("HUMAN vs LLM JUDGE AGREEMENT")
    print("=" * 60)
    print(f"Samples compared: {len(sample_ids)}")
    print()

    for dimension in DIMENSIONS:
        human_scores = [
            human[sample_id][dimension]
            for sample_id in sample_ids
        ]

        llm_scores = [
            llm[sample_id][dimension]
            for sample_id in sample_ids
        ]

        exact_agreement = sum(
            h == l
            for h, l in zip(human_scores, llm_scores)
        )

        agreement_rate = exact_agreement / len(sample_ids)

        kappa = cohen_kappa_score(
            human_scores,
            llm_scores
        )

        print(
            f"{dimension.capitalize():15s}: "
            f"{exact_agreement}/{len(sample_ids)} "
            f"({agreement_rate:.1%}) | "
            f"Cohen's kappa: {kappa:.3f}"
        )

        disagreements = [
            sample_id
            for sample_id in sample_ids
            if human[sample_id][dimension]
            != llm[sample_id][dimension]
        ]

        if disagreements:
            print(
                f"  Disagreements: "
                f"{', '.join(map(str, disagreements))}"
            )

    print()
    print("Detailed disagreements:")

    found = False

    for sample_id in sample_ids:
        differences = []

        for dimension in DIMENSIONS:
            h = human[sample_id][dimension]
            l = llm[sample_id][dimension]

            if h != l:
                differences.append(
                    f"{dimension}: human={h}, llm={l}"
                )

        if differences:
            found = True
            print(
                f"Sample {sample_id}: "
                + "; ".join(differences)
            )

    if not found:
        print("None")


if __name__ == "__main__":
    main()