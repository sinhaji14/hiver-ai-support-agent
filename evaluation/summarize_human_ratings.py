import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RATINGS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "human_generation_ratings.json"
)


def main():

    with open(
        RATINGS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        ratings = json.load(f)

    print("=" * 60)
    print("HUMAN GENERATION EVALUATION")
    print("=" * 60)

    total = len(ratings)

    metrics = [
        "helpfulness",
        "groundedness",
        "safety",
    ]

    for metric in metrics:

        scores = [
            row[metric]
            for row in ratings
        ]

        average = sum(scores) / len(scores)

        score_3 = sum(
            1
            for score in scores
            if score == 3
        )

        print(
            f"\n{metric.title()}"
        )

        print(
            f"Average score: "
            f"{average:.2f}/3"
        )

        print(
            f"Score of 3: "
            f"{score_3}/{total} "
            f"({score_3 / total:.2%})"
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    all_scores = []

    for row in ratings:

        all_scores.extend(
            [
                row["helpfulness"],
                row["groundedness"],
                row["safety"],
            ]
        )

    overall_average = (
        sum(all_scores)
        / len(all_scores)
    )

    print(
        "\nOverall average: "
        f"{overall_average:.2f}/3"
    )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()