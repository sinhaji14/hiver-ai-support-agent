import json
from pathlib import Path

RESULTS_PATH = Path("data/processed/generation_results.jsonl")
OUTPUT_PATH = Path("evaluation/human_generation_ratings_independent.json")


def load_results():
    results = []

    with open(RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))

    return results


def get_score(prompt):
    while True:
        value = input(prompt).strip()

        if value in {"1", "2", "3"}:
            return int(value)

        print("Please enter 1, 2, or 3.")


def main():
    results = load_results()

    print("=" * 70)
    print("INDEPENDENT HUMAN EVALUATION")
    print("=" * 70)
    print()
    print("Rate each generated response independently.")
    print()
    print("1 = Poor")
    print("2 = Partially acceptable")
    print("3 = Good")
    print()
    print("Dimensions:")
    print("  Helpfulness  - Does the response help resolve the customer's issue?")
    print("  Groundedness - Is the response supported by the historical evidence?")
    print("  Correctness  - Is the response appropriate/correct for the request?")
    print("  Safety       - Does it avoid unsafe or unsupported actions/claims?")
    print()
    print("Your ratings are independent human ratings and will be compared")
    print("against the LLM judge.")
    print()

    ratings = []

    for i, item in enumerate(results, start=1):
        sample_id = item["sample_id"]

        print("\n" + "=" * 70)
        print(f"SAMPLE {i}/{len(results)} — ID: {sample_id}")
        print("=" * 70)

        print("\nCUSTOMER:")
        print(item["customer_text"])

        print("\nGENERATED RESPONSE:")
        print(item["generated_reply"])

        print("\n--- RATE THIS RESPONSE ---")

        helpfulness = get_score("Helpfulness (1-3): ")
        groundedness = get_score("Groundedness (1-3): ")
        correctness = get_score("Correctness (1-3): ")
        safety = get_score("Safety (1-3): ")

        notes = input("Optional notes (press Enter to skip): ").strip()

        ratings.append({
            "sample_id": sample_id,
            "helpfulness": helpfulness,
            "groundedness": groundedness,
            "correctness": correctness,
            "safety": safety,
            "notes": notes
        })

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "description": (
                        "Independent human ratings of generated responses. "
                        "These ratings are collected separately from the "
                        "LLM-as-a-judge evaluation."
                    ),
                    "ratings": ratings
                },
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"\nSaved progress: {len(ratings)}/{len(results)}")

    print("\n" + "=" * 70)
    print("HUMAN EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Ratings saved to: {OUTPUT_PATH}")
    print(f"Total ratings: {len(ratings)}")


if __name__ == "__main__":
    main()