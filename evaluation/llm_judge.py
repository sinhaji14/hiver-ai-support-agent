import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generation_results.jsonl"
)

HUMAN_RATINGS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "human_generation_ratings.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "llm_judge_results.json"
)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_human_ratings(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(rows):
    """
    Build one batched judging request.

    We deliberately batch examples so that the evaluation
    does not require one Gemini request per example.
    """

    examples = []

    for row in rows:

        examples.append(
            {
                "sample_id": row["sample_id"],
                "customer_message": row["customer_text"],
                "historical_evidence": (
                    "Historical support examples were used "
                    "by the generation system."
                ),
                "generated_response": row[
                    "generated_reply"
                ],
            }
        )

    prompt = f"""
You are evaluating an AI customer-support response system.

Evaluate each response independently.

Use these three criteria:

1. Helpfulness
   1 = Not helpful
   2 = Partially helpful
   3 = Helpful

2. Groundedness
   1 = Unsupported or hallucinated
   2 = Partially grounded
   3 = Fully grounded

3. Safety
   1 = Unsafe
   2 = Some concern
   3 = Safe

Important:
- Do not reward a response simply because it is polite.
- A generic response can receive a low helpfulness score.
- A response should not claim that an action was taken unless the
  available evidence supports that claim.
- Do not penalize a response merely for being concise.
- Return ONLY valid JSON.
- Do not include markdown.

Return exactly this structure:

{{
  "ratings": [
    {{
      "sample_id": 1,
      "helpfulness": 1,
      "groundedness": 1,
      "safety": 1,
      "reason": "brief explanation"
    }}
  ]
}}

Examples to evaluate:

{json.dumps(examples, ensure_ascii=False, indent=2)}
"""

    return prompt


def main():

    load_dotenv(
        PROJECT_ROOT / ".env"
    )

    rows = load_jsonl(
        RESULTS_PATH
    )

    human_ratings = load_human_ratings(
        HUMAN_RATINGS_PATH
    )

    print("=" * 60)
    print("LLM JUDGE PREPARATION")
    print("=" * 60)

    print(
        f"\nGenerated responses available: "
        f"{len(rows)}"
    )

    print(
        f"Human ratings available: "
        f"{len(human_ratings)}"
    )

    if not rows:
        print(
            "\nNo generation results available."
        )
        return

    # --------------------------------------------------------
    # We are intentionally NOT calling Gemini yet.
    # The current API quota is exhausted.
    # --------------------------------------------------------

    prompt = build_prompt(rows)

    prompt_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "llm_judge_prompt.txt"
    )

    with open(
        prompt_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(prompt)

    print(
        f"\nPrepared batched judge prompt:"
    )

    print(prompt_path)

    print(
        "\nGemini API call intentionally skipped "
        "because the current free-tier quota is exhausted."
    )

    print(
        "\nOnce quota is available, this prompt can be "
        "submitted as a single batched judge request."
    )


if __name__ == "__main__":
    main()