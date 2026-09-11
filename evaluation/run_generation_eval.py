import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from intent.hybrid import HybridIntentClassifier
from generation.generate_reply import (
    load_retrieval_index,
    retrieve_examples,
    generate_reply,
)


INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generation_sample.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generation_results.jsonl"
)

TOP_K = 5


def load_jsonl(path):

    rows = []

    if not path.exists():
        return rows

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def save_jsonl(path, rows):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


def main():

    samples = load_jsonl(INPUT_PATH)

    print(
        f"Generation examples: "
        f"{len(samples)}"
    )

    # --------------------------------------------------------
    # Load existing results so the script can resume
    # --------------------------------------------------------

    results = load_jsonl(OUTPUT_PATH)

    completed_ids = {
        row["sample_id"]
        for row in results
    }

    if results:
        print(
            f"Already completed: "
            f"{len(results)}"
        )

    remaining = [
        sample
        for sample in samples
        if sample["sample_id"]
        not in completed_ids
    ]

    print(
        f"Remaining: "
        f"{len(remaining)}"
    )

    if not remaining:
        print("\nAll examples already completed.")
        return

    # --------------------------------------------------------
    # Load production classifier
    # --------------------------------------------------------

    print(
        "\nLoading hybrid intent classifier..."
    )

    classifier = HybridIntentClassifier()
    classifier.fit()

    # --------------------------------------------------------
    # Load leakage-safe retrieval index
    # --------------------------------------------------------

    print(
        "\nLoading leakage-safe retrieval index..."
    )

    embeddings, metadata = load_retrieval_index()

    print(
        f"Historical interactions: "
        f"{len(metadata)}"
    )

    # --------------------------------------------------------
    # Process remaining examples
    # --------------------------------------------------------

    for sample in remaining:

        sample_id = sample["sample_id"]

        print(
            f"\nGenerating sample "
            f"{sample_id}/{len(samples)}..."
        )

        customer_text = sample[
            "customer_text"
        ]

        try:

            # ------------------------------------------------
            # Intent
            # ------------------------------------------------

            intent_result = classifier.predict(
                customer_text
            )

            # ------------------------------------------------
            # Retrieval
            # ------------------------------------------------

            examples = retrieve_examples(
                customer_text,
                embeddings,
                metadata,
                classifier.model,
                top_k=TOP_K,
            )

            # ------------------------------------------------
            # Generation
            # ------------------------------------------------

            reply = generate_reply(
                customer_text,
                examples
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            result = {
                "sample_id": sample_id,
                "interaction_id": sample[
                    "interaction_id"
                ],
                "customer_text": customer_text,
                "human_escalation_label": sample[
                    "human_escalation_label"
                ],
                "gold_intent": sample.get(
                    "intent"
                ),
                "predicted_intent": intent_result[
                    "intent"
                ],
                "intent_margin": intent_result[
                    "margin"
                ],
                "top_retrieval_similarity": (
                    examples[0]["score"]
                    if examples
                    else 0.0
                ),
                "generated_reply": reply,
            }

            results.append(result)

            # ------------------------------------------------
            # SAVE IMMEDIATELY
            # ------------------------------------------------

            save_jsonl(
                OUTPUT_PATH,
                results
            )

            print(
                f"Saved successfully. "
                f"Completed: {len(results)}/"
                f"{len(samples)}"
            )

            # ------------------------------------------------
            # Small delay to avoid hitting the RPM limit
            # ------------------------------------------------

            time.sleep(15)

        except Exception as e:

            error_message = str(e)

            print(
                "\nGeneration failed."
            )

            print(
                f"Error: {error_message}"
            )

            # Save everything completed so far
            save_jsonl(
                OUTPUT_PATH,
                results
            )

            print(
                f"\nProgress saved: "
                f"{len(results)}/{len(samples)}"
            )

            print(
                "\nStop here and rerun the same "
                "command after waiting."
            )

            return

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Saved: {OUTPUT_PATH}"
    )

    print(
        f"Examples: {len(results)}"
    )


if __name__ == "__main__":
    main()