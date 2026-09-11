import json
import sys
from pathlib import Path

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intent.hybrid import HybridIntentClassifier
from generation.generate_reply import (
    load_retrieval_index,
    retrieve_examples,
)
from escalation.decide import decide_escalation


GOLDEN_PATH = PROJECT_ROOT / "data" / "golden" / "golden_set.jsonl"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escalation_scores.jsonl"
)

TOP_K = 5


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def main():

    print("Loading golden set...")

    golden = load_jsonl(GOLDEN_PATH)

    print(f"Golden examples: {len(golden)}")

    # --------------------------------------------------------
    # Load production intent classifier
    # --------------------------------------------------------

    print("\nLoading hybrid intent classifier...")

    classifier = HybridIntentClassifier()
    classifier.fit()

    # --------------------------------------------------------
    # Load leakage-safe retrieval index
    # --------------------------------------------------------

    print("\nLoading leakage-safe retrieval index...")

    embeddings, metadata = load_retrieval_index()

    print(f"Historical interactions: {len(metadata)}")

    # --------------------------------------------------------
    # Score every golden example
    # --------------------------------------------------------

    results = []

    for i, item in enumerate(golden, start=1):

        customer_text = item["customer_text"]
        actual_escalation = item["escalation"]

        # Intent prediction
        intent_result = classifier.predict(customer_text)

        intent = intent_result["intent"]
        intent_confidence = intent_result["confidence"]
        intent_margin = intent_result["margin"]

        # Historical retrieval
        examples = retrieve_examples(
            customer_text,
            embeddings,
            metadata,
            classifier.model,
            top_k=TOP_K,
        )

        # Production escalation policy
        decision = decide_escalation(
            intent=intent,
            intent_confidence=intent_confidence,
            retrieved_examples=examples,
            intent_margin=intent_margin,
        )

        predicted_escalation = decision["decision"]

        top_similarity = (
            examples[0]["score"]
            if examples
            else 0.0
        )

        results.append(
            {
                "interaction_id": item.get("interaction_id"),
                "customer_text": customer_text,
                "actual_intent": item.get("intent"),
                "predicted_intent": intent,
                "intent_confidence": intent_confidence,
                "intent_similarity": intent_result["best_similarity"],
                "intent_margin": intent_margin,
                "top_retrieval_similarity": top_similarity,
                "actual_escalation": actual_escalation,
                "predicted_escalation": predicted_escalation,
                "escalation_reason": decision["reason"],
            }
        )

        if i % 10 == 0:
            print(f"Processed {i}/{len(golden)}")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for row in results:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Examples scored: {len(results)}")


if __name__ == "__main__":
    main()