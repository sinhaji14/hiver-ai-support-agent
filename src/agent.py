import sys

from intent.hybrid import HybridIntentClassifier
from generation.generate_reply import (
    load_retrieval_index,
    retrieve_examples,
    generate_reply,
    sanitize_text,
)
from escalation.decide import decide_escalation


TOP_K = 5


def main():
    if len(sys.argv) < 2:
        print('Usage:\npython src/agent.py "customer message"')
        raise SystemExit(1)

    customer_message = sys.argv[1]

    print("=" * 60)
    print("HIVER AI SUPPORT AGENT")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Intent classification
    # --------------------------------------------------------

    print("\nLoading intent classifier...")
    classifier = HybridIntentClassifier()
    classifier.fit()

    print("\nClassifying customer message...")

    intent_result = classifier.predict(customer_message)

    intent = intent_result["intent"]
    confidence = intent_result["confidence"]
    margin = intent_result["margin"]

    print(f"\nIntent: {intent}")
    print(f"Intent confidence: {confidence:.3f}")
    print(f"Intent similarity: {intent_result['best_similarity']:.3f}")
    print(f"Intent margin: {margin:.3f}")

    # --------------------------------------------------------
    # 2. Historical retrieval
    # --------------------------------------------------------

    print("\nLoading historical retrieval index...")

    embeddings, metadata = load_retrieval_index()

    print(f"Loaded {len(metadata)} historical interactions.")

    print("\nRetrieving similar historical cases...")

    examples = retrieve_examples(
        customer_message,
        embeddings,
        metadata,
        classifier.model,
        top_k=TOP_K,
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Sanitize historical examples before displaying them.
    # --------------------------------------------------------

    print("\nTop historical matches:")

    for i, example in enumerate(examples, start=1):

        safe_customer_text = sanitize_text(
            example["customer_text"]
        )

        safe_agent_text = sanitize_text(
            example["agent_text"]
        )

        print(f"\n{i}. Similarity: {example['score']:.3f}")
        print(f"Customer: {safe_customer_text}")
        print(f"Agent: {safe_agent_text}")

    # --------------------------------------------------------
    # 3. Escalation decision
    # --------------------------------------------------------

    escalation = decide_escalation(
        intent,
        confidence,
        examples,
        intent_margin=margin,
    )

    print("\n" + "-" * 60)
    print("ESCALATION DECISION")
    print("-" * 60)

    print(f"Decision: {escalation['decision']}")
    print(f"Reason: {escalation['reason']}")

    # --------------------------------------------------------
    # 4. Generate response only for AUTO-HANDLE
    # --------------------------------------------------------

    if escalation["decision"] == "AUTO-HANDLE":

        print("\nGenerating grounded response...")

        reply = generate_reply(
            customer_message,
            examples
        )

        print("\n" + "=" * 60)
        print("GENERATED CUSTOMER RESPONSE")
        print("=" * 60)

        print(reply)

    else:

        print("\nNo automatic customer response was generated.")
        print("The case should be reviewed by a human support agent.")


if __name__ == "__main__":
    main()