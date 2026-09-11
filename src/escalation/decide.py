HIGH_RISK_INTENTS = {
    "billing_payment_charges",
    "account_access_security",
    "missing_misdelivered_package",
    "returns_refunds_replacements",
}


def decide_escalation(
    intent,
    intent_confidence,
    retrieved_examples,
    intent_margin=None,
):
    """
    Decide whether a customer request can be handled automatically.

    The policy is intentionally conservative:
    - weak retrieval -> HUMAN
    - ambiguous intent -> HUMAN
    - high-risk intents -> HUMAN
    - otherwise require strong retrieval and intent separation
    """

    if not retrieved_examples:
        return {
            "decision": "HUMAN",
            "reason": (
                "No relevant historical support examples were found."
            ),
        }

    top_similarity = retrieved_examples[0]["score"]

    if intent_margin is None:
        intent_margin = intent_confidence

    # --------------------------------------------------------
    # 1. Weak historical evidence
    # --------------------------------------------------------

    if top_similarity < 0.75:
        return {
            "decision": "HUMAN",
            "reason": (
                f"Weak historical match (similarity "
                f"{top_similarity:.2f}); there is not enough "
                f"precedent for safe automatic handling."
            ),
        }

    # --------------------------------------------------------
    # 2. Ambiguous intent
    # --------------------------------------------------------

    if intent_margin < 0.05:
        return {
            "decision": "HUMAN",
            "reason": (
                f"Ambiguous intent classification "
                f"(margin {intent_margin:.3f}); the system is "
                f"not sufficiently certain about the customer's issue."
            ),
        }

    # --------------------------------------------------------
    # 3. High-risk intents
    # --------------------------------------------------------

    if intent in HIGH_RISK_INTENTS:
        return {
            "decision": "HUMAN",
            "reason": (
                f"{intent} is treated as high risk because it "
                f"may require account-specific, financial, security, "
                f"delivery-investigation, or return/refund actions."
            ),
        }

    # --------------------------------------------------------
    # 4. Require stronger evidence for automation
    # --------------------------------------------------------

    if top_similarity < 0.82:
        return {
            "decision": "HUMAN",
            "reason": (
                f"Historical similarity ({top_similarity:.2f}) "
                f"is not strong enough for automatic handling."
            ),
        }

    if intent_margin < 0.08:
        return {
            "decision": "HUMAN",
            "reason": (
                f"Intent margin ({intent_margin:.3f}) is too small "
                f"for safe automatic handling."
            ),
        }

    # --------------------------------------------------------
    # 5. AUTO-HANDLE
    # --------------------------------------------------------

    return {
        "decision": "AUTO-HANDLE",
        "reason": (
            f"Strong historical match ({top_similarity:.2f}) "
            f"and clearly separated intent classification "
            f"(margin {intent_margin:.3f})."
        ),
    }