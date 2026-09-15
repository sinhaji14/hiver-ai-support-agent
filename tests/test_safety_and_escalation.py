from escalation.decide import decide_escalation
from generation.generate_reply import sanitize_text


def test_no_retrieval_escalates_to_human():
    result = decide_escalation(
        intent="delivery_delay_tracking",
        intent_confidence=0.9,
        retrieved_examples=[],
        intent_margin=0.2,
    )

    assert result["decision"] == "HUMAN"


def test_weak_retrieval_escalates_to_human():
    result = decide_escalation(
        intent="delivery_delay_tracking",
        intent_confidence=0.9,
        retrieved_examples=[{"score": 0.70}],
        intent_margin=0.20,
    )

    assert result["decision"] == "HUMAN"


def test_ambiguous_intent_escalates_to_human():
    result = decide_escalation(
        intent="delivery_delay_tracking",
        intent_confidence=0.6,
        retrieved_examples=[{"score": 0.90}],
        intent_margin=0.03,
    )

    assert result["decision"] == "HUMAN"


def test_high_risk_intent_escalates_to_human():
    result = decide_escalation(
        intent="billing_payment_charges",
        intent_confidence=0.9,
        retrieved_examples=[{"score": 0.95}],
        intent_margin=0.20,
    )

    assert result["decision"] == "HUMAN"


def test_strong_low_risk_case_can_auto_handle():
    result = decide_escalation(
        intent="technical_digital_services",
        intent_confidence=0.9,
        retrieved_examples=[{"score": 0.90}],
        intent_margin=0.20,
    )

    assert result["decision"] == "AUTO-HANDLE"


def test_sanitize_removes_sensitive_information():
    text = (
        "Please visit https://example.com/order "
        "or email customer@example.com. "
        "Order 404-6800877-5877920 "
        "and call +1 555-123-4567 @AmazonHelp."
    )

    sanitized = sanitize_text(text)

    assert "https://" not in sanitized
    assert "customer@example.com" not in sanitized
    assert "404-6800877-5877920" not in sanitized
    assert "+1 555-123-4567" not in sanitized
    assert "@AmazonHelp" not in sanitized

    assert "[URL REMOVED]" in sanitized
    assert "[EMAIL REMOVED]" in sanitized
    assert "[ORDER NUMBER REMOVED]" in sanitized
    assert "[PHONE NUMBER REMOVED]" in sanitized
    assert "@customer" in sanitized
