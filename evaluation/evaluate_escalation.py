import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCORES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "escalation_scores.jsonl"
)


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def main():

    print("=" * 60)
    print("ESCALATION EVALUATION")
    print("=" * 60)

    rows = load_jsonl(SCORES_PATH)

    print(f"\nTotal examples: {len(rows)}")

    # --------------------------------------------------------
    # Ground truth and predictions
    # --------------------------------------------------------

    y_true = [
        row["actual_escalation"]
        for row in rows
    ]

    y_pred = [
        row["predicted_escalation"]
        for row in rows
    ]

    labels = [
        "AUTO-HANDLE",
        "HUMAN",
    ]

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    print("\nConfusion Matrix")
    print("(rows = actual, columns = predicted)")
    print()
    print(f"Labels: {labels}")
    print(cm)

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy:.2%}")

    # --------------------------------------------------------
    # Safety metrics
    # --------------------------------------------------------

    false_auto_handling = sum(
        1
        for actual, predicted in zip(y_true, y_pred)
        if actual == "HUMAN"
        and predicted == "AUTO-HANDLE"
    )

    actual_human = sum(
        1
        for actual in y_true
        if actual == "HUMAN"
    )

    false_auto_rate = (
        false_auto_handling / actual_human
        if actual_human > 0
        else 0.0
    )

    predicted_auto = sum(
        1
        for predicted in y_pred
        if predicted == "AUTO-HANDLE"
    )

    total = len(y_pred)

    auto_coverage = (
        predicted_auto / total
        if total > 0
        else 0.0
    )

    predicted_human = sum(
        1
        for predicted in y_pred
        if predicted == "HUMAN"
    )

    human_rate = (
        predicted_human / total
        if total > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Print safety / coverage metrics
    # --------------------------------------------------------

    print("\nSafety / Coverage Metrics")
    print("-" * 60)

    print(
        f"False auto-handling: "
        f"{false_auto_handling}/{actual_human}"
    )

    print(
        f"False auto-handling rate: "
        f"{false_auto_rate:.4f}"
    )

    print(
        f"False auto-handling rate: "
        f"{false_auto_rate:.2%}"
    )

    print(
        f"\nPredicted AUTO-HANDLE: "
        f"{predicted_auto}/{total}"
    )

    print(
        f"Auto-handle coverage: "
        f"{auto_coverage:.4f}"
    )

    print(
        f"Auto-handle coverage: "
        f"{auto_coverage:.2%}"
    )

    print(
        f"\nPredicted HUMAN: "
        f"{predicted_human}/{total}"
    )

    print(
        f"Human escalation rate: "
        f"{human_rate:.4f}"
    )

    print(
        f"Human escalation rate: "
        f"{human_rate:.2%}"
    )


if __name__ == "__main__":
    main()