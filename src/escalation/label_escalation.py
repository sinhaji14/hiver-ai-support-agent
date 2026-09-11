import json
import os


GOLDEN_FILE = "data/golden/golden_set.jsonl"


def load_jsonl(path):
    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            data.append(json.loads(line))

    return data


def save_jsonl(path, data):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for item in data:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                ) + "\n"
            )


def print_guidelines():

    print("\n" + "=" * 70)
    print("ESCALATION LABELING GUIDELINES")
    print("=" * 70)

    print("""
AUTO-HANDLE:
Use when the historical support response provides a clear,
low-risk answer or instruction that an AI agent can safely give.

Examples:
- Basic delivery/tracking information
- Simple general questions
- Straightforward technical guidance
- General order information where no sensitive action is required

HUMAN:
Use when the case requires account-specific investigation,
financial/security handling, unclear investigation, or an action
that the AI cannot safely perform.

Examples:
- Unauthorized charges or payment disputes
- Account security/fraud
- Cases requiring access to order/account information
- Complex unresolved complaints
- Situations where the historical response does not provide
  enough information for a safe answer

Choose based on whether an AI agent could safely handle the
customer's request using the historical response as evidence.
""")


def main():

    print_guidelines()

    data = load_jsonl(
        GOLDEN_FILE
    )

    print(
        f"\nLoaded {len(data)} golden examples."
    )

    # Existing labels are preserved.
    # We only add/update the escalation field.

    for item in data:

        if "escalation" not in item:
            item["escalation"] = ""

    labeled = sum(
        1
        for item in data
        if item.get("escalation") in {
            "AUTO-HANDLE",
            "HUMAN"
        }
    )

    print(
        f"Already labelled: "
        f"{labeled}/{len(data)}"
    )

    for index, item in enumerate(data):

        # Skip already-labelled examples
        if item.get("escalation") in {
            "AUTO-HANDLE",
            "HUMAN"
        }:
            continue

        print("\n" + "=" * 70)

        print(
            f"Example {index + 1}/{len(data)}"
        )

        print(
            f"Intent: "
            f"{item.get('intent', 'N/A')}"
        )

        print("\nCUSTOMER:")

        print(
            item.get(
                "customer_text",
                ""
            )
        )

        print("\nHISTORICAL SUPPORT RESPONSE:")

        print(
            item.get(
                "agent_text",
                ""
            )
        )

        print("\n" + "-" * 70)

        while True:

            choice = input(
                "\n1 = AUTO-HANDLE | "
                "2 = HUMAN | "
                "s = skip | "
                "q = save & quit: "
            ).strip().lower()

            if choice == "1":

                item["escalation"] = "AUTO-HANDLE"
                break

            elif choice == "2":

                item["escalation"] = "HUMAN"
                break

            elif choice == "s":

                print("Skipped.")
                break

            elif choice == "q":

                save_jsonl(
                    GOLDEN_FILE,
                    data
                )

                print(
                    f"\nProgress saved to "
                    f"{GOLDEN_FILE}"
                )

                return

            else:

                print(
                    "Invalid choice. "
                    "Use 1, 2, s, or q."
                )

        # Save after every label so progress isn't lost.
        save_jsonl(
            GOLDEN_FILE,
            data
        )

    print("\n" + "=" * 70)
    print("ESCALATION LABELING COMPLETE")
    print("=" * 70)

    auto_handle = sum(
        1
        for item in data
        if item.get("escalation") == "AUTO-HANDLE"
    )

    human = sum(
        1
        for item in data
        if item.get("escalation") == "HUMAN"
    )

    print(
        f"\nAUTO-HANDLE: {auto_handle}"
    )

    print(
        f"HUMAN: {human}"
    )

    print(
        f"Total labelled: "
        f"{auto_handle + human}"
    )


if __name__ == "__main__":
    main()