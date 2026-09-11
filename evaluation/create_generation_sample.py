import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GOLDEN_PATH = (
    PROJECT_ROOT
    / "data"
    / "golden"
    / "golden_set.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "generation_sample.jsonl"
)


SAMPLE_SIZE = 20
SEED = 42


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def main():

    random.seed(SEED)

    golden = load_jsonl(GOLDEN_PATH)

    # --------------------------------------------------------
    # We evaluate only cases labelled AUTO-HANDLE by humans.
    #
    # These are the cases where the system would actually
    # need to generate an automatic customer response.
    # --------------------------------------------------------

    auto_cases = [
        row
        for row in golden
        if row["escalation"] == "AUTO-HANDLE"
    ]

    print(f"Golden examples: {len(golden)}")
    print(f"Human-labelled AUTO-HANDLE cases: {len(auto_cases)}")

    if len(auto_cases) < SAMPLE_SIZE:
        raise ValueError(
            f"Not enough AUTO-HANDLE examples. "
            f"Need {SAMPLE_SIZE}, found {len(auto_cases)}."
        )

    sample = random.sample(
        auto_cases,
        SAMPLE_SIZE
    )

    # Keep ordering deterministic
    sample.sort(
        key=lambda x: str(x.get("interaction_id", ""))
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        for i, row in enumerate(sample, start=1):

            output = {
                "sample_id": i,
                "interaction_id": row.get("interaction_id"),
                "customer_text": row["customer_text"],
                "historical_agent_text": row.get(
                    "agent_text",
                    ""
                ),
                "intent": row.get("intent"),
                "human_escalation_label": row["escalation"],
            }

            f.write(
                json.dumps(
                    output,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("\n" + "=" * 60)
    print("GENERATION SAMPLE CREATED")
    print("=" * 60)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Examples: {len(sample)}")


if __name__ == "__main__":
    main()