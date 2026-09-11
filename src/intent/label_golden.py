import json

INPUT_PATH = "data/golden/golden_set.jsonl"
TAXONOMY_PATH = "src/intent/taxonomy.json"

# Load taxonomy
with open(TAXONOMY_PATH, "r", encoding="utf-8") as f:
    taxonomy = json.load(f)

# Load golden set
examples = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        examples.append(json.loads(line))

print("=" * 70)
print("AMAZONHELP GOLDEN SET LABELING")
print("=" * 70)

print("\nINTENTS:\n")

for i, intent in enumerate(taxonomy, 1):
    print(f"{i}. {intent['intent_name']}")
    print(f"   {intent['description']}")

print("\nCommands:")
print("  Enter 1-10 = assign intent")
print("  s = skip")
print("  q = quit and save progress")
print("=" * 70)

for i, example in enumerate(examples):

    # Skip already-labelled examples
    if example.get("intent"):
        continue

    print("\n" + "-" * 70)
    print(f"Example {i + 1}/{len(examples)}")
    print("-" * 70)

    print("\nCUSTOMER:")
    print(example["customer_text"])

    print("\nHISTORICAL AGENT RESPONSE:")
    print(example["agent_text"])

    while True:

        choice = input("\nIntent (1-10 / s / q): ").strip().lower()

        if choice == "q":
            with open(INPUT_PATH, "w", encoding="utf-8") as f:
                for item in examples:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print("\nProgress saved.")
            print(f"Labelled: {sum(bool(x.get('intent')) for x in examples)}/{len(examples)}")
            raise SystemExit

        if choice == "s":
            break

        if choice.isdigit() and 1 <= int(choice) <= len(taxonomy):
            intent = taxonomy[int(choice) - 1]

            example["intent"] = intent["intent_id"]

            note = input("Optional note (Enter to skip): ").strip()
            example["notes"] = note

            break

        print("Please enter a number from 1-10, 's', or 'q'.")

# Save
with open(INPUT_PATH, "w", encoding="utf-8") as f:
    for item in examples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

labelled = sum(bool(x.get("intent")) for x in examples)

print("\n" + "=" * 70)
print("LABELING COMPLETE")
print("=" * 70)
print(f"Labelled: {labelled}/{len(examples)}")
print(f"Saved to: {INPUT_PATH}")