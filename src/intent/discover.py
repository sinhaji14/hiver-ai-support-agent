import json
import os
import time
from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# Configuration
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

INPUT_PATH = "data/processed/intent_sample.jsonl"
OUTPUT_PATH = "data/processed/intent_candidates.json"

BATCH_SIZE = 50


# --------------------------------------------------
# Load customer messages
# --------------------------------------------------

print("Loading intent sample...")

interactions = []

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    for line in f:
        interactions.append(json.loads(line))

print(f"Loaded {len(interactions):,} interactions.")


# --------------------------------------------------
# Analyze one batch
# --------------------------------------------------

def analyze_batch(messages):

    numbered_messages = []

    for i, message in enumerate(messages, 1):
        text = message["customer_text"].replace("\n", " ")
        numbered_messages.append(f"{i}. {text}")

    joined = "\n".join(numbered_messages)

    prompt = f"""
You are analyzing real Amazon customer-support conversations.

Below are customer messages from AmazonHelp support interactions.

Your task is to identify the recurring CUSTOMER PROBLEMS in this batch.

Do NOT create overly specific categories.

Group messages that represent the same underlying support problem.

For each category provide:

1. category_name
2. description
3. example_message_numbers
4. approximate_frequency

Aim for 5–10 categories for this batch.

Do not invent information that isn't present in the messages.

Customer messages:

{joined}

Return ONLY valid JSON in this format:

[
  {{
    "category_name": "...",
    "description": "...",
    "example_message_numbers": [1, 4, 10],
    "approximate_frequency": "high"
  }}
]
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    text = response.text.strip()

    # Remove markdown fences if Gemini adds them
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    return json.loads(text)


# --------------------------------------------------
# Process batches
# --------------------------------------------------

all_candidates = []

for start in range(0, len(interactions), BATCH_SIZE):

    batch = interactions[start:start + BATCH_SIZE]

    print(
        f"Processing batch "
        f"{start // BATCH_SIZE + 1}/"
        f"{(len(interactions) + BATCH_SIZE - 1) // BATCH_SIZE}"
    )

    try:

        result = analyze_batch(batch)

        all_candidates.extend(result)

    except Exception as e:

        print(f"ERROR: {e}")

    # Avoid hitting rate limits
    time.sleep(1)


# --------------------------------------------------
# Save candidates
# --------------------------------------------------

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_candidates,
        f,
        ensure_ascii=False,
        indent=2
    )

print()
print("=" * 60)
print("INTENT DISCOVERY COMPLETE")
print("=" * 60)
print(f"Candidate categories: {len(all_candidates):,}")
print(f"Saved to: {OUTPUT_PATH}")