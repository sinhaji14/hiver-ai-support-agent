import pandas as pd
import json
from collections import defaultdict

FILE_PATH = "data/raw/twcs/twcs.csv"
OUTPUT_PATH = "data/processed/amazon_conversations.jsonl"

BRAND = "AmazonHelp"
CHUNKSIZE = 100_000

print("Loading AmazonHelp tweets...")

chunks = []

for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNKSIZE):
    mask = (
        (chunk["author_id"] == BRAND)
        | (
            (chunk["inbound"] == True)
            & chunk["text"].fillna("").str.contains(
                "@AmazonHelp",
                case=False,
                regex=False
            )
        )
    )

    selected = chunk.loc[
        mask,
        [
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "response_tweet_id",
            "in_response_to_tweet_id",
        ],
    ]

    chunks.append(selected)

df = pd.concat(chunks, ignore_index=True)

print(f"Loaded {len(df):,} relevant tweets.")

# --------------------------------------------------
# Create lookup by tweet ID
# --------------------------------------------------

tweets = {}

for _, row in df.iterrows():
    tweets[int(row["tweet_id"])] = {
        "tweet_id": int(row["tweet_id"]),
        "author_id": row["author_id"],
        "inbound": bool(row["inbound"]),
        "created_at": row["created_at"],
        "text": str(row["text"]),
        "parent": (
            int(row["in_response_to_tweet_id"])
            if pd.notna(row["in_response_to_tweet_id"])
            else None
        ),
    }

# --------------------------------------------------
# Find conversation root for each tweet
# --------------------------------------------------

def find_root(tweet_id):
    visited = set()
    current = tweet_id

    while current in tweets:
        if current in visited:
            break

        visited.add(current)

        parent = tweets[current]["parent"]

        if parent is None or parent not in tweets:
            return current

        current = parent

    return current


# --------------------------------------------------
# Group tweets by conversation root
# --------------------------------------------------

conversations = defaultdict(list)

print("Reconstructing conversations...")

for tweet_id in tweets:
    root = find_root(tweet_id)
    conversations[root].append(tweets[tweet_id])


# --------------------------------------------------
# Sort messages chronologically
# --------------------------------------------------

clean_conversations = []

for root, messages in conversations.items():

    messages.sort(key=lambda x: x["created_at"])

    # Keep only conversations containing both
    # customer and AmazonHelp messages.
    has_customer = any(m["inbound"] for m in messages)
    has_agent = any(
        not m["inbound"] and m["author_id"] == BRAND
        for m in messages
    )

    if not (has_customer and has_agent):
        continue

    clean_conversations.append({
        "conversation_id": root,
        "messages": messages,
    })


# --------------------------------------------------
# Save JSONL
# --------------------------------------------------

print(f"Usable conversations: {len(clean_conversations):,}")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for conversation in clean_conversations:
        f.write(
            json.dumps(
                conversation,
                ensure_ascii=False
            )
            + "\n"
        )

print(f"Saved to: {OUTPUT_PATH}")