import pandas as pd
import json

FILE_PATH = "data/raw/twcs/twcs.csv"
OUTPUT_PATH = "data/processed/amazon_interactions.jsonl"

BRAND = "AmazonHelp"
CHUNKSIZE = 100_000

print("Loading AmazonHelp support tweets...")

support_chunks = []

for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNKSIZE):

    mask = (
        (chunk["inbound"] == False)
        & (chunk["author_id"] == BRAND)
        & chunk["in_response_to_tweet_id"].notna()
    )

    support_chunks.append(
        chunk.loc[
            mask,
            [
                "tweet_id",
                "author_id",
                "created_at",
                "text",
                "in_response_to_tweet_id",
            ],
        ]
    )

support = pd.concat(support_chunks, ignore_index=True)

print(f"AmazonHelp replies: {len(support):,}")

# IDs of customer tweets that AmazonHelp replied to
parent_ids = set(
    support["in_response_to_tweet_id"]
    .astype(int)
)

print(f"Customer parent IDs: {len(parent_ids):,}")

# Find those customer tweets
customer_chunks = []

print("Finding customer messages...")

for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNKSIZE):

    selected = chunk[
        chunk["tweet_id"].isin(parent_ids)
        & (chunk["inbound"] == True)
    ]

    if len(selected) > 0:
        customer_chunks.append(
            selected[
                [
                    "tweet_id",
                    "author_id",
                    "created_at",
                    "text",
                ]
            ]
        )

customers = pd.concat(
    customer_chunks,
    ignore_index=True
)

print(f"Matched customer messages: {len(customers):,}")

# Create lookup
customer_lookup = {
    int(row["tweet_id"]): row
    for _, row in customers.iterrows()
}

# Create clean interaction pairs
interactions = []

for _, row in support.iterrows():

    parent_id = int(row["in_response_to_tweet_id"])

    customer = customer_lookup.get(parent_id)

    if customer is None:
        continue

    interactions.append({
        "customer_tweet_id": parent_id,
        "customer_author_id": customer["author_id"],
        "customer_created_at": customer["created_at"],
        "customer_text": customer["text"],

        "agent_tweet_id": int(row["tweet_id"]),
        "agent_created_at": row["created_at"],
        "agent_text": row["text"],
    })

print(f"Clean interactions: {len(interactions):,}")

# Save
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:

    for item in interactions:
        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            ) + "\n"
        )

print(f"Saved to: {OUTPUT_PATH}")