import pandas as pd

FILE_PATH = "data/raw/twcs/twcs.csv"
BRAND = "AmazonHelp"
CHUNKSIZE = 100_000

# --------------------------------------------------
# Pass 1: Find AmazonHelp support tweets
# --------------------------------------------------

print("Pass 1: Finding AmazonHelp support tweets...")

amazon_tweets = []

for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNKSIZE):

    mask = (
        (chunk["inbound"] == False)
        & (chunk["author_id"] == BRAND)
    )

    selected = chunk.loc[
        mask,
        [
            "tweet_id",
            "author_id",
            "inbound",
            "created_at",
            "text",
            "in_response_to_tweet_id",
            "response_tweet_id",
        ],
    ]

    amazon_tweets.append(selected)

amazon = pd.concat(amazon_tweets, ignore_index=True)

print(f"AmazonHelp support tweets: {len(amazon):,}")


# --------------------------------------------------
# Pass 2: Find the customer tweets they responded to
# --------------------------------------------------

parent_ids = set(
    amazon["in_response_to_tweet_id"]
    .dropna()
    .astype(int)
)

print(f"Unique parent tweet IDs: {len(parent_ids):,}")

customer_parents = []

print("\nPass 2: Finding customer parent tweets...")

for chunk in pd.read_csv(FILE_PATH, chunksize=CHUNKSIZE):

    selected = chunk[
        chunk["tweet_id"].isin(parent_ids)
        & (chunk["inbound"] == True)
    ]

    if len(selected) > 0:
        customer_parents.append(
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
    customer_parents,
    ignore_index=True
)


# --------------------------------------------------
# Statistics
# --------------------------------------------------

print("\n" + "=" * 65)
print("AMAZONHELP CONVERSATION ANALYSIS")
print("=" * 65)

print(f"AmazonHelp support tweets:       {len(amazon):,}")
print(f"Support tweets with parent ID:   {amazon['in_response_to_tweet_id'].notna().sum():,}")
print(f"Matched customer parent tweets:  {len(customers):,}")

if len(amazon) > 0:
    response_rate = (
        amazon["in_response_to_tweet_id"].notna().mean() * 100
    )

    print(f"Support tweets with parent:      {response_rate:.2f}%")

print("=" * 65)


# --------------------------------------------------
# Show real examples
# --------------------------------------------------

merged = amazon.merge(
    customers,
    left_on="in_response_to_tweet_id",
    right_on="tweet_id",
    suffixes=("_support", "_customer"),
)

print("\nREAL CUSTOMER → AMAZONHELP EXAMPLES\n")

for _, row in merged.head(10).iterrows():

    print("-" * 65)

    print("CUSTOMER:")
    print(row["text_customer"])

    print("\nAMAZONHELP:")
    print(row["text_support"])

print("-" * 65)