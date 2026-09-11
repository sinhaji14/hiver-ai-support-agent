import pandas as pd
from collections import defaultdict

FILE_PATH = "data/raw/twcs/twcs.csv"

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
]

stats = {
    brand: {
        "inbound": 0,
        "outbound": 0,
        "customers": set(),
        "tweets": 0,
    }
    for brand in BRANDS
}

chunksize = 100_000

print("Analyzing candidate brands...\n")

for chunk in pd.read_csv(FILE_PATH, chunksize=chunksize):

    # Support tweets from these brands
    outbound = chunk[
        (chunk["inbound"] == False)
        & (chunk["author_id"].isin(BRANDS))
    ]

    for brand in BRANDS:
        brand_outbound = outbound[outbound["author_id"] == brand]

        stats[brand]["outbound"] += len(brand_outbound)

        # Customer tweets mentioning the brand
        mask = (
            (chunk["inbound"] == True)
            & chunk["text"].fillna("").str.contains(
                "@" + brand,
                case=False,
                regex=False
            )
        )

        brand_inbound = chunk[mask]

        stats[brand]["inbound"] += len(brand_inbound)

        stats[brand]["customers"].update(
            brand_inbound["author_id"].dropna().tolist()
        )

for brand in BRANDS:
    stats[brand]["tweets"] = (
        stats[brand]["inbound"]
        + stats[brand]["outbound"]
    )

print("=" * 75)
print(f"{'Brand':20} {'Customer':>12} {'Support':>12} {'Customers':>12}")
print("=" * 75)

for brand in BRANDS:
    s = stats[brand]

    print(
        f"{brand:20} "
        f"{s['inbound']:>12,} "
        f"{s['outbound']:>12,} "
        f"{len(s['customers']):>12,}"
    )

print("=" * 75)