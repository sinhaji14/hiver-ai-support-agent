import pandas as pd
from collections import Counter

FILE_PATH = "data/raw/twcs/twcs.csv"

brand_customer_counts = Counter()
brand_support_counts = Counter()

chunksize = 100_000

print("Scanning dataset...")

for chunk in pd.read_csv(FILE_PATH, chunksize=chunksize):
    # Customer tweets
    inbound = chunk[chunk["inbound"] == True]

    # Support/brand tweets
    outbound = chunk[chunk["inbound"] == False]

    brand_customer_counts.update(inbound["text"].dropna().apply(
        lambda x: x.split()[0].replace("@", "") if x.startswith("@") else ""
    ))

    brand_support_counts.update(outbound["author_id"].dropna())

print("\nTop brands by support responses:")
for brand, count in brand_support_counts.most_common(30):
    print(f"{brand:30} {count:,}")

print("\nTop customer-mentioned brands:")
for brand, count in brand_customer_counts.most_common(30):
    if brand:
        print(f"{brand:30} {count:,}")