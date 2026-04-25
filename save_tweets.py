from collector_tweepy import fetch_tweets
import pandas as pd
import os
from datetime import datetime

# List all queries you want to fetch
queries = [
    "birthday lang:en -is:retweet",
    "holiday lang:en -is:retweet",
    # Add more queries here
]

# Master CSV file
master_csv = "all_tweets.csv"

# Load existing CSV to avoid duplicates
if os.path.exists(master_csv):
    master_df = pd.read_csv(master_csv, encoding="utf-8-sig")
else:
    master_df = pd.DataFrame()  # Empty DataFrame for first run

for q in queries:
    # Fetch tweets
    df = fetch_tweets(q, limit=100, since_minutes=60)
    df['query'] = q  # Track which query
    df['fetched_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp

    # Avoid duplicates based on tweet id
    if not master_df.empty:
        df = df[~df['id'].isin(master_df['id'])]

    # Append to master DataFrame
    master_df = pd.concat([master_df, df], ignore_index=True)

# Save/overwrite the master CSV
master_df.to_csv(master_csv, index=False, encoding="utf-8-sig")
print(f"Saved/updated master CSV with {len(master_df)} total tweets.")
