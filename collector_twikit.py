import asyncio
import pandas as pd
from pathlib import Path
from twikit import Client

# ---------- CONFIG ----------
AUTH_TOKEN = "47b9f56cb1cdd6709ad9c85da1c54a5ba84d6c58"
CT0_TOKEN  = "24f18905f5bcd0a2f40cc0e793b3baf8368d7b17a4acbe0860dac208c1feecb127bfb7711405c41492cf758591b190122c50a4de90dbe4c770bc61efe55983518b74f87b09286b5a6f3d6cc13d83c647"
CSV_PATH   = Path("all_tweets.csv")
# ----------------------------

async def _fetch_async(query: str, limit: int = 100) -> pd.DataFrame:
    """Internal async call to Twikit to fetch tweets."""
    client = Client('en-US')
    client.set_cookies({'auth_token': AUTH_TOKEN, 'ct0': CT0_TOKEN})

    # Twikit search returns a list directly (not an async iterator)
    results = await client.search_tweet(query, 'Latest', count=limit)

    rows = []
    for t in results:
        rows.append({
            "id":       t.id,
            "date":     t.created_at,
            "username": t.user.screen_name,
            "content":  t.text,
            "lang":     getattr(t, "lang", ""),
            "url":      f"https://twitter.com/{t.user.screen_name}/status/{t.id}"
        })
    return pd.DataFrame(rows)

def fetch_tweets_twikit(query: str, limit: int = 100) -> pd.DataFrame:
    """
    Public function to fetch tweets and append to all_tweets.csv.
    Safe to call from Streamlit (runs the async code internally).
    """
    df_new = asyncio.run(_fetch_async(query, limit))

    if df_new.empty:
        return df_new

    if CSV_PATH.exists():
        df_old = pd.read_csv(CSV_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    # save combined data
    df_all.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    return df_new
