import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import logging
from transformers import pipeline

# local modules
from preprocessor import preprocess_dataframe
from collector_twikit import fetch_tweets_twikit

# Logging
logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("app.log"), logging.StreamHandler()])
logger = logging.getLogger("realtime_sentiment_app")

st.set_page_config(page_title="Real-Time Sentiment Dashboard", layout="wide")
st.title("Real-Time Sentiment Analysis of Tweets")

# Simple auth
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
app_password = st.secrets.get("app_password", None) or "demo"
if not st.session_state["authenticated"]:
    st.sidebar.header("🔒 Login")
    pw = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if pw == app_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.sidebar.error("Wrong password")
    st.warning("Please login")
    st.stop()

# Sidebar controls
st.sidebar.header("Controls")
DATA_SOURCE = st.sidebar.selectbox("Data source", ["Demo CSV / Sample", "Live (Twitter via Twikit)"])
query = st.sidebar.text_input("Query / Hashtag", value="#AsianGames")
limit = st.sidebar.slider("Max tweets to fetch", min_value=5, max_value=200, value=20)
since_minutes = st.sidebar.slider("Look back (minutes) — Demo only", min_value=15, max_value=7*24*60, value=180, step=15)
auto_fetch = st.sidebar.checkbox("Auto fetch at load (only for Live)", value=False)

candidate_files = ["sample_tweets.csv", "all_tweets.csv"] + [p.name for p in Path(".").glob("*_tweets.csv")]
candidate_files = list(dict.fromkeys(candidate_files))
sample_file = st.sidebar.selectbox("Choose demo CSV file", candidate_files)

# Helpers
def load_demo_df(sample_filename: str, n: int, q: str = "") -> pd.DataFrame:
    p = Path(sample_filename)
    if not p.exists():
        alt = Path("data") / sample_filename
        p = alt if alt.exists() else None
    if p is None or not p.exists():
        now = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        return pd.DataFrame({
            "id": [1,2,3],
            "date": [now, now, now],
            "username": ["alice","bob","carol"],
            "content": ["I love this! 😍", "Not good... :(", "Time to celebrate 🎉"],
            "lang": ["en","en","en"],
            "url": ["#","#","#"]
        }).head(n)
    df = pd.read_csv(p)
    expected = ["id", "date", "username", "content", "lang", "url"]
    for c in expected:
        if c not in df.columns:
            df[c] = ""
    if q:
        qlow = q.lower()
        mask = df["content"].astype(str).str.lower().str.contains(qlow, na=False) | df["username"].astype(str).str.lower().str.contains(qlow, na=False)
        df = df[mask]
    return df[expected].head(n)

def fetch_live_tweets_twikit(q: str, limit_val: int) -> pd.DataFrame:
    try:
        with st.spinner("Fetching live tweets (Twikit)…"):
            df = fetch_tweets_twikit(q, limit_val)
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame(columns=["id","date","username","content","lang","url"])
        expected = ["id","date","username","content","lang","url"]
        for c in expected:
            if c not in df.columns:
                df[c] = ""
        return df[expected].reset_index(drop=True)
    except Exception as e:
        st.error(f"Error calling Twikit: {e}")
        return pd.DataFrame(columns=["id","date","username","content","lang","url"])

def style_sentiment_col(val):
    if val == "Positive": return "background-color: #d4f7dc; color: #0b6623"
    if val == "Negative": return "background-color: #ffd6d6; color: #8b0000"
    if val == "Neutral":  return "background-color: #fff7cc; color: #7a5c00"
    return ""

# Session state / fetch logic
current_query_hash = f"{DATA_SOURCE}|{query}|{limit}|{since_minutes}|{sample_file}"
last_query = st.session_state.get("last_query", None)
do_fetch = False
fetch_btn = st.button("Fetch Tweets")
if fetch_btn:
    do_fetch = True
elif auto_fetch and last_query != current_query_hash and DATA_SOURCE == "Live (Twitter via Twikit)":
    do_fetch = True

if do_fetch:
    if DATA_SOURCE == "Demo CSV / Sample":
        df = load_demo_df(sample_file, limit, query)
    else:
        df = fetch_live_tweets_twikit(query, limit)
    st.session_state["last_df"] = df
    st.session_state["last_query"] = current_query_hash
    st.session_state["last_sample_file"] = sample_file
else:
    if "last_df" in st.session_state:
        df = st.session_state["last_df"]
        last_sample_file = st.session_state.get("last_sample_file", None)
        if DATA_SOURCE == "Demo CSV / Sample" and last_sample_file != sample_file:
            st.warning(f"Demo file changed to '{sample_file}'. Click 'Fetch Tweets' to load the new file.")
    else:
        df = load_demo_df(sample_file, limit, query)
        st.session_state["last_df"] = df
        st.session_state["last_sample_file"] = sample_file

# Top info
st.subheader("Query panel")
st.markdown(f"**Source:** {DATA_SOURCE} • **Query:** `{query}` • **Limit:** {limit} • **Look back:** {since_minutes} minutes")
st.markdown(f"**Showing {len(df)} fetched tweets**")
st.dataframe(df, use_container_width=True, height=300)

# Sample tweets scrollable
if not df.empty:
    st.markdown("---")
    st.subheader("Sample tweets (expand to see content)")
    st.markdown("""
    <style>
    .scrollable-tweets { max-height: 360px; overflow-y: auto; border: 1px solid rgba(49,51,63,0.12); padding: 0.5rem; border-radius:4px }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="scrollable-tweets">', unsafe_allow_html=True)
    for _, row in df.head(min(len(df), 50)).iterrows():
        with st.expander(f"{row.get('username','')} • {row.get('date','')}"):
            st.write(row.get("content",""))
            if row.get("url"):
                st.markdown(f"[Open tweet]({row.get('url')})")
    st.markdown('</div>', unsafe_allow_html=True)

# Equal columns: preprocessing and sentiment
col_left, col_right = st.columns([1,1])

with col_left:
    st.subheader("🔧 Preprocessing")
    if st.button("Run preprocessing on current tweets"):
        df_local = df.copy()
        df_local = preprocess_dataframe(df_local)
        st.session_state["preprocessed_demo"] = df_local
        st.success(f"✅ Preprocessed {len(df_local)} rows")
    if "preprocessed_demo" in st.session_state:
        df_pre = st.session_state["preprocessed_demo"]
        st.dataframe(df_pre[["content","content_clean"]], use_container_width=True, height=400)
    else:
        st.info("Run preprocessing to see cleaned tweets here.")

with col_right:
    st.subheader("💬 Run Vertex AI Inference")

    if st.button("Run Vertex Ai Sentiment Inference"):
        df_local = st.session_state.get("preprocessed_demo", df).copy()
        texts = [
            row["content_clean"] if isinstance(row["content_clean"], str) and len(row["content_clean"].strip()) > 3
            else row["content"]
            for _, row in df_local.iterrows()
        ]

        with st.spinner("Running Vertex AI inference…"):
            try:
                # Load sentiment model from Hugging Face
                model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
                sentiment_analyzer = pipeline("sentiment-analysis", model=model)

                preds = []
                for text in texts:
                    res = sentiment_analyzer(text)[0]
                    label = res["label"].capitalize()
                    score = round(float(res["score"]), 2)
                    if label.lower() in ["positive", "negative", "neutral"]:
                        preds.append((label, score))
                    else:
                        preds.append(("Neutral", score))
            except Exception as e:
                st.error(f"Error in Vertex AI inference: {e}")
                preds = [("Neutral", 0.5)] * len(texts)

        df_local["predicted_sentiment"], df_local["confidence"] = zip(*preds)
        st.session_state["demo_result"] = df_local
        st.success("✅ Vertex AI sentiment inference done")

    if "demo_result" in st.session_state:
        df_res = st.session_state["demo_result"]
        st.dataframe(df_res[["content","predicted_sentiment","confidence"]], use_container_width=True, height=400)
    else:
        st.info("Run Vertex AI inference to see predictions")

# Export
st.markdown("---")
st.subheader("Export / Save")
if st.button("Save last results to CSV"):
    df_save = st.session_state.get("demo_result", st.session_state.get("preprocessed_demo", df))
    out = Path("results")
    out.mkdir(exist_ok=True)
    fname = out / f"tweets_result_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    df_save.to_csv(fname, index=False, encoding="utf-8-sig")
    with open(fname, "rb") as f:
        data = f.read()
    st.download_button("Download saved CSV", data=data, file_name=fname.name, mime="text/csv")
    st.success(f"Saved {len(df_save)} rows to {fname}")

st.caption("Realtime Sentiment Analysis using Twikit and GCP")
