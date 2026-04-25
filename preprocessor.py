'''
# preprocessor.py (emoji lexicon version)
import re
import pandas as pd
import emoji
from pathlib import Path

LEXICON_PATH = Path("lexicon.csv")
EMOJI_LEXICON_PATH = Path("emoji_lexicon.csv")

# -----------------------------
# Load Hinglish/Telugu Lexicon
# -----------------------------
def load_lexicon():
    if not LEXICON_PATH.exists():
        print("⚠️  Lexicon file not found — Hinglish/Telugu mapping skipped.")
        return {}
    df = pd.read_csv(LEXICON_PATH).dropna(subset=["word", "english_equivalent"])
    lex_map = {
        str(row["word"]).strip().lower(): str(row["english_equivalent"]).strip().lower()
        for _, row in df.iterrows()
    }
    print(f"✅ Loaded {len(lex_map)} entries from lexicon.csv")
    return lex_map

LEXICON_MAP = load_lexicon()

# -----------------------------
# Load Emoji Lexicon
# -----------------------------
def load_emoji_lexicon():
    if not EMOJI_LEXICON_PATH.exists():
        print("⚠️  Emoji lexicon not found — using defaults.")
        return {}
    df = pd.read_csv(EMOJI_LEXICON_PATH).dropna(subset=["emoji", "sentiment"])
    emo_map = {str(r["emoji"]).strip(): str(r["sentiment"]).strip() for _, r in df.iterrows()}
    print(f"✅ Loaded {len(emo_map)} emojis from emoji_lexicon.csv")
    return emo_map

EMOJI_MAP = load_emoji_lexicon()

# -----------------------------
# Emoji Replacement
# -----------------------------
def replace_emojis(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for emo, sentiment in EMOJI_MAP.items():
        if emo in text:
            text = text.replace(emo, f" {sentiment} ")
    text = emoji.replace_emoji(text, replace="")  # remove leftover emojis
    return text

# -----------------------------
# Lexicon Replacement
# -----------------------------
def apply_lexicon(text: str) -> str:
    if not text:
        return ""
    text_low = text.lower()
    sorted_keys = sorted(LEXICON_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        replacement = LEXICON_MAP[key]
        pattern = r'\b' + re.escape(key) + r'\b'
        text_low = re.sub(pattern, replacement, text_low, flags=re.IGNORECASE)
    return text_low

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = replace_emojis(text)
    text = apply_lexicon(text)
    text = re.sub(r"(emo_[a-z]+)", r"@@\1@@", text)
    text = re.sub(r"[^a-z_\s@]", " ", text)
    text = text.replace("@@", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# DataFrame Preprocessing
# -----------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "content" not in df.columns:
        raise ValueError("Input DataFrame must have a 'content' column.")
    df = df.copy()
    df["content_clean"] = df["content"].apply(clean_text)
    return df

# -----------------------------
# Test Run
# -----------------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        "content": [
            "I love this movie 😍🔥",
            "Feeling 😞 and 😭 today...",
            "😐 Just a normal day.",
            "🥳 Celebration time!"
        ]
    })
    print("\nBefore preprocessing:\n", df)
    out = preprocess_dataframe(df)
    print("\nAfter preprocessing:\n", out[["content", "content_clean"]])
'''



# preprocessor.py
import re
import pandas as pd
import emoji
from pathlib import Path

LEXICON_PATH = Path("lexicon.csv")
EMOJI_LEXICON_PATH = Path("emoji_lexicon.csv")

# -----------------------------
# Load Hinglish/Telugu Lexicon
# -----------------------------
def load_lexicon():
    if not LEXICON_PATH.exists():
        print("⚠️ Lexicon file not found — Hinglish/Telugu mapping skipped.")
        return {}
    df = pd.read_csv(LEXICON_PATH).dropna(subset=["word", "english_equivalent"])
    lex_map = {str(row["word"]).strip().lower(): str(row["english_equivalent"]).strip().lower()
               for _, row in df.iterrows()}
    print(f"✅ Loaded {len(lex_map)} entries from lexicon.csv")
    return lex_map

LEXICON_MAP = load_lexicon()

# -----------------------------
# Load Emoji Lexicon
# -----------------------------
def load_emoji_lexicon():
    if not EMOJI_LEXICON_PATH.exists():
        print("⚠️ Emoji lexicon not found — using defaults.")
        return {}
    df = pd.read_csv(EMOJI_LEXICON_PATH).dropna(subset=["emoji", "sentiment"])
    emo_map = {str(r["emoji"]).strip(): str(r["sentiment"]).strip() for _, r in df.iterrows()}
    print(f"✅ Loaded {len(emo_map)} emojis from emoji_lexicon.csv")
    return emo_map

EMOJI_MAP = load_emoji_lexicon()

# -----------------------------
# Emoji Replacement
# -----------------------------
def replace_emojis(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for emo, sentiment in EMOJI_MAP.items():
        if emo in text:
            text = text.replace(emo, f" {sentiment} ")
    text = emoji.replace_emoji(text, replace="")  # remove leftover emojis
    return text

# -----------------------------
# Lexicon Replacement
# -----------------------------
def apply_lexicon(text: str) -> str:
    if not text:
        return ""
    text_low = text.lower()
    sorted_keys = sorted(LEXICON_MAP.keys(), key=len, reverse=True)
    for key in sorted_keys:
        replacement = LEXICON_MAP[key]
        pattern = r'\b' + re.escape(key) + r'\b'
        text_low = re.sub(pattern, replacement, text_low, flags=re.IGNORECASE)
    return text_low

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = replace_emojis(text)
    text = apply_lexicon(text)
    text = re.sub(r"(emo_[a-z]+)", r"@@\1@@", text)
    text = re.sub(r"[^a-z_\s@]", " ", text)
    text = text.replace("@@", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------------
# DataFrame Preprocessing
# -----------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "content" not in df.columns:
        raise ValueError("Input DataFrame must have a 'content' column.")
    df = df.copy()
    df["content_clean"] = df["content"].apply(clean_text)
    return df

# -----------------------------
# Test Run
# -----------------------------
if __name__ == "__main__":
    df = pd.DataFrame({
        "content": [
            "I love this movie 😍🔥",
            "Feeling 😞 and 😭 today...",
            "😐 Just a normal day.",
            "🥳 Celebration time!"
        ]
    })
    print("\nBefore preprocessing:\n", df)
    out = preprocess_dataframe(df)
    print("\nAfter preprocessing:\n", out[["content", "content_clean"]])
