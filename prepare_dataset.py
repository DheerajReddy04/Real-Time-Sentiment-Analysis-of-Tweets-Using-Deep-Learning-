# prepare_dataset.py
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# 1️⃣ Load CSV
df = pd.read_csv("preprocessed_tweets.csv")  # replace with your file path

# 2️⃣ Map sentiment to numeric labels
label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}
id2label = {v: k for k, v in label2id.items()}
df['label_id'] = df['predicted_sentiment'].map(label2id)

# 3️⃣ Select only needed columns for Hugging Face Dataset
df = df[['content_clean', 'label_id']].rename(columns={'content_clean': 'text', 'label_id': 'label'})

# 4️⃣ Split into train and validation
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['label'])

# 5️⃣ Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print("✅ Train dataset size:", len(train_dataset))
print("✅ Validation dataset size:", len(val_dataset))

# 6️⃣ Tokenization
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# 7️⃣ Set format for PyTorch
train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

print("✅ Dataset ready for DistilBERT training")
