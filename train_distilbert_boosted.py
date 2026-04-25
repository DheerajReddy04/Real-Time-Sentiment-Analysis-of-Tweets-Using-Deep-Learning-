# train_distilbert_boosted.py
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import evaluate
import torch

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# 1️⃣ Load the full TweetEval sentiment dataset
print("🔹 Loading TweetEval sentiment dataset...")
ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
train_ds = ds["train"].shuffle(seed=42).select(range(20000))  # 20k samples
val_ds = ds["validation"]

print(f"✅ Train dataset size: {len(train_ds)}")
print(f"✅ Validation dataset size: {len(val_ds)}")

# 2️⃣ Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
).to(device)

# 3️⃣ Tokenize data
def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_enc = train_ds.map(tokenize, batched=True)
val_enc = val_ds.map(tokenize, batched=True)

train_enc = train_enc.rename_column("label", "labels")
val_enc = val_enc.rename_column("label", "labels")

train_enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 4️⃣ Metrics
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
        "f1": f1.compute(predictions=preds, references=p.label_ids, average="weighted")["f1"]
    }

# 5️⃣ Training configuration
args = TrainingArguments(
    output_dir="./results_boosted",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs_boosted",
    logging_steps=50
)

# 6️⃣ Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_enc,
    eval_dataset=val_enc,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

# 7️⃣ Train and evaluate
print("🚀 Starting boosted training...")
trainer.train()

print("✅ Training complete! Evaluating best model...")
results = trainer.evaluate()
print(f"🎯 Final Evaluation Results: {results}")

# 8️⃣ Save model
trainer.save_model("./distilbert_tweeteval_boosted")
print("✅ Model saved at ./distilbert_tweeteval_boosted")
