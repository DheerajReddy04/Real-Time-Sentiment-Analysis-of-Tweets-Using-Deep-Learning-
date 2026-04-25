from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datasets import load_dataset
import evaluate
import torch

print("🔹 Loading model and tokenizer...")
model_name = "distilbert_tweeteval_boosted"
model = DistilBertForSequenceClassification.from_pretrained(model_name)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

print("🔹 Loading TweetEval validation dataset...")
dataset = load_dataset("tweet_eval", "sentiment")
eval_ds = dataset["validation"].shuffle(seed=42).select(range(1000))  # Evaluate on 1k samples for speed

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding=True, max_length=128)

eval_encoded = eval_ds.map(tokenize, batched=True)
eval_encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

print("🔹 Evaluating model...")
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

model.eval()
for batch in torch.utils.data.DataLoader(eval_encoded, batch_size=16):
    with torch.no_grad():
        outputs = model(**{k: v for k, v in batch.items() if k != "label"})
    preds = outputs.logits.argmax(dim=-1)
    accuracy_metric.add_batch(predictions=preds, references=batch["label"])
    f1_metric.add_batch(predictions=preds, references=batch["label"])

acc = accuracy_metric.compute()["accuracy"]
f1 = f1_metric.compute(average="weighted")["f1"]

print(f"✅ Final Evaluation Results:")
print(f"   • Accuracy: {acc * 100:.2f}%")
print(f"   • F1 Score: {f1 * 100:.2f}%")
