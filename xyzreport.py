# train_distilbert_boosted.py – Fine-tuning DistilBERT on Tweet Dataset
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast
from datasets import load_dataset
import torch

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

def tokenize(batch): return tokenizer(batch["text"], truncation=True, padding=True)
dataset = dataset.map(tokenize, batched=True)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
args = TrainingArguments(output_dir="./distilbert_tweeteval_boosted", evaluation_strategy="epoch", num_train_epochs=3)
trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["test"])
trainer.train()
trainer.save_model("./distilbert_tweeteval_boosted")
