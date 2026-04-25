# train_distilbert_quick_fixed.py
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Using device: {device}")

# 1️⃣ Load TweetEval sentiment dataset (small subset for quick training)
print("🔹 Loading TweetEval sentiment dataset...")
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
train_dataset = dataset['train'].shuffle(seed=42).select(range(5000))
val_dataset = dataset['validation'].shuffle(seed=42).select(range(500))

print(f"✅ Train dataset size: {len(train_dataset)}")
print(f"✅ Validation dataset size: {len(val_dataset)}")

# 2️⃣ Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', 
    num_labels=3
)
model.to(device)

# 3️⃣ Tokenize datasets with proper formatting
def tokenize_function(examples):
    # Tokenize the texts
    tokenized = tokenizer(
        examples['text'], 
        truncation=True, 
        padding=False,  # We'll use data collator for dynamic padding
        max_length=128
    )
    return tokenized

# Apply tokenization
print("🔹 Tokenizing datasets...")
train_dataset = train_dataset.map(
    tokenize_function, 
    batched=True,
    batch_size=1000
)
val_dataset = val_dataset.map(
    tokenize_function, 
    batched=True,
    batch_size=1000
)

# 4️⃣ Format datasets for PyTorch
def format_dataset(dataset):
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return dataset

train_dataset = format_dataset(train_dataset)
val_dataset = format_dataset(val_dataset)

print("🔹 Dataset columns:", train_dataset.column_names)

# 5️⃣ Use DataCollatorWithPadding for dynamic padding
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

# 6️⃣ Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# 7️⃣ Training arguments with optimized settings
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,               # Increased for better learning
    per_device_train_batch_size=16,   # Increased batch size
    per_device_eval_batch_size=16,
    learning_rate=2e-5,               # Slightly lower learning rate
    warmup_steps=500,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_strategy="steps",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
    dataloader_pin_memory=False,      # Fix for the pin_memory warning
    remove_unused_columns=True,       # Important for proper training
    label_names=["labels"]            # Explicitly specify label column
)

# 8️⃣ Trainer with proper configuration
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,      # ✅ Fixes the variable-length tensor error
    compute_metrics=compute_metrics
)

# 9️⃣ Train the model
print("🚀 Starting training...")
try:
    trainer.train()
    
    # 🔟 Save model
    trainer.save_model("./distilbert_tweeteval_quick_fixed")
    tokenizer.save_pretrained("./distilbert_tweeteval_quick_fixed")
    print("✅ Model saved at ./distilbert_tweeteval_quick_fixed")
    
    # Evaluate final model
    print("🔹 Evaluating final model...")
    eval_results = trainer.evaluate()
    print(f"✅ Final evaluation results: {eval_results}")
    
except Exception as e:
    print(f"❌ Training failed with error: {e}")
    print("🔹 Debug info:")
    print(f"   - Train dataset features: {train_dataset.features}")
    print(f"   - Sample item: {train_dataset[0]}")