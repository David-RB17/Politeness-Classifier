import pandas as pd
import numpy as np
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Config
MODEL_NAME = "cl-tohoku/bert-base-japanese"
DATA_PATH = r"G:\Python Projects\politeness-classifier-jp\models"



NUM_LABELS = 2 
SAVE_PATH = r"G:\Python Projects\politeness-classifier-jp\models\bert-models"

# Create incrementing subfolder
i = 1
while True:
    numbered_path = f"{SAVE_PATH}_{i:03d}"
    if not os.path.exists(numbered_path):
        os.makedirs(numbered_path)
        SAVE_PATH = numbered_path
        break
    i += 1

# 1. Load data
df = pd.read_csv(DATA_PATH)
df = df[["sentence", "label"]]  # adjust if your column names differ
train_df, test_df = train_test_split(df, stratify=df["label"], test_size=0.2)

# 2. Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length")

train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)
train_ds = train_ds.map(tokenize_fn)
test_ds = test_ds.map(tokenize_fn)

# 3. Load model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# 4. Define metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# 5. Training
training_args = TrainingArguments(
    output_dir=SAVE_PATH,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",
    greater_is_better=True,
    logging_dir="./logs",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

trainer.train()

# 6. Save final model
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
