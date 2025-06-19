import pandas as pd
import numpy as np
import torch
import os
import json
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split

# === CONFIG ===
MODEL_PATH = r"G:\Python Projects\politeness-classifier-jp\models\bert-finetunedv2"
DATA_PATH = r"data/processed/BunnyGirl300-Preprocessed.csv"
OUTPUT_DIR = MODEL_PATH  # saves alongside the model
NUM_LABELS = 3
CLASS_NAMES = ["Casual", "Neutral", "Polite"]  # change to ["Informal", "Formal"] if binary

# === LOAD MODEL & TOKENIZER ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# === LOAD AND PREP TEST DATA ===
df = pd.read_csv(DATA_PATH)
df = df[["sentence", "label"]]
_, test_df = train_test_split(df, stratify=df["label"], test_size=0.2, random_state=42)

test_ds = Dataset.from_pandas(test_df)

def tokenize_fn(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length")

test_ds = test_ds.map(tokenize_fn)

# === PREDICT ===
trainer = Trainer(model=model, tokenizer=tokenizer)
pred_output = trainer.predict(test_ds)

preds = np.argmax(pred_output.predictions, axis=1)
labels = pred_output.label_ids

# === SAVE CLASSIFICATION REPORT ===
report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)

# === SAVE METRICS.JSON ===
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
acc = accuracy_score(labels, preds)

metrics = {
    "accuracy": acc,
    "precision": precision,
    "recall": recall,
    "f1": f1
}
with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

# === SAVE CONFUSION MATRIX ===
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
plt.close()
