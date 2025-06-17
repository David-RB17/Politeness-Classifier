# Japanese Politeness Level Classifier (WIP)

This project aims to develop a deep learning model that can classify short Japanese text inputs based on their politeness level:

- 0 – Casual (タメ口)
- 1 – Neutral (普通)
- 2 – Polite / Keigo (敬語)

---

## Goal

The goal is to build an NLP classifier that understands the formality and tone of Japanese sentences. This will involve:

- Collecting and annotating a custom dataset from Japanese subtitles, dialogues, or transcripts
- Preprocessing Japanese text using tokenizers like MeCab, SudachiPy, or Fugashi
- Training a classification model (starting with BERT for Japanese)
- Evaluating how well the model can distinguish subtle shifts in tone

---

## Research & Development Tasks

- Research politeness markers and speech patterns in Japanese (e.g., verb forms, honorifics)
- Investigate reliable sources of conversational Japanese data (anime/drama subs, chat logs)
- Experiment with both traditional ML (e.g., TF-IDF + Logistic Regression) and transformer-based models
- Explore tokenization and text normalization for Japanese input
- Optionally: build a small Streamlit app for real-time predictions

---

## Status

Currently in the data collection and research phase.  
Model training and deployment will follow.

---

## Author

David — Applied Data Science & AI student