# Japanese Formality Classifier (WIP)

This project aims to develop a deep learning model that can classify short Japanese text inputs as either **formal** or **informal**.

---

## Goal

The goal is to build an NLP classifier that understands the formality and tone of Japanese sentences. This involves:

- Collecting and annotating a custom dataset from Japanese subtitles, dialogues, or transcripts
- Preprocessing Japanese text using tokenizers like MeCab, SudachiPy, or Fugashi
- Training a classification model (starting with BERT for Japanese)
- Evaluating the model’s ability to distinguish between **formal** (敬語・丁寧語) and **informal** (ため口・カジュアル) language

---

## Research & Development Tasks

- Research formality markers and speech patterns in Japanese (e.g., verb endings, honorifics, particles)
- Investigate reliable sources of conversational Japanese data (anime/drama subs, chat logs, games)
- Experiment with both traditional ML (e.g., TF-IDF + Logistic Regression) and transformer-based models
- Explore tokenization and text normalization techniques for Japanese
- Optionally: Build a Streamlit or web-based app for real-time predictions

---

## Background & Inspiration

This project is inspired by recent studies on Japanese politeness and formality classification using deep learning. Prior research has shown that transformer models (e.g., BERT) can effectively capture nuances in Japanese tone and register, especially when fine-tuned on curated or labeled datasets.

By simplifying the classification into **binary formality levels**, this project aims to deliver a lightweight yet practical tool for tone-aware Japanese NLP applications.

---

## Status

🟡 **Currently in the data collection and research phase**  
Model training and evaluation will follow soon.

---

## Author

David — Applied Data Science & AI student  
Hogeschool Rotterdam
