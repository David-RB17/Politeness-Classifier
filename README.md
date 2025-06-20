# Japanese Formality Classifier

This project aims to develop a deep learning model that can classify short Japanese text inputs as either **formal** or **informal**.

---

## Goal

The goal is to build an NLP classifier that understands the formality and tone of Japanese sentences. This involves:

- Collecting and annotating a custom dataset from Japanese subtitles, dialogues, or transcripts
- Preprocessing Japanese text using a MeCab-based tokenizer (`fugashi`) compatible with BERT
- Training a classification model (starting with `cl-tohoku/bert-base-japanese`)
- Evaluating the model’s ability to distinguish between **formal** (敬語・丁寧語) and **informal** (ため口・カジュアル) language

---

## Research & Development Tasks

- Research formality markers and speech patterns in Japanese (e.g., verb endings, honorifics, particles)
- Investigate reliable sources of conversational Japanese data (anime/drama subs, chat logs, games)
- Explore tokenization and text normalization techniques for Japanese (used `fugashi` via AutoTokenizer)
- Build a Streamlit-based app for real-time predictions

---

## Background & Inspiration

This project is inspired by recent studies on Japanese politeness and formality classification using deep learning. Prior research has shown that transformer models (e.g., BERT) can effectively capture nuances in Japanese tone and register, especially when fine-tuned on curated or labeled datasets.

By simplifying the classification into **binary formality levels**, this project delivers a lightweight yet practical tool for tone-aware Japanese NLP applications.

---

## Status

**Current Model:**  
Fine-tuned Japanese BERT (`cl-tohoku/bert-base-japanese`) on ~800 manually labeled sentences from japanese anime subtitles, reduced to a **binary classification**:
- `0` = Informal (タメ口 / 普通)
- `1` = Formal (敬語)

**Evaluation Results (Test Set):**
- **Accuracy:** 96%
- **F1 Score:** 0.96
- **Precision:** 0.98 (Formal), 0.94 (Informal)
- **Recall:** 0.92 (Formal), 0.99 (Informal)

**App:**  
A fully functional **Streamlit app** is included for interactive testing. Users can input Japanese sentences and instantly receive predictions.

---

## Author

David — Applied Data Science & AI student  
Hogeschool Rotterdam
