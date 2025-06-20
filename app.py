import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.predict import predict_formality

# === Load model and tokenizer ===
# MODEL_PATH = r"G:\Python Projects\politeness-classifier-jp\models\bert-finetunedv2"
MODEL_PATH = "David-RB/japanese-formality-classifier"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# === Streamlit page config ===
st.set_page_config(page_title="Japanese Formality Classifier", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://wallpapersok.com/images/hd/japanese-minimalist-black-white-landscape-euqli3al5e2ykm13.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Header ===
st.markdown("<h1 style='text-align: center;'>Japanese Formality Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a Japanese sentence to classify it as <strong>Formal</strong> or <strong>Informal</strong>.</p>", unsafe_allow_html=True)
st.markdown("---")

# === Input ===
text_input = st.text_input("Input your Japanese sentence:")

if text_input:
    prediction = predict_formality(text_input, model, tokenizer)

    st.markdown("---")
    st.markdown(f"<h3>Prediction: <code>{prediction}</code></h3>", unsafe_allow_html=True)

    if prediction.lower() == "formal":
        st.info("This sentence uses polite/formal language.")
    else:
        st.warning("This sentence uses informal or casual language.")

# === Footer ===
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.9em;'>Built using BERT and Streamlit</p>", unsafe_allow_html=True)
