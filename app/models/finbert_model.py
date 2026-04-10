# Load the fine-tuned FinBERT model and define prediction functions

import os
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.preprocessing.text_preprocessing import preprocess_text


# MODEL PATH CONFIG (SAFE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "KemuntoJudith/finbert-sentiment"

# Only check if it's a local path
if os.path.isdir(MODEL_PATH):
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model folder not found at {MODEL_PATH}")


# LOAD MODEL
tokenizer = None
model = None

@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.eval()

    return tokenizer, model


# GET LABELS
def get_labels():
    _, model = load_model()
    return model.config.id2label


# PREDICTION FUNCTION
def predict_sentiment(text: str):
    tokenizer, model = load_model()
    
    # Preprocess
    cleaned_text = preprocess_text(text)

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    # LABEL MAPPING
    sentiment = model.config.id2label[predicted_class].lower()
    confidence = probs[0][predicted_class].item()

    return {
    "text": text,
    "sentiment": sentiment,
    "confidence": round(confidence, 3)
}


# BATCH PREDICTION
def predict_batch(text_list):
    results = []
    for text in text_list:
        result = predict_sentiment(text)
        results.append(result)
    return results


# TEST BLOCK
if __name__ == "__main__":

    print("\n=== DEBUG: MODEL LABELS ===\n")

    tokenizer, model = load_model()
    print("MODEL LABELS:", model.config.id2label)

    print("\n=== TEST PREDICTIONS ===\n")

    test_texts = [
        "This is the worst bank!",
        "The service was excellent",
        "The app is slow"
    ]

    predictions = predict_batch(test_texts)

    for result in predictions:
        print("Text:", result["text"])
        print("Sentiment:", result["sentiment"])
        print("Confidence:", result["confidence"])
        print("-------------------------------")