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
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, output_attentions=True)

    model.eval()

    return tokenizer, model


# GET LABELS
def get_labels():
    _, model = load_model()
    return model.config.id2label


# Attention Processing Function
def get_attention_scores(attentions, tokens):
    try:
        if attentions is None or len(attentions) == 0:
            return []

        attn = torch.stack(attentions)

        attn_mean = attn.mean(dim=(0, 2))  # batch x seq x seq
        cls_attention = attn_mean[0][0]

        return list(zip(tokens, cls_attention.tolist()))

    except Exception as e:
        return []


# Get attention weights for visualization
def get_attention_weights(text):
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # ADD SAFETY CHECK
    if outputs.attentions is None or len(outputs.attentions) == 0:
        return [], []

    attentions = outputs.attentions[-1]  # last layer

    # safer averaging
    try:
        scores = attentions.mean(dim=1).squeeze().mean(dim=0)
    except:
        return [], []

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    if len(tokens) != len(scores):
        return [], []

    return tokens, scores.tolist()


# PREDICTION FUNCTION
def predict_sentiment(text: str):
    tokenizer, model = load_model()
    
    # Preprocess
    cleaned_text = preprocess_text(text)

    if not cleaned_text or cleaned_text.strip() == "":
        return {
            "text": text,
            "sentiment": "neutral",
            "confidence": 0.0
        }

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512
    )

    with torch.no_grad():
        # Enable attention output
        outputs = model(**inputs, output_attentions=True)

        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[0]

        predicted_class = torch.argmax(probs).item()

    # LABEL MAPPING
    sentiment = model.config.id2label[predicted_class].lower()
    confidence = probs[predicted_class].item()

    # Convert probabilities
    prob_negative = probs[0].item()
    prob_neutral = probs[1].item()
    prob_positive = probs[2].item()

    # Extract tokens + attention
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    try:
        attention_scores = get_attention_scores(outputs.attentions, tokens)
    except:
        attention_scores = []

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 3),

        # For LIME and probability display
        "prob_negative": prob_negative,
        "prob_neutral": prob_neutral,
        "prob_positive": prob_positive,

        # For attention visualization
        "attention": attention_scores
    }

# LIME-compatible function
def predict_proba(texts):
    tokenizer, model = load_model()

    if isinstance(texts, str):
        texts = [texts]

    texts = [t if t and t.strip() else "empty input" for t in texts]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    return probs.numpy()

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
        print("Probabilities:", result["prob_negative"], result["prob_neutral"], result["prob_positive"])
        print("-------------------------------")