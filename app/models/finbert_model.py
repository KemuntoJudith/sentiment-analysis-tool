# Load the fine-tuned FinBERT model and define prediction functions

import os
import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.preprocessing.text_preprocessing import preprocess_text


# MODEL PATH CONFIG (SAFE)

# Detect Streamlit Cloud
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_SERVER_PORT") is not None

# Use relative path
LOCAL_MODEL_PATH = "models/finbert_final"

# Fallback to Hugging Face if local model missing
HF_MODEL_PATH = "ProsusAI/finbert"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "models", "finbert_final")
MODEL_PATH = os.path.abspath(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print(f"⚠ Local model not found at {MODEL_PATH}, will use Hugging Face fallback")


# LOAD MODEL
tokenizer = None
model = None


@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                local_files_only=os.path.exists(MODEL_PATH)
            )

            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_PATH,
                local_files_only=os.path.exists(MODEL_PATH)
            )

            model.eval()
            print("✅ Local model loaded")

        except Exception as e:
            print(f"❌ Local load failed: {e}")
            print("🔁 Falling back to Hugging Face")

            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)
            model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_PATH)
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
        

    labels = model.config.id2label
    raw_label = labels[predicted_class] 
    sentiment = raw_label.lower()
    confidence = probs[0][predicted_class].item()

    
    # DEBUG BLOCK
    try:
        st.write({
        "text": text,
        "predicted_index": predicted_class,
        "raw_label": raw_label,
        "final_sentiment": sentiment,
        "confidence": confidence
        })
    except:
        print({
        "text": text,
        "predicted_index": predicted_class,
        "raw_label": raw_label,
        "final_sentiment": sentiment,
        "confidence": confidence
    })

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
