# Load the fine-tuned FinBERT model and define prediction functions

import torch
import torch.nn.functional as F
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.preprocessing.text_preprocessing import preprocess_text

# Path to model (use Hugging Face for deployment)
MODEL_PATH = "ProsusAI/finbert"

# Load tokenizer and model
tokenizer = None
model = None


@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
    return tokenizer, model


labels = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def predict_sentiment(text: str):
    tokenizer, model = load_model()

    # Preprocess text
    cleaned_text = preprocess_text(text)

    # Tokenize
    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)

        predicted_class = torch.argmax(probs, dim=1).item()
        sentiment = labels[predicted_class]
        confidence = probs[0, predicted_class].item()

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": round(confidence, 3)
    }


def predict_batch(text_list):
    results = []

    for text in text_list:
        result = predict_sentiment(text)
        results.append(result)

    return results


# Test block
if __name__ == "__main__":

    test_texts = [
        "The mobile banking app is very slow",
        "Customer service was excellent today",
        "The ATM was out of cash",
        "I am satisfied with my loan application",
        "The branch opened late"
    ]

    print("\n=== FinBERT Sentiment Test ===\n")

    predictions = predict_batch(test_texts)

    for result in predictions:
        print("Text:", result["text"])
        print("Sentiment:", result["sentiment"])
        print("Confidence:", result["confidence"])
        print("-------------------------------")