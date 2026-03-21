# ABSA model for aspect-based sentiment analysis (ABSA) using a fine-tuned BERT model

# Aspect Based Sentiment Analysis Model

from app.models.finbert_model import predict_sentiment
from app.preprocessing.text_preprocessing import preprocess_text


# Define aspect categories

aspect_categories = {

    "mobile_banking_app": [
        "app",
        "application",
        "mobile app",
        "login",
        "crash",
        "password",
        "secret question",
        "mobile banking",
        "bug"
    ],

    "customer_service": [
        "customer service",
        "support",
        "agent",
        "service",
        "contact center",
        "helpdesk",
        "call center"
    ],

    "payments": [
        "payment",
        "transfer",
        "transaction",
        "sent money",
        "reversal",
        "deposit",
        "withdraw"
    ],

    "loans": [
        "loans",
        "repayment",
        "loan application",
        "facility",
        "credit"
    ],

    "insurance_services": [
        "insurance",
        "cover",
        "premium",
        "insurance claim",
        "place cover"
    ],

    "account_issues": [
        "account",
        "login",
        "number",
        "activation"
    ],

    "cards": [
        "card",
        "credit card",
        "debit card",
        "atm"
    ]
}


# Detect aspects present in the text

def detect_aspects(text):

    detected_aspects = []

    clean_text = preprocess_text(text)

    for aspect, keywords in aspect_categories.items():

        for keyword in keywords:

            if keyword in clean_text:

                detected_aspects.append(aspect)

                break

    return detected_aspects


# Perform aspect-based sentiment analysis

def analyze_aspects(text):

    aspects_found = detect_aspects(text)

    results = []

    for aspect in aspects_found:

        sentiment_result = predict_sentiment(text)

        results.append({

            "aspect": aspect,

            "sentiment": sentiment_result["sentiment"],

            "confidence": sentiment_result["confidence"]

        })

    return {

        "text": text,

        "aspects": results

    }


# Test the ABSA model

if __name__ == "__main__":

    test_texts = [

        "The mobile banking app keeps crashing",

        "Customer service helped me resolve my issue quickly",

        "My transfer failed and the payment was not reversed",

        "The loan application process is very slow",

        "My credit card was declined at the ATM",

        "I cannot login to my account after activation"
    ]

    print("\n=== Aspect Based Sentiment Analysis Test ===\n")

    for text in test_texts:

        result = analyze_aspects(text)

        print(result)

        print("-------------------------------------------")

