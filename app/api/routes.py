# Create a Flask Blueprint for API routes
# This module defines the API routes for the application.

from flask import Blueprint, request, jsonify

from app.models.finbert_model import predict_sentiment
from app.models.absa_model import analyze_aspects

from app.utils.db import save_result

api_routes = Blueprint("api_routes", __name__)


# Route 1: Overall Sentiment
@api_routes.route("/predict-sentiment", methods=["POST"])
def predict_sentiment_route():

    data = request.get_json()

    text = data.get("text")

    result = predict_sentiment(text)

        # Save overall sentiment
    save_result(
        text=text,
        aspect="overall",
        sentiment=result["sentiment"],
        confidence=result["confidence"]
    )

    return jsonify(result)


# Route 2: Aspect-Based Sentiment
@api_routes.route("/predict-aspects", methods=["POST"])
def predict_aspects_route():

    data = request.get_json()
    text = data.get("text")

    result = analyze_aspects(text)

    aspects = result["aspects"]

    for item in aspects:

        # Save aspect-based sentiment
        save_result(
            text=text,
            aspect=item["aspect"],
            sentiment=item["sentiment"],
            confidence=item["confidence"]
        )

    return jsonify(result)

@api_routes.route("/", methods=["GET"])
def home():
    return {"message": "Sentiment API is running"}