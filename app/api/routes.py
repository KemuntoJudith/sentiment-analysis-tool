# Flask API Routes for Sentiment Analysis Tool

from flask import Blueprint, request, jsonify
from werkzeug.security import check_password_hash

from app.models.finbert_model import predict_sentiment
from app.models.absa_model import analyze_aspects
from app.utils.db import save_result, SessionLocal, InferenceResult, User

api_routes = Blueprint("api_routes", __name__)


# Route 1: Overall Sentiment
@api_routes.route("/predict-sentiment", methods=["POST"])
def predict_sentiment_route():

    print("Request received")

    data = request.get_json()
    text = data.get("text")

    # default user_id if not provided
    user_id = data.get("user_id", 1)

    print("Text received:", text)

    # Run FinBERT
    print("Running FinBERT...")
    sentiment_result = predict_sentiment(text)
    print("FinBERT finished")

    sentiment = sentiment_result["sentiment"]
    confidence = sentiment_result["confidence"]

    # Run Aspect Detection
    print("Running ABSA...")
    aspect_result = analyze_aspects(text)
    print("ABSA finished")

    aspects = aspect_result.get("aspects", [])

    # Default aspect
    aspect = "general"
    if aspects:
        aspect = aspects[0]["aspect"]

    print("Saving result...")

    # ✅ Save to DB with user_id
    save_result(
        text=text,
        aspect=aspect,
        sentiment=sentiment,
        confidence=confidence,
        user_id=user_id
    )

    print("Returning response")

    return jsonify({
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "aspect": aspect
    })


# Route 2: Aspect-Based Sentiment
@api_routes.route("/predict-aspects", methods=["POST"])
def predict_aspects_route():

    data = request.get_json()
    text = data.get("text")

    # ✅ FIX: default user_id
    user_id = data.get("user_id", 1)

    result = analyze_aspects(text)
    aspects = result.get("aspects", [])

    for item in aspects:
        save_result(
            text=text,
            aspect=item["aspect"],
            sentiment=item["sentiment"],
            confidence=item["confidence"],
            user_id=user_id
        )

    return jsonify(result)


# Route 3: Home
@api_routes.route("/", methods=["GET"])
def home():
    return {"message": "Sentiment API is running"}


# Route 4: User Analytics
@api_routes.route("/analytics/user/<int:user_id>", methods=["GET"])
def analytics_user(user_id):
    """
    Fetch all predictions for a given user.
    Returns list of dicts: text, aspect, sentiment, confidence, timestamp
    """
    session = SessionLocal()
    try:
        results = session.query(InferenceResult).filter(
            InferenceResult.user_id == user_id
        ).all()

        data = [
            {
                "text": r.text,
                "aspect": r.aspect,
                "sentiment": r.sentiment,
                "confidence": r.confidence,
                "timestamp": r.timestamp.isoformat()
            }
            for r in results
        ]

        return jsonify(data)

    finally:
        session.close()


# Route 5: User Login
@api_routes.route("/login", methods=["POST"])
def login():

    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    session = SessionLocal()

    try:
        user = session.query(User).filter(User.username == username).first()

        if user and check_password_hash(user.password, password):
            return jsonify({
                "user_id": user.id,
                "username": user.username
            })

        return jsonify({"error": "Invalid credentials"}), 401

    finally:
        session.close()