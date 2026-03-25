# For streamlit app - moved core logic to separate functions for reuse in both Flask API and Streamlit UI

from app.models.finbert_model import predict_sentiment as finbert_predict
from app.models.absa_model import analyze_aspects
from app.utils.db import save_result, SessionLocal, InferenceResult
from app.utils.db import User
from werkzeug.security import check_password_hash


# Function 1: Predict overall sentiment
def predict_sentiment_local(text: str, user_id: int = None):
    """
    Run sentiment prediction + aspect detection and optionally save to DB.
    Returns a dict similar to Flask API.
    """

    try:
        # FinBERT sentiment
        sentiment_result = finbert_predict(text)
        sentiment = sentiment_result["sentiment"]
        confidence = sentiment_result["confidence"]
    except Exception as e:
        # 🔥 Prevent app crash if model fails
        return {
            "text": text,
            "sentiment": "error",
            "confidence": 0.0,
            "aspect": "error"
        }

    # Aspect detection
    aspect_result = analyze_aspects(text)
    aspects = aspect_result.get("aspects", [])

    aspect = "general"
    if aspects:
        aspect = aspects[0]["aspect"]

    # Save to DB if user_id provided
    if user_id is not None:
        try:
            save_result(
                text=text,
                aspect=aspect,
                sentiment=sentiment,
                confidence=confidence,
                user_id=user_id
            )
        except Exception:
            pass  # 🔥 Prevent DB crash

    # Return structured response
    return {
        "text": text,
        "sentiment": sentiment,
        "confidence": confidence,
        "aspect": aspect
    }


# Function 2: Predict aspects only
def predict_aspects_local(text: str, user_id: int = None):
    """
    Run aspect-based sentiment detection.
    Saves each aspect result if user_id is provided.
    """

    try:
        result = analyze_aspects(text)
    except Exception:
        return {"text": text, "aspects": []}

    aspects = result.get("aspects", [])

    if user_id is not None:
        for item in aspects:
            try:
                save_result(
                    text=text,
                    aspect=item["aspect"],
                    sentiment=item["sentiment"],
                    confidence=item["confidence"],
                    user_id=user_id
                )
            except Exception:
                pass  # Prevent DB crash

    return result


# Function 3: Fetch user analytics
def analytics_user_local(user_id: int):
    """
    Fetch all predictions for a given user.
    """
    session = SessionLocal()
    try:
        results = session.query(InferenceResult).filter(InferenceResult.user_id == user_id).all()
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
        return data
    except Exception:
        return []  # Prevent crash if DB fails
    finally:
        session.close()


# Function 4: User login
def login_local(username: str, password: str):
    """
    Authenticate user against DB.
    Returns user info dict or None if invalid.
    """
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.username == username).first()
        if user and check_password_hash(user.password, password):
            return {
                "user_id": user.id,
                "username": user.username
            }
        return None
    except Exception:
        return None  # Prevent crash if DB fails
    finally:
        session.close()