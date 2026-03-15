# Create analytics queries

# Install  the required libraries
from flask import Blueprint, jsonify
from sqlalchemy import text
from app.utils.db import SessionLocal

# Create a Blueprint for analytics routes
analytics_routes = Blueprint("analytics_routes", __name__)

# Sentiment distribution query (pie chart)
@analytics_routes.route("/sentiment-distribution", methods=["GET"])
def sentiment_distribution():

    session = SessionLocal()

    try:

        query = text("""
        SELECT sentiment, COUNT(*) as count
        FROM inference_results
        GROUP BY sentiment
        """)

        results = session.execute(query).fetchall()

        data = [{"sentiment": r[0], "count": r[1]} for r in results]

        return jsonify(data)

    finally:
        session.close()


# Aspect distribution query (bar chart)
@analytics_routes.route("/aspect-distribution", methods=["GET"])
def aspect_distribution():

    session = SessionLocal()

    try:

        query = text("""
        SELECT aspect, COUNT(*) as count
        FROM inference_results
        GROUP BY aspect
        """)

        results = session.execute(query).fetchall()

        data = [{"aspect": r[0], "count": r[1]} for r in results]

        return jsonify(data)

    finally:
        session.close()
        

# Negative alerts panel
@analytics_routes.route("/alerts", methods=["GET"])
def alerts():

    session = SessionLocal()

    try:

        query = text("""
        SELECT text, aspect, sentiment, confidence, timestamp
        FROM inference_results
        WHERE sentiment = 'negative'
        ORDER BY timestamp DESC
        LIMIT 10
        """)

        results = session.execute(query).fetchall()

        alerts = []

        for r in results:
            alerts.append({
                "text": r[0],
                "aspect": r[1],
                "sentiment": r[2],
                "confidence": float(r[3]),
                "timestamp": str(r[4])
            })

        return jsonify(alerts)

    finally:
        session.close()