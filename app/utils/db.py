import os
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import streamlit as st


# DETECT ENVIRONMENT
IS_CLOUD = os.getenv("STREAMLIT_SERVER_PORT") is not None


# DATABASE CONNECTION
DATABASE_URL = os.getenv("DATABASE_URL") or \
    "postgresql+psycopg2://postgres:6063@localhost:5432/sentiment_analysis_tool"

engine = None
SessionLocal = None

if not IS_CLOUD:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()



# MODELS
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)


class InferenceResult(Base):
    __tablename__ = "inference_results"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    text = Column(String)
    aspect = Column(String)
    sentiment = Column(String)
    confidence = Column(Float)



# CREATE TABLES (ONLY LOCAL)
if engine:
    Base.metadata.create_all(bind=engine)



# SAFE FUNCTIONS

import traceback

def save_result(text, aspect, sentiment, confidence, user_id):
    if user_id is None:
        print("🚨 save_result called WITHOUT user_id!")
        traceback.print_stack()   # 🔥 THIS SHOWS EXACT CALL LOCATION

    # existing code below...
def save_result(text, aspect, sentiment, confidence, user_id=None):
    if not SessionLocal:
        return  # Skip in cloud

    session = SessionLocal()
    try:
        record = InferenceResult(
            text=text, aspect=aspect,
            sentiment=sentiment, confidence=confidence
        )
        session.add(record)
        session.commit()
    except:
        session.rollback()
    finally:
        session.close()


def get_all_results():
    if not SessionLocal:
        return []  # Return empty in cloud

    session = SessionLocal()
    try:
        records = session.query(InferenceResult).all()
        return [
            {
                "id": r.id,
                "text": r.text,
                "aspect": r.aspect,
                "sentiment": r.sentiment,
                "confidence": r.confidence,
                "timestamp": r.timestamp
            } for r in records
        ]
    finally:
        session.close()