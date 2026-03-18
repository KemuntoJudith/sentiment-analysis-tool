# Database utilities for result storage only

import os
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import streamlit as st

# DATABASE CONNECTION
DATABASE_URL = (
    st.secrets.get("DB_URL")
    or os.getenv("DATABASE_URL")
    or "postgresql+psycopg2://postgres:6063@localhost:5432/sentiment_analysis_tool"
)
st.write("DATABASE_URL:", DATABASE_URL)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# MODELS
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)


class InferenceResult(Base):
    __tablename__ = "inference_results"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    text = Column(String)
    aspect = Column(String)
    sentiment = Column(String)
    confidence = Column(Float)


# CREATE TABLES
Base.metadata.create_all(bind=engine)


# SAVE INFERENCE RESULTS
def save_result(text, aspect, sentiment, confidence):
    session = SessionLocal()
    try:
        record = InferenceResult(
            text=text, aspect=aspect, sentiment=sentiment, confidence=confidence
        )
        session.add(record)
        session.commit()
        return {
            "id": record.id,
            "text": record.text,
            "aspect": record.aspect,
            "sentiment": record.sentiment,
            "confidence": record.confidence,
            "timestamp": record.timestamp
        }
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# FETCH DATA FOR ANALYTICS
def get_all_results():
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