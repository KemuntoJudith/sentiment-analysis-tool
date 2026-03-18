# Database utilities for user management and result storage

import os
import streamlit as st
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime
from passlib.hash import bcrypt

# DATABASE CONNECTION
try:
    DATABASE_URL = st.secrets["DB_URL"]
except Exception:
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://postgres:6063@localhost:5432/sentiment_analysis_tool"
    )

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# MODELS
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    role = Column(String, default="analyst")
    results = relationship("InferenceResult", back_populates="user")


class InferenceResult(Base):
    __tablename__ = "inference_results"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    text = Column(String)
    aspect = Column(String)
    sentiment = Column(String)
    confidence = Column(Float)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="results")


# CREATE TABLES
Base.metadata.create_all(bind=engine)


# HELPER: UTF-8 truncation for bcrypt
def truncate_utf8(s, max_bytes=72):
    """
    Truncate a string so its UTF-8 byte length does not exceed max_bytes.
    Ensures bcrypt password limit is respected.
    """
    encoded = s.encode("utf-8")[:max_bytes]
    return encoded.decode("utf-8", errors="ignore")


# USER MANAGEMENT
def create_user(username, password, role="analyst"):
    session = SessionLocal()
    try:
        hashed_password = bcrypt.hash(truncate_utf8(password))
        user = User(username=username, password=hashed_password, role=role)
        session.add(user)
        session.commit()
        return {"id": user.id, "username": user.username, "role": user.role}
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def authenticate_user(username, password):
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.username == username).first()
        if user and bcrypt.verify(truncate_utf8(password), user.password):
            return {"id": user.id, "username": user.username, "role": user.role}
        return None
    finally:
        session.close()


def get_user_by_username(username):
    session = SessionLocal()
    try:
        user = session.query(User).filter(User.username == username).first()
        if user:
            return {"id": user.id, "username": user.username, "role": user.role}
        return None
    finally:
        session.close()


# DASHBOARD-FRIENDLY FUNCTIONS
def call_register(username, password, role="analyst"):
    """Used by Streamlit dashboard to register a new user safely."""
    existing_user = get_user_by_username(username)
    if existing_user:
        raise Exception("Username already exists")
    return create_user(username, password, role)


def call_login(username, password):
    """Used by Streamlit dashboard to authenticate a user."""
    user_dict = authenticate_user(username, password)
    if not user_dict:
        raise Exception("Invalid username or password")
    return user_dict


# SAVE INFERENCE RESULTS
def save_result(text, aspect, sentiment, confidence, user_id):
    session = SessionLocal()
    try:
        record = InferenceResult(
            text=text, aspect=aspect, sentiment=sentiment, confidence=confidence, user_id=user_id
        )
        session.add(record)
        session.commit()
        return {
            "id": record.id,
            "text": record.text,
            "aspect": record.aspect,
            "sentiment": record.sentiment,
            "confidence": record.confidence,
            "user_id": record.user_id,
            "timestamp": record.timestamp
        }
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# FETCH DATA FOR ANALYTICS
def get_user_results(user_id):
    session = SessionLocal()
    try:
        records = session.query(InferenceResult).filter(InferenceResult.user_id == user_id).all()
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
                "timestamp": r.timestamp,
                "user_id": r.user_id
            } for r in records
        ]
    finally:
        session.close()