# SQLAlchemy setup for PostgreSQL

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql+psycopg2://postgres:6063@localhost:5432/sentiment_analysis_tool"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String, unique=True, nullable=False)

    password = Column(String, nullable=False)

    role = Column(String, default="analyst")


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(DateTime, default=datetime.utcnow)

    text = Column(String)

    aspect = Column(String)

    sentiment = Column(String)

    confidence = Column(Float)

    user_id = Column(Integer, ForeignKey("users.id"))


Base.metadata.create_all(bind=engine)


def save_result(text, aspect, sentiment, confidence, user_id):

    session = SessionLocal()

    try:

        record = InferenceResult(
            text=text,
            aspect=aspect,
            sentiment=sentiment,
            confidence=confidence,
            user_id=user_id
        )

        session.add(record)

        session.commit()

    finally:

        session.close()
