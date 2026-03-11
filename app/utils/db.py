# Configure SLQAlchemy to connect to PostgreSQL and define the table structure for storing inference results.

from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "postgresql+psycopg2://postgres:6063@localhost:5432/sentiment_analysis_tool"

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(DateTime, default=datetime.utcnow)

    text = Column(String)

    aspect = Column(String)

    sentiment = Column(String)

    confidence = Column(Float)


Base.metadata.create_all(bind=engine)


def save_result(text, aspect, sentiment, confidence):

    session = SessionLocal()

    try:

        record = InferenceResult(
            text=text,
            aspect=aspect,
            sentiment=sentiment,
            confidence=confidence
        )

        session.add(record)

        session.commit()

    finally:

        session.close()