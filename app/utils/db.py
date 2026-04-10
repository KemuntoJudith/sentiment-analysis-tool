import os
import logging
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime


# LOGGING SETUP
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ENVIRONMENT DETECTION
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_SERVER_PORT") is not None


# DATABASE CONFIG
if IS_STREAMLIT_CLOUD:
    # Use SQLite in Streamlit Cloud
    DATABASE_URL = "sqlite:///sentiment.db"
else:
    # Use PostgreSQL locally (fallback to SQLite if not set)
    DATABASE_URL = os.getenv("DATABASE_URL") or \
        "postgresql+psycopg2://postgres:6063@localhost:5432/sentiment_analysis_tool"

logger.info(f"Using DATABASE_URL: {DATABASE_URL}")


# ENGINE & SESSION
try:
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False}  # REQUIRED for Streamlit
        )
    else:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)

    SessionLocal = sessionmaker(bind=engine)
    logger.info("✅ Database engine created successfully")

except Exception as e:
    logger.error(f"❌ Failed to create engine: {e}")
    engine = None
    SessionLocal = None

Base = declarative_base()


# MODELS
class InferenceResult(Base):
    __tablename__ = "inference_results"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    text = Column(String)
    aspect = Column(String)
    sentiment = Column(String)
    confidence = Column(Float)


# CREATE TABLES
if engine:
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables created / verified")
    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}")


# SAVE FUNCTION
def save_result(text, aspect, sentiment, confidence, timestamp=None):
    if not SessionLocal:
        logger.error("❌ SessionLocal is not initialized")
        return False

    session = SessionLocal()

    try:
        # If timestamp not provided, use current time
        if timestamp is None:
            timestamp = datetime.utcnow()

        record = InferenceResult(
            text=text,
            aspect=aspect,
            sentiment=sentiment,
            confidence=confidence,
            timestamp=timestamp  
        )

        session.add(record)
        session.commit()

        logger.info(f"✅ Saved to DB: {text[:40]} | {sentiment}")
        return True

    except Exception as e:
        session.rollback()
        logger.error(f"❌ Failed to save result: {e}")
        return False

    finally:
        session.close()


# FETCH FUNCTION
def get_all_results():
    if not SessionLocal:
        logger.error("❌ SessionLocal is not initialized")
        return []

    session = SessionLocal()

    try:
        records = session.query(InferenceResult).all()

        logger.info(f"📊 Retrieved {len(records)} records from DB")

        return [
            {
                "id": r.id,
                "text": r.text,
                "aspect": r.aspect,
                "sentiment": r.sentiment,
                "confidence": r.confidence,
                "timestamp": r.timestamp
            }
            for r in records
        ]

    except Exception as e:
        logger.error(f"❌ Failed to fetch results: {e}")
        return []

    finally:
        session.close()