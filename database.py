"""
Database models and configuration for Whitespace Microservice
SQLAlchemy models for PostgreSQL
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from sqlalchemy import text
from typing import Generator
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/whitespace_db")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Account(Base):
    """Account model - stores CRM account data"""
    __tablename__ = "accounts"
    
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    industry = Column(String)
    organization_name = Column(String, nullable=False, index=True)  # Multi-tenant field
    
    # Contact and opportunity metrics
    contacts = Column(Integer, default=0)
    active_opps = Column(Integer, default=0)
    won_opps = Column(Integer, default=0)
    lost_opps = Column(Integer, default=0)
    
    # Activity metrics
    tasks = Column(Integer, default=0)
    events = Column(Integer, default=0)
    
    # Revenue metrics
    pipeline_revenue = Column(Float, default=0.0)
    won_revenue = Column(Float, default=0.0)
    lost_revenue = Column(Float, default=0.0)
    
    # Account characteristics
    is_existing_customer = Column(Boolean, default=True)
    account_age_days = Column(Integer, default=0)
    
    # Product information (JSON string)
    offerings_sold = Column(Text, default='[]')  # JSON array of product names
    
    # Timestamps
    created_date = Column(DateTime)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Account(id='{self.id}', name='{self.name}', org='{self.organization_name}')>"

class Offering(Base):
    """Offering/Product model - master catalog of all products"""
    __tablename__ = "offerings"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text)  # Optional product description
    category = Column(String)   # Optional product category
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<Offering(id={self.id}, name='{self.name}')>"

class Recommendation(Base):
    """Recommendation model - stores ML predictions"""
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Account information
    account_id = Column(String, nullable=False, index=True)
    account_name = Column(String, nullable=False)
    organization_name = Column(String, nullable=False, index=True)
    
    # Recommendation details
    recommended_offering = Column(String, nullable=False)
    confidence_score = Column(Float, nullable=False)
    method_used = Column(String, nullable=False)  # 'Random Forest', 'KNN', 'Association Rules'
    reasoning = Column(Text)  # Human-readable explanation
    
    # Filtering
    threshold_passed = Column(Boolean, default=False)  # Above threshold flag
    threshold_value = Column(Float, default=0.5)  # Threshold used
    
    # Status tracking
    is_active = Column(Boolean, default=True)  # Can be deactivated
    was_acted_upon = Column(Boolean, default=False)  # Did sales team act on this?
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)
    expires_at = Column(DateTime)  # Optional expiration
    
    def __repr__(self):
        return (f"<Recommendation(account='{self.account_name}', "
                f"offering='{self.recommended_offering}', "
                f"confidence={self.confidence_score:.3f})>")

class ModelTraining(Base):
    """Model training history - track retraining events"""
    __tablename__ = "model_training"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Training details
    training_data_count = Column(Integer, nullable=False)
    models_trained = Column(Integer, nullable=False)  # Number of RF models
    offerings_count = Column(Integer, nullable=False)
    threshold_used = Column(Float, default=0.5)
    
    # Training results
    training_success = Column(Boolean, default=False)
    error_message = Column(Text)  # If training failed
    
    # Performance metrics (optional)
    avg_cv_score = Column(Float)  # Average cross-validation score
    training_duration_seconds = Column(Float)
    
    # Timestamps
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    
    def __repr__(self):
        return f"<ModelTraining(id={self.id}, success={self.training_success}, models={self.models_trained})>"

# Database utility functions
def get_database_session() -> Generator[Session, None, None]:
    """
    Dependency function for FastAPI to get database session
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to create tables: {str(e)}")
        return False

def drop_tables():
    """Drop all database tables (use with caution!)"""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped")
        return True
    except Exception as e:
        logger.error(f"Failed to drop tables: {str(e)}")
        return False

def test_database_connection() -> bool:
    """Test database connectivity"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        logger.info("Database connection successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False

def get_account_count(db: Session) -> int:
    """Get total number of accounts in database"""
    try:
        return db.query(Account).count()
    except Exception as e:
        logger.error(f"Error counting accounts: {str(e)}")
        return 0

def get_organization_accounts(db: Session, org_name: str) -> list:
    """Get all accounts for a specific organization (multi-tenant)"""
    try:
        accounts = db.query(Account).filter(Account.organization_name == org_name).all()
        return accounts
    except Exception as e:
        logger.error(f"Error retrieving accounts for {org_name}: {str(e)}")
        return []

def get_all_offerings(db: Session) -> list:
    """Get all available offerings"""
    try:
        offerings = db.query(Offering).filter(Offering.is_active == True).all()
        return [offering.name for offering in offerings]
    except Exception as e:
        logger.error(f"Error retrieving offerings: {str(e)}")
        return []

def clear_old_recommendations(db: Session, days_old: int = 30):
    """Clear recommendations older than specified days"""
    try:
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        deleted_count = db.query(Recommendation).filter(
            Recommendation.created_at < cutoff_date
        ).delete()
        
        db.commit()
        logger.info(f"Cleared {deleted_count} old recommendations")
        return deleted_count
    except Exception as e:
        logger.error(f"Error clearing old recommendations: {str(e)}")
        db.rollback()
        return 0

# Database initialization
def initialize_database():
    """Initialize database with tables and basic setup"""
    logger.info("Initializing database...")
    
    # Test connection
    if not test_database_connection():
        logger.error("Cannot connect to database")
        return False
    
    # Create tables
    if not create_tables():
        logger.error("Failed to create database tables")
        return False
    
    logger.info("Database initialization completed")
    return True