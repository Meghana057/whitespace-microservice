"""
FastAPI Microservice for Whitespace Recommendations
Clean separation: API layer only, ML logic in whitespace_ml.py
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import json
import logging
import os
from dotenv import load_dotenv

# Import our custom modules
from database import (
    get_database_session, Account, Offering, Recommendation, ModelTraining,
    get_account_count, get_all_offerings, initialize_database
)
from whitespace_ml import WhitespaceRecommendationEngine

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app configuration
app = FastAPI(
    title="Whitespace Recommendation Microservice",
    description="AI-powered product recommendation system for CRM whitespace analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML engine
ml_engine = WhitespaceRecommendationEngine(
    threshold=float(os.getenv('ML_THRESHOLD', 0.5))
)

# Pydantic Models for API requests/responses
class AccountCreate(BaseModel):
    """Request model for creating accounts"""
    id: Optional[str] = None
    name: str = Field(..., min_length=1, max_length=200)
    industry: Optional[str] = Field(None, max_length=100)
    organization_name: str = Field(..., min_length=1, max_length=200)
    
    # Metrics
    contacts: int = Field(0, ge=0)
    active_opps: int = Field(0, ge=0)
    won_opps: int = Field(0, ge=0)
    lost_opps: int = Field(0, ge=0)
    tasks: int = Field(0, ge=0)
    events: int = Field(0, ge=0)
    
    # Revenue
    pipeline_revenue: float = Field(0.0, ge=0)
    won_revenue: float = Field(0.0, ge=0)
    lost_revenue: float = Field(0.0, ge=0)
    
    # Characteristics
    is_existing_customer: bool = True
    account_age_days: int = Field(0, ge=0)
    offerings_sold: Optional[List[str]] = []

class RecommendationRequest(BaseModel):
    """Request model for generating recommendations"""
    account: AccountCreate
    max_recommendations: int = Field(5, ge=1, le=20)

class RecommendationItem(BaseModel):
    """Single recommendation item"""
    offering: str
    confidence_score: float
    method: str
    reasoning: str
    threshold_passed: bool

class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    account_id: str
    account_name: str
    organization_name: str
    recommendations: List[RecommendationItem]
    total_recommendations: int
    threshold_used: float
    timestamp: datetime

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    service: str
    database_connected: bool
    models_trained: bool
    total_accounts: int
    total_offerings: int
    last_training: Optional[datetime] = None

class OrganizationAccountsResponse(BaseModel):
    """Response for organization accounts"""
    organization: str
    account_count: int
    accounts: List[Dict]

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    logger.info("Starting Whitespace Microservice...")
    
    # Initialize database
    if not initialize_database():
        logger.error("Database initialization failed")
        raise RuntimeError("Cannot start service without database")
    
    # Load models if available or train new ones
    try:
        # Try to load existing models
        if os.path.exists("models/whitespace_models.joblib"):
            logger.info("Loading existing models...")
            if ml_engine.load_models("models/whitespace_models.joblib"):
                logger.info("Models loaded successfully")
            else:
                logger.warning("Failed to load existing models, will train new ones")
        else:
            logger.info("No existing models found, will train when data is available")
            
    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}")
    
    logger.info("Whitespace Microservice started successfully")

# API Endpoints

@app.get("/health", response_model=HealthCheck)
async def health_check(db: Session = Depends(get_database_session)):
    """Service health check endpoint"""
    try:
        # Test database connection
        account_count = get_account_count(db)
        offerings = get_all_offerings(db)
        db_connected = True
        
        # Get last training info
        last_training = None
        try:
            training_record = db.query(ModelTraining).order_by(
                ModelTraining.completed_at.desc()
            ).first()
            if training_record and training_record.completed_at:
                last_training = training_record.completed_at
        except:
            pass
        
        return HealthCheck(
            status="healthy" if db_connected and ml_engine.is_trained else "partial",
            service="Whitespace Recommendation Microservice",
            database_connected=db_connected,
            models_trained=ml_engine.is_trained,
            total_accounts=account_count,
            total_offerings=len(offerings),
            last_training=last_training
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthCheck(
            status="unhealthy",
            service="Whitespace Recommendation Microservice",
            database_connected=False,
            models_trained=False,
            total_accounts=0,
            total_offerings=0
        )

@app.post("/accounts", status_code=201)
async def create_account(
    account_data: AccountCreate,
    db: Session = Depends(get_database_session)
):
    """Create a new account in the database"""
    try:
        # Generate ID if not provided
        account_id = account_data.id or f"acc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create database record
        db_account = Account(
            id=account_id,
            name=account_data.name,
            industry=account_data.industry,
            organization_name=account_data.organization_name,
            contacts=account_data.contacts,
            active_opps=account_data.active_opps,
            won_opps=account_data.won_opps,
            lost_opps=account_data.lost_opps,
            tasks=account_data.tasks,
            events=account_data.events,
            pipeline_revenue=account_data.pipeline_revenue,
            won_revenue=account_data.won_revenue,
            lost_revenue=account_data.lost_revenue,
            is_existing_customer=account_data.is_existing_customer,
            account_age_days=account_data.account_age_days,
            offerings_sold=json.dumps(account_data.offerings_sold or []),
            created_date=datetime.now()
        )
        
        db.add(db_account)
        db.commit()
        db.refresh(db_account)
        
        logger.info(f"Created account: {account_data.name} ({account_id})")
        
        return {
            "message": "Account created successfully",
            "account_id": account_id,
            "account_name": account_data.name
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating account: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to create account: {str(e)}")

@app.post("/predict", response_model=RecommendationResponse)
async def generate_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_database_session)
):
    """Generate whitespace recommendations for an account"""
    try:
        if not ml_engine.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="ML models not trained yet. Use /train endpoint first."
            )
        
        # Convert request to format expected by ML engine
        account_data = {
            'id': request.account.id or f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'name': request.account.name,
            'industry': request.account.industry or 'Unknown',
            'organization_name': request.account.organization_name,
            'contacts': request.account.contacts,
            'active_opps': request.account.active_opps,
            'won_opps': request.account.won_opps,
            'lost_opps': request.account.lost_opps,
            'tasks': request.account.tasks,
            'events': request.account.events,
            'pipeline_revenue': request.account.pipeline_revenue,
            'won_revenue': request.account.won_revenue,
            'lost_revenue': request.account.lost_revenue,
            'is_existing_customer': request.account.is_existing_customer,
            'account_age_days': request.account.account_age_days,
            'offerings_sold': json.dumps(request.account.offerings_sold or [])
        }
        
        # Generate recommendations
        recommendations = ml_engine.predict_for_account(
            account_data, 
            top_n=request.max_recommendations
        )
        
        # Store recommendations in database (background task)
        background_tasks.add_task(
            store_recommendations, 
            db, account_data, recommendations
        )
        
        # Format response
        recommendation_items = [
            RecommendationItem(
                offering=rec['offering'],
                confidence_score=rec['confidence_score'],
                method=rec['method'],
                reasoning=rec['reasoning'],
                threshold_passed=rec['threshold_passed']
            ) for rec in recommendations
        ]
        
        logger.info(f"Generated {len(recommendations)} recommendations for {request.account.name}")
        
        return RecommendationResponse(
            account_id=account_data['id'],
            account_name=request.account.name,
            organization_name=request.account.organization_name,
            recommendations=recommendation_items,
            total_recommendations=len(recommendations),
            threshold_used=ml_engine.threshold,
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/recommendations/{account_id}")
async def get_stored_recommendations(
    account_id: str,
    db: Session = Depends(get_database_session)
):
    """Retrieve stored recommendations for an account"""
    try:
        recommendations = db.query(Recommendation).filter(
            Recommendation.account_id == account_id,
            Recommendation.is_active == True
        ).order_by(Recommendation.confidence_score.desc()).all()
        
        if not recommendations:
            raise HTTPException(
                status_code=404, 
                detail=f"No recommendations found for account {account_id}"
            )
        
        return {
            "account_id": account_id,
            "recommendation_count": len(recommendations),
            "recommendations": [
                {
                    "id": rec.id,
                    "offering": rec.recommended_offering,
                    "confidence_score": rec.confidence_score,
                    "method": rec.method_used,
                    "reasoning": rec.reasoning,
                    "threshold_passed": rec.threshold_passed,
                    "created_at": rec.created_at,
                    "was_acted_upon": rec.was_acted_upon
                } for rec in recommendations
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/organizations/{org_name}/accounts", response_model=OrganizationAccountsResponse)
async def get_organization_accounts(
    org_name: str,
    db: Session = Depends(get_database_session)
):
    """Get all accounts for an organization (multi-tenant support)"""
    try:
        accounts = db.query(Account).filter(
            Account.organization_name == org_name
        ).all()
        
        if not accounts:
            logger.warning(f"No accounts found for organization: {org_name}")
        
        account_summaries = []
        for acc in accounts:
            offerings_sold = []
            try:
                offerings_sold = json.loads(acc.offerings_sold or '[]')
            except:
                offerings_sold = []
            
            account_summaries.append({
                "id": acc.id,
                "name": acc.name,
                "industry": acc.industry,
                "contacts": acc.contacts,
                "won_opps": acc.won_opps,
                "won_revenue": acc.won_revenue,
                "pipeline_revenue": acc.pipeline_revenue,
                "offerings_count": len(offerings_sold),
                "account_age_days": acc.account_age_days,
                "updated_at": acc.updated_at
            })
        
        return OrganizationAccountsResponse(
            organization=org_name,
            account_count=len(accounts),
            accounts=account_summaries
        )
        
    except Exception as e:
        logger.error(f"Error retrieving accounts for {org_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/offerings")
async def get_available_offerings(db: Session = Depends(get_database_session)):
    """Get all available offerings/products"""
    try:
        offerings = get_all_offerings(db)
        return {
            "offerings": offerings,
            "total_count": len(offerings)
        }
    except Exception as e:
        logger.error(f"Error retrieving offerings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_models(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_database_session)
):
    """Train or retrain ML models"""
    try:
        # Get all accounts for training
        accounts = db.query(Account).all()
        
        if len(accounts) < 10:  # Minimum training data requirement
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient training data. Found {len(accounts)} accounts, need at least 10"
            )
        
        # Convert to format expected by ML engine
        training_data = []
        for acc in accounts:
            training_data.append({
                'id': acc.id,
                'name': acc.name,
                'industry': acc.industry or 'Unknown',
                'organization_name': acc.organization_name,
                'contacts': acc.contacts,
                'active_opps': acc.active_opps,
                'won_opps': acc.won_opps,
                'lost_opps': acc.lost_opps,
                'tasks': acc.tasks,
                'events': acc.events,
                'pipeline_revenue': acc.pipeline_revenue,
                'won_revenue': acc.won_revenue,
                'lost_revenue': acc.lost_revenue,
                'is_existing_customer': acc.is_existing_customer,
                'account_age_days': acc.account_age_days,
                'offerings_sold': acc.offerings_sold or '[]'
            })
        
        # Train models in background
        background_tasks.add_task(train_models_background, db, training_data)
        
        return {
            "message": "Model training started",
            "training_data_count": len(training_data),
            "status": "in_progress"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training initiation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks
async def store_recommendations(
    db: Session, 
    account_data: dict, 
    recommendations: List[dict]
):
    """Store recommendations in database (background task)"""
    try:
        for rec in recommendations:
            if rec.get('threshold_passed', False):
                db_rec = Recommendation(
                    account_id=account_data['id'],
                    account_name=account_data['name'],
                    organization_name=account_data['organization_name'],
                    recommended_offering=rec['offering'],
                    confidence_score=rec['confidence_score'],
                    method_used=rec['method'],
                    reasoning=rec['reasoning'],
                    threshold_passed=rec['threshold_passed'],
                    threshold_value=ml_engine.threshold
                )
                db.add(db_rec)
        
        db.commit()
        logger.info(f"Stored {len([r for r in recommendations if r.get('threshold_passed')])} recommendations")
        
    except Exception as e:
        logger.error(f"Error storing recommendations: {str(e)}")
        db.rollback()

async def train_models_background(db: Session, training_data: List[dict]):
    """Train models in background (async task)"""
    training_start = datetime.now()
    
    # Create training record
    training_record = ModelTraining(
        training_data_count=len(training_data),
        models_trained=0,
        offerings_count=0,
        threshold_used=ml_engine.threshold,
        started_at=training_start
    )
    db.add(training_record)
    db.commit()
    
    try:
        logger.info(f"Starting background training with {len(training_data)} accounts")
        
        # Train the models
        success = ml_engine.train_models(training_data)
        
        if success:
            # Save models to disk
            os.makedirs("models", exist_ok=True)
            ml_engine.save_models("models/whitespace_models.joblib")
            
            # Update training record
            model_info = ml_engine.get_model_info()
            training_record.training_success = True
            training_record.models_trained = model_info['num_rf_models']
            training_record.offerings_count = model_info['num_offerings']
            training_record.completed_at = datetime.now()
            training_record.training_duration_seconds = (
                datetime.now() - training_start
            ).total_seconds()
            
            logger.info("Background model training completed successfully")
            
        else:
            training_record.training_success = False
            training_record.error_message = "Training failed - insufficient data"
            training_record.completed_at = datetime.now()
            logger.error("Background model training failed")
        
        db.commit()
        
    except Exception as e:
        error_msg = f"Training error: {str(e)}"
        logger.error(error_msg)
        
        training_record.training_success = False
        training_record.error_message = error_msg
        training_record.completed_at = datetime.now()
        db.commit()

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("SERVICE_PORT", 8000))
    
    # Run the application
    uvicorn.run(app, host=host, port=port)