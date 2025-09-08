# Whitespace Recommendation Microservice

AI-powered product recommendation system for CRM account whitespace analysis.

## Features

- **FastAPI** REST endpoints for easy integration
- **PostgreSQL** database with SQLAlchemy ORM
- **Machine Learning** recommendation engine (Random Forest, KNN, Association Rules)
- **Multi-tenant** architecture supporting multiple organizations
- **Docker** containerization for microservice deployment
- **UV** package management for modern Python dependency handling

## Architecture

- **API Layer**: FastAPI with async endpoints
- **ML Engine**: Scikit-learn models with 0.5 confidence threshold
- **Database**: PostgreSQL with proper schema design
- **Deployment**: Docker Compose for development, container-ready for production

## Quick Start

### Prerequisites
- Python 3.9+
- Docker & Docker Compose
- UV package manager

### Setup

1. **Clone repository**:
```bash
git clone https://github.com/YOUR_USERNAME/whitespace-microservice.git
cd whitespace-microservice
```

2. **Install dependencies**:
```bash
pip install uv
uv sync
```

3. **Configure environment**:
```bash
cp .env.template .env
# Edit .env with your database credentials
```

4. **Start PostgreSQL**:
```bash
docker-compose up -d postgres
```

5. **Load your data**:
```bash
# Place your Excel file as HC.xlsx in project root
uv run python data_loader.py
```

6. **Start the microservice**:
```bash
uv run python main.py
```

## API Endpoints

- `GET /health` - Service health check
- `POST /predict` - Generate whitespace recommendations
- `POST /accounts` - Create new account
- `GET /recommendations/{account_id}` - Get stored recommendations
- `GET /organizations/{org_name}/accounts` - Multi-tenant account access
- `POST /train` - Retrain ML models
- `GET /offerings` - Available products/services

## API Documentation

Interactive API docs available at: `http://localhost:8000/docs`

## Example Usage

### Generate Recommendations

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "account": {
      "name": "Acme Corp",
      "organization_name": "TechCorp Solutions",
      "industry": "Technology",
      "contacts": 15,
      "won_opps": 5,
      "won_revenue": 50000,
      "offerings_sold": ["Digital Foundation Services (DFS)"]
    }
  }'
```

### Check Service Health

```bash
curl http://localhost:8000/health
```

## Data Requirements

The system expects an Excel file (`HC.xlsx`) with these sheets:
- **Account Summary**: Account details, contacts, opportunities, revenue
- **All offerings**: Master catalog of products/services

*Note: HC.xlsx is excluded from git for data privacy*

## ML Models

- **Random Forest**: Primary recommendation engine (50% weight)
- **KNN Similarity**: Similar account patterns (30% weight) 
- **Association Rules**: Product co-occurrence (20% weight)
- **Threshold**: 0.5 minimum confidence for recommendations

## Multi-Tenant Support

The system supports multiple organizations:
- TechCorp Solutions
- MedHealth Systems  
- IndustrialWorks Inc
- FinanceFirst Corp
- LogiFlow Enterprises

## Environment Variables

Key configuration in `.env`:
```bash
DATABASE_URL=postgresql://postgres:password@localhost:5432/whitespace_db
ML_THRESHOLD=0.5
SERVICE_PORT=8000
```

## Development

### Project Structure
```
├── main.py              # FastAPI application
├── whitespace_ml.py     # ML recommendation engine
├── database.py          # Database models & config
├── data_loader.py       # Data processing pipeline
├── docker-compose.yml   # PostgreSQL container
└── .env                 # Configuration (not in git)
```

### Testing
```bash
# Test all endpoints
uv run python test_api.py

# Manual health check
curl http://localhost:8000/health
```

## Deployment

### Docker (Development)
```bash
docker-compose up -d
```

### Production
The microservice is container-ready and follows microservice architecture patterns for integration with existing systems.
