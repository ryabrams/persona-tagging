# Persona Classification UI - Phased Implementation Plan

**Document Version**: 1.0
**Created**: 2025-10-22
**Purpose**: Detailed, actionable implementation plan for building the web UI, broken into token-efficient phases

---

## Overview

This document breaks down the UI implementation (from [interface.md](interface.md)) into **small, manageable phases** designed to work within Claude's token budget. Each phase includes:

- **Clear scope** and deliverables
- **Specific implementation tasks** with file-level instructions
- **Testing checkpoints** to validate progress
- **Token efficiency strategies** (minimal context switching, focused work)

---

## Implementation Strategy

### Principles

1. **Incremental Development**: Build vertically (end-to-end features) rather than horizontally (all backends, then all frontends)
2. **Token Budget Management**:
   - Each phase targets 20K-40K tokens
   - Minimize file reads by working on related components together
   - Clear handoff points between phases
3. **Testing as You Go**: Each phase includes validation before moving forward
4. **Leverage Existing Code**: Reuse all existing Python scripts ([scripts/predict.py](scripts/predict.py), [scripts/train_model.py](scripts/train_model.py), etc.)

### Tech Stack (Confirmed)

**Backend**: FastAPI + Celery + Redis + PostgreSQL
**Frontend**: React + TypeScript + Material-UI (MUI)
**Deployment**: Docker Compose
**Testing**: pytest (backend), Jest + React Testing Library (frontend)

---

## Phase Breakdown

### Phase 0: Project Setup & Environment (Week 1)
**Goal**: Get all infrastructure running locally
**Token Budget**: ~15K tokens
**No existing code modification needed**

### Phase 1A: Backend - Prediction API Foundation (Week 2-3)
**Goal**: Basic prediction endpoint that wraps existing predict.py
**Token Budget**: ~35K tokens

### Phase 1B: Backend - Job Queue & Status (Week 3-4)
**Goal**: Async job processing with Celery
**Token Budget**: ~30K tokens

### Phase 1C: Frontend - Prediction Upload UI (Week 4-5)
**Goal**: Upload CSV and submit prediction jobs
**Token Budget**: ~35K tokens

### Phase 1D: Frontend - Results Display (Week 5-6)
**Goal**: View and download prediction results
**Token Budget**: ~35K tokens

### Phase 1E: Integration & Polish (Week 6-7)
**Goal**: Connect all pieces, add error handling, real-time updates
**Token Budget**: ~30K tokens

### Phase 2A: Backend - Training API (Week 8-9)
**Goal**: Model training endpoints
**Token Budget**: ~30K tokens

### Phase 2B: Frontend - Training UI (Week 9-10)
**Goal**: Training page with progress tracking
**Token Budget**: ~30K tokens

### Phase 2C: Configuration Management (Week 11-12)
**Goal**: Keyword rules and standardization CRUD
**Token Budget**: ~35K tokens

### Phase 3: Production Features (Week 13+)
**Goal**: Auth, monitoring, deployment
**Split into smaller sub-phases as needed**

---

## Detailed Phase Instructions

---

## Phase 0: Project Setup & Environment

### Objectives
- Create project structure
- Set up Docker Compose for local development
- Initialize backend and frontend projects
- Verify all services can communicate

### Tasks

#### Task 0.1: Create Project Structure
```
persona-tagging/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── api/                 # API routes
│   │   │   ├── __init__.py
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       └── endpoints/
│   │   ├── core/                # Config, database, celery
│   │   ├── models/              # SQLAlchemy models
│   │   ├── schemas/             # Pydantic schemas
│   │   ├── services/            # Business logic
│   │   └── workers/             # Celery tasks
│   ├── tests/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── alembic/                 # Database migrations
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/          # Reusable components
│   │   ├── pages/               # Page components
│   │   ├── services/            # API client
│   │   ├── hooks/               # Custom React hooks
│   │   ├── types/               # TypeScript types
│   │   ├── utils/               # Utilities
│   │   ├── App.tsx
│   │   └── index.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
├── nginx/
│   └── nginx.conf
├── docker-compose.yml
├── .env.example
└── README_UI.md                 # New README for UI project
```

**Instructions for Claude**:
```
Create the following files:
1. docker-compose.yml - Define services: backend, frontend, postgres, redis, celery_worker, nginx
2. backend/requirements.txt - Add: fastapi, uvicorn, celery, redis, sqlalchemy, psycopg2-binary, alembic, pydantic, python-multipart, pandas, joblib, scikit-learn
3. backend/Dockerfile - Python 3.11 base, copy requirements, install deps, copy app code
4. frontend/package.json - Create React app with TypeScript, MUI, react-router-dom, react-query, axios
5. frontend/Dockerfile - Node 18 base, multi-stage build (build -> nginx serve)
6. nginx/nginx.conf - Proxy /api to backend:8000, serve frontend static files
7. .env.example - Template for environment variables (DB credentials, Redis URL, etc.)
```

#### Task 0.2: Initialize Backend FastAPI App

**File**: `backend/app/main.py`
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Persona Classification API",
    version="1.0.0",
    description="API for persona classification system"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "api": "healthy"
        }
    }
```

**File**: `backend/app/core/config.py`
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://dev:dev@db:5432/persona_classifier"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # Celery
    CELERY_BROKER_URL: str = "redis://redis:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"

    # File upload
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    UPLOAD_DIR: str = "/app/uploads"
    RESULTS_DIR: str = "/app/results"

    # Model paths
    MODEL_PATH: str = "/app/model/persona_classifier.pkl"

    class Config:
        env_file = ".env"

settings = Settings()
```

#### Task 0.3: Initialize Frontend React App

**File**: `frontend/src/App.tsx`
```typescript
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

const theme = createTheme({
  palette: {
    primary: {
      main: '#2563EB',
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Routes>
          <Route path="/" element={<div>Home Page - Coming Soon</div>} />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
```

**File**: `frontend/src/services/api.ts`
```typescript
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check endpoint
export const healthCheck = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};
```

#### Task 0.4: Verification Steps

**Instructions for Claude**:
```
1. Run `docker-compose up --build` to start all services
2. Verify each service is healthy:
   - PostgreSQL: docker-compose exec db psql -U dev -d persona_classifier -c "SELECT 1;"
   - Redis: docker-compose exec redis redis-cli ping
   - Backend: curl http://localhost/api/v1/health
   - Frontend: Open http://localhost in browser
3. Check logs for each service (docker-compose logs <service>)
4. Document any issues in a setup_notes.md file
```

### Deliverables
- ✅ All Docker services running
- ✅ Backend returns healthy status
- ✅ Frontend loads in browser
- ✅ Database and Redis accessible
- ✅ Documentation of setup process

---

## Phase 1A: Backend - Prediction API Foundation

### Objectives
- Create database models for prediction jobs
- Implement file upload endpoint
- Integrate with existing [scripts/predict.py](scripts/predict.py)
- Return job status

### Tasks

#### Task 1A.1: Database Models

**File**: `backend/app/models/prediction.py`
```python
from sqlalchemy import Column, String, Integer, DateTime, JSON, Enum, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()

class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class PredictionJob(Base):
    __tablename__ = "prediction_jobs"

    id = Column(String(36), primary_key=True)  # UUID
    filename = Column(String(255), nullable=True)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, index=True)
    total_records = Column(Integer, nullable=True)
    processed_records = Column(Integer, default=0)
    progress = Column(Float, default=0.0)
    config = Column(JSON, nullable=True)  # Store prediction config
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(String(1000), nullable=True)
    result_file_path = Column(String(500), nullable=True)
```

**File**: `backend/app/core/database.py`
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Instructions**:
- Create Alembic migration: `alembic revision --autogenerate -m "Create prediction_jobs table"`
- Apply migration: `alembic upgrade head`

#### Task 1A.2: Pydantic Schemas

**File**: `backend/app/schemas/prediction.py`
```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime

class PredictionConfig(BaseModel):
    confidence_threshold: int = Field(default=50, ge=0, le=100)
    duplicate_handling: Literal["keep_first", "keep_last", "keep_all"] = "keep_first"
    priority_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    use_keyword_matching: bool = True
    use_title_standardization: bool = True

class PredictionJobCreate(BaseModel):
    method: Literal["file_upload", "manual_entry"]
    config: Optional[PredictionConfig] = None

class PredictionJobResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class PredictionJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    created_at: datetime
    completed_at: Optional[datetime] = None
    total_records: Optional[int] = None
    processed_records: int = 0
    error: Optional[str] = None
    result_url: Optional[str] = None
```

#### Task 1A.3: File Upload Endpoint

**File**: `backend/app/api/v1/endpoints/predictions.py`
```python
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.schemas.prediction import PredictionJobCreate, PredictionJobResponse, PredictionConfig
from app.services.prediction_service import PredictionService
import uuid
import json

router = APIRouter()

@router.post("/predictions", response_model=PredictionJobResponse, status_code=201)
async def create_prediction_job(
    file: UploadFile = File(...),
    config: str = Form(default="{}"),  # JSON string
    db: Session = Depends(get_db)
):
    """
    Submit a prediction job with CSV file upload.
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    # Parse config
    try:
        config_dict = json.loads(config) if config else {}
        prediction_config = PredictionConfig(**config_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid configuration: {str(e)}")

    # Create job
    service = PredictionService(db)
    job = await service.create_job(file, prediction_config)

    return PredictionJobResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at
    )

@router.get("/predictions/{job_id}", response_model=PredictionJobStatus)
async def get_prediction_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get prediction job status.
    """
    service = PredictionService(db)
    job = service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return PredictionJobStatus(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        created_at=job.created_at,
        completed_at=job.completed_at,
        total_records=job.total_records,
        processed_records=job.processed_records,
        error=job.error_message,
        result_url=f"/api/v1/predictions/{job_id}/results" if job.status == "completed" else None
    )
```

#### Task 1A.4: Prediction Service (Business Logic)

**File**: `backend/app/services/prediction_service.py`
```python
import os
import uuid
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import UploadFile
from app.models.prediction import PredictionJob, JobStatus
from app.schemas.prediction import PredictionConfig
from app.core.config import settings

class PredictionService:
    def __init__(self, db: Session):
        self.db = db

    async def create_job(self, file: UploadFile, config: PredictionConfig) -> PredictionJob:
        """
        Create a prediction job and save the uploaded file.
        """
        job_id = str(uuid.uuid4())

        # Ensure upload directory exists
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        # Save uploaded file
        file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}_{file.filename}")
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Quick validation: check if it's a valid CSV with required columns
        try:
            df = pd.read_csv(file_path)
            # Standardize column names (case-insensitive)
            df.columns = [col.strip() for col in df.columns]
            col_map = {col.lower(): col for col in df.columns}

            required_cols = ['record id', 'job title']
            for req_col in required_cols:
                if req_col not in col_map:
                    raise ValueError(f"Missing required column: {req_col}")

            total_records = len(df)
        except Exception as e:
            os.remove(file_path)  # Clean up
            raise ValueError(f"Invalid CSV file: {str(e)}")

        # Create job record
        job = PredictionJob(
            id=job_id,
            filename=file.filename,
            status=JobStatus.PENDING,
            total_records=total_records,
            config=config.model_dump(),
        )

        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)

        # TODO: Queue Celery task (Phase 1B)

        return job

    def get_job(self, job_id: str) -> PredictionJob:
        """
        Get prediction job by ID.
        """
        return self.db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
```

#### Task 1A.5: Register Routes

**File**: `backend/app/api/v1/api.py`
```python
from fastapi import APIRouter
from app.api.v1.endpoints import predictions

api_router = APIRouter()
api_router.include_router(predictions.router, tags=["predictions"])
```

**Update**: `backend/app/main.py`
```python
# Add after CORS middleware
from app.api.v1.api import api_router
app.include_router(api_router, prefix="/api/v1")
```

#### Task 1A.6: Testing

**File**: `backend/tests/test_predictions.py`
```python
from fastapi.testclient import TestClient
from app.main import app
import io

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_prediction_job():
    # Create a simple CSV file
    csv_content = "Record ID,Job Title\n1,AI Engineer\n2,Product Manager"
    csv_file = io.BytesIO(csv_content.encode('utf-8'))

    response = client.post(
        "/api/v1/predictions",
        files={"file": ("test.csv", csv_file, "text/csv")},
        data={"config": '{"confidence_threshold": 60}'}
    )

    assert response.status_code == 201
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "pending"

def test_get_prediction_status():
    # First create a job
    csv_content = "Record ID,Job Title\n1,AI Engineer"
    csv_file = io.BytesIO(csv_content.encode('utf-8'))

    create_response = client.post(
        "/api/v1/predictions",
        files={"file": ("test.csv", csv_file, "text/csv")}
    )
    job_id = create_response.json()["job_id"]

    # Then check status
    response = client.get(f"/api/v1/predictions/{job_id}")
    assert response.status_code == 200
    assert response.json()["job_id"] == job_id
```

**Run tests**: `pytest backend/tests/test_predictions.py -v`

### Deliverables
- ✅ Database schema for prediction jobs
- ✅ File upload endpoint working
- ✅ Basic validation (CSV format, required columns)
- ✅ Job creation returns job_id
- ✅ Status endpoint returns job info
- ✅ Tests passing

---

## Phase 1B: Backend - Job Queue & Status

### Objectives
- Set up Celery worker to process predictions asynchronously
- Integrate existing [scripts/predict.py](scripts/predict.py) into Celery task
- Update job status in real-time
- Handle errors gracefully

### Tasks

#### Task 1B.1: Celery Configuration

**File**: `backend/app/core/celery_app.py`
```python
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "persona_classifier",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=['app.workers.prediction_tasks']
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
)
```

#### Task 1B.2: Prediction Task (Wraps existing predict.py)

**File**: `backend/app/workers/prediction_tasks.py`
```python
import os
import sys
import logging
from celery import Task
from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.prediction import PredictionJob, JobStatus
from datetime import datetime

# Import existing prediction script
# Add the scripts directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from scripts.predict import run_prediction  # Assuming we refactor predict.py to expose this function

logger = logging.getLogger(__name__)

class PredictionTask(Task):
    """Custom task class for handling job status updates."""

    def on_success(self, retval, task_id, args, kwargs):
        """Update job status on success."""
        job_id = args[0]
        db = SessionLocal()
        try:
            job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                job.progress = 100.0
                db.commit()
        finally:
            db.close()

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Update job status on failure."""
        job_id = args[0]
        db = SessionLocal()
        try:
            job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
            if job:
                job.status = JobStatus.FAILED
                job.error_message = str(exc)[:1000]
                job.completed_at = datetime.utcnow()
                db.commit()
        finally:
            db.close()

@celery_app.task(base=PredictionTask, bind=True)
def process_prediction(self, job_id: str):
    """
    Process a prediction job using the existing predict.py script.
    """
    db = SessionLocal()

    try:
        # Get job details
        job = db.query(PredictionJob).filter(PredictionJob.id == job_id).first()
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Update status to processing
        job.status = JobStatus.PROCESSING
        job.progress = 10.0
        db.commit()

        # Prepare file paths
        input_file = os.path.join(settings.UPLOAD_DIR, f"{job_id}_{job.filename}")
        output_file = os.path.join(settings.RESULTS_DIR, f"{job_id}_results.csv")
        os.makedirs(settings.RESULTS_DIR, exist_ok=True)

        # Set environment variables for prediction configuration
        config = job.config or {}
        os.environ['PC_CONFIDENCE_THRESHOLD'] = str(config.get('confidence_threshold', 50))
        os.environ['PC_DUPLICATE_HANDLING'] = config.get('duplicate_handling', 'keep_first')

        # Update progress
        job.progress = 30.0
        db.commit()

        # Run prediction using existing script
        # NOTE: You'll need to refactor predict.py to expose a function that accepts input/output paths
        run_prediction(
            input_file=input_file,
            output_file=output_file,
            model_path=settings.MODEL_PATH,
            # Add other parameters as needed
        )

        # Update progress
        job.progress = 90.0
        job.result_file_path = output_file
        db.commit()

        # Final status update handled by on_success

    except Exception as e:
        logger.error(f"Error processing prediction {job_id}: {str(e)}")
        raise
    finally:
        db.close()
```

**Important**: You'll need to refactor [scripts/predict.py](scripts/predict.py) to expose a `run_prediction()` function. Current script runs as main. Extract core logic into a function.

**File**: `scripts/predict.py` (refactored - partial example)
```python
# Add this function to existing predict.py
def run_prediction(input_file: str, output_file: str, model_path: str,
                   keyword_file: str = None, standardization_file: str = None):
    """
    Run prediction on input file and save results to output file.

    Args:
        input_file: Path to input CSV with Record ID and Job Title
        output_file: Path to save results CSV
        model_path: Path to trained model pickle file
        keyword_file: Optional path to keyword matching CSV
        standardization_file: Optional path to title reference CSV
    """
    # Move all the logic from if __name__ == "__main__" here
    # Keep the same functionality, just make it callable
    # ... existing predict.py code ...
    pass

# Keep existing if __name__ == "__main__" for CLI usage
if __name__ == "__main__":
    # Parse args and call run_prediction()
    pass
```

#### Task 1B.3: Trigger Celery Task from API

**Update**: `backend/app/services/prediction_service.py`
```python
# Add import at top
from app.workers.prediction_tasks import process_prediction

# Update create_job method - add at the end before return:
        # Queue Celery task
        process_prediction.delay(job_id)

        return job
```

#### Task 1B.4: Results Endpoint

**File**: `backend/app/api/v1/endpoints/predictions.py` (add new endpoint)
```python
from fastapi.responses import FileResponse
import os

@router.get("/predictions/{job_id}/results")
async def get_prediction_results(
    job_id: str,
    format: str = "csv",  # csv or json
    db: Session = Depends(get_db)
):
    """
    Download prediction results.
    """
    service = PredictionService(db)
    job = service.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job is {job.status}, not completed")

    if not job.result_file_path or not os.path.exists(job.result_file_path):
        raise HTTPException(status_code=404, detail="Results file not found")

    if format == "csv":
        return FileResponse(
            job.result_file_path,
            media_type="text/csv",
            filename=f"results_{job_id}.csv"
        )
    elif format == "json":
        # Read CSV and convert to JSON
        df = pd.read_csv(job.result_file_path)
        return df.to_dict(orient='records')
    else:
        raise HTTPException(status_code=400, detail="Format must be 'csv' or 'json'")
```

#### Task 1B.5: Testing

**Test Celery worker locally**:
```bash
# Terminal 1: Start Celery worker
docker-compose exec backend celery -A app.core.celery_app worker --loglevel=info

# Terminal 2: Submit a test job
curl -X POST http://localhost/api/v1/predictions \
  -F "file=@data/input.csv" \
  -F 'config={"confidence_threshold": 50}'

# Terminal 3: Check job status (use job_id from response)
curl http://localhost/api/v1/predictions/{job_id}
```

### Deliverables
- ✅ Celery worker processes jobs asynchronously
- ✅ Existing predict.py integrated into task
- ✅ Job status updates (pending → processing → completed/failed)
- ✅ Results can be downloaded
- ✅ Error handling works

---

## Phase 1C: Frontend - Prediction Upload UI

### Objectives
- Create prediction page with file upload
- Configuration form
- Submit job and navigate to status page
- Form validation

### Tasks

#### Task 1C.1: Type Definitions

**File**: `frontend/src/types/api.ts`
```typescript
export interface PredictionConfig {
  confidence_threshold: number;
  duplicate_handling: 'keep_first' | 'keep_last' | 'keep_all';
  priority_threshold: number;
  use_keyword_matching: boolean;
  use_title_standardization: boolean;
}

export interface PredictionJob {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
}

export interface PredictionJobStatus extends PredictionJob {
  progress: number;
  completed_at?: string;
  total_records?: number;
  processed_records: number;
  error?: string;
  result_url?: string;
}
```

#### Task 1C.2: API Service

**File**: `frontend/src/services/predictionApi.ts`
```typescript
import { apiClient } from './api';
import { PredictionConfig, PredictionJob, PredictionJobStatus } from '../types/api';

export const predictionApi = {
  async createJob(file: File, config?: Partial<PredictionConfig>): Promise<PredictionJob> {
    const formData = new FormData();
    formData.append('file', file);

    if (config) {
      formData.append('config', JSON.stringify(config));
    }

    const response = await apiClient.post<PredictionJob>('/predictions', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  },

  async getJobStatus(jobId: string): Promise<PredictionJobStatus> {
    const response = await apiClient.get<PredictionJobStatus>(`/predictions/${jobId}`);
    return response.data;
  },

  async downloadResults(jobId: string, format: 'csv' | 'json' = 'csv'): Promise<Blob> {
    const response = await apiClient.get(`/predictions/${jobId}/results`, {
      params: { format },
      responseType: format === 'csv' ? 'blob' : 'json',
    });
    return response.data;
  },
};
```

#### Task 1C.3: File Upload Component

**File**: `frontend/src/components/FileUploadZone.tsx`
```typescript
import React, { useCallback, useState } from 'react';
import { Box, Typography, Paper } from '@mui/material';
import { CloudUpload as UploadIcon } from '@mui/icons-material';

interface FileUploadZoneProps {
  onFileSelect: (file: File) => void;
  accept?: string;
}

export const FileUploadZone: React.FC<FileUploadZoneProps> = ({
  onFileSelect,
  accept = '.csv'
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith('.csv')) {
      setFile(droppedFile);
      onFileSelect(droppedFile);
    }
  }, [onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      onFileSelect(selectedFile);
    }
  }, [onFileSelect]);

  return (
    <Paper
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      sx={{
        border: `2px dashed ${isDragging ? '#2563EB' : '#E5E7EB'}`,
        backgroundColor: isDragging ? '#F0F9FF' : '#F9FAFB',
        padding: 4,
        textAlign: 'center',
        cursor: 'pointer',
        transition: 'all 0.2s',
      }}
      onClick={() => document.getElementById('file-input')?.click()}
    >
      <input
        id="file-input"
        type="file"
        accept={accept}
        onChange={handleFileInput}
        style={{ display: 'none' }}
      />

      <UploadIcon sx={{ fontSize: 48, color: '#9CA3AF', mb: 2 }} />

      {file ? (
        <Typography variant="body1">
          Selected: {file.name} ({(file.size / 1024).toFixed(2)} KB)
        </Typography>
      ) : (
        <>
          <Typography variant="h6" gutterBottom>
            Drag & drop CSV file here
          </Typography>
          <Typography variant="body2" color="text.secondary">
            or click to browse
          </Typography>
          <Typography variant="caption" display="block" sx={{ mt: 2 }}>
            Required columns: Record ID, Job Title
          </Typography>
        </>
      )}
    </Paper>
  );
};
```

#### Task 1C.4: Configuration Form Component

**File**: `frontend/src/components/PredictionConfigForm.tsx`
```typescript
import React from 'react';
import {
  Box,
  Slider,
  Typography,
  FormControl,
  FormLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  Paper,
} from '@mui/material';
import { PredictionConfig } from '../types/api';

interface PredictionConfigFormProps {
  config: Partial<PredictionConfig>;
  onChange: (config: Partial<PredictionConfig>) => void;
}

const defaultConfig: PredictionConfig = {
  confidence_threshold: 50,
  duplicate_handling: 'keep_first',
  priority_threshold: 0.7,
  use_keyword_matching: true,
  use_title_standardization: true,
};

export const PredictionConfigForm: React.FC<PredictionConfigFormProps> = ({
  config,
  onChange,
}) => {
  const currentConfig = { ...defaultConfig, ...config };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Configuration
      </Typography>

      <Box sx={{ mb: 3 }}>
        <FormLabel>Confidence Threshold: {currentConfig.confidence_threshold}%</FormLabel>
        <Slider
          value={currentConfig.confidence_threshold}
          onChange={(_, value) =>
            onChange({ ...config, confidence_threshold: value as number })
          }
          min={0}
          max={100}
          marks={[
            { value: 0, label: '0%' },
            { value: 50, label: '50%' },
            { value: 100, label: '100%' },
          ]}
        />
        <Typography variant="caption" color="text.secondary">
          Minimum confidence score to assign a persona
        </Typography>
      </Box>

      <Box sx={{ mb: 3 }}>
        <FormControl fullWidth>
          <FormLabel>Duplicate Handling</FormLabel>
          <Select
            value={currentConfig.duplicate_handling}
            onChange={(e) =>
              onChange({
                ...config,
                duplicate_handling: e.target.value as PredictionConfig['duplicate_handling'],
              })
            }
          >
            <MenuItem value="keep_first">Keep First</MenuItem>
            <MenuItem value="keep_last">Keep Last</MenuItem>
            <MenuItem value="keep_all">Keep All</MenuItem>
          </Select>
        </FormControl>
      </Box>

      <Box sx={{ mb: 2 }}>
        <FormControlLabel
          control={
            <Switch
              checked={currentConfig.use_keyword_matching}
              onChange={(e) =>
                onChange({ ...config, use_keyword_matching: e.target.checked })
              }
            />
          }
          label="Use Keyword Matching"
        />
      </Box>

      <Box>
        <FormControlLabel
          control={
            <Switch
              checked={currentConfig.use_title_standardization}
              onChange={(e) =>
                onChange({ ...config, use_title_standardization: e.target.checked })
              }
            />
          }
          label="Use Title Standardization"
        />
      </Box>
    </Paper>
  );
};
```

#### Task 1C.5: Prediction Page

**File**: `frontend/src/pages/PredictionPage.tsx`
```typescript
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Box,
  Typography,
  Button,
  Stepper,
  Step,
  StepLabel,
  Alert,
} from '@mui/material';
import { FileUploadZone } from '../components/FileUploadZone';
import { PredictionConfigForm } from '../components/PredictionConfigForm';
import { predictionApi } from '../services/predictionApi';
import { PredictionConfig } from '../types/api';

const steps = ['Upload File', 'Configure', 'Submit'];

export const PredictionPage: React.FC = () => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [config, setConfig] = useState<Partial<PredictionConfig>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setError(null);
  };

  const handleNext = () => {
    if (activeStep === 0 && !file) {
      setError('Please select a file');
      return;
    }
    setActiveStep((prev) => prev + 1);
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
    setError(null);
  };

  const handleSubmit = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const job = await predictionApi.createJob(file, config);
      navigate(`/predictions/${job.job_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to submit job');
    } finally {
      setLoading(false);
    }
  };

  const renderStepContent = () => {
    switch (activeStep) {
      case 0:
        return <FileUploadZone onFileSelect={handleFileSelect} />;
      case 1:
        return <PredictionConfigForm config={config} onChange={setConfig} />;
      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Review & Submit
            </Typography>
            <Typography>File: {file?.name}</Typography>
            <Typography>
              Confidence Threshold: {config.confidence_threshold || 50}%
            </Typography>
            <Typography>
              Duplicate Handling: {config.duplicate_handling || 'keep_first'}
            </Typography>
          </Box>
        );
      default:
        return null;
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Typography variant="h4" gutterBottom>
        New Prediction
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Box sx={{ mb: 4 }}>{renderStepContent()}</Box>

      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button disabled={activeStep === 0} onClick={handleBack}>
          Back
        </Button>
        {activeStep === steps.length - 1 ? (
          <Button
            variant="contained"
            onClick={handleSubmit}
            disabled={loading || !file}
          >
            {loading ? 'Submitting...' : 'Submit'}
          </Button>
        ) : (
          <Button variant="contained" onClick={handleNext}>
            Next
          </Button>
        )}
      </Box>
    </Container>
  );
};
```

#### Task 1C.6: Update Routing

**Update**: `frontend/src/App.tsx`
```typescript
import { PredictionPage } from './pages/PredictionPage';

// Add route
<Route path="/predictions/new" element={<PredictionPage />} />
```

### Deliverables
- ✅ File upload UI with drag-and-drop
- ✅ Configuration form with all settings
- ✅ Step-by-step wizard
- ✅ Form validation
- ✅ Submit job and get job_id

---

## Phase 1D: Frontend - Results Display

**(Continues with similar detailed instructions for results page, job status polling, etc.)**

### Objectives
- Create results page showing job status
- Display classification results in a table
- Download results as CSV
- Filter and sort results

**[Detailed implementation instructions would continue here...]**

---

## Token Budget Management Strategy

### Between Phases
1. **Checkpoint**: After each phase, document what was completed in a `progress.md` file
2. **Minimal Context**: Start next phase with only necessary file reads
3. **Incremental Testing**: Test each component as you build (don't wait for full integration)

### During Implementation
1. **Batch Related Tasks**: Work on all backend models together, then all frontend components
2. **Avoid Re-reads**: Once you read a file, complete all tasks for that file
3. **Use Code References**: Reference existing patterns instead of re-reading similar files

---

## Success Criteria for MVP (End of Phase 1E)

- [ ] User can upload CSV file via web UI
- [ ] Predictions run asynchronously in background
- [ ] User can see real-time progress
- [ ] Results display in table with sorting/filtering
- [ ] User can download results as CSV
- [ ] Errors are handled gracefully with helpful messages
- [ ] All existing predict.py functionality works through UI
- [ ] System deployed and running via Docker Compose

---

## Next Steps After Reading This

**For Phase 0 (Project Setup)**:
1. Confirm tech stack choices (React + FastAPI + Docker)
2. I'll create all initial project structure files
3. Set up Docker Compose
4. Verify all services start successfully
5. Create initial smoke tests

**Do you want me to start with Phase 0 now?**
