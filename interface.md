# UI Implementation Specification for Persona Classification System

## Executive Summary

This document outlines the technical and design requirements for implementing a web-based user interface for the Persona Classification System. The UI will provide an accessible, user-friendly alternative to the current CLI-based workflow while maintaining the system's robust classification capabilities.

---

## 1. Technical Architecture

### 1.1 Technology Stack Recommendations

#### Frontend
- **Framework**: React 18+ with TypeScript
  - **Rationale**: Component reusability, strong typing, large ecosystem, excellent file upload handling
  - **Alternative**: Vue.js 3 (lighter weight, gentler learning curve)

- **State Management**: React Context API + React Query
  - **Rationale**: Built-in for simple state, React Query for async data fetching and caching
  - **Alternative**: Redux Toolkit (for more complex state requirements)

- **UI Component Library**: Material-UI (MUI) v5 or shadcn/ui
  - **Rationale**: Professional appearance, accessibility built-in, comprehensive component set
  - **Alternative**: Ant Design (more enterprise-focused)

- **File Handling**: react-dropzone + Papa Parse (CSV parsing)
  - **Rationale**: Drag-and-drop UX, robust CSV parsing with validation

- **Data Visualization**: Recharts or Chart.js
  - **Rationale**: Display confidence distributions, persona breakdowns, model metrics

- **Table Component**: TanStack Table (formerly React Table)
  - **Rationale**: Powerful sorting, filtering, pagination for results display

#### Backend API Layer
- **Framework**: FastAPI (Python)
  - **Rationale**:
    - Native async support for handling large file uploads
    - Automatic OpenAPI documentation
    - Type hints align with existing Python codebase
    - Fast performance with async workers
  - **Alternative**: Flask (simpler but synchronous)

- **Task Queue**: Celery + Redis
  - **Rationale**: Handle long-running training/prediction jobs asynchronously
  - **Alternative**: RQ (simpler but less feature-rich)

- **File Storage**: Local filesystem with configurable path + optional cloud storage (S3/Azure Blob)
  - **Rationale**: Match current system architecture, add cloud option for scalability

- **Database**: PostgreSQL
  - **Rationale**: Store job history, user configurations, model metadata
  - **Alternative**: SQLite (simpler deployment, single-user scenarios)

#### Infrastructure
- **Containerization**: Docker + Docker Compose
  - **Services**: Web frontend, API backend, Celery workers, Redis, PostgreSQL
  - **Rationale**: Consistent deployment, easy scaling, development/production parity

- **Web Server**: Nginx (reverse proxy)
  - **Rationale**: Serve static files, handle large uploads, SSL termination

### 1.2 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User Browser                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         React Frontend (TypeScript)                 │   │
│  │  • Upload Interface  • Results Dashboard            │   │
│  │  • Model Training UI • Configuration Panel          │   │
│  └──────────────────┬──────────────────────────────────┘   │
└────────────────────┼────────────────────────────────────────┘
                     │ REST API (JSON)
                     │
┌────────────────────▼────────────────────────────────────────┐
│                   Nginx Reverse Proxy                        │
│           (Static Files + API Proxy + WebSockets)            │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼────┐  ┌───▼───┐  ┌────▼─────┐
   │ FastAPI │  │ Redis │  │PostgreSQL│
   │   API   │  │ Queue │  │    DB    │
   └────┬────┘  └───┬───┘  └──────────┘
        │           │
        │      ┌────▼─────────────┐
        │      │ Celery Workers   │
        │      │ (Background Jobs)│
        │      └────┬─────────────┘
        │           │
   ┌────▼───────────▼────────────────┐
   │    Existing Python Scripts      │
   │  • train_model.py               │
   │  • predict.py                   │
   │  • title_standardizer.py        │
   └─────────────────────────────────┘
```

---

## 2. Backend API Specifications

### 2.1 API Endpoints

#### 2.1.1 Prediction Endpoints

**POST /api/v1/predictions**
- **Purpose**: Submit job titles for classification
- **Request**:
  ```json
  {
    "method": "file_upload" | "manual_entry",
    "file": "multipart/form-data CSV file (if method=file_upload)",
    "entries": [                         // if method=manual_entry
      {
        "record_id": "string",
        "job_title": "string"
      }
    ],
    "config": {
      "confidence_threshold": 50,        // 0-100
      "duplicate_handling": "keep_first", // keep_first|keep_last|keep_all
      "priority_threshold": 0.7,         // 0.0-1.0
      "use_keyword_matching": true,
      "use_title_standardization": true
    }
  }
  ```
- **Response**:
  ```json
  {
    "job_id": "uuid",
    "status": "pending",
    "created_at": "ISO8601 timestamp"
  }
  ```
- **Status Codes**: 201 (Created), 400 (Invalid input), 413 (File too large), 500 (Server error)

**GET /api/v1/predictions/{job_id}**
- **Purpose**: Check prediction job status
- **Response**:
  ```json
  {
    "job_id": "uuid",
    "status": "pending" | "processing" | "completed" | "failed",
    "progress": 75,                      // 0-100
    "created_at": "ISO8601 timestamp",
    "completed_at": "ISO8601 timestamp",
    "total_records": 1000,
    "processed_records": 750,
    "result_url": "/api/v1/predictions/{job_id}/results",
    "error": "Error message if failed"
  }
  ```

**GET /api/v1/predictions/{job_id}/results**
- **Purpose**: Retrieve prediction results
- **Query Parameters**:
  - `page=1` (pagination)
  - `limit=100` (results per page)
  - `format=json|csv` (response format)
  - `filter_segment=GenAI` (filter by persona)
  - `min_confidence=50` (filter by confidence)
- **Response** (JSON format):
  ```json
  {
    "job_id": "uuid",
    "pagination": {
      "page": 1,
      "limit": 100,
      "total_records": 1000,
      "total_pages": 10
    },
    "summary": {
      "total_records": 1000,
      "classified": 850,
      "not_classified": 150,
      "keyword_matched": 200,
      "ml_predicted": 650,
      "persona_distribution": {
        "GenAI": 120,
        "Engineering": 350,
        "Product": 200,
        "Cyber Security": 80,
        "Trust & Safety": 50,
        "Legal & Compliance": 30,
        "Executive": 20,
        "Not Classified": 150
      },
      "confidence_distribution": {
        "50-60": 100,
        "60-70": 150,
        "70-80": 200,
        "80-90": 250,
        "90-100": 300
      }
    },
    "results": [
      {
        "record_id": "string",
        "job_title": "string",
        "persona_segment": "GenAI",
        "confidence_score": 85,
        "classification_method": "ml" | "keyword" | "unclassified"
      }
    ]
  }
  ```
- **Response** (CSV format): Standard CSV download with headers

**GET /api/v1/predictions**
- **Purpose**: List all prediction jobs (with pagination and filtering)
- **Query Parameters**:
  - `page=1`
  - `limit=20`
  - `status=completed` (filter by status)
  - `from_date=ISO8601` (filter by creation date)
- **Response**:
  ```json
  {
    "pagination": { "page": 1, "limit": 20, "total": 45 },
    "jobs": [
      {
        "job_id": "uuid",
        "status": "completed",
        "created_at": "ISO8601",
        "total_records": 1000,
        "filename": "original_filename.csv"
      }
    ]
  }
  ```

**DELETE /api/v1/predictions/{job_id}**
- **Purpose**: Delete a prediction job and its results
- **Response**: 204 (No Content)

#### 2.1.2 Training Endpoints

**POST /api/v1/training**
- **Purpose**: Train a new model
- **Request**:
  ```json
  {
    "training_file": "multipart/form-data CSV file",
    "config": {
      "test_size": 0.2,                 // 0.1-0.5
      "max_features": 5000,             // min 100
      "model_name": "custom_model_v1"   // optional identifier
    }
  }
  ```
- **Response**:
  ```json
  {
    "job_id": "uuid",
    "status": "pending"
  }
  ```

**GET /api/v1/training/{job_id}**
- **Purpose**: Check training job status
- **Response**:
  ```json
  {
    "job_id": "uuid",
    "status": "pending" | "training" | "completed" | "failed",
    "progress": 60,
    "stage": "data_validation" | "model_training" | "evaluation" | "saving",
    "error": "Error message if failed",
    "metrics": {                        // only when completed
      "test_accuracy": 0.883,
      "cv_mean_accuracy": 0.876,
      "cv_std": 0.042,
      "total_samples": 1630,
      "unique_titles": 1245,
      "persona_distribution": {
        "Engineering": 523,
        "Product": 387
      },
      "classification_report": {
        "Engineering": {
          "precision": 0.92,
          "recall": 0.89,
          "f1_score": 0.90,
          "support": 105
        }
      }
    }
  }
  ```

**GET /api/v1/training**
- **Purpose**: List training jobs
- **Response**: Similar to predictions list with training-specific fields

#### 2.1.3 Configuration Endpoints

**GET /api/v1/config**
- **Purpose**: Get current system configuration
- **Response**:
  ```json
  {
    "personas": ["GenAI", "Engineering", "Product", ...],
    "priority_order": ["GenAI", "Engineering", ...],
    "default_config": {
      "confidence_threshold": 50,
      "duplicate_handling": "keep_first",
      "priority_threshold": 0.7,
      "similarity_range": 0.1,
      "max_title_length": 500
    },
    "model_info": {
      "last_trained": "ISO8601",
      "model_version": "v1.2.3",
      "training_samples": 1630
    },
    "features_enabled": {
      "keyword_matching": true,
      "title_standardization": true
    }
  }
  ```

**GET /api/v1/config/keywords**
- **Purpose**: Get keyword matching rules
- **Response**:
  ```json
  {
    "rules": [
      {
        "id": 1,
        "keyword": "chief executive",
        "rule": "contains",
        "persona_segment": "Executive",
        "exclude_keyword": ""
      }
    ]
  }
  ```

**PUT /api/v1/config/keywords**
- **Purpose**: Update keyword matching rules
- **Request**: Array of keyword rules
- **Response**: Updated rules

**GET /api/v1/config/standardization**
- **Purpose**: Get title standardization mappings
- **Response**:
  ```json
  {
    "mappings": [
      {
        "id": 1,
        "reference": "Sr. PM",
        "standardization": "Senior Product Manager"
      }
    ],
    "stats": {
      "total_mappings": 150,
      "cache_hits": 450,
      "cache_misses": 50
    }
  }
  ```

**PUT /api/v1/config/standardization**
- **Purpose**: Update standardization mappings
- **Request**: Array of mappings
- **Response**: Updated mappings

#### 2.1.4 Model Management Endpoints

**GET /api/v1/models**
- **Purpose**: List all trained models
- **Response**:
  ```json
  {
    "models": [
      {
        "id": "uuid",
        "name": "custom_model_v1",
        "created_at": "ISO8601",
        "is_active": true,
        "metrics": {
          "test_accuracy": 0.883,
          "total_samples": 1630
        }
      }
    ]
  }
  ```

**POST /api/v1/models/{model_id}/activate**
- **Purpose**: Set a model as the active model for predictions
- **Response**: 200 (OK)

**DELETE /api/v1/models/{model_id}**
- **Purpose**: Delete a trained model
- **Response**: 204 (No Content)

#### 2.1.5 Validation Endpoints

**POST /api/v1/validate/input**
- **Purpose**: Validate input CSV before submission
- **Request**: Multipart form data with CSV file
- **Response**:
  ```json
  {
    "valid": true,
    "errors": [],
    "warnings": [
      "5 duplicate Record IDs found"
    ],
    "stats": {
      "total_records": 1000,
      "columns": ["Record ID", "Job Title"],
      "duplicate_ids": 5,
      "empty_titles": 2,
      "max_title_length": 120
    }
  }
  ```

**POST /api/v1/validate/training**
- **Purpose**: Validate training CSV before submission
- **Response**: Similar validation structure with training-specific checks

#### 2.1.6 Health & Monitoring

**GET /api/v1/health**
- **Purpose**: Health check endpoint
- **Response**:
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "components": {
      "database": "healthy",
      "redis": "healthy",
      "celery": "healthy",
      "model": "loaded"
    }
  }
  ```

**GET /api/v1/stats**
- **Purpose**: System statistics
- **Response**:
  ```json
  {
    "total_predictions": 50000,
    "total_training_jobs": 25,
    "active_jobs": 3,
    "disk_usage": {
      "models": "250MB",
      "results": "1.2GB"
    }
  }
  ```

### 2.2 WebSocket Support

**WS /api/v1/ws/jobs/{job_id}**
- **Purpose**: Real-time job progress updates
- **Messages**:
  ```json
  {
    "type": "progress",
    "job_id": "uuid",
    "progress": 75,
    "stage": "processing",
    "processed_records": 750,
    "total_records": 1000,
    "timestamp": "ISO8601"
  }
  ```

### 2.3 Authentication & Authorization (Optional, Phase 2)

**POST /api/v1/auth/login**
- **Purpose**: User authentication
- **Request**: `{ "username": "user", "password": "pass" }`
- **Response**: `{ "access_token": "JWT", "refresh_token": "JWT" }`

**POST /api/v1/auth/logout**
- **Purpose**: Invalidate tokens

**Middleware**: JWT bearer token authentication for all endpoints except /health

---

## 3. Frontend Design Specifications

### 3.1 Page Structure

#### 3.1.1 Main Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Header: Logo | Navigation | Model Status | Settings        │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────────────────────────┐  │
│  │   Sidebar     │  │         Main Content Area         │  │
│  │               │  │                                   │  │
│  │ • Home        │  │  Dynamic content based on         │  │
│  │ • Predict     │  │  selected navigation item         │  │
│  │ • Train       │  │                                   │  │
│  │ • Jobs        │  │                                   │  │
│  │ • Config      │  │                                   │  │
│  │ • History     │  │                                   │  │
│  └───────────────┘  └───────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│  Footer: Version | API Status | Documentation Link          │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Core Pages/Views

#### 3.2.1 Home/Dashboard

**Purpose**: System overview and quick actions

**Components**:
- **Hero Section**: Welcome message, quick start buttons
- **Model Status Card**:
  - Current model accuracy
  - Last trained date
  - Training samples count
  - Quick retrain button
- **Recent Activity**: List of recent prediction/training jobs
- **Statistics Overview**:
  - Total predictions processed (all-time)
  - Total jobs today
  - Most common persona (pie chart)
  - Average confidence score (gauge)
- **Quick Actions**:
  - "New Prediction" button (primary CTA)
  - "Train Model" button
  - "View Results" button

**Design Notes**:
- Use card-based layout for easy scanning
- Color-coded status indicators (green=healthy, yellow=warning, red=error)
- Responsive grid system (3 columns desktop, 1 column mobile)

#### 3.2.2 Prediction Page

**Purpose**: Upload files and configure predictions

**Layout**: Wizard-style multi-step form

**Step 1: Upload Input**
- **File Upload Zone**:
  - Drag-and-drop area with click-to-browse fallback
  - Shows file name, size, record count after upload
  - CSV format validation (instant feedback)
  - Preview first 5 rows in table format
  - "Validate File" button (optional pre-check)
- **Manual Entry Option** (collapsible):
  - Data grid for entering records manually
  - Add/remove row buttons
  - CSV paste support (detect and parse)

**Step 2: Configure Settings**
- **Configuration Panel** (form with tooltips):
  - Confidence Threshold slider (0-100, default: 50)
  - Duplicate Handling dropdown (keep_first|keep_last|keep_all)
  - Priority Threshold slider (0.0-1.0, default: 0.7)
  - Toggle switches:
    - Use Keyword Matching (default: on)
    - Use Title Standardization (default: on)
  - "Use Default Settings" button
  - "Save as Preset" button (future feature)
- **Preview Panel**:
  - Shows how settings affect classification
  - Example record with predicted outcome

**Step 3: Review & Submit**
- **Summary Section**:
  - File name and record count
  - Selected configuration
  - Estimated processing time
- **Action Buttons**:
  - "Back" button
  - "Submit" button (primary, large)

**Step 4: Processing & Results**
- **Progress View** (while processing):
  - Progress bar with percentage
  - Current stage indicator (e.g., "Applying keyword rules...")
  - Records processed count (e.g., "750/1000")
  - Estimated time remaining
  - Cancel job button
- **Results View** (on completion):
  - Automatically navigate to results page
  - Success notification with summary stats

**Design Notes**:
- Clear step indicators (1/4, 2/4, etc.)
- Breadcrumb navigation
- Form validation with inline error messages
- Auto-save draft (localStorage)
- Keyboard shortcuts (Enter to proceed, Esc to cancel)

#### 3.2.3 Results Page

**Purpose**: Display and analyze classification results

**Layout**: Split view with summary and details

**Summary Panel** (top section):
- **Key Metrics Row** (large number displays):
  - Total Records
  - Classified (with percentage)
  - Not Classified (with percentage)
  - Average Confidence Score
- **Classification Breakdown** (2-column layout):
  - **Left**: Persona Distribution (bar chart or pie chart)
    - Each persona with count and percentage
    - Color-coded by persona
    - Interactive (click to filter table)
  - **Right**: Confidence Distribution (histogram)
    - Buckets: 50-60, 60-70, 70-80, 80-90, 90-100
    - Show average by persona
- **Method Breakdown** (small cards):
  - Keyword Matched count
  - ML Predicted count
  - Unclassified count

**Actions Bar**:
- **Export Options**:
  - Download CSV button (full results)
  - Download JSON button
  - Download Report PDF (with charts)
- **Filter Controls**:
  - Persona multi-select dropdown
  - Confidence range slider
  - Classification method filter (keyword|ml|unclassified)
  - Search box (filter by job title or record ID)
- **View Options**:
  - Toggle: "Show only unclassified"
  - Toggle: "Group by persona"
  - Sort dropdown (confidence, record ID, persona)

**Results Table** (main content):
- **Columns**:
  1. Record ID
  2. Job Title (with truncation, hover for full text)
  3. Persona Segment (color-coded badge)
  4. Confidence Score (progress bar visualization)
  5. Method (icon: keyword/ML/unclassified)
  6. Actions (eye icon for details)
- **Features**:
  - Sortable columns
  - Sticky header
  - Pagination (100 records per page, configurable)
  - Row selection (for bulk operations)
  - Highlight rows below confidence threshold
  - Expandable rows for additional details

**Details Modal** (click on eye icon):
- Record information
- All probability scores for each persona (bar chart)
- Applied keyword rules (if any)
- Standardized title (if applied)
- Edit classification option (manual override)

**Design Notes**:
- Responsive: Summary becomes vertical on mobile
- Loading states for async operations
- Empty state when no results
- Tooltips on all metrics
- Accessibility: keyboard navigation, ARIA labels

#### 3.2.4 Training Page

**Purpose**: Train new models with custom data

**Layout**: Similar wizard structure to Prediction page

**Step 1: Upload Training Data**
- **File Upload Zone**:
  - Drag-and-drop with validation
  - Preview first 10 rows
  - Data quality checks (instant feedback):
    - Column validation (Job Title, Persona Segment)
    - Minimum sample requirements (10 total, 2+ personas)
    - Duplicate detection
    - Invalid persona warnings
- **Data Quality Report** (collapsible):
  - Persona distribution table
  - Class imbalance warnings (ratio > 10:1)
  - Missing personas list
  - Recommendations for improvement

**Step 2: Configure Training**
- **Training Settings**:
  - Model Name (text input)
  - Test Size slider (0.1-0.5, default: 0.2)
  - Max Features slider (100-10000, default: 5000)
  - Advanced options (collapsible):
    - N-gram range (default: 1-3)
    - Min document frequency (default: 2)
    - Random seed (default: 42)
- **Expected Outcomes**:
  - Training samples count
  - Test samples count
  - Estimated training time
  - Stratification indicator (yes/no based on data)

**Step 3: Review & Train**
- Summary of training configuration
- "Start Training" button (primary)

**Step 4: Training Progress**
- **Progress Indicator**:
  - Stage-based progress (4 stages):
    1. Data Validation (0-25%)
    2. Model Training (25-60%)
    3. Evaluation (60-90%)
    4. Saving (90-100%)
  - Current stage highlighted
  - Live logs (optional, collapsible)
- **Cannot navigate away warning** (optional cancellation)

**Step 5: Training Results**
- **Model Performance Metrics**:
  - Test Accuracy (large display)
  - Cross-Validation Mean ± Std
  - Training Time
- **Classification Report** (table):
  - Per-persona precision, recall, F1-score, support
  - Color-coded performance (green >0.8, yellow 0.6-0.8, red <0.6)
  - Overall accuracy at bottom
- **Confusion Matrix** (optional, collapsible)
- **Data Statistics**:
  - Total samples used
  - Unique titles
  - Persona distribution (bar chart)
- **Actions**:
  - "Activate Model" button (make it the active model)
  - "Download Model" button
  - "Train Another Model" button
  - "View Model Details" link

**Design Notes**:
- Clear warnings for insufficient data
- Inline help text for technical parameters
- Success animation on training completion
- Compare with previous model option

#### 3.2.5 Jobs/History Page

**Purpose**: View and manage all prediction and training jobs

**Layout**: Tabbed interface

**Tab 1: Predictions**
- **Filter Bar**:
  - Status filter (all|pending|processing|completed|failed)
  - Date range picker
  - Search by job ID or filename
- **Jobs Table**:
  - Columns: Job ID, Filename, Status, Created At, Records, Actions
  - Status badges (color-coded)
  - Actions: View Results, Delete, Retry (if failed)
- **Pagination**

**Tab 2: Training Jobs**
- Similar structure to predictions tab
- Additional columns: Model Name, Accuracy
- Action: Activate Model, View Metrics, Delete

**Bulk Actions**:
- Select multiple jobs (checkbox)
- Bulk delete button

**Design Notes**:
- Auto-refresh for pending/processing jobs
- Sort by date (newest first)
- Quick filters (Today, This Week, This Month)
- Empty state illustrations

#### 3.2.6 Configuration Page

**Purpose**: Manage system settings and reference data

**Layout**: Tabbed settings panel

**Tab 1: Keyword Rules**
- **Table View**:
  - Columns: Keyword, Rule Type, Persona, Exclude Keyword, Actions
  - Inline editing
  - Add New Rule button
- **Bulk Import**:
  - Upload keyword_matching.csv
  - Download current rules as CSV
- **Validation**:
  - Check for invalid rule types
  - Warn about unknown personas
  - Duplicate keyword warnings

**Tab 2: Title Standardization**
- **Table View**:
  - Columns: Reference Title, Standardization, Actions
  - Search/filter
  - Inline editing
- **Bulk Import**:
  - Upload title_reference.csv
  - Download current mappings as CSV
- **Statistics**:
  - Total mappings
  - Cache hit/miss ratio
  - Most used mappings

**Tab 3: System Settings**
- **Default Configuration** (form):
  - All prediction configuration options
  - Set as system defaults
- **Model Management**:
  - Active model indicator
  - List of all models with activation option
  - Delete old models
- **Personas** (read-only):
  - Display current persona list and priority order
  - Warning if changes require code modification

**Tab 4: Advanced**
- **File Upload Limits**:
  - Max file size
  - Max records per file
- **Job Retention**:
  - Days to keep completed jobs
  - Auto-cleanup toggle
- **API Settings** (if multi-user):
  - Rate limiting
  - Concurrent jobs limit

**Design Notes**:
- Save buttons per tab
- Confirmation dialogs for destructive actions
- Form validation with helpful error messages
- "Reset to Defaults" option

### 3.3 UI Components Library

#### 3.3.1 Reusable Components

**JobStatusBadge**
- Props: status (pending|processing|completed|failed)
- Displays color-coded badge with icon
- Variants: small, medium, large

**PersonaBadge**
- Props: persona name
- Color-coded by persona
- Optional: count display

**ConfidenceBar**
- Props: score (0-100)
- Visual progress bar
- Color gradient (red < 50, yellow 50-70, green > 70)

**FileUploadZone**
- Drag-and-drop functionality
- File validation
- Preview support
- Error/success states

**ProgressStepper**
- Props: steps array, current step
- Visual step indicators
- Clickable previous steps (navigation)

**DataTable**
- Generic table component
- Sorting, filtering, pagination
- Row selection
- Export functionality

**ChartCard**
- Container for charts with title, description
- Loading state
- Empty state
- Download chart option

**ConfigurationForm**
- Reusable form for settings
- Validation support
- Tooltips integration
- Reset to defaults

**JobProgressTracker**
- Real-time progress display
- Stage indicators
- WebSocket integration
- Time remaining estimate

### 3.4 Design System

#### 3.4.1 Color Palette

**Primary Colors** (Brand):
- Primary Blue: `#2563EB` (buttons, links, accents)
- Primary Hover: `#1D4ED8`
- Primary Light: `#DBEAFE`

**Persona Colors**:
- GenAI: `#8B5CF6` (Purple)
- Engineering: `#3B82F6` (Blue)
- Product: `#10B981` (Green)
- Cyber Security: `#EF4444` (Red)
- Trust & Safety: `#F59E0B` (Orange)
- Legal & Compliance: `#6366F1` (Indigo)
- Executive: `#EC4899` (Pink)
- Not Classified: `#6B7280` (Gray)

**Semantic Colors**:
- Success: `#10B981` (Green)
- Warning: `#F59E0B` (Amber)
- Error: `#EF4444` (Red)
- Info: `#3B82F6` (Blue)

**Neutrals**:
- Background: `#F9FAFB`
- Surface: `#FFFFFF`
- Border: `#E5E7EB`
- Text Primary: `#111827`
- Text Secondary: `#6B7280`
- Text Disabled: `#9CA3AF`

#### 3.4.2 Typography

**Font Family**: Inter (sans-serif) or system font stack

**Scale**:
- H1: 36px, bold (page titles)
- H2: 30px, semibold (section headers)
- H3: 24px, semibold (card headers)
- H4: 20px, medium (subsections)
- Body: 16px, regular (default text)
- Small: 14px, regular (captions, labels)
- Tiny: 12px, regular (metadata)

#### 3.4.3 Spacing

**Scale**: 4px base unit
- xs: 4px
- sm: 8px
- md: 16px
- lg: 24px
- xl: 32px
- 2xl: 48px

#### 3.4.4 Shadows

- Card: `0 1px 3px rgba(0,0,0,0.1)`
- Hover: `0 4px 6px rgba(0,0,0,0.1)`
- Modal: `0 20px 25px rgba(0,0,0,0.15)`

#### 3.4.5 Border Radius

- Small: 4px (badges, buttons)
- Medium: 8px (cards, inputs)
- Large: 12px (modals, large cards)

### 3.5 Responsive Breakpoints

- Mobile: < 640px (single column, simplified navigation)
- Tablet: 640px - 1024px (2-column layouts, collapsible sidebar)
- Desktop: > 1024px (full layout, all features visible)

### 3.6 Accessibility Requirements

- WCAG 2.1 Level AA compliance
- Keyboard navigation support (Tab, Enter, Esc, Arrow keys)
- Screen reader support (ARIA labels, semantic HTML)
- Color contrast ratio >= 4.5:1 for text
- Focus indicators on all interactive elements
- Skip to main content link
- Error announcements for form validation
- Alt text for all images and icons

---

## 4. User Experience (UX) Flows

### 4.1 Primary User Flow: Run a Prediction

```
1. User lands on Home page
   → Sees "New Prediction" button

2. Clicks "New Prediction"
   → Navigates to Prediction page

3. Uploads CSV file via drag-and-drop
   → File validates automatically
   → Shows preview of first 5 rows
   → Displays record count

4. Reviews default configuration
   → Adjusts confidence threshold if needed
   → Clicks "Next"

5. Reviews summary
   → Clicks "Submit"

6. Job starts processing
   → Progress bar appears with real-time updates
   → Can navigate away (job continues in background)

7. Receives notification when complete
   → Clicks notification or "View Results" button

8. Views results page
   → Sees summary statistics
   → Browses classification table
   → Filters by low confidence
   → Downloads CSV

9. Done
```

**Success Criteria**:
- < 30 seconds from upload to submission
- Clear progress indication at all times
- Easy access to results

### 4.2 Secondary Flow: Train a Model

```
1. User navigates to Training page
   → Upload training data CSV
   → Sees data quality report (class distribution, warnings)

2. Adjusts training parameters
   → Names the model
   → Sets test size (optional)

3. Starts training
   → Progress indicator shows stages
   → Estimated time remaining

4. Training completes
   → Views performance metrics
   → Compares to previous model (if exists)

5. Activates new model
   → Confirmation dialog
   → Success message

6. New model is now used for predictions
```

### 4.3 Error Handling Flows

**File Upload Error**:
- Invalid format → Show inline error with example format
- File too large → Show error with size limit, suggest filtering data
- Missing columns → Highlight missing columns, show requirements

**Processing Error**:
- Model not trained → Redirect to training page with explanation
- Insufficient data → Show specific data requirements, offer guidance
- Server error → Show friendly error message, offer retry button

**Network Error**:
- Lost connection → Show offline indicator, queue actions for retry
- Timeout → Show timeout message, offer to check job status

### 4.4 Empty States

- No jobs history → "No jobs yet. Start by running a prediction."
- No results → "No results match your filters. Try adjusting filters."
- No keyword rules → "No keyword rules configured. Add rules to improve accuracy."

---

## 5. Data Flow & State Management

### 5.1 Frontend State Structure

```typescript
// Global App State (React Context)
interface AppState {
  user: {
    authenticated: boolean;
    preferences: UserPreferences;
  };
  config: {
    personas: string[];
    priorityOrder: string[];
    defaultSettings: PredictionConfig;
    modelInfo: ModelInfo;
  };
  notifications: Notification[];
}

// Prediction State (React Query)
interface PredictionJob {
  jobId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  totalRecords: number;
  processedRecords: number;
  createdAt: string;
  completedAt?: string;
  error?: string;
}

// Results State (React Query)
interface PredictionResults {
  jobId: string;
  summary: ResultsSummary;
  results: ClassificationResult[];
  pagination: Pagination;
}

// Training State (React Query)
interface TrainingJob {
  jobId: string;
  status: 'pending' | 'training' | 'completed' | 'failed';
  progress: number;
  stage: string;
  metrics?: TrainingMetrics;
  error?: string;
}
```

### 5.2 Data Caching Strategy

**React Query Configuration**:
- Cache time: 5 minutes for results, 10 minutes for config
- Stale time: 30 seconds for job status, 5 minutes for results
- Refetch on window focus for active jobs
- Optimistic updates for configuration changes

### 5.3 Real-time Updates

**WebSocket Integration**:
- Connect to WS on job submission
- Receive progress updates
- Update UI reactively
- Disconnect on job completion or page unmount
- Fallback to polling if WS unavailable (every 2 seconds)

---

## 6. Performance Considerations

### 6.1 Frontend Optimization

- **Code Splitting**: Lazy load routes (React.lazy)
- **Bundle Size**:
  - Target: < 300KB gzipped main bundle
  - Separate vendor bundle (React, MUI)
  - Dynamic imports for charts and heavy components
- **Virtualization**: Use react-window for large result tables (1000+ rows)
- **Image Optimization**: Use WebP format, lazy loading
- **API Response Caching**: React Query with appropriate stale times
- **Debouncing**: Search inputs (300ms delay)
- **Throttling**: Scroll and resize handlers

### 6.2 Backend Optimization

- **Async Processing**: All predictions and training run in Celery workers
- **Pagination**: Max 1000 results per request, default 100
- **Database Indexing**:
  - job_id (primary key)
  - status (for filtering)
  - created_at (for sorting)
- **File Upload**:
  - Stream large files (avoid loading into memory)
  - Max size: 100MB (configurable)
  - Virus scanning for enterprise deployments
- **Caching**:
  - Redis cache for config endpoints (1 hour TTL)
  - Model loaded once, shared across workers
- **Rate Limiting**:
  - 100 requests/minute per IP for API
  - 5 concurrent jobs per user

### 6.3 Scalability Targets

- **Concurrent Users**: 50-100
- **File Size**: Up to 100MB (50,000 records)
- **Response Time**:
  - API endpoints: < 200ms (p95)
  - File upload: < 5s for 10MB
  - Predictions: < 30s for 1,000 records
  - Training: < 5 minutes for 2,000 samples
- **Job Queue**: Handle 100 queued jobs
- **Storage**: 10GB for results and models

---

## 7. Security Requirements

### 7.1 Input Validation

- **File Uploads**:
  - Validate file type (CSV only)
  - Scan for malicious content
  - Limit file size (100MB)
  - Sanitize file names
- **API Inputs**:
  - Validate all JSON payloads against schemas
  - Sanitize SQL inputs (use parameterized queries)
  - Escape HTML in user-generated content
- **Configuration**:
  - Validate ranges for numeric settings
  - Whitelist allowed persona names
  - Prevent path traversal in file operations

### 7.2 Authentication & Authorization (Phase 2)

- **Authentication**: JWT tokens with 1-hour expiry
- **Authorization**: Role-based access control (RBAC)
  - Admin: Full access (train, configure, delete)
  - User: Predictions and viewing results
  - Viewer: Read-only access
- **Password Requirements**: Min 8 chars, uppercase, lowercase, number
- **Session Management**: Secure cookies, HttpOnly, SameSite

### 7.3 Data Protection

- **At Rest**: Encrypt sensitive data in database (AES-256)
- **In Transit**: HTTPS/TLS 1.2+ for all communications
- **File Storage**: Separate directories per user/job, prevent cross-access
- **Logging**: Redact sensitive data (PII) from logs
- **Data Retention**: Auto-delete old jobs (configurable, default 30 days)

### 7.4 API Security

- **CORS**: Whitelist allowed origins
- **CSRF Protection**: Token-based for state-changing operations
- **Rate Limiting**: Prevent abuse and DoS
- **API Versioning**: /api/v1/ for backwards compatibility
- **Error Messages**: Generic errors in production (no stack traces)

### 7.5 Dependency Management

- **Frontend**: npm audit before deployment
- **Backend**: Python safety check, Dependabot alerts
- **Docker Images**: Use official base images, scan for vulnerabilities

---

## 8. Testing Requirements

### 8.1 Frontend Testing

**Unit Tests** (Jest + React Testing Library):
- Component rendering
- User interactions (clicks, form inputs)
- State management
- Utility functions
- Target: > 80% code coverage

**Integration Tests**:
- API integration (mocked)
- Form submission flows
- Navigation between pages
- File upload handling

**E2E Tests** (Cypress or Playwright):
- Complete prediction flow
- Complete training flow
- Results filtering and export
- Configuration updates
- Target: Cover all critical paths

### 8.2 Backend Testing

**Unit Tests** (pytest):
- API endpoint logic
- Data validation functions
- Model loading and prediction
- Configuration parsing
- Target: > 85% code coverage

**Integration Tests**:
- API endpoints with database
- Celery task execution
- File upload and processing
- Model training pipeline

**Load Tests** (Locust):
- Concurrent API requests
- Multiple file uploads
- Worker throughput
- Target: 100 concurrent users

### 8.3 Acceptance Criteria

- All unit tests pass
- E2E tests cover happy paths and error cases
- Load tests meet performance targets
- Manual UAT by stakeholders
- Accessibility audit passes (axe DevTools)

---

## 9. Deployment Architecture

### 9.1 Development Environment

**Docker Compose Setup**:
```yaml
services:
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    volumes: ["./frontend:/app"]

  backend:
    build: ./backend
    ports: ["8000:8000"]
    volumes: ["./backend:/app"]
    depends_on: [db, redis]

  celery_worker:
    build: ./backend
    command: celery -A app worker
    depends_on: [backend, redis]

  db:
    image: postgres:15
    environment:
      POSTGRES_DB: persona_classifier
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
    volumes: ["postgres_data:/var/lib/postgresql/data"]

  redis:
    image: redis:7-alpine

  nginx:
    image: nginx:alpine
    ports: ["80:80"]
    volumes: ["./nginx.conf:/etc/nginx/nginx.conf"]
    depends_on: [frontend, backend]
```

**Access**: http://localhost (Nginx proxies to frontend/backend)

### 9.2 Production Environment

**Option A: Single Server Deployment**
- Docker Compose on single VPS (DigitalOcean, AWS EC2)
- Nginx as reverse proxy with SSL (Let's Encrypt)
- PostgreSQL and Redis in Docker containers
- Celery workers (2-4 workers)
- Suitable for: < 100 users, < 1000 predictions/day

**Option B: Container Orchestration (Kubernetes)**
- Frontend: 2 replicas (load balanced)
- Backend API: 3 replicas (autoscaling based on CPU)
- Celery Workers: 5 replicas (autoscaling based on queue length)
- PostgreSQL: Managed service (AWS RDS, Google Cloud SQL)
- Redis: Managed service (ElastiCache, Cloud Memorystore)
- Ingress with SSL termination
- Suitable for: > 100 users, > 5000 predictions/day

**Option C: Serverless (Advanced)**
- Frontend: Static hosting (Vercel, Netlify, CloudFront + S3)
- Backend API: AWS Lambda + API Gateway (with FastAPI)
- Workers: Lambda functions triggered by SQS
- Database: Aurora Serverless
- Redis: ElastiCache or DynamoDB (for task queue)
- Suitable for: Variable load, cost optimization

### 9.3 CI/CD Pipeline

**GitHub Actions Workflow**:

```yaml
# .github/workflows/deploy.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm test
      - run: npm run lint

  test-backend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: pytest
      - run: flake8

  build-and-push:
    needs: [test-frontend, test-backend]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - uses: docker/build-push-action@v4
        with:
          push: true
          tags: myregistry/persona-ui:latest

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          # Trigger deployment (Kubernetes, Docker Compose, etc.)
```

### 9.4 Monitoring & Observability

**Logging**:
- **Backend**: Structured JSON logs (timestamp, level, message, context)
- **Collection**: Fluentd or Vector → Elasticsearch or Loki
- **Retention**: 30 days

**Metrics** (Prometheus + Grafana):
- API request rate, latency, error rate
- Job queue length and processing time
- Worker utilization
- Database connections and query time
- Redis cache hit/miss ratio
- System resources (CPU, memory, disk)

**Tracing** (Optional, Jaeger or Zipkin):
- End-to-end request tracing
- Identify bottlenecks

**Alerting** (Alertmanager):
- API error rate > 5%
- Job queue length > 100
- Worker failures
- Disk usage > 85%
- Database connection failures

**Health Checks**:
- /api/v1/health endpoint
- Kubernetes liveness and readiness probes
- Uptime monitoring (UptimeRobot, Pingdom)

---

## 10. Development Phases & Milestones

### Phase 1: MVP (Minimum Viable Product) - 8-10 weeks

**Goal**: Core functionality for predictions

**Week 1-2: Setup & Architecture**
- Project scaffolding (React + FastAPI)
- Docker Compose environment
- Database schema design
- API endpoint planning
- Design system setup (MUI integration)

**Week 3-4: Backend Foundation**
- FastAPI routes for predictions
- Celery worker integration
- File upload handling
- Database models and migrations
- Integration with existing predict.py script

**Week 5-6: Frontend Core**
- Prediction page (upload + configuration)
- Job progress tracker
- Results page (table view)
- Basic navigation and layout

**Week 7-8: Integration & Polish**
- WebSocket for real-time updates
- CSV export functionality
- Error handling and validation
- Responsive design
- Basic testing (unit + integration)

**Week 9-10: Testing & Deployment**
- End-to-end testing
- Documentation
- Deployment setup
- UAT with stakeholders

**Deliverables**:
- ✅ File upload and prediction
- ✅ Real-time progress tracking
- ✅ Results viewing and export
- ✅ Basic configuration (confidence threshold)
- ✅ Job history

### Phase 2: Enhancement - 6-8 weeks

**Goal**: Model training and advanced features

**Week 1-2: Training UI**
- Training page (upload + configuration)
- Training progress tracker
- Model evaluation display
- Model management (list, activate, delete)

**Week 3-4: Configuration Management**
- Keyword rules CRUD interface
- Title standardization CRUD interface
- System settings panel
- Configuration import/export

**Week 5-6: Advanced Results**
- Charts and visualizations (Recharts)
- Advanced filtering and search
- Bulk operations
- Manual classification overrides

**Week 7-8: Polish & Optimize**
- Performance optimization (virtualization)
- Accessibility audit and fixes
- Enhanced error handling
- Comprehensive testing

**Deliverables**:
- ✅ Model training via UI
- ✅ Keyword and standardization management
- ✅ Advanced analytics and charts
- ✅ Configuration presets

### Phase 3: Production-Ready - 4-6 weeks

**Goal**: Security, scalability, and enterprise features

**Week 1-2: Authentication & Authorization**
- User authentication (JWT)
- Role-based access control
- User management interface
- Session management

**Week 3-4: Enterprise Features**
- API rate limiting
- Audit logging
- Data retention policies
- Scheduled jobs (cron predictions)

**Week 5-6: Deployment & Monitoring**
- Kubernetes deployment
- CI/CD pipeline
- Monitoring dashboards (Grafana)
- Alerting setup
- Load testing

**Deliverables**:
- ✅ Multi-user support
- ✅ Production-grade security
- ✅ Scalable architecture
- ✅ Monitoring and alerting

### Phase 4: Advanced Features (Optional) - Ongoing

**Possible Enhancements**:
- Batch job scheduling (run predictions daily)
- API key management (for external integrations)
- Webhooks (notify external systems on completion)
- Model versioning and A/B testing
- Custom persona creation (dynamic personas)
- Advanced analytics (trend analysis over time)
- Integration with HubSpot API (direct sync)
- Mobile app (React Native)
- Collaborative features (team sharing)

---

## 11. Documentation Requirements

### 11.1 User Documentation

**User Guide** (in-app help + separate docs site):
- Getting started tutorial
- Step-by-step guides for each feature
- Configuration reference
- Troubleshooting guide
- FAQ
- Video tutorials (screen recordings)

**API Documentation**:
- OpenAPI/Swagger UI (auto-generated by FastAPI)
- Authentication guide
- Code examples (cURL, Python, JavaScript)
- Webhooks documentation (Phase 3)

### 11.2 Developer Documentation

**README.md**:
- Project overview
- Setup instructions
- Architecture diagram
- Development workflow
- Contributing guidelines

**Architecture Documentation**:
- System design overview
- Database schema
- API design decisions
- State management approach
- Deployment architecture

**Code Documentation**:
- JSDoc for React components
- Python docstrings for functions
- Inline comments for complex logic

**Runbooks**:
- Deployment procedures
- Backup and restore
- Scaling guidelines
- Incident response

---

## 12. Success Metrics & KPIs

### 12.1 User Adoption Metrics

- Number of active users (daily, weekly, monthly)
- Number of predictions run
- Number of models trained
- Conversion rate (visitors → predictions)
- User retention rate (week-over-week)

### 12.2 Performance Metrics

- Average prediction processing time
- Average training time
- API response times (p50, p95, p99)
- Error rate (< 1% target)
- System uptime (99.9% target)

### 12.3 Quality Metrics

- User satisfaction score (CSAT survey)
- Number of support tickets
- Bug report rate
- Feature request volume
- Classification accuracy (compared to CLI)

### 12.4 Business Metrics

- Time saved vs CLI workflow (target: 70% reduction)
- Number of new use cases enabled
- Cost per prediction
- Infrastructure costs

---

## 13. Risk Assessment & Mitigation

### 13.1 Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Model compatibility issues | High | Medium | Thorough testing, version pinning |
| Large file upload failures | Medium | Medium | Chunked uploads, retry logic |
| WebSocket connection drops | Low | High | Polling fallback, reconnection logic |
| Database performance degradation | High | Low | Indexing, query optimization, caching |
| Celery worker failures | High | Medium | Health checks, auto-restart, monitoring |

### 13.2 User Experience Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Confusing UI for non-technical users | High | Medium | User testing, simplified defaults |
| Slow page load times | Medium | Medium | Code splitting, optimization |
| Mobile usability issues | Low | Low | Responsive design, mobile testing |
| Inconsistent results vs CLI | High | Low | Rigorous validation, parallel testing |

### 13.3 Security Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Unauthorized access to results | High | Medium | Authentication, authorization |
| Malicious file uploads | High | Low | File validation, virus scanning |
| API abuse/DoS | Medium | Medium | Rate limiting, monitoring |
| Data leakage | High | Low | Access controls, logging, encryption |

---

## 14. Open Questions & Decisions Needed

1. **Deployment Target**:
   - Single-user (localhost) or multi-user (cloud)?
   - Decision impacts auth, scaling, and infrastructure

2. **Authentication Approach**:
   - Phase 1: No auth (single user)?
   - Phase 2: Simple password or enterprise SSO?

3. **Data Persistence**:
   - How long to retain prediction results?
   - Should users be able to delete their own jobs?

4. **Branding**:
   - Product name for the UI?
   - Logo and color scheme preferences?

5. **Budget & Timeline**:
   - Available development resources?
   - Hard deadlines for specific features?

6. **Integration Requirements**:
   - Need to integrate with existing systems (HubSpot, Salesforce)?
   - API-first for external consumption?

7. **Compliance**:
   - GDPR, CCPA, or other data privacy requirements?
   - Need for audit trails?

---

## 15. Estimated Effort

### 15.1 Development Effort (assuming 1 full-time developer)

- **Phase 1 (MVP)**: 8-10 weeks
- **Phase 2 (Enhancement)**: 6-8 weeks
- **Phase 3 (Production-Ready)**: 4-6 weeks
- **Total**: 18-24 weeks (4.5-6 months)

### 15.2 Team Composition (Recommended)

**Minimum Team (MVP)**:
- 1 Full-stack Developer (React + Python)
- 1 UX/UI Designer (part-time, first 4 weeks)

**Ideal Team (Full Product)**:
- 1 Frontend Developer (React/TypeScript)
- 1 Backend Developer (Python/FastAPI)
- 1 UX/UI Designer (part-time throughout)
- 1 DevOps Engineer (part-time, Phase 3)
- 1 QA Engineer (part-time, all phases)

**Accelerated Timeline**: 2 full-stack developers → 12-16 weeks

### 15.3 Infrastructure Costs (Estimated Monthly)

**Development**:
- Local development: $0

**Production (Small Scale)**:
- VPS (4 vCPU, 8GB RAM): $40/month
- Database (managed): $15/month
- Redis (managed): $10/month
- Domain + SSL: $2/month
- **Total**: ~$70/month

**Production (Enterprise Scale)**:
- Kubernetes cluster: $200-500/month
- Managed database: $100/month
- Managed Redis: $50/month
- Load balancer: $20/month
- Monitoring (Grafana Cloud): $50/month
- **Total**: ~$420-720/month

---

## 16. Next Steps

1. **Review & Approve**: Stakeholder review of this specification
2. **Prioritize Features**: Confirm Phase 1 scope and priorities
3. **Design Mockups**: Create high-fidelity UI designs based on wireframes in this spec
4. **Technical Proof-of-Concept**:
   - FastAPI + existing predict.py integration
   - React file upload + results display
   - Celery async job processing
5. **Project Setup**: Initialize repositories, Docker environment, CI/CD
6. **Sprint Planning**: Break down Phase 1 into 2-week sprints
7. **Begin Development**: Start with backend foundation and frontend scaffolding in parallel

---

## Appendix A: Technology Alternatives Comparison

### Frontend Frameworks

| Framework | Pros | Cons | Recommendation |
|-----------|------|------|----------------|
| **React** | Huge ecosystem, flexibility, job market | Boilerplate, decision fatigue | ✅ **Recommended** |
| **Vue.js** | Easier learning curve, clean syntax | Smaller ecosystem | Good alternative |
| **Angular** | Full framework, TypeScript native | Heavy, opinionated | Overkill for this project |
| **Svelte** | Best performance, less boilerplate | Smaller ecosystem, newer | Too risky for production |

### Backend Frameworks

| Framework | Pros | Cons | Recommendation |
|-----------|------|------|----------------|
| **FastAPI** | Modern, async, auto-docs, type hints | Newer (but mature) | ✅ **Recommended** |
| **Flask** | Simple, mature, flexible | Synchronous, manual setup | Good for MVP |
| **Django** | Batteries included, admin panel | Heavy, ORM lock-in | Overkill |
| **Node.js (Express)** | JavaScript everywhere | Weak typing, different language | Not ideal |

### Database

| Database | Pros | Cons | Recommendation |
|----------|------|------|----------------|
| **PostgreSQL** | Feature-rich, reliable, JSON support | Requires setup | ✅ **Recommended** |
| **SQLite** | Zero-config, embedded | Limited concurrency | Good for MVP/localhost |
| **MySQL** | Mature, widely supported | Less feature-rich | Alternative |
| **MongoDB** | Flexible schema | Overkill for structured data | Not recommended |

---

## Appendix B: Sample UI Wireframes (ASCII Art)

### Home Page (Dashboard)
```
┌────────────────────────────────────────────────────────┐
│ [Logo] Persona Classifier     [Model: v1.2 ✓] [⚙️]     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Welcome back! Ready to classify job titles?          │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │ New          │  │ Train        │                  │
│  │ Prediction   │  │ Model        │                  │
│  └──────────────┘  └──────────────┘                  │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Model Status                                    │  │
│  │ Accuracy: 88.3% | Trained: 2 days ago          │  │
│  │ Samples: 1,630 | Personas: 7                   │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │ Recent Activity                                 │  │
│  │ • Prediction #123 - Completed (1,000 records)  │  │
│  │ • Training Job #45 - Completed (88.3%)         │  │
│  │ • Prediction #122 - Failed                     │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  Statistics Overview                                   │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│  │ 50,000  │ │   125   │ │  GenAI  │ │  85%    │    │
│  │ Total   │ │ Today   │ │ Top     │ │ Avg     │    │
│  │ Predict │ │ Jobs    │ │ Persona │ │ Conf    │    │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘    │
└────────────────────────────────────────────────────────┘
```

### Prediction Page (Step 1: Upload)
```
┌────────────────────────────────────────────────────────┐
│ New Prediction                        Step 1 of 4      │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Upload Input File                                     │
│  ┌────────────────────────────────────────────────┐  │
│  │                                                │  │
│  │          📁 Drag & drop CSV file here          │  │
│  │                                                │  │
│  │              or click to browse                │  │
│  │                                                │  │
│  │    Required: Record ID, Job Title columns     │  │
│  │                                                │  │
│  └────────────────────────────────────────────────┘  │
│                                                        │
│  📋 Or enter job titles manually                      │
│  ┌────────────────────────────────────────────────┐  │
│  │ Record ID       | Job Title                    │  │
│  │ ─────────────────────────────────────────────  │  │
│  │ [         ]     | [                        ]   │  │
│  │ [+ Add Row]                                   │  │
│  └────────────────────────────────────────────────┘  │
│                                                        │
│                      [Validate] [Next →]              │
└────────────────────────────────────────────────────────┘
```

### Results Page
```
┌────────────────────────────────────────────────────────┐
│ Prediction Results - Job #123                          │
├────────────────────────────────────────────────────────┤
│  Summary                                               │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐         │
│  │ 1,000  │ │  850   │ │  150   │ │  85%   │         │
│  │ Total  │ │ Class  │ │ Not    │ │ Avg    │         │
│  │        │ │ ified  │ │ Class  │ │ Conf   │         │
│  └────────┘ └────────┘ └────────┘ └────────┘         │
│                                                        │
│  [Download CSV ⬇] [Download JSON]  [Filter ▼]        │
│                                                        │
│  ┌────────────────────────────────────────────────┐  │
│  │ Record ID | Job Title      | Persona  | Conf  │  │
│  │───────────────────────────────────────────────  │  │
│  │ 12345     │ AI Engineer    │ GenAI    │ ████  │  │
│  │ 67890     │ Software Dev   │ Engineer │ ████  │  │
│  │ 11223     │ Product Lead   │ Product  │ ███   │  │
│  │ ...                                             │  │
│  └────────────────────────────────────────────────┘  │
│                                                        │
│  [← Prev]  Page 1 of 10  [Next →]                     │
└────────────────────────────────────────────────────────┘
```

---

## Appendix C: API Request/Response Examples

### Example: Submit Prediction Job

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/predictions \
  -H "Content-Type: multipart/form-data" \
  -F "file=@input.csv" \
  -F 'config={"confidence_threshold": 60, "duplicate_handling": "keep_first"}'
```

**Response** (201 Created):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2025-10-22T14:30:00Z"
}
```

### Example: Get Results

**Request**:
```bash
curl -X GET "http://localhost:8000/api/v1/predictions/550e8400-e29b-41d4-a716-446655440000/results?page=1&limit=100&format=json"
```

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "pagination": {
    "page": 1,
    "limit": 100,
    "total_records": 1000,
    "total_pages": 10
  },
  "summary": {
    "total_records": 1000,
    "classified": 850,
    "not_classified": 150,
    "keyword_matched": 200,
    "ml_predicted": 650,
    "persona_distribution": {
      "GenAI": 120,
      "Engineering": 350,
      "Product": 200,
      "Cyber Security": 80,
      "Trust & Safety": 50,
      "Legal & Compliance": 30,
      "Executive": 20,
      "Not Classified": 150
    }
  },
  "results": [
    {
      "record_id": "37462838462827",
      "job_title": "AI Engineer",
      "persona_segment": "GenAI",
      "confidence_score": 90,
      "classification_method": "ml"
    },
    {
      "record_id": "82736482736473",
      "job_title": "Senior Software Developer",
      "persona_segment": "Engineering",
      "confidence_score": 85,
      "classification_method": "keyword"
    }
  ]
}
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-22
**Author**: Technical Specification for Persona Classification UI
**Status**: Draft - Pending Review
