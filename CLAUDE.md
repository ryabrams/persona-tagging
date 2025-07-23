# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Training and Prediction
- `make train` - Retrain the persona classification model using data/training_data.csv
- `make predict` - Run predictions on data/input.csv and output to tagged_personas.csv
- `python scripts/train_model.py` - Direct training script execution
- `python scripts/predict.py` - Direct prediction script execution

### Dependencies
- `pip install -r requirements.txt` - Install required Python packages (scikit-learn, pandas, numpy, joblib)

## Architecture Overview

This is a machine learning system for classifying job titles into predefined persona segments using scikit-learn. The system has a two-tier classification approach:

1. **Keyword-based rules** (highest priority) - Exact matches from data/keyword_matching.csv
2. **ML model predictions** - TF-IDF + Logistic Regression pipeline with confidence thresholds

### Core Components

**scripts/predict.py** - Main prediction engine that:
- Loads trained model from model/persona_classifier.pkl
- Applies keyword matching rules first (if data/keyword_matching.csv exists)
- Standardizes job titles using title_standardizer.py
- Runs ML predictions on remaining entries
- Enforces persona segment priority ordering
- Filters out low-confidence predictions (< 50%)

**scripts/train_model.py** - Model training pipeline:
- Uses TF-IDF vectorizer with 1-2 ngrams
- Logistic Regression classifier (max_iter=1000, random_state=42)
- Saves trained pipeline to model/persona_classifier.pkl

**scripts/title_standardizer.py** - Title preprocessing utility:
- Maps job title variations to standardized forms using data/title_reference.csv
- Simple dictionary lookup with fallback to original title

### Persona Segment Priority Order
The system enforces this hierarchy when multiple classifications are possible:
1. GenAI
2. Engineering  
3. Product
4. Cyber Security
5. Trust & Safety
6. Legal & Compliance
7. Executive

### Data Files Structure
- **data/input.csv** - Input for predictions (Record ID, Job Title)
- **data/training_data.csv** - Training data (Job Title, Persona Segment)
- **data/keyword_matching.csv** - Rule-based matching (Keyword, Rule, Persona Segment, Exclude Keyword)
- **data/title_reference.csv** - Title standardization mapping (Reference, Standardization)
- **tagged_personas.csv** - Prediction output (Record ID, Job Title, Persona Segment, Confidence Score)

### Key Implementation Details
- Confidence scores are rounded to nearest 5% and capped at 100%
- Low confidence threshold is 50% (configurable in predict.py:131)
- Keyword rules support both "contains" and "equals" matching with optional exclusion
- All column matching is case-insensitive
- Model uses TF-IDF features with English stop words removal