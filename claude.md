# Persona Classification System

## Project Overview

This is a production-ready machine learning system that classifies job titles into predefined persona segments. The system is designed for workforce analytics and data enrichment workflows, particularly for enterprise data from platforms like HubSpot.

**Purpose**: Automatically categorize job titles using a hybrid approach combining rule-based keyword matching with machine learning predictions.

**Technology Stack**: Python, scikit-learn (Logistic Regression + TF-IDF), pandas, joblib

## Architecture

### Classification Pipeline (3-Stage Approach)

The system processes job titles through three sequential layers:

1. **Keyword Matching** (Optional, 100% confidence)
   - Applied first to original job titles
   - Case-insensitive rule matching (contains/equals)
   - Supports exclusion keywords
   - Configuration: [data/keyword_matching.csv](data/keyword_matching.csv)

2. **Title Standardization** (Optional)
   - Normalizes job title variations
   - Case-insensitive dictionary lookup
   - Thread-safe LRU caching (500 entries)
   - Configuration: [data/title_reference.csv](data/title_reference.csv)

3. **ML Classification** (Fallback)
   - TF-IDF vectorization (1-3 grams, 5000 max features, min_df=2)
   - Logistic Regression (balanced class weights, 1000 iterations)
   - Priority enforcement for uncertain predictions (< 70% confidence)
   - Confidence threshold filtering (default: 50%)

### Persona Segments (Priority Order)

1. GenAI
2. Engineering
3. Product
4. Cyber Security
5. Trust & Safety
6. Legal & Compliance
7. Executive

**Note**: Priority order resolves ambiguous classifications when model confidence is < 70%.

## Key Components

### Core Scripts

- [scripts/train_model.py](scripts/train_model.py) - Model training pipeline with cross-validation
- [scripts/predict.py](scripts/predict.py) - Prediction pipeline with multi-stage classification
- [scripts/title_standardizer.py](scripts/title_standardizer.py) - Thread-safe title normalization module

### Data Files

**Required for Training:**
- [data/training_data.csv](data/training_data.csv) - Labeled training data (Job Title, Persona Segment)

**Required for Prediction:**
- [data/input.csv](data/input.csv) - Input data (Record ID, Job Title)

**Optional Configuration:**
- [data/keyword_matching.csv](data/keyword_matching.csv) - Keyword rules (Keyword, Rule, Persona Segment, Exclude Keyword)
- [data/title_reference.csv](data/title_reference.csv) - Title mappings (Reference, Standardization)

**Generated Outputs:**
- [model/persona_classifier.pkl](model/persona_classifier.pkl) - Trained scikit-learn Pipeline
- [model/model_metadata.txt](model/model_metadata.txt) - Training metadata and metrics
- [tagged_personas.csv](tagged_personas.csv) - Classification results

### Makefile Targets

```bash
make train      # Train model
make predict    # Run predictions
make all        # Full pipeline (train + predict)
make check      # Verify system setup
make validate   # Check file formats
make clean      # Remove generated files
make test       # Test on training data
make retrain    # Force retraining
```

## Configuration

### Environment Variables

All settings have sensible defaults but can be overridden:

- `PC_CONFIDENCE_THRESHOLD` (0-100, default: 50) - Minimum confidence for assignment
- `PC_DUPLICATE_HANDLING` (keep_first|keep_last|keep_all, default: keep_first)
- `PC_PRIORITY_THRESHOLD` (0.0-1.0, default: 0.7) - Threshold for priority enforcement
- `PC_SIMILARITY_RANGE` (0.0-1.0, default: 0.1) - Range for considering similar probabilities
- `PC_MAX_TITLE_LENGTH` (10-10000, default: 500) - Max job title length
- `PC_TEST_SIZE` (0.1-0.5, default: 0.2) - Train/test split ratio
- `PC_MAX_FEATURES` (100+, default: 5000) - TF-IDF max features

### Model Parameters

Defined in [scripts/train_model.py](scripts/train_model.py):
- TF-IDF: 1-3 grams, English stop words, min_df=2
- Logistic Regression: max_iter=1000, class_weight='balanced', random_state=42
- Cross-validation: 5-fold (adaptive based on class sizes)
- Data validation: Min 10 samples, 2+ personas, class imbalance detection

## Code Patterns & Conventions

### Data Validation

All scripts perform extensive validation:
- Column name standardization (case-insensitive)
- Missing value handling
- Duplicate detection and configurable handling
- Data type conversion and sanitization
- Class distribution analysis

### Error Handling

- Comprehensive logging at INFO/WARNING/ERROR levels
- Graceful degradation for optional features
- Validation at multiple pipeline stages
- User-friendly error messages with emoji indicators (✅/❌)

### Performance Optimizations

- LRU caching for title standardization (maxsize=500)
- Thread-safe operations with locks
- Vectorized pandas operations
- Lazy loading of reference dictionaries
- Cache statistics tracking

### Logging Philosophy

```python
logger.info()    # Normal operations, statistics, progress
logger.warning() # Potential issues, missing optional files, data quality alerts
logger.error()   # Fatal errors that stop execution
```

## Development Guidelines

### When Training Models

1. Ensure training data has >= 10 samples total, 2+ personas
2. Check for class imbalance (system warns if ratio > 10:1)
3. Review classification report for per-persona performance
4. Validate persona segments match VALID_PERSONAS exactly
5. Use title standardization to improve generalization

### When Making Predictions

1. Validate input CSV has 'Record ID' and 'Job Title' columns
2. Handle duplicate Record IDs appropriately (see PC_DUPLICATE_HANDLING)
3. Keyword matches always override ML predictions (100% confidence)
4. Low confidence predictions (< PC_CONFIDENCE_THRESHOLD) are left unassigned
5. Priority enforcement automatically adjusts ambiguous predictions

### When Adding Features

- Maintain backward compatibility with existing CSV formats
- Add environment variable configuration for new parameters
- Include validation with clear error messages
- Update model metadata for training parameters
- Log configuration changes at INFO level

### Common Modification Points

**Add new persona segment:**
1. Update PRIORITY_ORDER in [scripts/predict.py](scripts/predict.py:20)
2. Update VALID_PERSONAS in [scripts/train_model.py](scripts/train_model.py:29)
3. Add training samples to [data/training_data.csv](data/training_data.csv)
4. Update README.md documentation

**Adjust ML model:**
1. Modify Pipeline in [scripts/train_model.py](scripts/train_model.py:144-156)
2. Update environment variable validation if adding configurable params
3. Document changes in model metadata

**Enhance keyword matching:**
1. Edit [data/keyword_matching.csv](data/keyword_matching.csv) (no code changes needed)
2. Use 'contains' for substring matching, 'equals' for exact matches
3. Add 'Exclude Keyword' to prevent false positives

## Testing & Validation

### Data Quality Checks

- Duplicate detection (training and input data)
- Class distribution analysis
- Class imbalance warnings (> 10:1 ratio)
- Invalid persona segment detection
- Missing value handling
- Title length validation

### Model Evaluation

- Classification report (precision, recall, F1-score per persona)
- K-fold cross-validation (adaptive folds based on class sizes)
- Hold-out test set accuracy
- Metadata tracking (training date, parameters, dataset stats)

### Quick Validation Commands

```bash
make validate  # Check all file formats and requirements
make test      # Test prediction on training data (smoke test)
make check     # Verify all required files exist
```

## Troubleshooting

### Common Issues

**"Model file not found"**
- Solution: Run `make train` first

**"Invalid persona segments in training data"**
- Solution: Ensure persona names match VALID_PERSONAS exactly (case-sensitive)

**"Cannot use stratified split"**
- Cause: Too few samples in some classes (< 10 per class)
- Solution: Add more training samples for underrepresented personas

**Low confidence scores across the board**
- Solution: Add more diverse training examples, check data quality

**High class imbalance detected**
- Solution: Add more samples for underrepresented personas
- Note: System uses class_weight='balanced' to mitigate this

### Debug Mode

Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Security & Data Handling

- All CSV files use UTF-8 encoding
- No sensitive data is logged (only counts and statistics)
- Model files use joblib serialization (standard sklearn practice)
- No external network calls or API dependencies
- Input validation prevents excessively long titles (configurable max length)

## Performance Characteristics

- **Training**: Scales linearly with number of samples (1600+ samples trains in seconds)
- **Prediction**: Processes thousands of records in seconds
- **Memory**: LRU cache limited to 500 entries for title standardization
- **Thread Safety**: Title standardization module is thread-safe

## License

MIT License (Copyright 2025 Ryan Abrams)
