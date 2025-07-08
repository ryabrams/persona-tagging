import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import os
import sys
import logging
from datetime import datetime
from title_standardizer import standardize_title, get_standardization_stats

# Configure logging

logging.basicConfig(level=logging.INFO, format=’%(asctime)s - %(levelname)s - %(message)s’)
logger = logging.getLogger(**name**)

# File paths

TRAINING_FILE = “data/training_data.csv”
MODEL_FILE = “model/persona_classifier.pkl”
MODEL_METADATA_FILE = “model/model_metadata.txt”

# Minimum samples per class

MIN_SAMPLES_PER_CLASS = 5
MIN_SAMPLES_FOR_STRATIFY = 10  # Need at least 10 samples per class for meaningful 80/20 split
MIN_SAMPLES_FOR_CV = 5  # Minimum samples per class for cross-validation

# Valid persona segments (same as PRIORITY_ORDER in predict.py)

VALID_PERSONAS = [“GenAI”, “Engineering”, “Product”, “Cyber Security”, “Trust & Safety”, “Legal & Compliance”, “Executive”]

# Model parameters

MAX_FEATURES = 5000
MIN_DF = 2
NGRAM_RANGE = (1, 3)
TEST_SIZE = 0.2
RANDOM_STATE = 42
CLASS_IMBALANCE_THRESHOLD = 10

def load_and_prepare_data(file_path):
“”“Load and prepare training data with quality checks.”””
logger.info(f”Loading training data from {file_path}”)

```
# Load data
df = pd.read_csv(file_path, encoding='utf-8')
initial_count = len(df)
logger.info(f"Loaded {initial_count} records")

# Standardize column names
df.columns = df.columns.str.lower()

# Ensure necessary columns exist
if not {'job title', 'persona segment'}.issubset(df.columns):
    raise ValueError("❌ Training file must contain 'Job Title' and 'Persona Segment' columns (case-insensitive).")

# Drop missing values
df.dropna(subset=['job title', 'persona segment'], inplace=True)
dropped_count = initial_count - len(df)
if dropped_count > 0:
    logger.warning(f"Dropped {dropped_count} rows with missing values")

if df.empty:
    raise ValueError("❌ Training data is empty after dropping missing values.")

# Check for sufficient data
if len(df) < 10:
    raise ValueError(f"❌ Insufficient training data. Found only {len(df)} samples, need at least 10.")

# Apply title standardization
logger.info("Applying title standardization...")
standardization_stats = get_standardization_stats()
logger.info(f"Standardization stats: {standardization_stats}")

df['original_title'] = df['job title']
df['job title'] = df['job title'].apply(standardize_title)

# Log standardization impact
standardized_count = (df['original_title'] != df['job title']).sum()
logger.info(f"Standardized {standardized_count} out of {len(df)} titles ({standardized_count/len(df)*100:.1f}%)")

return df
```

def perform_data_quality_checks(df):
“”“Perform data quality checks and log statistics.”””
logger.info(”\n=== Data Quality Report ===”)

```
# Check for duplicates
duplicate_count = df.duplicated(subset=['job title', 'persona segment']).sum()
if duplicate_count > 0:
    logger.warning(f"Found {duplicate_count} duplicate entries")
    df = df.drop_duplicates(subset=['job title', 'persona segment'])
    logger.info(f"Removed duplicates, {len(df)} unique entries remain")

# Validate persona segments
unique_personas = df['persona segment'].unique()
invalid_personas = set(unique_personas) - set(VALID_PERSONAS)
if invalid_personas:
    logger.error(f"❌ Invalid persona segments found: {invalid_personas}")
    logger.error(f"Valid personas are: {VALID_PERSONAS}")
    raise ValueError(f"❌ Invalid persona segments in training data: {invalid_personas}")

# Check for missing valid personas
missing_personas = set(VALID_PERSONAS) - set(unique_personas)
if missing_personas:
    logger.warning(f"Missing persona segments in training data: {missing_personas}")
    logger.warning("Model will not be able to predict these personas accurately")

# Analyze persona distribution
persona_counts = df['persona segment'].value_counts()

# Check for single class
if len(persona_counts) == 1:
    raise ValueError(f"❌ Training data contains only one persona segment: {persona_counts.index[0]}. Need at least 2 different personas for classification.")

logger.info("\nPersona Segment Distribution:")
for persona, count in persona_counts.items():
    percentage = count / len(df) * 100
    logger.info(f"  {persona}: {count} samples ({percentage:.1f}%)")

# Check for classes with too few samples
low_sample_classes = persona_counts[persona_counts < MIN_SAMPLES_PER_CLASS]
if not low_sample_classes.empty:
    logger.warning(f"\nClasses with fewer than {MIN_SAMPLES_PER_CLASS} samples:")
    for persona, count in low_sample_classes.items():
        logger.warning(f"  {persona}: {count} samples")

# Check class imbalance
if len(persona_counts) > 1:
    imbalance_ratio = persona_counts.max() / persona_counts.min()
    if imbalance_ratio > CLASS_IMBALANCE_THRESHOLD:
        logger.warning(f"\nHigh class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
        logger.warning("Consider using class weights or resampling techniques")

# Title diversity check
unique_titles = df['job title'].nunique()
logger.info(f"\nUnique job titles: {unique_titles}")

return df
```

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
“”“Train the model and evaluate its performance.”””
logger.info(”\n=== Model Training ===”)

```
# Define the model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=NGRAM_RANGE,  # Include trigrams for better context
        stop_words='english',
        max_features=MAX_FEATURES,
        min_df=MIN_DF  # Ignore terms that appear in less than MIN_DF documents
    )),
    ('clf', LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight='balanced'  # Handle class imbalance
    ))
])

# Train the model
logger.info("Training model...")
pipeline.fit(X_train, y_train)

# Evaluate on test set
logger.info("\n=== Model Evaluation ===")
y_pred = pipeline.predict(X_test)

# Classification report
logger.info("\nClassification Report:")
report = classification_report(y_test, y_pred)
logger.info("\n" + report)

# Cross-validation scores
# Determine number of CV folds based on smallest class size in training set
train_persona_counts = pd.Series(y_train).value_counts()
min_train_class_size = train_persona_counts.min()

# Use fewer folds if necessary
n_folds = min(5, min_train_class_size)
if n_folds < 2:
    logger.warning(f"Skipping cross-validation (smallest class has only {min_train_class_size} samples in training set)")
else:
    logger.info(f"\nCross-validation scores ({n_folds}-fold):")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=n_folds, scoring='accuracy')
    logger.info(f"  Mean accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Test set accuracy
test_accuracy = pipeline.score(X_test, y_test)
logger.info(f"  Test set accuracy: {test_accuracy:.3f}")

return pipeline, test_accuracy
```

def save_model_and_metadata(pipeline, test_accuracy, df, persona_counts):
“”“Save the trained model and metadata.”””
# Ensure model directory exists
os.makedirs(“model”, exist_ok=True)

```
# Save the model
joblib.dump(pipeline, MODEL_FILE)
logger.info(f"\nModel saved to {MODEL_FILE}")

# Save metadata
metadata = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_samples': len(df),
    'unique_titles': df['job title'].nunique(),
    'test_accuracy': f"{test_accuracy:.3f}",
    'persona_distribution': persona_counts.to_dict(),
    'model_parameters': {
        'tfidf_ngram_range': pipeline.named_steps['tfidf'].ngram_range,
        'tfidf_max_features': pipeline.named_steps['tfidf'].max_features,
        'classifier': type(pipeline.named_steps['clf']).__name__
    }
}

with open(MODEL_METADATA_FILE, 'w') as f:
    f.write("=== Model Training Metadata ===\n\n")
    for key, value in metadata.items():
        if isinstance(value, dict):
            f.write(f"{key}:\n")
            for k, v in value.items():
                f.write(f"  {k}: {v}\n")
        else:
            f.write(f"{key}: {value}\n")

logger.info(f"Metadata saved to {MODEL_METADATA_FILE}")
```

def main():
try:
# Check if training file exists
if not os.path.exists(TRAINING_FILE):
raise FileNotFoundError(f”❌ Training file not found: {TRAINING_FILE}”)

```
    # Load and prepare data
    df = load_and_prepare_data(TRAINING_FILE)
    
    # Perform data quality checks
    df = perform_data_quality_checks(df)
    
    # Get persona counts for metadata
    persona_counts = df['persona segment'].value_counts()
    
    # Prepare features and labels
    X = df['job title']
    y = df['persona segment']
    
    # Split data for validation
    # Check if we can safely stratify
    min_class_size = persona_counts.min()
    can_stratify = min_class_size >= MIN_SAMPLES_FOR_STRATIFY  # Need enough samples per class for split
    
    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        logger.info(f"\nTrain/test split (stratified): {len(X_train)} training, {len(X_test)} test samples")
    else:
        logger.warning(f"Cannot use stratified split (smallest class has {min_class_size} samples)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logger.info(f"\nTrain/test split (random): {len(X_train)} training, {len(X_test)} test samples")
    
    # Train and evaluate model
    pipeline, test_accuracy = train_and_evaluate_model(X_train, y_train, X_test, y_test)
    
    # Save model and metadata
    save_model_and_metadata(pipeline, test_accuracy, df, persona_counts)
    
    logger.info("\n✅ Model training completed successfully!")
    
except FileNotFoundError as e:
    logger.error(f"❌ {e}")
    sys.exit(1)
except ValueError as e:
    logger.error(f"❌ {e}")
    sys.exit(1)
except Exception as e:
    logger.error("❌ There was an error retraining the model.")
    logger.error(f"Error type: {type(e).__name__}")
    logger.error(f"Error details: {e}")
    sys.exit(1)
```

if **name** == “**main**”:
main()