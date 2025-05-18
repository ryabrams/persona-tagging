import pandas as pd
import joblib
import os
import sys
import numpy as np
import logging
from title_standardizer import standardize_title

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File paths
MODEL_FILE = "model/persona_classifier.pkl"
INPUT_FILE = "data/input.csv"
OUTPUT_FILE = "tagged_personas.csv"

# Keyword matching rules file
KEYWORD_FILE = "data/keyword_matching.csv"

# Define Persona Segment Priority Order
priority_order = ["GenAI", "Engineering", "Product", "Cyber Security", "Trust & Safety", "Legal & Compliance", "Executive"]

try:
    # Load model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("❌ Model file not found. Train the model first using the `make train` command.")

    model = joblib.load(MODEL_FILE)

    # Load input data
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Standardize column names to lower case for case-insensitivity
    df.columns = df.columns.str.lower()

    # Ensure necessary columns exist
    if not {'record id', 'job title'}.issubset(df.columns):
        raise ValueError("❌ Input file must contain both 'Record ID' and 'Job Title' columns (case-insensitive).")

    df.dropna(subset=['record id', 'job title'], inplace=True)

    if df.empty:
        raise ValueError("❌ Input data is empty after dropping missing values.")

    # Keyword-based Persona Segment assignment
    if os.path.exists(KEYWORD_FILE):
        keyword_df = pd.read_csv(KEYWORD_FILE)
        # Allow optional exclusion keyword for combined include/exclude rules
        if 'Exclude Keyword' not in keyword_df.columns:
            keyword_df['Exclude Keyword'] = ""
        # Prepare columns for keyword matches
        df['Persona Segment'] = ""
        df['Confidence Score'] = 0
        # Apply rules in order listed
        for _, rule in keyword_df.iterrows():
            include_kw = str(rule['Keyword']).strip()
            exclude_kw = str(rule.get('Exclude Keyword', "")).strip()
            rule_type = str(rule['Rule']).lower()
            segment = str(rule['Persona Segment'])
            if rule_type == 'contains':
                mask = df['job title'].astype(str).str.lower().str.contains(include_kw.lower(), na=False)
            elif rule_type == 'equals':
                mask = df['job title'].astype(str).str.lower() == include_kw.lower()
            else:
                continue
            # Apply exclusion if specified
            if exclude_kw:
                mask &= ~df['job title'].astype(str).str.lower().str.contains(exclude_kw.lower(), na=False)
            df.loc[mask & (df['Persona Segment'] == ""), 'Persona Segment'] = segment
            df.loc[mask & (df['Confidence Score'] == 0), 'Confidence Score'] = 100
        # Separate out keyword-assigned rows
        df_matched = df[df['Persona Segment'] != ""].copy()
        df = df[df['Persona Segment'] == ""].copy()
        # If all entries were keyword-assigned, save and exit early
        if df.empty:
            df_matched.to_csv(OUTPUT_FILE, index=False)
            logger.info("All entries were categorized by keywords.")
            sys.exit(0)
    else:
        # No keyword file: prepare empty columns for later prediction
        df['Persona Segment'] = ""
        df['Confidence Score'] = 0

    # Standardize job titles
    df['job title'] = df['job title'].apply(standardize_title)

    # Make predictions
    probabilities = model.predict_proba(df['job title'])
    predicted_labels = model.classes_[np.argmax(probabilities, axis=1)]
    confidence_scores = np.max(probabilities, axis=1) * 100

    # Round confidence scores to the nearest multiple of 5
    confidence_scores = np.round(confidence_scores / 5) * 5

    # Ensure confidence scores are within 0-100 range
    confidence_scores = np.clip(confidence_scores, 0, 100)

    # Enforce persona segment priority
    def enforce_priority(pred_labels, probs):
        adjusted_labels = []
        for i, label in enumerate(pred_labels):
            # Get all probabilities for this input
            label_probs = probs[i]

            # Sort labels based on model probabilities
            sorted_indices = np.argsort(label_probs)[::-1]
            sorted_labels = model.classes_[sorted_indices]

            # Find the highest-priority label based on predefined order
            for candidate in sorted_labels:
                if candidate in priority_order:
                    adjusted_labels.append(candidate)
                    break
            else:
                adjusted_labels.append(label)  # Default to model prediction if no match found

        return np.array(adjusted_labels)

    # Apply priority enforcement
    adjusted_predictions = enforce_priority(predicted_labels, probabilities)

    # Combine keyword-assigned entries with model predictions
    # Use adjusted_predictions and confidence_scores for the unmatched subset
    df_unmatched = df.copy()  # contains only rows not assigned via keywords
    df_unmatched['Persona Segment'] = adjusted_predictions
    df_unmatched['Confidence Score'] = confidence_scores.astype(int)
    df_unmatched.loc[df_unmatched['Confidence Score'] < 60, 'Persona Segment'] = ""
    if 'df_matched' in locals():
        df = pd.concat([df_matched, df_unmatched], ignore_index=True)
    else:
        df = df_unmatched

    # Log basic metrics
    n_keyword_matched = df_matched.shape[0] if 'df_matched' in locals() else 0
    n_model_predicted = df_unmatched.shape[0]
    n_low_confidence = (df_unmatched['Confidence Score'] < 60).sum()
    logger.info(f"{n_keyword_matched} rows keyword-matched, {n_model_predicted} rows model-predicted, {n_low_confidence} rows low-confidence")

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)

    logger.info("The input file data has been categorized.")

except Exception as e:
    logger.error("There was an error categorizing the input file.")
    logger.error(f"Error: {e}")