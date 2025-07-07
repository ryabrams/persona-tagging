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
KEYWORD_FILE = "data/keyword_matching.csv"

# Define Persona Segment Priority Order
PRIORITY_ORDER = ["GenAI", "Engineering", "Product", "Cyber Security", "Trust & Safety", "Legal & Compliance", "Executive"]


def apply_keyword_matching(df, keyword_file):
    """Apply keyword-based persona assignment rules."""
    if not os.path.exists(keyword_file):
        return df, pd.DataFrame()
    
    keyword_df = pd.read_csv(keyword_file)
    
    # Ensure optional exclusion column exists
    if 'Exclude Keyword' not in keyword_df.columns:
        keyword_df['Exclude Keyword'] = ""
    
    # Prepare columns for keyword matches
    df['Persona Segment'] = ""
    df['Confidence Score'] = 0
    
    # Convert job titles to lowercase once for efficiency
    job_titles_lower = df['job title'].astype(str).str.lower()
    
    # Apply rules in order listed
    for _, rule in keyword_df.iterrows():
        include_kw = str(rule['Keyword']).strip().lower()
        exclude_kw = str(rule.get('Exclude Keyword', "")).strip().lower()
        rule_type = str(rule['Rule']).lower()
        segment = str(rule['Persona Segment'])
        
        # Skip if invalid rule type
        if rule_type not in ['contains', 'equals']:
            logger.warning(f"Skipping invalid rule type: {rule_type}")
            continue
        
        # Create mask based on rule type
        if rule_type == 'contains':
            mask = job_titles_lower.str.contains(include_kw, na=False)
        else:  # equals
            mask = job_titles_lower == include_kw
        
        # Apply exclusion if specified
        if exclude_kw:
            mask &= ~job_titles_lower.str.contains(exclude_kw, na=False)
        
        # Apply segment only to unassigned rows
        unassigned_mask = df['Persona Segment'] == ""
        final_mask = mask & unassigned_mask
        
        df.loc[final_mask, 'Persona Segment'] = segment
        df.loc[final_mask, 'Confidence Score'] = 100
    
    # Split matched and unmatched
    df_matched = df[df['Persona Segment'] != ""].copy()
    df_unmatched = df[df['Persona Segment'] == ""].copy()
    
    return df_unmatched, df_matched


def enforce_priority(pred_labels, probs, classes, priority_order):
    """Enforce persona segment priority when model is uncertain."""
    adjusted_labels = []
    
    for i, label in enumerate(pred_labels):
        # Get probability for the predicted label
        max_prob = probs[i].max()
        
        # Only adjust if confidence is relatively low (< 70%)
        if max_prob < 0.7:
            # Find labels with similar probabilities (within 10% of max)
            close_indices = np.where(probs[i] >= max_prob - 0.1)[0]
            close_labels = classes[close_indices]
            
            # Find the highest priority label among close candidates
            for priority_label in priority_order:
                if priority_label in close_labels:
                    adjusted_labels.append(priority_label)
                    break
            else:
                adjusted_labels.append(label)  # Keep original if no priority match
        else:
            adjusted_labels.append(label)  # Keep high-confidence predictions
    
    return np.array(adjusted_labels)


def main():
    try:
        # Load model
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError("❌ Model file not found. Train the model first using the `make train` command.")
        
        model = joblib.load(MODEL_FILE)
        logger.info("Model loaded successfully.")
        
        # Load input data
        if not os.path.exists(INPUT_FILE):
            raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")
        
        df = pd.read_csv(INPUT_FILE)
        logger.info(f"Loaded {len(df)} records from input file.")
        
        # Standardize column names to lower case
        df.columns = df.columns.str.lower()
        
        # Ensure necessary columns exist
        required_columns = {'record id', 'job title'}
        if not required_columns.issubset(df.columns):
            raise ValueError("❌ Input file must contain both 'Record ID' and 'Job Title' columns (case-insensitive).")
        
        # Drop rows with missing required data
        initial_count = len(df)
        df.dropna(subset=['record id', 'job title'], inplace=True)
        dropped_count = initial_count - len(df)
        
        if dropped_count > 0:
            logger.warning(f"Dropped {dropped_count} rows with missing data.")
        
        if df.empty:
            raise ValueError("❌ Input data is empty after dropping missing values.")
        
        # Apply keyword-based classification
        df_unmatched, df_matched = apply_keyword_matching(df, KEYWORD_FILE)
        
        # Log keyword matching results
        n_keyword_matched = len(df_matched)
        n_to_predict = len(df_unmatched)
        
        if n_keyword_matched > 0:
            logger.info(f"Keyword matching: {n_keyword_matched} rows assigned.")
        
        # If all entries were keyword-assigned, save and exit
        if df_unmatched.empty:
            df_matched.to_csv(OUTPUT_FILE, index=False)
            logger.info("All entries were categorized by keywords.")
            logger.info(f"✅ Output saved to {OUTPUT_FILE}")
            return
        
        # Prepare for ML prediction
        logger.info(f"Applying ML model to {n_to_predict} remaining rows.")
        
        # Standardize job titles for ML model
        df_unmatched['job title'] = df_unmatched['job title'].apply(standardize_title)
        
        # Make predictions
        probabilities = model.predict_proba(df_unmatched['job title'])
        predicted_labels = model.classes_[np.argmax(probabilities, axis=1)]
        confidence_scores = np.max(probabilities, axis=1) * 100
        
        # Clip first, then round confidence scores
        confidence_scores = np.clip(confidence_scores, 0, 100)
        confidence_scores = np.round(confidence_scores / 5) * 5
        
        # Apply priority enforcement
        adjusted_predictions = enforce_priority(
            predicted_labels, 
            probabilities, 
            model.classes_, 
            PRIORITY_ORDER
        )
        
        # Assign predictions
        df_unmatched['Persona Segment'] = adjusted_predictions
        df_unmatched['Confidence Score'] = confidence_scores.astype(int)
        
        # Clear low-confidence predictions
        low_conf_mask = df_unmatched['Confidence Score'] < 50
        n_low_confidence = low_conf_mask.sum()
        df_unmatched.loc[low_conf_mask, 'Persona Segment'] = ""
        
        # Combine results
        if not df_matched.empty:
            df_final = pd.concat([df_matched, df_unmatched], ignore_index=True)
        else:
            df_final = df_unmatched
        
        # Sort by record ID for consistent output
        df_final = df_final.sort_values('record id').reset_index(drop=True)
        
        # Save to CSV
        df_final.to_csv(OUTPUT_FILE, index=False)
        
        # Log summary statistics
        logger.info("Classification complete:")
        logger.info(f"  - Keyword matched: {n_keyword_matched}")
        logger.info(f"  - ML predicted: {n_to_predict}")
        logger.info(f"  - Low confidence (unassigned): {n_low_confidence}")
        logger.info(f"  - Total processed: {len(df_final)}")
        logger.info(f"✅ Output saved to {OUTPUT_FILE}")
        
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("There was an error categorizing the input file.")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()