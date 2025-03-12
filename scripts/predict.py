import pandas as pd
import joblib
import os
import numpy as np
from scripts.title_standardizer import standardize_title

# File paths
MODEL_FILE = "model/persona_classifier.pkl"
INPUT_FILE = "data/input.csv"
OUTPUT_FILE = "tagged_personas.csv"

try:
    # Load model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Model file not found. Train the model first using `make train`.")

    model = joblib.load(MODEL_FILE)

    # Load input data
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Ensure necessary columns exist
    if not {'Record ID', 'Job title'}.issubset(df.columns):
        raise ValueError("Input file must contain 'Record ID' and 'Job title' columns.")

    df.dropna(subset=['Record ID', 'Job title'], inplace=True)

    if df.empty:
        raise ValueError("Input data is empty after dropping missing values.")

    # Standardize job titles
    df['Job title'] = df['Job title'].apply(standardize_title)

    # Make predictions
    probabilities = model.predict_proba(df['Job title'])
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    confidence_scores = np.max(probabilities, axis=1) * 100

    # Round confidence scores to nearest multiple of 5
    confidence_scores = np.round(confidence_scores / 5) * 5

    # Ensure confidence scores are within 0-100 range
    confidence_scores = np.clip(confidence_scores, 0, 100)

    # Add results to DataFrame
    df['Persona Segment'] = predictions
    df['Confidence Score'] = confidence_scores.astype(int)

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ The input file has been tagged.")

except Exception as e:
    print("❌ There was an error tagging the input file.")
    print(f"Error: {e}")