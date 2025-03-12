import pandas as pd
import joblib
import os
import numpy as np
from title_standardizer import standardize_title

# File paths
MODEL_FILE = "model/persona_classifier.pkl"
INPUT_FILE = "data/input.csv"
OUTPUT_FILE = "tagged_personas.csv"

# Load model
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Model file not found. Train the model first using `make train`.")

model = joblib.load(MODEL_FILE)

# Load input data
df = pd.read_csv(INPUT_FILE)
df.dropna(inplace=True)

# Standardize job titles
df['Job title'] = df['Job title'].apply(standardize_title)

# Make predictions
probabilities = model.predict_proba(df['Job title'])
predictions = model.classes_[np.argmax(probabilities, axis=1)]
confidence_scores = np.max(probabilities, axis=1) * 100

# Round confidence scores to nearest multiple of 5
confidence_scores = np.round(confidence_scores / 5) * 5

# Ensure minimum confidence of 65%
confidence_scores[confidence_scores < 65] = 65

# Add results to DataFrame
df['Persona Segment'] = predictions
df['Confidence Score'] = confidence_scores.astype(int)

# Save to CSV
df.to_csv(OUTPUT_FILE, index=False)
print(f"Predictions saved to {OUTPUT_FILE}")