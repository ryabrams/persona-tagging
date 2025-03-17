import pandas as pd
import joblib
import os
import numpy as np
from title_standardizer import standardize_title

# File paths
MODEL_FILE = "model/persona_classifier.pkl"
INPUT_FILE = "data/input.csv"
OUTPUT_FILE = "tagged_personas.csv"

# Define Persona Segment Priority Order
priority_order = ["GenAI", "Engineering", "Product", "Trust & Safety", "Legal & Compliance", "Executive"]

try:
    # Load model
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("❌ Model file not found. Train the model first using the `make train` command.")

    model = joblib.load(MODEL_FILE)

    # Load input data
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # Ensure necessary columns exist
    if not {'Record ID', 'Job Title'}.issubset(df.columns):
        raise ValueError("❌ Input file must contain both 'Record ID' and 'Job Title' columns.")

    df.dropna(subset=['Record ID', 'Job Title'], inplace=True)

    if df.empty:
        raise ValueError("❌ Input data is empty after dropping missing values.")

    # Standardize job titles
    df['Job Title'] = df['Job Title'].apply(standardize_title)

    # Make predictions
    probabilities = model.predict_proba(df['Job Title'])
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

    # Add results to DataFrame
    df['Persona Segment'] = adjusted_predictions
    df['Confidence Score'] = confidence_scores.astype(int)

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)

    print("✅ Success! The input file data has been categorized.")

except Exception as e:
    print("❌ There was an error categorizing the input file.")
    print(f"Error: {e}")