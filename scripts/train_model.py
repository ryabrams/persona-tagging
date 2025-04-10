import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

# File paths
TRAINING_FILE = "data/training_data.csv"
MODEL_FILE = "model/persona_classifier.pkl"

try:
    # Ensure training file exists
    if not os.path.exists(TRAINING_FILE):
        raise FileNotFoundError(f"Training file not found: {TRAINING_FILE}")

    # Load training data
    df = pd.read_csv(TRAINING_FILE)

    # Standardize column names to lowercase for case-insensitivity
    df.columns = df.columns.str.lower()

    # Ensure necessary columns exist
    if not {'job title', 'persona segment'}.issubset(df.columns):
        raise ValueError("Training file must contain 'Job Title' and 'Persona Segment' columns (case-insensitive).")

    df.dropna(subset=['job title', 'persona segment'], inplace=True)

    if df.empty:
        raise ValueError("Training data is empty after dropping missing values.")

    X_train, y_train = df['job title'], df['persona segment']

    # Define the model pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Ensure model directory exists
    os.makedirs("model", exist_ok=True)

    # Save the model
    joblib.dump(pipeline, MODEL_FILE)

    print("✅ The model has been retrained.")

except Exception as e:
    print("❌ There was an error retraining the model.")
    print(f"Error: {e}")