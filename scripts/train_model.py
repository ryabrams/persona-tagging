import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os

# File paths
TRAINING_FILE = "data/training_data.csv"
MODEL_FILE = "model/persona_classifier.pkl"

# Load training data
df = pd.read_csv(TRAINING_FILE)

# Ensure data format
df.dropna(inplace=True)
X_train, y_train = df['Job title'], df['Persona Segment']

# Define the model pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Save the model
joblib.dump(pipeline, MODEL_FILE)
print(f"Model trained and saved at {MODEL_FILE}")