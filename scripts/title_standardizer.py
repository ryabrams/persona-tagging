import pandas as pd

# File path
REFERENCE_FILE = "data/title_reference.csv"

# Load title reference data
try:
    reference_df = pd.read_csv(REFERENCE_FILE)
    reference_dict = dict(zip(reference_df['Reference'], reference_df['Standardization']))
except FileNotFoundError:
    reference_dict = {}

def standardize_title(title):
    return reference_dict.get(title, title)  # Return standardized title if available, else return original