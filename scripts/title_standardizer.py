import pandas as pd
import os

# File path
REFERENCE_FILE = "data/title_reference.csv"

# Load title reference data
reference_dict = {}

if os.path.exists(REFERENCE_FILE):
    reference_df = pd.read_csv(REFERENCE_FILE)
    if {'Reference', 'Standardization'}.issubset(reference_df.columns):
        reference_dict = dict(zip(reference_df['Reference'], reference_df['Standardization']))
    else:
        print("Warning: title_reference.csv does not contain expected columns. Standardization skipped.")

def standardize_title(title):
    return reference_dict.get(title, title)  # Return standardized title if available, else return original