import pandas as pd
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# File path
REFERENCE_FILE = "data/title_reference.csv"

# Load title reference data
reference_dict = {}

try:
    if os.path.exists(REFERENCE_FILE):
        reference_df = pd.read_csv(REFERENCE_FILE)
        if {'Reference', 'Standardization'}.issubset(reference_df.columns):
            reference_dict = dict(zip(reference_df['Reference'], reference_df['Standardization']))
        else:
            logger.warning("title_reference.csv does not contain expected columns ('Reference', 'Standardization'). Standardization disabled.")
    else:
        logger.info("title_reference.csv not found. Standardization disabled.")
except Exception as e:
    logger.error(f"Error loading title reference file: {e}. Standardization disabled.")

def standardize_title(title):
    if not isinstance(title, str):
        return str(title) if title is not None else ""
    return reference_dict.get(title, title)  # Return standardized title if available, else return original