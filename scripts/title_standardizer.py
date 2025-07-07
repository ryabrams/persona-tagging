import pandas as pd
import os
import logging
from functools import lru_cache
from difflib import get_close_matches

# Configure logging
logger = logging.getLogger(__name__)

# File path
REFERENCE_FILE = "data/title_reference.csv"

# Global cache for reference dictionary
_reference_dict = None
_reference_loaded = False


def load_reference_dict():
    """Load the reference dictionary from CSV file with caching."""
    global _reference_dict, _reference_loaded
    
    if _reference_loaded:
        return _reference_dict
    
    _reference_dict = {}
    
    if os.path.exists(REFERENCE_FILE):
        try:
            reference_df = pd.read_csv(REFERENCE_FILE)
            
            # Check for required columns
            if {'Reference', 'Standardization'}.issubset(reference_df.columns):
                # Create case-insensitive dictionary
                for ref, std in zip(reference_df['Reference'], reference_df['Standardization']):
                    if pd.notna(ref) and pd.notna(std):
                        # Store with lowercase key for case-insensitive lookup
                        _reference_dict[str(ref).lower().strip()] = str(std).strip()
                
                logger.info(f"Loaded {len(_reference_dict)} title mappings from {REFERENCE_FILE}")
            else:
                logger.warning(f"title_reference.csv does not contain expected columns 'Reference' and 'Standardization'. Standardization will be skipped.")
                
        except Exception as e:
            logger.error(f"Error loading title reference file: {e}")
    else:
        logger.info(f"Title reference file not found at {REFERENCE_FILE}. Standardization will be skipped.")
    
    _reference_loaded = True
    return _reference_dict


@lru_cache(maxsize=1000)
def standardize_title(title, use_fuzzy=False, fuzzy_threshold=0.8):
    """
    Standardize a job title based on reference mappings.
    
    Args:
        title: The job title to standardize
        use_fuzzy: Whether to use fuzzy matching for close matches
        fuzzy_threshold: Minimum similarity score for fuzzy matching (0-1)
    
    Returns:
        Standardized title if found, otherwise the original title
    """
    if pd.isna(title) or not isinstance(title, str):
        return title
    
    # Load reference dictionary if not already loaded
    reference_dict = load_reference_dict()
    
    if not reference_dict:
        return title
    
    # Clean and lowercase the input title
    clean_title = str(title).lower().strip()
    
    # Exact match (case-insensitive)
    if clean_title in reference_dict:
        return reference_dict[clean_title]
    
    # Fuzzy matching if enabled
    if use_fuzzy:
        # Get close matches
        close_matches = get_close_matches(
            clean_title, 
            reference_dict.keys(), 
            n=1, 
            cutoff=fuzzy_threshold
        )
        
        if close_matches:
            matched_key = close_matches[0]
            logger.debug(f"Fuzzy match: '{title}' -> '{reference_dict[matched_key]}' (matched with '{matched_key}')")
            return reference_dict[matched_key]
    
    # Return original title if no match found
    return title


def get_standardization_stats():
    """Get statistics about the loaded standardizations."""
    reference_dict = load_reference_dict()
    
    if not reference_dict:
        return {
            'loaded': False,
            'count': 0,
            'unique_standardizations': 0
        }
    
    unique_standardizations = len(set(reference_dict.values()))
    
    return {
        'loaded': True,
        'count': len(reference_dict),
        'unique_standardizations': unique_standardizations
    }


def reload_reference():
    """Force reload of the reference dictionary."""
    global _reference_loaded
    _reference_loaded = False
    standardize_title.cache_clear()
    logger.info("Reference dictionary cache cleared and will be reloaded on next use.")