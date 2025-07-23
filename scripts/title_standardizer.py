import pandas as pd
import os
import logging
from functools import lru_cache
from difflib import get_close_matches
import threading

logger = logging.getLogger(__name__)
REFERENCE_FILE = "data/title_reference.csv"

_reference_dict = None
_reference_loaded = False
_reference_lock = threading.Lock()

def load_reference_dict():
    """Load the reference dictionary from CSV file with thread-safe caching."""
    global _reference_dict, _reference_loaded

    if _reference_loaded:
        return _reference_dict

    with _reference_lock:
        if _reference_loaded:
            return _reference_dict

        _reference_dict = {}

        if os.path.exists(REFERENCE_FILE):
            try:
                reference_df = pd.read_csv(REFERENCE_FILE, encoding='utf-8')
                if {'Reference', 'Standardization'}.issubset(reference_df.columns):
                    for ref, std in zip(reference_df['Reference'], reference_df['Standardization']):
                        if pd.notna(ref) and pd.notna(std):
                            _reference_dict[str(ref).lower().strip()] = str(std).strip()
                    if logger.isEnabledFor(logging.INFO):
                        logger.info(f"Loaded {len(_reference_dict)} title mappings from {REFERENCE_FILE}")
                else:
                    if logger.isEnabledFor(logging.WARNING):
                        logger.warning("title_reference.csv does not contain expected columns 'Reference' and 'Standardization'. Standardization will be skipped.")
            except Exception as e:
                if logger.isEnabledFor(logging.ERROR):
                    logger.error(f"Error loading title reference file: {e}")
        else:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Title reference file not found at {REFERENCE_FILE}. Standardization will be skipped.")

        _reference_loaded = True
        return _reference_dict

def _standardize_title_impl(title_lower, reference_dict, use_fuzzy, fuzzy_threshold):
    """Internal implementation without caching to avoid memory leak."""
    # Exact match (case-insensitive)
    if title_lower in reference_dict:
        return reference_dict[title_lower]

    # Fuzzy matching if enabled
    if use_fuzzy and reference_dict:
        close_matches = get_close_matches(
            title_lower,
            reference_dict.keys(),
            n=1,
            cutoff=fuzzy_threshold
        )
        if close_matches:
            matched_key = close_matches[0]
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Fuzzy match: '{title_lower}' -> '{reference_dict[matched_key]}' (matched with '{matched_key}')")
            return reference_dict[matched_key]
    return None

@lru_cache(maxsize=1000)
def _cached_standardize(title_lower):
    """Cached version for exact matches only."""
    reference_dict = load_reference_dict()
    if not reference_dict:
        return None
    return reference_dict.get(title_lower)

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

    clean_title = str(title).lower().strip()

    if not use_fuzzy:
        result = _cached_standardize(clean_title)
        return result if result is not None else title

    reference_dict = load_reference_dict()
    if not reference_dict:
        return title

    result = _standardize_title_impl(clean_title, reference_dict, use_fuzzy, fuzzy_threshold)
    return result if result is not None else title

def is_standardization_available():
    """Check if title standardization is available (faster than getting full stats)."""
    reference_dict = load_reference_dict()
    return reference_dict is not None and len(reference_dict) > 0

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
    with _reference_lock:
        _reference_loaded = False
        _cached_standardize.cache_clear()
        if logger.isEnabledFor(logging.INFO):
            logger.info("Reference dictionary cache cleared and will be reloaded on next use.")