import pandas as pd
import os
import logging
from functools import lru_cache
from difflib import get_close_matches
import threading
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
REFERENCE_FILE = Path("data") / "title_reference.csv"

# Global variables for thread-safe caching
_reference_dict = None
_reference_loaded = False
_reference_lock = threading.Lock()

def load_reference_dict() -> Dict[str, str]:
    """Load the reference dictionary from CSV file with thread-safe caching."""
    global _reference_dict, _reference_loaded

    if _reference_loaded:
        return _reference_dict

    with _reference_lock:
        if _reference_loaded:
            return _reference_dict

        _reference_dict = {}

        if REFERENCE_FILE.exists():
            try:
                reference_df = pd.read_csv(REFERENCE_FILE, encoding='utf-8')
                
                # Check for required columns with case-insensitive matching
                columns_lower = [col.lower() for col in reference_df.columns]
                required_columns = {'reference', 'standardization'}
                
                if not required_columns.issubset(set(columns_lower)):
                    missing_cols = required_columns - set(columns_lower)
                    logger.warning(f"title_reference.csv is missing required columns: {missing_cols}. "
                                 f"Expected columns: Reference, Standardization (case-insensitive). "
                                 f"Found columns: {list(reference_df.columns)}. Standardization will be skipped.")
                else:
                    # Map to actual column names
                    ref_col = None
                    std_col = None
                    for i, col in enumerate(reference_df.columns):
                        if col.lower() == 'reference':
                            ref_col = col
                        elif col.lower() == 'standardization':
                            std_col = col
                    
                    if ref_col and std_col:
                        # Filter out rows with missing values
                        valid_rows = reference_df[[ref_col, std_col]].dropna()
                        initial_count = len(reference_df)
                        valid_count = len(valid_rows)
                        
                        if initial_count > valid_count:
                            logger.warning(f"Dropped {initial_count - valid_count} rows with missing values from title reference file")
                        
                        # Build dictionary
                        for ref, std in zip(valid_rows[ref_col], valid_rows[std_col]):
                            ref_str = str(ref).lower().strip()
                            std_str = str(std).strip()
                            
                            # Skip empty strings
                            if ref_str and std_str:
                                _reference_dict[ref_str] = std_str
                        
                        logger.info(f"Loaded {len(_reference_dict)} title mappings from {REFERENCE_FILE}")
                    
                        # Check for duplicate references
                        duplicates = valid_rows[ref_col].str.lower().str.strip().duplicated()
                        if duplicates.any():
                            duplicate_count = duplicates.sum()
                            logger.warning(f"Found {duplicate_count} duplicate reference titles in {REFERENCE_FILE}. "
                                         f"Only the last occurrence of each duplicate will be used.")
                                         
            except pd.errors.EmptyDataError:
                logger.warning(f"Title reference file {REFERENCE_FILE} is empty. Standardization will be skipped.")
            except pd.errors.ParserError as e:
                logger.error(f"Error parsing title reference file {REFERENCE_FILE}: {e}. Standardization will be skipped.")
            except Exception as e:
                logger.error(f"Unexpected error loading title reference file: {e}. Standardization will be skipped.")
                
        else:
            logger.info(f"Title reference file not found at {REFERENCE_FILE}. Standardization will be skipped.")

        _reference_loaded = True
        return _reference_dict

def _standardize_title_impl(title_lower: str, reference_dict: Dict[str, str], 
                           use_fuzzy: bool, fuzzy_threshold: float) -> Optional[str]:
    """Internal implementation without caching to avoid memory leak."""
    # Exact match (case-insensitive)
    if title_lower in reference_dict:
        return reference_dict[title_lower]

    # Fuzzy matching if enabled
    if use_fuzzy and reference_dict:
        try:
            close_matches = get_close_matches(
                title_lower,
                reference_dict.keys(),
                n=1,
                cutoff=fuzzy_threshold
            )
            if close_matches:
                matched_key = close_matches[0]
                logger.debug(f"Fuzzy match: '{title_lower}' -> '{reference_dict[matched_key]}' "
                           f"(matched with '{matched_key}')")
                return reference_dict[matched_key]
        except Exception as e:
            logger.warning(f"Error in fuzzy matching for '{title_lower}': {e}")
            
    return None

@lru_cache(maxsize=500)  # Reduced cache size to prevent memory issues
def _cached_standardize(title_lower: str, dict_id: int) -> Optional[str]:
    """
    Cached version for exact matches only.

    Args:
        title_lower: Lowercase job title to lookup
        dict_id: ID of the reference dict (for cache invalidation on reload)

    Returns:
        Standardized title if found, None otherwise
    """
    # Access the global reference dict (already loaded and thread-safe)
    if not _reference_dict:
        return None
    return _reference_dict.get(title_lower)

def standardize_title(title: Any, use_fuzzy: bool = False, fuzzy_threshold: float = 0.8) -> str:
    """
    Standardize a job title based on reference mappings.

    Args:
        title: The job title to standardize
        use_fuzzy: Whether to use fuzzy matching for close matches
        fuzzy_threshold: Minimum similarity score for fuzzy matching (0-1)

    Returns:
        Standardized title if found, otherwise the original title
    """
    # Handle non-string inputs
    if pd.isna(title):
        return ""
    
    if not isinstance(title, str):
        title = str(title)
    
    # Basic validation
    if not title.strip():
        return title

    clean_title = title.lower().strip()
    
    # Validate fuzzy threshold
    if use_fuzzy and not (0.0 <= fuzzy_threshold <= 1.0):
        logger.warning(f"Invalid fuzzy_threshold {fuzzy_threshold}, using default 0.8")
        fuzzy_threshold = 0.8

    # Load reference dict to ensure it's available
    reference_dict = load_reference_dict()
    if not reference_dict:
        return title

    if not use_fuzzy:
        # Use cached lookup with dict ID for cache invalidation
        result = _cached_standardize(clean_title, id(reference_dict))
        return result if result is not None else title

    result = _standardize_title_impl(clean_title, reference_dict, use_fuzzy, fuzzy_threshold)
    return result if result is not None else title

def is_standardization_available() -> bool:
    """Check if title standardization is available (faster than getting full stats)."""
    try:
        reference_dict = load_reference_dict()
        return reference_dict is not None and len(reference_dict) > 0
    except Exception as e:
        logger.error(f"Error checking standardization availability: {e}")
        return False

def get_standardization_stats() -> Dict[str, Any]:
    """Get statistics about the loaded standardizations."""
    try:
        reference_dict = load_reference_dict()
        if not reference_dict:
            return {
                'loaded': False,
                'count': 0,
                'unique_standardizations': 0,
                'file_exists': REFERENCE_FILE.exists()
            }
        
        unique_standardizations = len(set(reference_dict.values()))
        return {
            'loaded': True,
            'count': len(reference_dict),
            'unique_standardizations': unique_standardizations,
            'file_exists': True,
            'cache_info': _cached_standardize.cache_info()._asdict()
        }
    except Exception as e:
        logger.error(f"Error getting standardization stats: {e}")
        return {
            'loaded': False,
            'count': 0,
            'unique_standardizations': 0,
            'file_exists': REFERENCE_FILE.exists(),
            'error': str(e)
        }

def reload_reference() -> None:
    """Force reload of the reference dictionary."""
    global _reference_loaded
    with _reference_lock:
        _reference_loaded = False
        _cached_standardize.cache_clear()
        logger.info("Reference dictionary cache cleared and will be reloaded on next use.")

def clear_cache() -> None:
    """Clear the standardization cache to free memory."""
    _cached_standardize.cache_clear()
    logger.debug("Title standardization cache cleared")

def get_cache_info() -> Dict[str, Any]:
    """Get information about the current cache state."""
    try:
        cache_info = _cached_standardize.cache_info()
        return {
            'hits': cache_info.hits,
            'misses': cache_info.misses,
            'maxsize': cache_info.maxsize,
            'currsize': cache_info.currsize,
            'hit_rate': cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
        }
    except Exception as e:
        logger.error(f"Error getting cache info: {e}")
        return {'error': str(e)}

# Validation function for external use
def validate_reference_file(file_path: str = REFERENCE_FILE) -> Dict[str, Any]:
    """Validate the reference file and return detailed information."""
    validation_result = {
        'valid': False,
        'exists': False,
        'readable': False,
        'has_required_columns': False,
        'row_count': 0,
        'valid_rows': 0,
        'issues': []
    }
    
    try:
        # Check if file exists
        file_path_obj = Path(file_path) if isinstance(file_path, str) else file_path
        validation_result['exists'] = file_path_obj.exists()
        if not validation_result['exists']:
            validation_result['issues'].append(f"File does not exist: {file_path}")
            return validation_result
        
        # Try to read the file
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            validation_result['readable'] = True
            validation_result['row_count'] = len(df)
        except Exception as e:
            validation_result['issues'].append(f"File is not readable: {e}")
            return validation_result
        
        # Check for required columns
        columns_lower = [col.lower() for col in df.columns]
        required_columns = {'reference', 'standardization'}
        validation_result['has_required_columns'] = required_columns.issubset(set(columns_lower))
        
        if not validation_result['has_required_columns']:
            missing = required_columns - set(columns_lower)
            validation_result['issues'].append(f"Missing required columns: {missing}")
            return validation_result
        
        # Count valid rows
        ref_col = next(col for col in df.columns if col.lower() == 'reference')
        std_col = next(col for col in df.columns if col.lower() == 'standardization')
        valid_rows = df[[ref_col, std_col]].dropna()
        validation_result['valid_rows'] = len(valid_rows)
        
        if validation_result['valid_rows'] == 0:
            validation_result['issues'].append("No valid rows found (all rows have missing values)")
        elif validation_result['valid_rows'] < validation_result['row_count']:
            dropped = validation_result['row_count'] - validation_result['valid_rows']
            validation_result['issues'].append(f"{dropped} rows have missing values and will be ignored")
        
        # Check for duplicates
        duplicates = valid_rows[ref_col].str.lower().str.strip().duplicated().sum()
        if duplicates > 0:
            validation_result['issues'].append(f"{duplicates} duplicate reference titles found")
        
        validation_result['valid'] = validation_result['valid_rows'] > 0
        
    except Exception as e:
        validation_result['issues'].append(f"Unexpected error during validation: {e}")
    
    return validation_result
