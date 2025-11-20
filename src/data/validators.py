"""Data validation functions."""

import pandas as pd
from typing import List, Optional
import logging

from src.utils.exceptions import DataValidationError

logger = logging.getLogger("omnistats")


def validate_required_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    dataset_name: str = "dataset"
) -> None:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        dataset_name: Name of dataset for error messages
        
    Raises:
        DataValidationError: If required columns are missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        error_msg = (
            f"{dataset_name} is missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
        logger.error(error_msg)
        raise DataValidationError(error_msg)


def validate_date_column(
    df: pd.DataFrame,
    date_col: str,
    dataset_name: str = "dataset"
) -> None:
    """
    Validate that a column contains valid dates.
    
    Args:
        df: DataFrame to validate
        date_col: Name of date column
        dataset_name: Name of dataset for error messages
        
    Raises:
        DataValidationError: If date column is invalid
    """
    if date_col not in df.columns:
        error_msg = f"{dataset_name} missing date column: {date_col}"
        logger.error(error_msg)
        raise DataValidationError(error_msg)
    
    if df[date_col].isna().all():
        error_msg = f"{dataset_name} date column {date_col} contains only null values"
        logger.error(error_msg)
        raise DataValidationError(error_msg)


def validate_numeric_column(
    df: pd.DataFrame,
    numeric_col: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    dataset_name: str = "dataset"
) -> None:
    """
    Validate that a column contains valid numeric values.
    
    Args:
        df: DataFrame to validate
        numeric_col: Name of numeric column
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        dataset_name: Name of dataset for error messages
        
    Raises:
        DataValidationError: If numeric column is invalid
    """
    if numeric_col not in df.columns:
        error_msg = f"{dataset_name} missing numeric column: {numeric_col}"
        logger.error(error_msg)
        raise DataValidationError(error_msg)
    
    # Check for non-numeric values
    non_numeric = pd.to_numeric(df[numeric_col], errors='coerce').isna()
    if non_numeric.any():
        n_invalid = non_numeric.sum()
        error_msg = (
            f"{dataset_name} column {numeric_col} contains {n_invalid} "
            f"non-numeric values"
        )
        logger.warning(error_msg)
    
    # Check min/max bounds
    numeric_series = pd.to_numeric(df[numeric_col], errors='coerce')
    
    if min_value is not None and (numeric_series < min_value).any():
        n_below = (numeric_series < min_value).sum()
        logger.warning(
            f"{dataset_name} column {numeric_col} has {n_below} values "
            f"below minimum {min_value}"
        )
    
    if max_value is not None and (numeric_series > max_value).any():
        n_above = (numeric_series > max_value).sum()
        logger.warning(
            f"{dataset_name} column {numeric_col} has {n_above} values "
            f"above maximum {max_value}"
        )

