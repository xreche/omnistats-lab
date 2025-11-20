"""RFM (Recency, Frequency, Monetary) segmentation module."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime
import logging

from src.utils.exceptions import DataValidationError
from src.data.validators import (
    validate_required_columns,
    validate_date_column,
    validate_numeric_column
)

logger = logging.getLogger("omnistats")


def calculate_rfm_scores(
    orders_df: pd.DataFrame,
    customer_id_col: str = 'customer_id',
    date_col: str = 'order_purchase_timestamp',
    value_col: str = 'order_value',
    observation_date: Optional[datetime] = None,
    recency_bins: int = 5,
    frequency_bins: int = 5,
    monetary_bins: int = 5
) -> pd.DataFrame:
    """
    Calculate RFM scores for customer segmentation.
    
    RFM stands for:
    - Recency: Days since last purchase
    - Frequency: Number of purchases
    - Monetary: Total value of purchases
    
    Args:
        orders_df: DataFrame with customer orders
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        value_col: Name of order value column
        observation_date: Date to calculate RFM from. If None, uses max date in data
        recency_bins: Number of bins for recency score (default: 5)
        frequency_bins: Number of bins for frequency score (default: 5)
        monetary_bins: Number of bins for monetary score (default: 5)
    
    Returns:
        DataFrame with RFM scores and values per customer
    
    Raises:
        DataValidationError: If data is invalid
    """
    try:
        validate_required_columns(
            orders_df,
            [customer_id_col, date_col, value_col],
            "orders"
        )
        validate_numeric_column(orders_df, value_col, min_value=0)
        
        # Convert date column
        orders_df[date_col] = pd.to_datetime(orders_df[date_col], errors='coerce')
        orders_df = orders_df.dropna(subset=[date_col, value_col])
        
        if len(orders_df) == 0:
            raise DataValidationError("No valid order data found")
        
        # Set observation date
        if observation_date is None:
            observation_date = orders_df[date_col].max()
        else:
            observation_date = pd.to_datetime(observation_date)
        
        logger.info(f"Calculating RFM scores. Observation date: {observation_date.date()}")
        
        # Calculate Recency (days since last purchase)
        recency = orders_df.groupby(customer_id_col)[date_col].max().reset_index()
        recency['recency_days'] = (observation_date - recency[date_col]).dt.days
        recency = recency[[customer_id_col, 'recency_days']]
        
        # Calculate Frequency (number of purchases)
        frequency = orders_df.groupby(customer_id_col)[date_col].count().reset_index()
        frequency.columns = [customer_id_col, 'frequency']
        
        # Calculate Monetary (total value)
        monetary = orders_df.groupby(customer_id_col)[value_col].sum().reset_index()
        monetary.columns = [customer_id_col, 'monetary_value']
        
        # Merge RFM components
        rfm = recency.merge(frequency, on=customer_id_col).merge(monetary, on=customer_id_col)
        
        # Calculate RFM scores (1-5, where 5 is best)
        # Recency: Lower is better (more recent = higher score)
        rfm['recency_score'] = pd.qcut(
            rfm['recency_days'].rank(method='first'),
            q=recency_bins,
            labels=range(recency_bins, 0, -1),
            duplicates='drop'
        ).astype(int)
        
        # Frequency: Higher is better
        rfm['frequency_score'] = pd.qcut(
            rfm['frequency'].rank(method='first'),
            q=frequency_bins,
            labels=range(1, frequency_bins + 1),
            duplicates='drop'
        ).astype(int)
        
        # Monetary: Higher is better
        rfm['monetary_score'] = pd.qcut(
            rfm['monetary_value'].rank(method='first'),
            q=monetary_bins,
            labels=range(1, monetary_bins + 1),
            duplicates='drop'
        ).astype(int)
        
        # Calculate RFM score (concatenation of R, F, M)
        rfm['rfm_score'] = (
            rfm['recency_score'].astype(str) +
            rfm['frequency_score'].astype(str) +
            rfm['monetary_score'].astype(str)
        )
        
        logger.info(f"Calculated RFM scores for {len(rfm)} customers")
        return rfm
        
    except Exception as e:
        error_msg = f"Error calculating RFM scores: {str(e)}"
        logger.error(error_msg)
        raise DataValidationError(error_msg) from e


def assign_rfm_segments(
    rfm_df: pd.DataFrame,
    segment_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Assign customer segments based on RFM scores.
    
    Uses standard RFM segmentation rules or custom mapping.
    
    Args:
        rfm_df: DataFrame with RFM scores (from calculate_rfm_scores)
        segment_map: Optional custom mapping of RFM patterns to segment names.
                    If None, uses standard segmentation.
    
    Returns:
        DataFrame with assigned segments
    """
    if segment_map is None:
        # Standard RFM segmentation
        segment_map = {
            '555': 'Champions',
            '554': 'Champions',
            '544': 'Champions',
            '545': 'Champions',
            '454': 'Champions',
            '455': 'Champions',
            '445': 'Champions',
            '444': 'Champions',
            '543': 'Loyal Customers',
            '542': 'Loyal Customers',
            '541': 'Loyal Customers',
            '534': 'Loyal Customers',
            '535': 'Loyal Customers',
            '533': 'Loyal Customers',
            '532': 'Loyal Customers',
            '531': 'Loyal Customers',
            '525': 'Loyal Customers',
            '524': 'Loyal Customers',
            '523': 'Loyal Customers',
            '522': 'Loyal Customers',
            '521': 'Loyal Customers',
            '515': 'Loyal Customers',
            '514': 'Loyal Customers',
            '513': 'Loyal Customers',
            '512': 'Loyal Customers',
            '511': 'Loyal Customers',
            '444': 'Potential Loyalists',
            '443': 'Potential Loyalists',
            '434': 'Potential Loyalists',
            '433': 'Potential Loyalists',
            '432': 'Potential Loyalists',
            '431': 'Potential Loyalists',
            '424': 'Potential Loyalists',
            '423': 'Potential Loyalists',
            '422': 'Potential Loyalists',
            '421': 'Potential Loyalists',
            '414': 'Potential Loyalists',
            '413': 'Potential Loyalists',
            '412': 'Potential Loyalists',
            '411': 'Potential Loyalists',
            '344': 'New Customers',
            '343': 'New Customers',
            '334': 'New Customers',
            '333': 'New Customers',
            '323': 'New Customers',
            '322': 'New Customers',
            '321': 'New Customers',
            '312': 'New Customers',
            '311': 'New Customers',
            '233': 'Promising',
            '232': 'Promising',
            '231': 'Promising',
            '223': 'Promising',
            '222': 'Promising',
            '221': 'Promising',
            '212': 'Promising',
            '211': 'Promising',
            '155': 'Need Attention',
            '154': 'Need Attention',
            '144': 'Need Attention',
            '143': 'Need Attention',
            '134': 'Need Attention',
            '133': 'Need Attention',
            '124': 'Need Attention',
            '123': 'Need Attention',
            '115': 'About to Sleep',
            '114': 'About to Sleep',
            '113': 'About to Sleep',
            '112': 'About to Sleep',
            '111': 'About to Sleep',
            '55': 'Cannot Lose Them',
            '54': 'Cannot Lose Them',
            '53': 'Cannot Lose Them',
            '52': 'Cannot Lose Them',
            '51': 'Cannot Lose Them',
            '45': 'Cannot Lose Them',
            '44': 'Cannot Lose Them',
            '43': 'Cannot Lose Them',
            '42': 'Cannot Lose Them',
            '41': 'Cannot Lose Them',
            '35': 'Cannot Lose Them',
            '34': 'Cannot Lose Them',
            '33': 'Cannot Lose Them',
            '32': 'Cannot Lose Them',
            '31': 'Cannot Lose Them',
            '25': 'Cannot Lose Them',
            '24': 'Cannot Lose Them',
            '23': 'Cannot Lose Them',
            '22': 'Cannot Lose Them',
            '21': 'Cannot Lose Them',
            '15': 'Cannot Lose Them',
            '14': 'Cannot Lose Them',
            '13': 'Cannot Lose Them',
            '12': 'Cannot Lose Them',
            '11': 'Cannot Lose Them',
        }
    
    # Assign segments
    rfm_df['segment'] = rfm_df['rfm_score'].map(segment_map).fillna('Others')
    
    # Count segments
    segment_counts = rfm_df['segment'].value_counts()
    logger.info(f"Assigned segments: {dict(segment_counts)}")
    
    return rfm_df


def calculate_rfm_segments(
    orders_df: pd.DataFrame,
    customer_id_col: str = 'customer_id',
    date_col: str = 'order_purchase_timestamp',
    value_col: str = 'order_value',
    observation_date: Optional[datetime] = None,
    segment_map: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete RFM segmentation pipeline.
    
    Args:
        orders_df: DataFrame with customer orders
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        value_col: Name of order value column
        observation_date: Date to calculate RFM from
        segment_map: Optional custom segment mapping
    
    Returns:
        Tuple of:
        - DataFrame with RFM scores and segments
        - Dictionary with segmentation metrics
    """
    # Calculate RFM scores
    rfm = calculate_rfm_scores(
        orders_df,
        customer_id_col,
        date_col,
        value_col,
        observation_date
    )
    
    # Assign segments
    rfm = assign_rfm_segments(rfm, segment_map)
    
    # Calculate metrics
    metrics = {
        'total_customers': len(rfm),
        'segment_distribution': rfm['segment'].value_counts().to_dict(),
        'avg_recency_days': rfm['recency_days'].mean(),
        'avg_frequency': rfm['frequency'].mean(),
        'avg_monetary_value': rfm['monetary_value'].mean(),
        'total_monetary_value': rfm['monetary_value'].sum()
    }
    
    logger.info(f"RFM segmentation complete for {len(rfm)} customers")
    
    return rfm, metrics

