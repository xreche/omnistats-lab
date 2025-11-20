"""Churn rate calculation for non-subscription businesses."""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from datetime import datetime, timedelta
import logging

from src.utils.exceptions import DataValidationError
from src.data.validators import (
    validate_required_columns,
    validate_date_column,
    validate_numeric_column
)

logger = logging.getLogger("omnistats")


def calculate_churn_rate(
    orders_df: pd.DataFrame,
    customer_id_col: str = 'customer_id',
    date_col: str = 'order_purchase_timestamp',
    inactivity_window_days: int = 90,
    observation_date: Optional[datetime] = None,
    cohort_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate churn rate for non-subscription business.
    
    Churn is defined as customers who haven't made a purchase within
    the inactivity window (default: 90 days) from the observation date.
    
    Args:
        orders_df: DataFrame with customer orders
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        inactivity_window_days: Days of inactivity to consider churned (default: 90)
        observation_date: Date to calculate churn from. If None, uses max date in data
        cohort_col: Optional column to calculate churn by cohort (e.g., acquisition month)
    
    Returns:
        Tuple of:
        - DataFrame with churn status per customer
        - Dictionary with churn metrics
    
    Raises:
        DataValidationError: If data is invalid
    """
    try:
        validate_required_columns(
            orders_df,
            [customer_id_col, date_col],
            "orders"
        )
        
        # Convert date column
        orders_df[date_col] = pd.to_datetime(orders_df[date_col], errors='coerce')
        orders_df = orders_df.dropna(subset=[date_col])
        
        if len(orders_df) == 0:
            raise DataValidationError("No valid order data found")
        
        # Set observation date
        if observation_date is None:
            observation_date = orders_df[date_col].max()
        else:
            observation_date = pd.to_datetime(observation_date)
        
        logger.info(
            f"Calculating churn rate with {inactivity_window_days}-day inactivity window. "
            f"Observation date: {observation_date.date()}"
        )
        
        # Get last purchase date per customer
        customer_last_purchase = orders_df.groupby(customer_id_col)[date_col].max().reset_index()
        customer_last_purchase.columns = [customer_id_col, 'last_purchase_date']
        
        # Calculate days since last purchase
        customer_last_purchase['days_since_last_purchase'] = (
            observation_date - customer_last_purchase['last_purchase_date']
        ).dt.days
        
        # Determine churn status
        customer_last_purchase['is_churned'] = (
            customer_last_purchase['days_since_last_purchase'] > inactivity_window_days
        )
        
        # Add cohort if specified
        if cohort_col:
            if cohort_col not in orders_df.columns:
                logger.warning(f"Cohort column {cohort_col} not found. Ignoring cohort analysis.")
                cohort_col = None
            else:
                # Get customer's first purchase date for cohort
                customer_first_purchase = orders_df.groupby(customer_id_col)[date_col].min().reset_index()
                customer_first_purchase.columns = [customer_id_col, 'first_purchase_date']
                customer_first_purchase['cohort'] = (
                    customer_first_purchase['first_purchase_date'].dt.to_period('M')
                )
                customer_last_purchase = customer_last_purchase.merge(
                    customer_first_purchase[[customer_id_col, 'cohort']],
                    on=customer_id_col,
                    how='left'
                )
        
        # Calculate metrics
        total_customers = len(customer_last_purchase)
        churned_customers = customer_last_purchase['is_churned'].sum()
        churn_rate = churned_customers / total_customers if total_customers > 0 else 0
        
        metrics = {
            'total_customers': total_customers,
            'churned_customers': churned_customers,
            'active_customers': total_customers - churned_customers,
            'churn_rate': churn_rate,
            'inactivity_window_days': inactivity_window_days,
            'observation_date': observation_date
        }
        
        # Cohort-level metrics if cohort specified
        if cohort_col:
            cohort_metrics = customer_last_purchase.groupby('cohort').agg({
                'is_churned': ['sum', 'count']
            }).reset_index()
            cohort_metrics.columns = ['cohort', 'churned', 'total']
            cohort_metrics['churn_rate'] = cohort_metrics['churned'] / cohort_metrics['total']
            metrics['cohort_metrics'] = cohort_metrics.to_dict('records')
        
        logger.info(
            f"Churn calculation complete. "
            f"Churn rate: {churn_rate:.2%} ({churned_customers}/{total_customers} customers)"
        )
        
        return customer_last_purchase, metrics
        
    except Exception as e:
        error_msg = f"Error calculating churn rate: {str(e)}"
        logger.error(error_msg)
        raise DataValidationError(error_msg) from e

