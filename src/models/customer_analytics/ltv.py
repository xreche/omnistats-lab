"""Lifetime Value (LTV) calculation using BG/NBD and Gamma-Gamma models."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging

from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

from src.utils.exceptions import DataValidationError, ModelTrainingError
from src.data.validators import validate_required_columns, validate_date_column

logger = logging.getLogger("omnistats")


def prepare_transaction_data(
    orders_df: pd.DataFrame,
    customer_id_col: str = 'customer_id',
    date_col: str = 'order_purchase_timestamp',
    value_col: str = 'order_value'
) -> pd.DataFrame:
    """
    Prepare transaction data for LTV modeling.
    
    Args:
        orders_df: DataFrame with customer orders
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        value_col: Name of order value column
    
    Returns:
        DataFrame formatted for lifetimes library
    """
    try:
        validate_required_columns(
            orders_df,
            [customer_id_col, date_col, value_col],
            "orders"
        )
        
        # Convert date column
        orders_df[date_col] = pd.to_datetime(orders_df[date_col], errors='coerce')
        
        # Remove rows with invalid dates or values
        orders_df = orders_df.dropna(subset=[date_col, value_col])
        
        # Prepare data for lifetimes
        summary = summary_data_from_transaction_data(
            transactions=orders_df,
            customer_id_col=customer_id_col,
            datetime_col=date_col,
            monetary_value_col=value_col,
            observation_period_end=orders_df[date_col].max()
        )
        
        logger.info(f"Prepared transaction data for {len(summary)} customers")
        return summary
        
    except Exception as e:
        error_msg = f"Error preparing transaction data: {str(e)}"
        logger.error(error_msg)
        raise DataValidationError(error_msg) from e


def calculate_ltv(
    orders_df: pd.DataFrame,
    customer_id_col: str = 'customer_id',
    date_col: str = 'order_purchase_timestamp',
    value_col: str = 'order_value',
    prediction_period_days: int = 365,
    discount_rate: float = 0.1,
    frequency_weight: float = 4.0,
    recency_weight: float = 4.0,
    monetary_weight: float = 1.0
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate Customer Lifetime Value using BG/NBD and Gamma-Gamma models.
    
    Uses the lifetimes library to:
    1. Fit BG/NBD model to predict future purchase frequency
    2. Fit Gamma-Gamma model to predict average order value
    3. Combine predictions to estimate LTV
    
    Args:
        orders_df: DataFrame with customer orders
        customer_id_col: Name of customer ID column
        date_col: Name of date column
        value_col: Name of order value column
        prediction_period_days: Number of days to predict LTV for (default: 365)
        discount_rate: Discount rate for future cash flows (default: 0.1 = 10%)
        frequency_weight: Weight for frequency parameter in BG/NBD (default: 4.0)
        recency_weight: Weight for recency parameter in BG/NBD (default: 4.0)
        monetary_weight: Weight for monetary parameter in Gamma-Gamma (default: 1.0)
    
    Returns:
        Tuple of:
        - DataFrame with LTV predictions per customer
        - Dictionary with model parameters and metrics
    
    Raises:
        DataValidationError: If data is invalid
        ModelTrainingError: If model training fails
    """
    try:
        # Prepare transaction data
        summary = prepare_transaction_data(
            orders_df,
            customer_id_col,
            date_col,
            value_col
        )
        
        # Filter customers with at least one purchase
        summary = summary[summary['frequency'] >= 0]
        
        if len(summary) == 0:
            raise DataValidationError("No valid transaction data found")
        
        logger.info(f"Training LTV models on {len(summary)} customers")
        
        # Fit BG/NBD model for frequency prediction
        bgf = BetaGeoFitter(penalizer_coef=0.0)
        bgf.fit(
            frequency=summary['frequency'],
            recency=summary['recency'],
            T=summary['T']
        )
        
        # Predict number of transactions in prediction period
        summary['predicted_transactions'] = bgf.conditional_expected_number_of_purchases_up_to_time(
            t=prediction_period_days,
            frequency=summary['frequency'],
            recency=summary['recency'],
            T=summary['T']
        )
        
        # Fit Gamma-Gamma model for monetary value prediction
        # Only fit on customers with repeat purchases
        repeat_customers = summary[summary['frequency'] > 0]
        
        if len(repeat_customers) == 0:
            logger.warning("No repeat customers found. Using average order value for all.")
            avg_order_value = summary['monetary_value'].mean()
            summary['predicted_avg_order_value'] = avg_order_value
        else:
            ggf = GammaGammaFitter(penalizer_coef=0.0)
            ggf.fit(
                frequency=repeat_customers['frequency'],
                monetary_value=repeat_customers['monetary_value']
            )
            
            # Predict average order value for all customers
            summary['predicted_avg_order_value'] = ggf.conditional_expected_average_profit(
                summary['frequency'],
                summary['monetary_value']
            )
        
        # Calculate LTV: predicted transactions * predicted avg order value
        # Apply discount rate for future cash flows
        summary['ltv'] = (
            summary['predicted_transactions'] *
            summary['predicted_avg_order_value'] *
            (1 / (1 + discount_rate))  # Simple discount
        )
        
        # Add additional metrics
        summary['ltv_undiscounted'] = (
            summary['predicted_transactions'] *
            summary['predicted_avg_order_value']
        )
        
        # Model metrics
        model_metrics = {
            'bgf_alpha': bgf.params_['alpha'],
            'bgf_r': bgf.params_['r'],
            'bgf_a': bgf.params_['a'],
            'bgf_b': bgf.params_['b'],
            'n_customers': len(summary),
            'n_repeat_customers': len(repeat_customers),
            'avg_ltv': summary['ltv'].mean(),
            'median_ltv': summary['ltv'].median(),
            'total_ltv': summary['ltv'].sum()
        }
        
        if len(repeat_customers) > 0:
            model_metrics['ggf_p'] = ggf.params_['p']
            model_metrics['ggf_q'] = ggf.params_['q']
            model_metrics['ggf_v'] = ggf.params_['v']
        
        logger.info(
            f"LTV calculation complete. Avg LTV: ${model_metrics['avg_ltv']:.2f}, "
            f"Total LTV: ${model_metrics['total_ltv']:,.2f}"
        )
        
        return summary, model_metrics
        
    except DataValidationError:
        raise
    except Exception as e:
        error_msg = f"Error calculating LTV: {str(e)}"
        logger.error(error_msg)
        raise ModelTrainingError(error_msg) from e

