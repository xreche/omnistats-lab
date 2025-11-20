"""Customer Acquisition Cost (CAC) calculation module."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Union
import logging

from src.utils.exceptions import DataValidationError
from src.data.validators import validate_required_columns, validate_numeric_column

logger = logging.getLogger("omnistats")


def calculate_cac(
    marketing_spend: pd.DataFrame,
    new_customers: pd.DataFrame,
    date_col: str = 'date',
    spend_col: str = 'spend',
    customer_id_col: str = 'customer_id',
    channel_col: Optional[str] = None,
    group_by: Optional[str] = None
) -> Union[float, pd.DataFrame]:
    """
    Calculate Customer Acquisition Cost (CAC).
    
    CAC can be calculated as:
    - Blended CAC: Total marketing spend / Total new customers
    - CAC by Channel: Marketing spend per channel / New customers per channel
    
    Args:
        marketing_spend: DataFrame with marketing spend data.
                        Must contain date_col and spend_col.
        new_customers: DataFrame with new customer acquisitions.
                      Must contain date_col and customer_id_col.
        date_col: Name of date column in both DataFrames
        spend_col: Name of marketing spend column
        customer_id_col: Name of customer ID column
        channel_col: Optional channel column for channel-level CAC
        group_by: Optional column to group by (e.g., 'month', 'channel')
    
    Returns:
        If group_by is None: float (blended CAC)
        If group_by is specified: DataFrame with CAC by group
        
    Raises:
        DataValidationError: If required columns are missing or data is invalid
    """
    try:
        # Validate marketing spend data
        validate_required_columns(
            marketing_spend,
            [date_col, spend_col],
            "marketing_spend"
        )
        validate_numeric_column(marketing_spend, spend_col, min_value=0)
        
        # Validate new customers data
        validate_required_columns(
            new_customers,
            [date_col, customer_id_col],
            "new_customers"
        )
        
        # Convert date columns
        marketing_spend[date_col] = pd.to_datetime(marketing_spend[date_col])
        new_customers[date_col] = pd.to_datetime(new_customers[date_col])
        
        # Prepare grouping columns
        group_cols = []
        if group_by:
            group_cols.append(group_by)
        if channel_col:
            group_cols.append(channel_col)
            validate_required_columns(marketing_spend, [channel_col], "marketing_spend")
            validate_required_columns(new_customers, [channel_col], "new_customers")
        
        if not group_cols:
            # Calculate blended CAC
            total_spend = marketing_spend[spend_col].sum()
            n_new_customers = new_customers[customer_id_col].nunique()
            
            if n_new_customers == 0:
                logger.warning("No new customers found. Returning NaN.")
                return np.nan
            
            cac = total_spend / n_new_customers
            logger.info(f"Blended CAC: ${cac:.2f} (Spend: ${total_spend:,.2f}, Customers: {n_new_customers})")
            return cac
        
        else:
            # Calculate CAC by group
            # Aggregate spend
            spend_agg = marketing_spend.groupby(group_cols)[spend_col].sum().reset_index()
            spend_agg.columns = group_cols + ['total_spend']
            
            # Count new customers
            customers_agg = new_customers.groupby(group_cols)[customer_id_col].nunique().reset_index()
            customers_agg.columns = group_cols + ['n_customers']
            
            # Merge and calculate CAC
            cac_df = spend_agg.merge(customers_agg, on=group_cols, how='outer')
            cac_df['cac'] = cac_df['total_spend'] / cac_df['n_customers'].replace(0, np.nan)
            
            # Sort by date if grouping by time
            if group_by == 'month' or date_col in group_cols:
                cac_df = cac_df.sort_values(group_cols[0])
            
            logger.info(f"Calculated CAC for {len(cac_df)} groups")
            return cac_df
            
    except Exception as e:
        error_msg = f"Error calculating CAC: {str(e)}"
        logger.error(error_msg)
        raise DataValidationError(error_msg) from e


def calculate_cac_by_channel(
    marketing_spend: pd.DataFrame,
    new_customers: pd.DataFrame,
    channel_col: str = 'channel',
    date_col: str = 'date',
    spend_col: str = 'spend',
    customer_id_col: str = 'customer_id'
) -> pd.DataFrame:
    """
    Calculate CAC by marketing channel.
    
    Args:
        marketing_spend: DataFrame with marketing spend by channel
        new_customers: DataFrame with new customer acquisitions by channel
        channel_col: Name of channel column
        date_col: Name of date column
        spend_col: Name of spend column
        customer_id_col: Name of customer ID column
    
    Returns:
        DataFrame with CAC by channel
    """
    return calculate_cac(
        marketing_spend=marketing_spend,
        new_customers=new_customers,
        date_col=date_col,
        spend_col=spend_col,
        customer_id_col=customer_id_col,
        channel_col=channel_col,
        group_by=channel_col
    )

