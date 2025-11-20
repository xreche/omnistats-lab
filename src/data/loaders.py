"""Data loading functions for Brazilian E-commerce dataset."""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import logging

from src.utils.exceptions import DataLoadingError

logger = logging.getLogger("omnistats")


def load_olist_data(
    data_dir: Path,
    datasets: Optional[list[str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load Brazilian E-commerce (Olist) datasets.
    
    Args:
        data_dir: Path to data directory containing 'Brazilian E-commerce' folder
        datasets: List of dataset names to load. If None, loads all available.
                 Options: 'customers', 'orders', 'order_items', 'order_payments',
                         'order_reviews', 'products', 'sellers', 'geolocation'
    
    Returns:
        Dictionary mapping dataset names to DataFrames
        
    Raises:
        DataLoadingError: If data files cannot be loaded
    """
    base_path = Path(data_dir) / "Brazilian E-commerce"
    
    if not base_path.exists():
        raise DataLoadingError(f"Data directory not found: {base_path}")
    
    # Mapping of dataset names to file names
    dataset_files = {
        'customers': 'olist_customers_dataset.csv',
        'orders': 'olist_orders_dataset.csv',
        'order_items': 'olist_order_items_dataset.csv',
        'order_payments': 'olist_order_payments_dataset.csv',
        'order_reviews': 'olist_order_reviews_dataset.csv',
        'products': 'olist_products_dataset.csv',
        'sellers': 'olist_sellers_dataset.csv',
        'geolocation': 'olist_geolocation_dataset.csv',
    }
    
    # If no specific datasets requested, load all
    if datasets is None:
        datasets = list(dataset_files.keys())
    
    data = {}
    
    for dataset_name in datasets:
        if dataset_name not in dataset_files:
            logger.warning(f"Unknown dataset: {dataset_name}. Skipping.")
            continue
        
        file_path = base_path / dataset_files[dataset_name]
        
        try:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}. Skipping.")
                continue
            
            logger.info(f"Loading {dataset_name} from {file_path}")
            df = pd.read_csv(file_path)
            data[dataset_name] = df
            logger.info(f"Loaded {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            error_msg = f"Error loading {dataset_name} from {file_path}: {str(e)}"
            logger.error(error_msg)
            raise DataLoadingError(error_msg) from e
    
    return data


def create_customer_order_aggregate(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    order_payments_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create aggregated customer-order dataset for analytics.
    
    Args:
        orders_df: Orders dataset
        order_items_df: Order items dataset
        order_payments_df: Order payments dataset
        
    Returns:
        Aggregated DataFrame with customer_id, order_id, order_date, 
        total_value, payment_value, etc.
    """
    try:
        # Aggregate order items by order_id
        order_totals = order_items_df.groupby('order_id').agg({
            'price': 'sum',
            'freight_value': 'sum'
        }).reset_index()
        order_totals['order_value'] = order_totals['price'] + order_totals['freight_value']
        
        # Aggregate payments by order_id
        payment_totals = order_payments_df.groupby('order_id')['payment_value'].sum().reset_index()
        
        # Merge with orders
        customer_orders = orders_df.merge(
            order_totals[['order_id', 'order_value']],
            on='order_id',
            how='left'
        ).merge(
            payment_totals,
            on='order_id',
            how='left'
        )
        
        # Convert date columns
        date_cols = [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_customer_date'
        ]
        for col in date_cols:
            if col in customer_orders.columns:
                customer_orders[col] = pd.to_datetime(customer_orders[col], errors='coerce')
        
        return customer_orders
        
    except Exception as e:
        logger.error(f"Error creating customer order aggregate: {str(e)}")
        raise DataLoadingError(f"Failed to aggregate customer orders: {str(e)}") from e

