"""Main script to run Customer Analytics pipeline."""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from src.data.loaders import load_olist_data, create_customer_order_aggregate
from src.models.customer_analytics import (
    calculate_cac,
    calculate_ltv,
    calculate_churn_rate,
    calculate_rfm_segments
)

logger = setup_logging()


def main():
    """Run complete Customer Analytics pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Customer Analytics Pipeline")
    logger.info("=" * 60)
    
    # Configuration
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Load data
        logger.info("\n[1/4] Loading data...")
        data = load_olist_data(
            data_dir,
            datasets=['orders', 'order_items', 'order_payments', 'customers']
        )
        
        # Create aggregated customer-order dataset
        customer_orders = create_customer_order_aggregate(
            data['orders'],
            data['order_items'],
            data['order_payments']
        )
        
        # Merge with customer data to get customer_unique_id
        customer_orders = customer_orders.merge(
            data['customers'][['customer_id', 'customer_unique_id']],
            on='customer_id',
            how='left'
        )
        
        # Use customer_unique_id for analysis (to identify repeat customers)
        customer_orders['customer_id'] = customer_orders['customer_unique_id']
        customer_orders = customer_orders.dropna(subset=['customer_id', 'order_purchase_timestamp'])
        
        logger.info(f"Loaded {len(customer_orders)} orders from {customer_orders['customer_id'].nunique()} unique customers")
        
        # 2. Calculate RFM Segments
        logger.info("\n[2/4] Calculating RFM segments...")
        rfm_df, rfm_metrics = calculate_rfm_segments(
            customer_orders,
            customer_id_col='customer_id',
            date_col='order_purchase_timestamp',
            value_col='order_value'
        )
        
        # Save RFM results
        rfm_output = output_dir / "reports" / "rfm_segments.csv"
        rfm_output.parent.mkdir(parents=True, exist_ok=True)
        rfm_df.to_csv(rfm_output, index=False)
        logger.info(f"Saved RFM segments to {rfm_output}")
        
        # 3. Calculate LTV
        logger.info("\n[3/4] Calculating Lifetime Value...")
        ltv_df, ltv_metrics = calculate_ltv(
            customer_orders,
            customer_id_col='customer_id',
            date_col='order_purchase_timestamp',
            value_col='order_value',
            prediction_period_days=365
        )
        
        # Save LTV results
        ltv_output = output_dir / "reports" / "ltv_predictions.csv"
        ltv_df.to_csv(ltv_output, index=False)
        logger.info(f"Saved LTV predictions to {ltv_output}")
        
        # 4. Calculate Churn Rate
        logger.info("\n[4/4] Calculating Churn Rate...")
        churn_df, churn_metrics = calculate_churn_rate(
            customer_orders,
            customer_id_col='customer_id',
            date_col='order_purchase_timestamp',
            inactivity_window_days=90
        )
        
        # Save churn results
        churn_output = output_dir / "reports" / "churn_analysis.csv"
        churn_df.to_csv(churn_output, index=False)
        logger.info(f"Saved churn analysis to {churn_output}")
        
        # 5. Summary Report
        logger.info("\n" + "=" * 60)
        logger.info("CUSTOMER ANALYTICS SUMMARY")
        logger.info("=" * 60)
        
        logger.info(f"\nRFM Segmentation:")
        logger.info(f"  Total customers: {rfm_metrics['total_customers']}")
        logger.info(f"  Top segments: {list(rfm_metrics['segment_distribution'].items())[:5]}")
        
        logger.info(f"\nLifetime Value:")
        logger.info(f"  Average LTV: ${ltv_metrics['avg_ltv']:.2f}")
        logger.info(f"  Median LTV: ${ltv_metrics['median_ltv']:.2f}")
        logger.info(f"  Total LTV: ${ltv_metrics['total_ltv']:,.2f}")
        
        logger.info(f"\nChurn Analysis:")
        logger.info(f"  Total customers: {churn_metrics['total_customers']}")
        logger.info(f"  Churned customers: {churn_metrics['churned_customers']}")
        logger.info(f"  Churn rate: {churn_metrics['churn_rate']:.2%}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

