"""Main script to run Marketing Science pipeline."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from src.models.marketing_science import (
    run_mmm_analysis,
    optimize_channel_budget,
    plot_mmm_results,
    calculate_price_elasticity,
    calculate_markov_attribution
)

logger = setup_logging()


def generate_synthetic_mmm_data(n_periods: int = 104) -> pd.DataFrame:
    """
    Generate synthetic Marketing Mix Modeling data for demonstration.
    
    Args:
        n_periods: Number of time periods (default: 104 = 2 years weekly)
    
    Returns:
        DataFrame with synthetic MMM data
    """
    np.random.seed(42)
    
    dates = pd.date_range(start='2022-01-01', periods=n_periods, freq='W')
    
    # Generate media spend (with some seasonality)
    tv_spend = 10000 + 5000 * np.sin(np.arange(n_periods) * 2 * np.pi / 52) + np.random.normal(0, 2000, n_periods)
    radio_spend = 5000 + 2000 * np.sin(np.arange(n_periods) * 2 * np.pi / 52 + np.pi/4) + np.random.normal(0, 1000, n_periods)
    digital_spend = 8000 + 3000 * np.sin(np.arange(n_periods) * 2 * np.pi / 26) + np.random.normal(0, 1500, n_periods)
    
    # Generate sales (with media effects + noise)
    base_sales = 50000
    tv_effect = 0.3 * tv_spend
    radio_effect = 0.2 * radio_spend
    digital_effect = 0.4 * digital_spend
    seasonality = 10000 * np.sin(np.arange(n_periods) * 2 * np.pi / 52)
    noise = np.random.normal(0, 5000, n_periods)
    
    sales = base_sales + tv_effect + radio_effect + digital_effect + seasonality + noise
    sales = np.maximum(sales, 0)  # Ensure non-negative
    
    # Generate price and promotion data
    price = 10 + np.random.normal(0, 1, n_periods)
    promotion = np.random.binomial(1, 0.2, n_periods)  # 20% promotion weeks
    
    df = pd.DataFrame({
        'date': dates,
        'sales': sales,
        'tv_spend': tv_spend,
        'radio_spend': radio_spend,
        'digital_spend': digital_spend,
        'price': price,
        'promotion': promotion,
        'competitor_price': price * 0.95 + np.random.normal(0, 0.5, n_periods)
    })
    
    return df


def main():
    """Run complete Marketing Science pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Marketing Science Pipeline")
    logger.info("=" * 60)
    
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Generate or load MMM data
        logger.info("\n[1/3] Preparing Marketing Mix Modeling data...")
        
        # Try to load real data, otherwise generate synthetic
        mmm_data_path = project_root / "data" / "raw" / "Marketing mix" / "mktmix.csv"
        
        if mmm_data_path.exists():
            logger.info(f"Loading MMM data from {mmm_data_path}")
            mmm_data = pd.read_csv(mmm_data_path)
            
            # Map columns to standard names
            column_mapping = {
                'NewVolSales': 'sales',
                'Base_Price': 'price',
                'TV': 'tv_spend',
                'Radio ': 'radio_spend',  # Note: column has trailing space
                'Radio': 'radio_spend',
                'Discount': 'promotion',
                'InStore': 'instore_spend',
                'Stout': 'stout_spend'
            }
            
            # Rename columns that exist
            for old_col, new_col in column_mapping.items():
                if old_col in mmm_data.columns:
                    mmm_data = mmm_data.rename(columns={old_col: new_col})
            
            # Clean column names (remove trailing spaces)
            mmm_data.columns = mmm_data.columns.str.strip()
            
            # Create quantity column from sales and price if both exist
            if 'sales' in mmm_data.columns and 'price' in mmm_data.columns:
                mmm_data['quantity'] = mmm_data['sales'] / mmm_data['price']
            elif 'sales' in mmm_data.columns:
                # Use sales as proxy for quantity if no price
                mmm_data['quantity'] = mmm_data['sales']
                logger.warning("No price column found. Using sales as proxy for quantity.")
            
            # Create date column if missing
            if 'date' not in mmm_data.columns:
                # Create weekly dates starting from a reference date
                n_periods = len(mmm_data)
                mmm_data['date'] = pd.date_range(start='2020-01-01', periods=n_periods, freq='W')
                logger.info(f"Created date column with {n_periods} weekly periods")
            
            logger.info(f"Loaded dataset with columns: {list(mmm_data.columns)}")
        else:
            logger.info("Real MMM data not found. Generating synthetic data...")
            mmm_data = generate_synthetic_mmm_data()
        
        logger.info(f"MMM data prepared: {len(mmm_data)} periods")
        
        # 2. Marketing Mix Modeling
        logger.info("\n[2/3] Running Marketing Mix Modeling...")
        
        # Identify available media channels
        available_media = []
        potential_channels = ['tv_spend', 'radio_spend', 'digital_spend', 'instore_spend', 'stout_spend']
        for ch in potential_channels:
            if ch in mmm_data.columns:
                available_media.append(ch)
        
        if not available_media:
            logger.warning("No media channels found. Skipping MMM analysis.")
        else:
            try:
                # Run MMM analysis with pymc-marketing
                # PARÁMETROS MÍNIMOS para pruebas rápidas
                logger.info("⚠️  Usando parámetros MÍNIMOS para pruebas rápidas")
                logger.info("   Para producción, aumentar: draws=1000, tune=1000, chains=2")
                mmm_results = run_mmm_analysis(
                    df=mmm_data,
                    target_col='sales',
                    media_channels=available_media,
                    control_vars=['price'] if 'price' in mmm_data.columns else None,
                    date_col='date' if 'date' in mmm_data.columns else None,
                    adstock_max_lag=4,  # Mínimo para pruebas rápidas
                    yearly_seasonality=1,  # Mínimo para pruebas rápidas
                    draws=50,  # MÍNIMO para pruebas rápidas (default producción: 1000)
                    chains=1,  # MÍNIMO para pruebas rápidas (default producción: 2)
                    tune=50,  # MÍNIMO para pruebas rápidas (default producción: 1000)
                    random_seed=42
                )
                
                # Save MMM results
                mmm_output = output_dir / "reports" / "mmm_results.txt"
                mmm_output.parent.mkdir(parents=True, exist_ok=True)
                
                with open(mmm_output, 'w', encoding='utf-8') as f:
                    f.write("MARKETING MIX MODELING RESULTS (PyMC-Marketing)\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("Powered by PyMC-Marketing (Bayesian Inference v5)\n\n")
                    f.write("Media Effectiveness:\n")
                    for channel, metrics in mmm_results['media_effectiveness'].items():
                        f.write(f"\n{channel}:\n")
                        if isinstance(metrics.get('mean'), (int, float)):
                            f.write(f"  Mean: {metrics['mean']:.4f}\n")
                            f.write(f"  Median: {metrics['median']:.4f}\n")
                            f.write(f"  Std: {metrics['std']:.4f}\n")
                            f.write(f"  95% HDI: [{metrics['hdi_3%']:.4f}, {metrics['hdi_97%']:.4f}]\n")
                        else:
                            f.write(f"  {metrics.get('note', 'N/A')}\n")
                    
                    f.write(f"\nModel Summary:\n")
                    f.write(f"  Samples: {mmm_results['summary_stats']['n_samples']}\n")
                    f.write(f"  Media Channels: {mmm_results['summary_stats']['n_media_channels']}\n")
                    f.write(f"  Draws: {mmm_results['summary_stats']['draws']}\n")
                    f.write(f"  Chains: {mmm_results['summary_stats']['chains']}\n")
                
                logger.info(f"Saved MMM results to {mmm_output}")
                
                # Optional: Generate visualizations
                try:
                    plot_output = output_dir / "visualizations" / "mmm"
                    plot_output.parent.mkdir(parents=True, exist_ok=True)
                    plot_mmm_results(
                        mmm_results['model'],
                        output_path=str(plot_output)
                    )
                    logger.info(f"Saved MMM visualizations to {plot_output}")
                except Exception as plot_error:
                    logger.warning(f"Could not generate visualizations: {plot_error}")
                
            except Exception as e:
                logger.error(f"MMM analysis failed: {str(e)}")
                logger.info("Skipping MMM analysis. Check that pymc-marketing is installed correctly.")
        
        # 3. Price Elasticity
        logger.info("\n[3/3] Calculating Price Elasticity...")
        
        # Check if we have required columns for price elasticity
        if 'quantity' not in mmm_data.columns or 'price' not in mmm_data.columns:
            logger.warning(
                "Missing required columns for price elasticity analysis. "
                "Need 'quantity' and 'price' columns."
            )
            logger.info("Skipping price elasticity analysis.")
            elasticity_results = None
        else:
            elasticity_results = calculate_price_elasticity(
                df=mmm_data,
                quantity_col='quantity',
                price_col='price',
                promotion_col='promotion' if 'promotion' in mmm_data.columns else None,
                control_vars=None  # Can add control vars if available
            )
            
            # Save elasticity results
            elasticity_output = output_dir / "reports" / "price_elasticity_results.txt"
            elasticity_output.parent.mkdir(parents=True, exist_ok=True)
            with open(elasticity_output, 'w') as f:
                f.write("PRICE ELASTICITY ANALYSIS\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Price Elasticity: {elasticity_results['elasticity']['coefficient']:.4f}\n")
                f.write(f"Std Error: {elasticity_results['elasticity']['std_error']:.4f}\n")
                f.write(f"P-value: {elasticity_results['elasticity']['p_value']:.4f}\n")
                f.write(f"\nInterpretation:\n{elasticity_results['elasticity']['interpretation']}\n")
                
                if elasticity_results['promotion_lift']:
                    f.write(f"\nPromotion Lift:\n")
                    f.write(f"  Coefficient: {elasticity_results['promotion_lift']['coefficient']:.4f}\n")
                    f.write(f"  {elasticity_results['promotion_lift']['interpretation']}\n")
                
                f.write(f"\nModel Metrics:\n")
                f.write(f"  R²: {elasticity_results['model_metrics']['r_squared']:.4f}\n")
                f.write(f"  Adj R²: {elasticity_results['model_metrics']['adj_r_squared']:.4f}\n")
        
            logger.info(f"Saved price elasticity results to {elasticity_output}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("MARKETING SCIENCE SUMMARY")
        logger.info("=" * 60)
        
        if elasticity_results:
            logger.info(f"\nPrice Elasticity:")
            logger.info(f"  Coefficient: {elasticity_results['elasticity']['coefficient']:.4f}")
            logger.info(f"  Interpretation: {elasticity_results['elasticity']['interpretation']}")
            logger.info(f"  R²: {elasticity_results['model_metrics']['r_squared']:.4f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

