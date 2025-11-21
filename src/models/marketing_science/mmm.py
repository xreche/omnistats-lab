"""Marketing Mix Modeling (MMM) using pymc-marketing.

This module implements Bayesian Marketing Mix Modeling using PyMC-Marketing,
the modern standard for MMM maintained by PyMC Labs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

try:
    from pymc_marketing.mmm import MMM
    from pymc_marketing.mmm.components.adstock import DelayedAdstock
    from pymc_marketing.mmm.components.saturation import HillSaturation
    import arviz as az
    PYMC_MARKETING_AVAILABLE = True
except ImportError:
    PYMC_MARKETING_AVAILABLE = False
    logger = logging.getLogger("omnistats")
    logger.warning(
        "pymc-marketing not available. Install with: pip install pymc-marketing"
    )

from src.utils.exceptions import ModelTrainingError, DataValidationError
from src.data.validators import validate_required_columns, validate_numeric_column

logger = logging.getLogger("omnistats")


def prepare_mmm_data(
    df: pd.DataFrame,
    target_col: str,
    media_channels: List[str],
    control_vars: Optional[List[str]] = None,
    date_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare data for Marketing Mix Modeling with pymc-marketing.
    
    pymc-marketing expects DataFrames with explicit date columns and
    properly formatted media channels.
    
    Args:
        df: DataFrame with target, media, and control variables
        target_col: Name of target variable (e.g., sales, revenue)
        media_channels: List of media channel column names
        control_vars: Optional list of control variable column names
        date_col: Optional date column for time series ordering
    
    Returns:
        Tuple of:
        - Prepared DataFrame with date column and all required columns
        - Name of the date column used
    
    Raises:
        DataValidationError: If required columns are missing or data is invalid
    """
    try:
        # Validate required columns
        required_cols = [target_col] + media_channels
        if control_vars:
            required_cols.extend(control_vars)
        
        validate_required_columns(df, required_cols, "MMM data")
        validate_numeric_column(df, target_col, min_value=0)
        
        # Create a copy to avoid modifying original
        df_prep = df.copy()
        
        # Ensure date column exists and is datetime
        if date_col and date_col in df_prep.columns:
            df_prep[date_col] = pd.to_datetime(df_prep[date_col])
            df_prep = df_prep.sort_values(date_col).reset_index(drop=True)
        else:
            # Create a default date column if not provided
            logger.warning("No date column provided. Creating default weekly dates.")
            df_prep['date'] = pd.date_range(
                start='2020-01-01',
                periods=len(df_prep),
                freq='W'
            )
            date_col = 'date'
        
        # Ensure all media channels are numeric and non-negative
        for channel in media_channels:
            validate_numeric_column(df_prep, channel, min_value=0)
        
        logger.info(
            f"Prepared MMM data: {len(df_prep)} samples, {len(media_channels)} media channels"
        )
        
        return df_prep, date_col
        
    except Exception as e:
        error_msg = f"Error preparing MMM data: {str(e)}"
        logger.error(error_msg)
        raise DataValidationError(error_msg) from e


def run_mmm_analysis(
    df: pd.DataFrame,
    target_col: str,
    media_channels: List[str],
    control_vars: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    adstock_max_lag: int = 8,
    yearly_seasonality: int = 2,
    draws: int = 1000,
    chains: int = 2,
    tune: int = 1000,
    random_seed: Optional[int] = None
) -> Dict:
    """
    Run Marketing Mix Modeling analysis using pymc-marketing.
    
    This function implements Bayesian MMM using MMM from
    pymc-marketing, which includes:
    - Adstock transformation (with configurable max lag)
    - Saturation curves (Hill function)
    - Yearly seasonality
    - Bayesian inference with PyMC
    
    Args:
        df: DataFrame with target, media, and control variables
        target_col: Name of target variable (e.g., 'sales')
        media_channels: List of media channel column names
        control_vars: Optional list of control variable column names
        date_col: Optional date column name (will be created if not provided)
        adstock_max_lag: Maximum lag for adstock transformation (default: 8)
        yearly_seasonality: Number of Fourier terms for yearly seasonality (default: 2)
        draws: Number of posterior samples (default: 1000)
        chains: Number of MCMC chains (default: 2)
        tune: Number of tuning/warmup samples (default: 1000)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with:
        - model: Trained MMM model
        - idata: InferenceData object with posterior samples
        - media_effectiveness: Media effectiveness estimates (posterior summaries)
        - summary_stats: Summary statistics
        - channel_names: List of channel names used in the model
    
    Raises:
        ModelTrainingError: If model training fails
    """
    if not PYMC_MARKETING_AVAILABLE:
        raise ModelTrainingError(
            "pymc-marketing not available. Install with: pip install pymc-marketing"
        )
    
    try:
        logger.info("Starting MMM analysis with pymc-marketing...")
        
        # Prepare data
        df_prep, date_col_name = prepare_mmm_data(
            df,
            target_col,
            media_channels,
            control_vars,
            date_col
        )
        
        # Initialize Adstock and Saturation transformations
        # DelayedAdstock requires l_max parameter (maximum lag)
        adstock_transform = DelayedAdstock(l_max=adstock_max_lag)
        saturation_transform = HillSaturation()
        
        # Initialize MMM model
        logger.info(
            f"Initializing MMM with {len(media_channels)} channels, "
            f"adstock_max_lag={adstock_max_lag}, yearly_seasonality={yearly_seasonality}"
        )
        
        mmm = MMM(
            date_column=date_col_name,
            channel_columns=media_channels,
            adstock=adstock_transform,
            saturation=saturation_transform,
            control_columns=control_vars,
            yearly_seasonality=yearly_seasonality if yearly_seasonality > 0 else None
        )
        
        # Prepare X (full DataFrame with date and channels) and y (target)
        # pymc-marketing expects full DataFrame with date column
        X = df_prep[[date_col_name] + media_channels + (control_vars or [])].copy()
        y = df_prep[target_col].copy()
        
        # Clean data: remove rows with NaN or inf values
        # pymc-marketing doesn't support NaN values directly
        valid_mask = ~(X.isna().any(axis=1) | np.isinf(X.select_dtypes(include=[np.number])).any(axis=1))
        valid_mask = valid_mask & ~(y.isna() | np.isinf(y))
        
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum()
            logger.warning(f"Removing {n_invalid} rows with NaN or inf values")
            X = X[valid_mask].reset_index(drop=True)
            y = y[valid_mask].reset_index(drop=True)
        
        if len(X) == 0:
            raise DataValidationError("No valid data remaining after cleaning NaN/inf values")
        
        # Handle zero values in media channels (add small epsilon to avoid division by zero)
        # pymc-marketing's adstock transformation can fail with exact zeros
        epsilon = 1e-6
        for channel in media_channels:
            if (X[channel] == 0).any():
                n_zeros = (X[channel] == 0).sum()
                if n_zeros > 0:
                    logger.warning(f"Channel {channel} has {n_zeros} zero values. Adding epsilon={epsilon} to avoid numerical issues.")
                    X[channel] = X[channel].replace(0, epsilon)
        
        logger.info(f"Using {len(X)} valid samples for MMM fitting")
        
        # Fit the model
        # pymc-marketing's fit() accepts sampler parameters as kwargs
        total_iterations = (tune + draws) * chains
        logger.info("=" * 60)
        logger.info("INICIANDO ENTRENAMIENTO DEL MODELO MMM")
        logger.info("=" * 60)
        logger.info(f"Configuración del modelo:")
        logger.info(f"  - Muestras de datos: {len(X)}")
        logger.info(f"  - Canales de medios: {len(media_channels)}")
        logger.info(f"  - Variables de control: {len(control_vars) if control_vars else 0}")
        logger.info(f"  - Cadenas MCMC: {chains}")
        logger.info(f"  - Iteraciones de ajuste (tune): {tune} por cadena")
        logger.info(f"  - Iteraciones de muestreo (draws): {draws} por cadena")
        logger.info(f"  - Total de iteraciones: {total_iterations}")
        logger.info("")
        logger.info("Fase 1/2: Ajuste (tuning/warmup)...")
        logger.info(f"  Procesando {tune} iteraciones por cadena ({chains} cadena(s))...")
        logger.info("  Esto puede tardar varios minutos. Por favor, espere...")
        logger.info("")
        
        import time
        start_time = time.time()
        
        mmm.fit(
            X=X,
            y=y,
            progressbar=True,  # PyMC mostrará barra de progreso automática
            random_seed=random_seed,
            draws=draws,
            tune=tune,
            chains=chains
        )
        
        elapsed_time = time.time() - start_time
        
        logger.info("")
        logger.info("Fase 2/2: Muestreo posterior completado")
        logger.info(f"Tiempo total de entrenamiento: {elapsed_time:.2f} segundos ({elapsed_time/60:.2f} minutos)")
        logger.info("=" * 60)
        logger.info("MMM model fitted successfully")
        logger.info("=" * 60)
        
        # Get inference data from fit_result property
        idata = mmm.fit_result
        
        # Extract media effectiveness from posterior
        # Use pymc-marketing's method to get time series contribution posterior
        media_effectiveness = {}
        try:
            # Get contribution posterior for all channels
            contribution_posterior = mmm.get_ts_contribution_posterior()
            
            for channel in media_channels:
                try:
                    # Extract samples for this channel
                    if channel in contribution_posterior:
                        samples = contribution_posterior[channel].values.flatten()
                        
                        media_effectiveness[channel] = {
                            'mean': float(np.mean(samples)),
                            'median': float(np.median(samples)),
                            'std': float(np.std(samples)),
                            'p5': float(np.percentile(samples, 5)),
                            'p95': float(np.percentile(samples, 95)),
                            'hdi_3%': float(np.percentile(samples, 3)),
                            'hdi_97%': float(np.percentile(samples, 97))
                        }
                    else:
                        # Try to find in posterior data variables
                        # Look for channel-related variables in posterior
                        posterior_vars = list(idata.posterior.data_vars.keys())
                        channel_vars = [v for v in posterior_vars if channel.lower() in v.lower()]
                        
                        if channel_vars:
                            # Use first matching variable
                            var_name = channel_vars[0]
                            samples = idata.posterior[var_name].values.flatten()
                            media_effectiveness[channel] = {
                                'mean': float(np.mean(samples)),
                                'median': float(np.median(samples)),
                                'std': float(np.std(samples)),
                                'p5': float(np.percentile(samples, 5)),
                                'p95': float(np.percentile(samples, 95))
                            }
                        else:
                            media_effectiveness[channel] = {
                                'mean': 'N/A',
                                'note': f'Channel contribution not found in posterior. Available vars: {posterior_vars[:5]}'
                            }
                except Exception as e:
                    logger.warning(f"Error extracting effectiveness for {channel}: {e}")
                    media_effectiveness[channel] = {
                        'mean': 'N/A',
                        'error': str(e)
                    }
        except Exception as e:
            logger.warning(f"Could not get contribution posterior: {e}. Using fallback method.")
            # Fallback: try to extract from idata directly
            for channel in media_channels:
                media_effectiveness[channel] = {
                    'mean': 'N/A',
                    'note': 'Use mmm.fit_result to access full posterior for detailed analysis'
                }
        
        # Summary statistics
        summary_stats = {
            'n_samples': len(y),
            'n_media_channels': len(media_channels),
            'n_control_vars': len(control_vars) if control_vars else 0,
            'target_mean': float(np.mean(y)),
            'target_std': float(np.std(y)),
            'draws': draws,
            'chains': chains,
            'tune': tune
        }
        
        results = {
            'model': mmm,
            'idata': idata,
            'media_effectiveness': media_effectiveness,
            'summary_stats': summary_stats,
            'channel_names': media_channels,
            'date_column': date_col_name
        }
        
        logger.info("MMM analysis completed successfully")
        return results
        
    except Exception as e:
        error_msg = f"Error running MMM analysis: {str(e)}"
        logger.error(error_msg)
        raise ModelTrainingError(error_msg) from e


def optimize_channel_budget(
    mmm_model: MMM,
    total_budget: float,
    budget_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    optimizer: str = "SCIPY"
) -> Dict:
    """
    Optimize channel budget allocation using pymc-marketing.
    
    Args:
        mmm_model: Trained DelayedSaturatedMMM model
        total_budget: Total budget to allocate across channels
        budget_bounds: Optional dict mapping channel names to (min, max) budget bounds
        optimizer: Optimization method ('SCIPY' or 'PYRO')
    
    Returns:
        Dictionary with:
        - optimal_allocation: Dict mapping channel names to optimal budget
        - expected_incremental_sales: Expected incremental sales from optimal allocation
        - optimization_status: Status of optimization
    """
    if not PYMC_MARKETING_AVAILABLE:
        raise ModelTrainingError(
            "pymc-marketing not available. Install with: pip install pymc-marketing"
        )
    
    try:
        logger.info(f"Optimizing budget allocation for total budget: {total_budget}")
        
        # Prepare budget bounds if not provided
        if budget_bounds is None:
            # Default: each channel gets at least 0 and at most total_budget
            budget_bounds = {
                channel: (0.0, total_budget)
                for channel in mmm_model.channel_columns
            }
        
        # Run optimization
        optimal_allocation = mmm_model.optimize_budget(
            budget=total_budget,
            budget_bounds=budget_bounds
        )
        
        logger.info("Budget optimization completed successfully")
        
        # Calculate expected incremental sales (if available from model)
        # This would require running predictions with the optimal allocation
        results = {
            'optimal_allocation': optimal_allocation,
            'total_budget': total_budget,
            'optimization_status': 'success'
        }
        
        return results
        
    except Exception as e:
        error_msg = f"Error optimizing channel budget: {str(e)}"
        logger.error(error_msg)
        raise ModelTrainingError(error_msg) from e


def plot_mmm_results(
    mmm_model: MMM,
    output_path: Optional[str] = None
) -> None:
    """
    Generate visualizations for MMM results using pymc-marketing.
    
    Args:
        mmm_model: Trained DelayedSaturatedMMM model
        output_path: Optional path to save plots
    """
    if not PYMC_MARKETING_AVAILABLE:
        raise ModelTrainingError(
            "pymc-marketing not available. Install with: pip install pymc-marketing"
        )
    
    try:
        logger.info("Generating MMM visualizations...")
        
        # Plot channel contribution grid
        try:
            import matplotlib.pyplot as plt
            mmm_model.plot_channel_contribution_grid()
            if output_path:
                plt.savefig(f"{output_path}_channel_contributions.png", dpi=300, bbox_inches='tight')
                logger.info(f"Saved channel contributions plot to {output_path}_channel_contributions.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot channel contributions: {e}")
        
        # Plot allocated contribution by channel
        try:
            import matplotlib.pyplot as plt
            mmm_model.plot_allocated_contribution_by_channel()
            if output_path:
                plt.savefig(f"{output_path}_allocated_contributions.png", dpi=300, bbox_inches='tight')
                logger.info(f"Saved allocated contributions plot to {output_path}_allocated_contributions.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not plot allocated contributions: {e}")
        
        logger.info("MMM visualizations generated successfully")
        
    except Exception as e:
        error_msg = f"Error generating MMM visualizations: {str(e)}"
        logger.error(error_msg)
        raise ModelTrainingError(error_msg) from e
