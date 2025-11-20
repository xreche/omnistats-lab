"""Marketing Mix Modeling (MMM) using lightweight_mmm."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

try:
    from lightweight_mmm import lightweight_mmm
    from lightweight_mmm import optimize_media
    from lightweight_mmm import plot
    LIGHTWEIGHT_MMM_AVAILABLE = True
except ImportError:
    LIGHTWEIGHT_MMM_AVAILABLE = False
    logger = logging.getLogger("omnistats")
    logger.warning(
        "lightweight_mmm not available. Install with: pip install lightweight-mmm"
    )

from src.utils.exceptions import ModelTrainingError, DataValidationError
from src.data.validators import validate_required_columns, validate_numeric_column
from src.features.marketing_features import apply_adstock_and_saturation

logger = logging.getLogger("omnistats")


def prepare_mmm_data(
    df: pd.DataFrame,
    target_col: str,
    media_channels: List[str],
    control_vars: Optional[List[str]] = None,
    date_col: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Prepare data for Marketing Mix Modeling.
    
    Args:
        df: DataFrame with target, media, and control variables
        target_col: Name of target variable (e.g., sales, revenue)
        media_channels: List of media channel column names
        control_vars: Optional list of control variable column names
        date_col: Optional date column for time series ordering
    
    Returns:
        Tuple of:
        - Target array (n_samples,)
        - Media array (n_samples, n_channels)
        - Control array (n_samples, n_controls) or None
        - List of media channel names
    """
    try:
        # Validate required columns
        required_cols = [target_col] + media_channels
        if control_vars:
            required_cols.extend(control_vars)
        
        validate_required_columns(df, required_cols, "MMM data")
        validate_numeric_column(df, target_col, min_value=0)
        
        # Sort by date if provided
        if date_col and date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)
        
        # Extract target
        target = df[target_col].values
        
        # Extract media channels
        media = df[media_channels].values
        
        # Extract control variables
        control = None
        if control_vars:
            control = df[control_vars].values
        
        logger.info(
            f"Prepared MMM data: {len(target)} samples, {len(media_channels)} media channels"
        )
        
        return target, media, control, media_channels
        
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
    n_samples: int = 1000,
    n_chains: int = 2,
    n_warmup: int = 500,
    apply_transformations: bool = True,
    adstock_decays: Optional[Dict[str, float]] = None,
    saturation_params: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict:
    """
    Run Marketing Mix Modeling analysis using lightweight_mmm.
    
    This function:
    1. Applies Adstock and Saturation transformations (optional)
    2. Fits a Bayesian MMM model
    3. Extracts media effectiveness and ROI estimates
    4. Provides optimization recommendations
    
    Args:
        df: DataFrame with target, media, and control variables
        target_col: Name of target variable
        media_channels: List of media channel column names
        control_vars: Optional list of control variable column names
        date_col: Optional date column
        n_samples: Number of MCMC samples (default: 1000)
        n_chains: Number of MCMC chains (default: 2)
        n_warmup: Number of warmup samples (default: 500)
        apply_transformations: Whether to apply Adstock and Saturation (default: True)
        adstock_decays: Dict of decay parameters per channel
        saturation_params: Dict of (half_saturation, slope) per channel
    
    Returns:
        Dictionary with:
        - model: Trained MMM model
        - media_effectiveness: Media effectiveness estimates
        - roi_estimates: ROI estimates per channel
        - optimization_results: Budget optimization results
        - summary_stats: Summary statistics
    
    Raises:
        ModelTrainingError: If model training fails
    """
    if not LIGHTWEIGHT_MMM_AVAILABLE:
        raise ModelTrainingError(
            "lightweight_mmm not available. Install with: pip install lightweight-mmm"
        )
    
    try:
        logger.info("Starting MMM analysis...")
        
        # Apply transformations if requested
        if apply_transformations:
            logger.info("Applying Adstock and Saturation transformations...")
            df_transformed = apply_adstock_and_saturation(
                df,
                media_channels,
                adstock_decays,
                saturation_params
            )
            # Use transformed columns
            transformed_channels = [f"{ch}_saturated" for ch in media_channels]
            # Check which transformed columns exist
            available_channels = [
                ch for ch in transformed_channels if ch in df_transformed.columns
            ]
            if available_channels:
                media_channels = available_channels
                df = df_transformed
        
        # Prepare data
        target, media, control, channel_names = prepare_mmm_data(
            df,
            target_col,
            media_channels,
            control_vars,
            date_col
        )
        
        # Initialize and fit model
        logger.info(f"Fitting MMM model with {n_samples} samples...")
        mmm = lightweight_mmm.LightweightMMM()
        mmm.fit(
            media=media,
            target=target,
            extra_features=control,
            number_samples=n_samples,
            number_chains=n_chains,
            number_warmup=n_warmup
        )
        
        logger.info("MMM model fitted successfully")
        
        # Extract media effectiveness
        media_effectiveness = {}
        for i, channel in enumerate(channel_names):
            # Get posterior distribution of media coefficients
            media_coef = mmm.trace['media_transformed'][:, :, i]
            media_effectiveness[channel] = {
                'mean': float(np.mean(media_coef)),
                'median': float(np.median(media_coef)),
                'std': float(np.std(media_coef)),
                'p5': float(np.percentile(media_coef, 5)),
                'p95': float(np.percentile(media_coef, 95))
            }
        
        # Calculate ROI estimates (simplified - would need spend data)
        roi_estimates = {}
        for channel in channel_names:
            # ROI = (Incremental Sales / Media Spend) * 100
            # This is a placeholder - actual ROI requires spend data
            roi_estimates[channel] = {
                'estimated_roi': 'N/A - requires spend data',
                'note': 'ROI calculation requires media spend data'
            }
        
        # Budget optimization (placeholder - would need budget constraints)
        optimization_results = {
            'status': 'Not run',
            'note': 'Budget optimization requires budget constraints and costs'
        }
        
        # Summary statistics
        summary_stats = {
            'n_samples': len(target),
            'n_media_channels': len(media_channels),
            'n_control_vars': len(control_vars) if control_vars else 0,
            'target_mean': float(np.mean(target)),
            'target_std': float(np.std(target))
        }
        
        results = {
            'model': mmm,
            'media_effectiveness': media_effectiveness,
            'roi_estimates': roi_estimates,
            'optimization_results': optimization_results,
            'summary_stats': summary_stats,
            'channel_names': channel_names
        }
        
        logger.info("MMM analysis completed successfully")
        return results
        
    except Exception as e:
        error_msg = f"Error running MMM analysis: {str(e)}"
        logger.error(error_msg)
        raise ModelTrainingError(error_msg) from e

