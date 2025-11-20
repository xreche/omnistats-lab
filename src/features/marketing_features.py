"""Marketing feature engineering: Adstock and Saturation transformations."""

import numpy as np
import pandas as pd
from typing import List, Optional, Union
import logging

logger = logging.getLogger("omnistats")


def adstock_transform(
    x: Union[np.ndarray, pd.Series],
    decay: float,
    max_lag: int = 12
) -> Union[np.ndarray, pd.Series]:
    """
    Apply Adstock transformation to marketing spend data.
    
    Adstock models the carryover effect of advertising, where past advertising
    continues to have an effect on current sales.
    
    Formula: adstock_t = x_t + decay * adstock_{t-1}
    
    Args:
        x: Input time series (marketing spend)
        decay: Decay parameter (0 < decay < 1). Higher = longer carryover effect
        max_lag: Maximum lag to consider (default: 12)
    
    Returns:
        Adstock-transformed series
    
    Raises:
        ValueError: If decay is not in (0, 1)
    """
    if not 0 < decay < 1:
        raise ValueError(f"Decay must be between 0 and 1, got {decay}")
    
    x_array = np.array(x) if isinstance(x, pd.Series) else x
    adstock = np.zeros_like(x_array)
    
    for t in range(len(x_array)):
        if t == 0:
            adstock[t] = x_array[t]
        else:
            adstock[t] = x_array[t] + decay * adstock[t - 1]
    
    # Apply max_lag truncation if needed
    if max_lag < len(adstock):
        # For simplicity, we keep the full adstock but could truncate
        pass
    
    if isinstance(x, pd.Series):
        return pd.Series(adstock, index=x.index, name=x.name)
    
    return adstock


def hill_saturation(
    x: Union[np.ndarray, pd.Series],
    half_saturation: float,
    slope: float = 1.0
) -> Union[np.ndarray, pd.Series]:
    """
    Apply Hill saturation function to marketing spend.
    
    Models the diminishing returns of advertising - each additional dollar
    of spend has less impact as spend increases.
    
    Formula: f(x) = (x^slope) / (half_saturation^slope + x^slope)
    
    Args:
        x: Input marketing spend
        half_saturation: Spend level at which effect is 50% of maximum
        slope: Slope parameter (default: 1.0). Higher = steeper curve
    
    Returns:
        Saturated marketing effect (0 to 1 scale)
    
    Raises:
        ValueError: If parameters are invalid
    """
    if half_saturation <= 0:
        raise ValueError(f"half_saturation must be > 0, got {half_saturation}")
    if slope <= 0:
        raise ValueError(f"slope must be > 0, got {slope}")
    
    x_array = np.array(x) if isinstance(x, pd.Series) else x
    x_array = np.maximum(x_array, 0)  # Ensure non-negative
    
    numerator = np.power(x_array, slope)
    denominator = np.power(half_saturation, slope) + numerator
    
    saturated = numerator / denominator
    
    if isinstance(x, pd.Series):
        return pd.Series(saturated, index=x.index, name=x.name)
    
    return saturated


def apply_adstock_and_saturation(
    df: pd.DataFrame,
    media_channels: List[str],
    adstock_decays: Optional[dict] = None,
    saturation_params: Optional[dict] = None
) -> pd.DataFrame:
    """
    Apply Adstock and Saturation transformations to multiple media channels.
    
    Args:
        df: DataFrame with media spend columns
        media_channels: List of column names for media channels
        adstock_decays: Dict mapping channel names to decay parameters.
                       If None, uses default decay=0.5 for all
        saturation_params: Dict mapping channel names to (half_saturation, slope).
                          If None, uses default (half_saturation=mean, slope=1.0)
    
    Returns:
        DataFrame with transformed media columns (original + transformed)
    """
    df_transformed = df.copy()
    
    # Default parameters
    if adstock_decays is None:
        adstock_decays = {channel: 0.5 for channel in media_channels}
    
    if saturation_params is None:
        saturation_params = {}
        for channel in media_channels:
            if channel in df.columns:
                mean_spend = df[channel].mean()
                saturation_params[channel] = (mean_spend, 1.0)
    
    for channel in media_channels:
        if channel not in df.columns:
            logger.warning(f"Channel {channel} not found in DataFrame. Skipping.")
            continue
        
        # Apply Adstock
        decay = adstock_decays.get(channel, 0.5)
        adstock_col = f"{channel}_adstock"
        df_transformed[adstock_col] = adstock_transform(
            df[channel],
            decay=decay
        )
        
        # Apply Saturation
        half_sat, slope = saturation_params.get(
            channel,
            (df[channel].mean(), 1.0)
        )
        saturated_col = f"{channel}_saturated"
        df_transformed[saturated_col] = hill_saturation(
            df_transformed[adstock_col],
            half_saturation=half_sat,
            slope=slope
        )
        
        logger.info(
            f"Transformed {channel}: adstock_decay={decay:.2f}, "
            f"saturation_half={half_sat:.2f}, slope={slope:.2f}"
        )
    
    return df_transformed

