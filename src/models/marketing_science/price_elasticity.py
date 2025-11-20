"""Price elasticity analysis using Log-Log regression models."""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import logging

import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

from src.utils.exceptions import ModelTrainingError, DataValidationError
from src.data.validators import (
    validate_required_columns,
    validate_numeric_column
)

logger = logging.getLogger("omnistats")


def calculate_price_elasticity(
    df: pd.DataFrame,
    quantity_col: str,
    price_col: str,
    promotion_col: Optional[str] = None,
    control_vars: Optional[list] = None,
    log_transform: bool = True
) -> Dict:
    """
    Calculate price elasticity of demand using Log-Log regression.
    
    Price elasticity measures how sensitive demand is to price changes.
    Elasticity < -1: Elastic (demand changes more than price)
    Elasticity > -1: Inelastic (demand changes less than price)
    
    Model: log(quantity) = α + β * log(price) + γ * promotion + controls + ε
    
    Price Elasticity = β (coefficient of log(price))
    
    Args:
        df: DataFrame with quantity, price, and optional promotion data
        quantity_col: Name of quantity/demand column
        price_col: Name of price column
        promotion_col: Optional name of promotion indicator column (0/1)
        control_vars: Optional list of control variable column names
        log_transform: Whether to apply log transformation (default: True)
    
    Returns:
        Dictionary with:
        - elasticity: Price elasticity coefficient
        - promotion_lift: Promotion lift coefficient (if promotion_col provided)
        - model_summary: OLS model summary
        - model_params: Model parameters
        - predictions: Model predictions
    
    Raises:
        DataValidationError: If data is invalid
        ModelTrainingError: If model training fails
    """
    try:
        # Validate required columns
        required_cols = [quantity_col, price_col]
        if promotion_col:
            required_cols.append(promotion_col)
        if control_vars:
            required_cols.extend(control_vars)
        
        validate_required_columns(df, required_cols, "Price elasticity data")
        validate_numeric_column(df, quantity_col, min_value=0)
        validate_numeric_column(df, price_col, min_value=0)
        
        # Remove rows with zero or negative values (for log transform)
        df_clean = df.copy()
        df_clean = df_clean[
            (df_clean[quantity_col] > 0) & (df_clean[price_col] > 0)
        ].copy()
        
        if len(df_clean) == 0:
            raise DataValidationError(
                "No valid data after removing zero/negative values"
            )
        
        logger.info(f"Preparing price elasticity model with {len(df_clean)} observations")
        
        # Prepare dependent variable
        if log_transform:
            y = np.log(df_clean[quantity_col].values)
            y_name = f"log({quantity_col})"
        else:
            y = df_clean[quantity_col].values
            y_name = quantity_col
        
        # Prepare independent variables
        X_vars = []
        X_names = []
        
        # Price (main variable)
        if log_transform:
            X_vars.append(np.log(df_clean[price_col].values))
            X_names.append(f"log({price_col})")
        else:
            X_vars.append(df_clean[price_col].values)
            X_names.append(price_col)
        
        # Promotion (if provided)
        if promotion_col:
            X_vars.append(df_clean[promotion_col].values)
            X_names.append(promotion_col)
        
        # Control variables
        if control_vars:
            for var in control_vars:
                if var in df_clean.columns:
                    X_vars.append(df_clean[var].values)
                    X_names.append(var)
        
        # Stack features
        X = np.column_stack(X_vars)
        X = sm.add_constant(X)  # Add intercept
        
        # Fit OLS model
        logger.info("Fitting Log-Log OLS model...")
        model = OLS(y, X)
        results = model.fit()
        
        # Extract price elasticity
        price_idx = 1  # After constant
        elasticity = results.params[price_idx]
        elasticity_se = results.bse[price_idx]
        elasticity_pvalue = results.pvalues[price_idx]
        
        # Extract promotion lift (if provided)
        promotion_lift = None
        if promotion_col:
            promo_idx = 2  # After constant and price
            promotion_lift = {
                'coefficient': float(results.params[promo_idx]),
                'std_error': float(results.bse[promo_idx]),
                'p_value': float(results.pvalues[promo_idx]),
                'interpretation': (
                    f"Promotion increases demand by "
                    f"{np.exp(results.params[promo_idx]) - 1:.1%}"
                    if log_transform else
                    f"Promotion increases demand by {results.params[promo_idx]:.2f} units"
                )
            }
        
        # Generate predictions
        predictions = results.predict(X)
        if log_transform:
            predictions = np.exp(predictions)  # Transform back
        
        # Model metrics
        model_metrics = {
            'r_squared': float(results.rsquared),
            'adj_r_squared': float(results.rsquared_adj),
            'f_statistic': float(results.fvalue),
            'f_pvalue': float(results.f_pvalue),
            'n_observations': len(df_clean),
            'n_features': len(X_names)
        }
        
        results_dict = {
            'elasticity': {
                'coefficient': float(elasticity),
                'std_error': float(elasticity_se),
                'p_value': float(elasticity_pvalue),
                'interpretation': (
                    f"Price elasticity: {elasticity:.3f}. "
                    f"A 1% increase in price leads to a {elasticity:.2f}% change in demand. "
                    f"{'Elastic' if elasticity < -1 else 'Inelastic'} demand."
                )
            },
            'promotion_lift': promotion_lift,
            'model_summary': str(results.summary()),
            'model_params': {
                'intercept': float(results.params[0]),
                'price_coefficient': float(elasticity),
                'feature_names': X_names
            },
            'model_metrics': model_metrics,
            'predictions': predictions,
            'actual': df_clean[quantity_col].values
        }
        
        logger.info(
            f"Price elasticity calculated: {elasticity:.3f} "
            f"(R² = {model_metrics['r_squared']:.3f})"
        )
        
        return results_dict
        
    except DataValidationError:
        raise
    except Exception as e:
        error_msg = f"Error calculating price elasticity: {str(e)}"
        logger.error(error_msg)
        raise ModelTrainingError(error_msg) from e

