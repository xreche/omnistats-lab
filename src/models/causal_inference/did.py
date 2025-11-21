"""Difference-in-Differences (DiD) for causal inference.

This module implements Difference-in-Differences analysis to estimate
causal effects using panel data with treatment and control groups
observed before and after treatment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import OLS
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger = logging.getLogger("omnistats")
    logger.warning(
        "statsmodels not available. Install with: pip install statsmodels"
    )

from src.utils.exceptions import ModelTrainingError, DataValidationError
from src.data.validators import (
    validate_required_columns,
    validate_numeric_column
)

logger = logging.getLogger("omnistats")


def run_did_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    time_col: str,
    group_col: Optional[str] = None,
    control_vars: Optional[List[str]] = None,
    post_treatment_period: Optional[str] = None,
    confidence_level: float = 0.95
) -> Dict:
    """
    Run Difference-in-Differences (DiD) analysis.
    
    DiD estimates causal effects by comparing the change in outcomes
    between treatment and control groups before and after treatment.
    
    Model: Y = α + β₁*Treatment + β₂*Post + β₃*(Treatment × Post) + controls + ε
    
    DiD Effect = β₃ (interaction coefficient)
    
    Args:
        df: DataFrame with treatment, outcome, time, and group variables
        treatment_col: Name of treatment indicator column (0/1)
        outcome_col: Name of outcome variable column
        time_col: Name of time period column (should be binary: 0=pre, 1=post)
        group_col: Optional name of group identifier column (if panel data)
        control_vars: Optional list of control variable column names
        post_treatment_period: Optional value indicating post-treatment period
        confidence_level: Confidence level for intervals (default: 0.95)
    
    Returns:
        Dictionary with:
        - did_effect: DiD effect estimate (interaction coefficient)
        - model_summary: Full regression model summary
        - coefficients: All model coefficients
        - summary_stats: Summary statistics
        - diagnostics: Model diagnostics
    
    Raises:
        ModelTrainingError: If DiD analysis fails
        DataValidationError: If data is invalid
    """
    if not STATSMODELS_AVAILABLE:
        raise ModelTrainingError(
            "statsmodels not available. Install with: pip install statsmodels"
        )
    
    try:
        logger.info("Starting Difference-in-Differences (DiD) analysis...")
        
        # Validate required columns
        required_cols = [treatment_col, outcome_col, time_col]
        if group_col:
            required_cols.append(group_col)
        if control_vars:
            required_cols.extend(control_vars)
        
        validate_required_columns(df, required_cols, "DiD data")
        
        # Validate treatment and time columns (should be binary)
        if df[treatment_col].nunique() != 2:
            raise DataValidationError(
                f"Treatment column '{treatment_col}' must be binary (0/1)"
            )
        
        if df[time_col].nunique() != 2:
            raise DataValidationError(
                f"Time column '{time_col}' must be binary (0=pre-treatment, 1=post-treatment)"
            )
        
        # Validate numeric columns
        validate_numeric_column(df, outcome_col)
        validate_numeric_column(df, treatment_col)
        validate_numeric_column(df, time_col)
        
        if control_vars:
            for var in control_vars:
                validate_numeric_column(df, var)
        
        # Prepare data
        df_clean = df[required_cols].copy()
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        
        if len(df_clean) < initial_rows:
            n_removed = initial_rows - len(df_clean)
            logger.warning(f"Removed {n_removed} rows with NaN values")
        
        if len(df_clean) == 0:
            raise DataValidationError("No valid data remaining after cleaning")
        
        # Ensure binary columns are numeric
        df_clean[treatment_col] = pd.to_numeric(df_clean[treatment_col], errors='coerce')
        df_clean[time_col] = pd.to_numeric(df_clean[time_col], errors='coerce')
        
        # Create interaction term (Treatment × Post)
        interaction_col = f"{treatment_col}_x_{time_col}"
        df_clean[interaction_col] = df_clean[treatment_col] * df_clean[time_col]
        
        # Prepare regression variables
        y = df_clean[outcome_col]
        X_vars = [treatment_col, time_col, interaction_col]
        
        if control_vars:
            X_vars.extend(control_vars)
        
        X = df_clean[X_vars].copy()
        X = sm.add_constant(X)  # Add intercept
        
        # Check for perfect multicollinearity
        if X.shape[1] > X.shape[0]:
            raise DataValidationError(
                "More variables than observations. Reduce number of control variables."
            )
        
        # Run OLS regression
        logger.info("Fitting DiD regression model...")
        model = OLS(y, X).fit()
        
        # Extract DiD effect (interaction coefficient)
        did_effect_coef = model.params[interaction_col]
        did_effect_se = model.bse[interaction_col]
        did_effect_pvalue = model.pvalues[interaction_col]
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        t_critical = 1.96  # Approximate for large samples
        did_effect_ci_lower = did_effect_coef - t_critical * did_effect_se
        did_effect_ci_upper = did_effect_coef + t_critical * did_effect_se
        
        # Summary statistics by group and time
        summary_stats = {
            'n_samples': len(df_clean),
            'n_treated': int((df_clean[treatment_col] == 1).sum()),
            'n_control': int((df_clean[treatment_col] == 0).sum()),
            'n_pre': int((df_clean[time_col] == 0).sum()),
            'n_post': int((df_clean[time_col] == 1).sum()),
            'outcome_by_group_time': {
                'treated_pre': float(df_clean[(df_clean[treatment_col] == 1) & (df_clean[time_col] == 0)][outcome_col].mean()),
                'treated_post': float(df_clean[(df_clean[treatment_col] == 1) & (df_clean[time_col] == 1)][outcome_col].mean()),
                'control_pre': float(df_clean[(df_clean[treatment_col] == 0) & (df_clean[time_col] == 0)][outcome_col].mean()),
                'control_post': float(df_clean[(df_clean[treatment_col] == 0) & (df_clean[time_col] == 1)][outcome_col].mean())
            }
        }
        
        # Calculate DiD manually for verification
        treated_change = (
            summary_stats['outcome_by_group_time']['treated_post'] -
            summary_stats['outcome_by_group_time']['treated_pre']
        )
        control_change = (
            summary_stats['outcome_by_group_time']['control_post'] -
            summary_stats['outcome_by_group_time']['control_pre']
        )
        did_manual = treated_change - control_change
        
        # Model diagnostics
        diagnostics = {
            'r_squared': float(model.rsquared),
            'adj_r_squared': float(model.rsquared_adj),
            'f_statistic': float(model.fvalue),
            'f_pvalue': float(model.f_pvalue),
            'n_observations': int(model.nobs),
            'n_variables': int(model.df_model + 1),  # +1 for intercept
            'did_manual_calculation': float(did_manual)
        }
        
        # All coefficients
        coefficients = {}
        for var in X_vars + ['const']:
            if var in model.params.index:
                coefficients[var] = {
                    'coefficient': float(model.params[var]),
                    'std_error': float(model.bse[var]),
                    'p_value': float(model.pvalues[var]),
                    'ci_lower': float(model.conf_int(alpha=alpha).loc[var, 0]),
                    'ci_upper': float(model.conf_int(alpha=alpha).loc[var, 1])
                }
        
        results = {
            'did_effect': {
                'coefficient': float(did_effect_coef),
                'std_error': float(did_effect_se),
                'p_value': float(did_effect_pvalue),
                'ci_lower': float(did_effect_ci_lower),
                'ci_upper': float(did_effect_ci_upper),
                'interpretation': _interpret_did_effect(did_effect_coef, did_effect_pvalue, outcome_col)
            },
            'model': model,
            'model_summary': str(model.summary()),
            'coefficients': coefficients,
            'summary_stats': summary_stats,
            'diagnostics': diagnostics,
            'treatment_col': treatment_col,
            'outcome_col': outcome_col,
            'time_col': time_col
        }
        
        logger.info(f"DiD analysis completed. DiD effect: {did_effect_coef:.4f} (p-value: {did_effect_pvalue:.4f})")
        logger.info(f"Interpretation: {results['did_effect']['interpretation']}")
        
        return results
        
    except Exception as e:
        error_msg = f"Error running DiD analysis: {str(e)}"
        logger.error(error_msg)
        raise ModelTrainingError(error_msg) from e


def _interpret_did_effect(effect_value: float, p_value: float, outcome_col: str) -> str:
    """Interpret DiD effect estimate."""
    significance_level = 0.05
    is_significant = p_value < significance_level
    
    if is_significant:
        if effect_value > 0:
            return f"Treatment has a statistically significant positive effect of {effect_value:.4f} on {outcome_col} (p={p_value:.4f}). Treatment increases {outcome_col} by {effect_value:.4f} units relative to control group."
        elif effect_value < 0:
            return f"Treatment has a statistically significant negative effect of {abs(effect_value):.4f} on {outcome_col} (p={p_value:.4f}). Treatment decreases {outcome_col} by {abs(effect_value):.4f} units relative to control group."
        else:
            return f"Treatment has no significant effect on {outcome_col} (p={p_value:.4f})."
    else:
        return f"Treatment effect is not statistically significant (p={p_value:.4f}). The estimated effect of {effect_value:.4f} may be due to chance."

