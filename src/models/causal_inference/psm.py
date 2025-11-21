"""Propensity Score Matching (PSM) for causal inference.

This module implements Propensity Score Matching using DoWhy library
to estimate causal effects of treatments/interventions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logger = logging.getLogger("omnistats")
    logger.warning(
        "DoWhy not available. Install with: pip install dowhy"
    )

from src.utils.exceptions import ModelTrainingError, DataValidationError
from src.data.validators import (
    validate_required_columns,
    validate_numeric_column
)

logger = logging.getLogger("omnistats")


def run_psm_analysis(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    confounders: List[str],
    common_causes: Optional[List[str]] = None,
    effect_modifiers: Optional[List[str]] = None,
    estimand_type: str = "nonparametric-ate",
    method_name: str = "backdoor.propensity_score_matching",
    random_seed: Optional[int] = None
) -> Dict:
    """
    Run Propensity Score Matching (PSM) analysis using DoWhy.
    
    PSM is used to estimate causal effects by matching treated and control
    units based on their propensity scores (probability of receiving treatment).
    
    Args:
        df: DataFrame with treatment, outcome, and confounder variables
        treatment_col: Name of treatment indicator column (0/1 or True/False)
        outcome_col: Name of outcome variable column
        confounders: List of confounder variable column names
        common_causes: Optional list of common causes (defaults to confounders)
        effect_modifiers: Optional list of effect modifier column names
        estimand_type: Type of causal estimand ('nonparametric-ate', 'ate', etc.)
        method_name: Estimation method ('backdoor.propensity_score_matching', etc.)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with:
        - causal_estimate: Causal effect estimate
        - psm_model: Fitted PSM model
        - matched_data: Matched dataset (if available)
        - summary_stats: Summary statistics
        - diagnostics: Model diagnostics
    
    Raises:
        ModelTrainingError: If PSM analysis fails
        DataValidationError: If data is invalid
    """
    if not DOWHY_AVAILABLE:
        raise ModelTrainingError(
            "DoWhy not available. Install with: pip install dowhy"
        )
    
    try:
        logger.info("Starting Propensity Score Matching (PSM) analysis...")
        
        # Validate required columns
        required_cols = [treatment_col, outcome_col] + confounders
        validate_required_columns(df, required_cols, "PSM data")
        
        # Validate treatment column (should be binary)
        if df[treatment_col].nunique() != 2:
            raise DataValidationError(
                f"Treatment column '{treatment_col}' must be binary (0/1 or True/False)"
            )
        
        # Validate numeric columns
        validate_numeric_column(df, outcome_col)
        for conf in confounders:
            validate_numeric_column(df, conf)
        
        # Prepare data (remove NaN values)
        df_clean = df[required_cols].copy()
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        
        if len(df_clean) < initial_rows:
            n_removed = initial_rows - len(df_clean)
            logger.warning(f"Removed {n_removed} rows with NaN values")
        
        if len(df_clean) == 0:
            raise DataValidationError("No valid data remaining after cleaning")
        
        # Convert treatment to binary if needed
        if df_clean[treatment_col].dtype == bool:
            df_clean[treatment_col] = df_clean[treatment_col].astype(int)
        elif df_clean[treatment_col].dtype == object:
            # Try to convert string booleans
            df_clean[treatment_col] = pd.to_numeric(df_clean[treatment_col], errors='coerce')
        
        # Check treatment balance
        n_treated = (df_clean[treatment_col] == 1).sum()
        n_control = (df_clean[treatment_col] == 0).sum()
        
        logger.info(f"Treatment groups: {n_treated} treated, {n_control} control")
        
        if n_treated == 0 or n_control == 0:
            raise DataValidationError(
                "Both treatment and control groups must have at least one observation"
            )
        
        # Use common_causes if provided, otherwise use confounders
        common_causes_list = common_causes if common_causes else confounders
        
        # Build causal graph string for DoWhy
        # Format: "treatment -> outcome; confounder -> treatment; confounder -> outcome"
        graph_edges = [f"{treatment_col} -> {outcome_col}"]
        for conf in common_causes_list:
            graph_edges.append(f"{conf} -> {treatment_col}")
            graph_edges.append(f"{conf} -> {outcome_col}")
        
        graph_str = "; ".join(graph_edges)
        
        logger.info(f"Building causal model with {len(common_causes_list)} confounders...")
        
        # Create causal model
        model = CausalModel(
            data=df_clean,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=common_causes_list,
            effect_modifiers=effect_modifiers if effect_modifiers else None,
            graph=graph_str
        )
        
        # Identify causal effect
        logger.info("Identifying causal effect...")
        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        
        # Estimate causal effect using PSM
        logger.info(f"Estimating causal effect using {method_name}...")
        causal_estimate = model.estimate_effect(
            identified_estimand,
            method_name=method_name,
            method_params={
                'num_ci_simulations': 100,
                'random_seed': random_seed
            } if random_seed else {'num_ci_simulations': 100}
        )
        
        # Refute estimate (sensitivity analysis)
        logger.info("Running refutation tests...")
        refutation_results = {}
        try:
            # Placebo treatment refutation
            refutation_placebo = model.refute_estimate(
                identified_estimand,
                causal_estimate,
                method_name="placebo_treatment_refuter",
                placebo_type="permute"
            )
            refutation_results['placebo_treatment'] = {
                'new_effect': float(refutation_placebo.new_effect),
                'p_value': float(refutation_placebo.refutation_result['p_value']) if hasattr(refutation_placebo, 'refutation_result') else None
            }
        except Exception as e:
            logger.warning(f"Placebo refutation failed: {e}")
            refutation_results['placebo_treatment'] = {'error': str(e)}
        
        # Summary statistics
        summary_stats = {
            'n_samples': len(df_clean),
            'n_treated': int(n_treated),
            'n_control': int(n_control),
            'n_confounders': len(common_causes_list),
            'treatment_rate': float(n_treated / len(df_clean)),
            'outcome_mean_treated': float(df_clean[df_clean[treatment_col] == 1][outcome_col].mean()),
            'outcome_mean_control': float(df_clean[df_clean[treatment_col] == 0][outcome_col].mean()),
            'outcome_mean_diff': float(
                df_clean[df_clean[treatment_col] == 1][outcome_col].mean() -
                df_clean[df_clean[treatment_col] == 0][outcome_col].mean()
            )
        }
        
        # Extract causal estimate details
        estimate_value = float(causal_estimate.value)
        estimate_ci_lower = float(causal_estimate.get_confidence_intervals()[0]) if hasattr(causal_estimate, 'get_confidence_intervals') else None
        estimate_ci_upper = float(causal_estimate.get_confidence_intervals()[1]) if hasattr(causal_estimate, 'get_confidence_intervals') else None
        
        results = {
            'causal_estimate': {
                'value': estimate_value,
                'ci_lower': estimate_ci_lower,
                'ci_upper': estimate_ci_upper,
                'interpretation': _interpret_causal_effect(estimate_value, outcome_col)
            },
            'psm_model': model,
            'identified_estimand': identified_estimand,
            'causal_estimate_object': causal_estimate,
            'summary_stats': summary_stats,
            'refutation_results': refutation_results,
            'treatment_col': treatment_col,
            'outcome_col': outcome_col,
            'confounders': common_causes_list
        }
        
        logger.info(f"PSM analysis completed. Causal effect: {estimate_value:.4f}")
        logger.info(f"Interpretation: {results['causal_estimate']['interpretation']}")
        
        return results
        
    except Exception as e:
        error_msg = f"Error running PSM analysis: {str(e)}"
        logger.error(error_msg)
        raise ModelTrainingError(error_msg) from e


def _interpret_causal_effect(effect_value: float, outcome_col: str) -> str:
    """Interpret causal effect estimate."""
    if effect_value > 0:
        return f"Treatment has a positive causal effect of {effect_value:.4f} on {outcome_col}. Treatment increases {outcome_col} by {effect_value:.4f} units on average."
    elif effect_value < 0:
        return f"Treatment has a negative causal effect of {abs(effect_value):.4f} on {outcome_col}. Treatment decreases {outcome_col} by {abs(effect_value):.4f} units on average."
    else:
        return f"Treatment has no significant causal effect on {outcome_col}."

