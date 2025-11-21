"""Main script to run Causal Inference pipeline (Pilar 3)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import logging

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging_config import setup_logging
from src.models.causal_inference import (
    run_psm_analysis,
    run_did_analysis
)
from src.utils.exceptions import DataValidationError, ModelTrainingError, MissingConfigurationError

logger = setup_logging()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise MissingConfigurationError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def generate_synthetic_psm_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic data for Propensity Score Matching demonstration.
    
    Simulates a treatment (e.g., marketing campaign) and outcome (e.g., sales)
    with confounders (e.g., customer age, income).
    
    Args:
        n_samples: Number of observations
    
    Returns:
        DataFrame with synthetic PSM data
    """
    np.random.seed(42)
    logger.info(f"Generating synthetic PSM data for {n_samples} samples...")
    
    # Generate confounders
    age = np.random.normal(35, 10, n_samples)
    age = np.clip(age, 18, 80)  # Clip to reasonable range
    
    income = np.random.normal(50000, 15000, n_samples)
    income = np.maximum(income, 20000)  # Minimum income
    
    # Generate propensity score (probability of treatment)
    # Treatment is more likely for younger, higher-income customers
    propensity = 1 / (1 + np.exp(-(0.02 * (income - 50000) / 1000 - 0.05 * (age - 35))))
    
    # Generate treatment assignment based on propensity
    treatment = np.random.binomial(1, propensity, n_samples)
    
    # Generate outcome (sales) with treatment effect
    # True treatment effect: +500 units
    true_effect = 500
    base_sales = 1000 + 0.1 * income + 5 * age + np.random.normal(0, 200, n_samples)
    treatment_effect = true_effect * treatment
    sales = base_sales + treatment_effect + np.random.normal(0, 100, n_samples)
    sales = np.maximum(sales, 0)  # Ensure non-negative
    
    df = pd.DataFrame({
        'customer_id': [f'CUST_{i:04d}' for i in range(n_samples)],
        'treatment': treatment,
        'sales': sales,
        'age': age,
        'income': income
    })
    
    logger.info(f"Synthetic PSM data generated: {treatment.sum()} treated, {n_samples - treatment.sum()} control")
    return df


def generate_synthetic_did_data(n_groups: int = 50, n_periods: int = 2) -> pd.DataFrame:
    """
    Generate synthetic panel data for Difference-in-Differences demonstration.
    
    Simulates treatment and control groups observed before and after treatment.
    
    Args:
        n_groups: Number of groups (e.g., stores, regions)
        n_periods: Number of time periods (2 = before/after)
    
    Returns:
        DataFrame with synthetic DiD panel data
    """
    np.random.seed(42)
    logger.info(f"Generating synthetic DiD data: {n_groups} groups, {n_periods} periods...")
    
    # Assign half to treatment, half to control
    n_treated = n_groups // 2
    n_control = n_groups - n_treated
    
    data_rows = []
    
    for group_id in range(n_groups):
        treatment_group = 1 if group_id < n_treated else 0
        
        # Group-specific characteristics
        group_size = np.random.normal(100, 20)
        group_quality = np.random.normal(0, 1)
        
        for period in range(n_periods):
            # Pre-treatment period (period=0)
            if period == 0:
                base_outcome = 1000 + 50 * group_size + 100 * group_quality + np.random.normal(0, 50)
            else:
                # Post-treatment period (period=1)
                # True DiD effect: +200 for treated groups
                true_did_effect = 200 if treatment_group == 1 else 0
                base_outcome = 1000 + 50 * group_size + 100 * group_quality + true_did_effect + np.random.normal(0, 50)
            
            data_rows.append({
                'group_id': f'GROUP_{group_id:03d}',
                'treatment': treatment_group,
                'period': period,
                'outcome': max(0, base_outcome),  # Ensure non-negative
                'group_size': group_size,
                'group_quality': group_quality
            })
    
    df = pd.DataFrame(data_rows)
    logger.info(f"Synthetic DiD data generated: {len(df)} observations")
    return df


def main():
    """Run complete Causal Inference pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Causal Inference Pipeline (Pilar 3)")
    logger.info("=" * 60)
    
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "raw"
    outputs_dir = base_dir / "outputs"
    config_path = base_dir / "config" / "model_configs" / "causal_inference_config.yaml"
    
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "reports").mkdir(parents=True, exist_ok=True)
    
    try:
        config = load_config(config_path)
        causal_config = config.get("causal_inference", {})
        
        # 1. Propensity Score Matching (PSM)
        logger.info("\n[1/2] Running Propensity Score Matching (PSM)...")
        
        # Try to load real PSM data, otherwise generate synthetic
        psm_data_path = data_dir / "causal_inference" / "psm_data.csv"
        
        if psm_data_path.exists():
            logger.info(f"Loading PSM data from {psm_data_path}")
            psm_data = pd.read_csv(psm_data_path)
            logger.info(f"Loaded dataset with columns: {list(psm_data.columns)}")
        else:
            logger.info("Real PSM data not found. Generating synthetic data...")
            psm_data = generate_synthetic_psm_data(n_samples=1000)
        
        psm_params = causal_config.get("psm", {})
        
        # Identify columns (adjust based on your data structure)
        treatment_col = 'treatment'
        outcome_col = 'sales'
        confounders = ['age', 'income']  # Adjust based on your data
        
        # Filter to available columns
        available_confounders = [c for c in confounders if c in psm_data.columns]
        
        if treatment_col not in psm_data.columns or outcome_col not in psm_data.columns:
            logger.warning("Required columns not found in PSM data. Skipping PSM analysis.")
        elif len(available_confounders) == 0:
            logger.warning("No confounders found in PSM data. Skipping PSM analysis.")
        else:
            try:
                psm_results = run_psm_analysis(
                    df=psm_data,
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    confounders=available_confounders,
                    estimand_type=psm_params.get("estimand_type", "nonparametric-ate"),
                    method_name=psm_params.get("method_name", "backdoor.propensity_score_matching"),
                    random_seed=psm_params.get("random_seed", 42)
                )
                
                # Save PSM results
                psm_output = outputs_dir / "reports" / "psm_results.txt"
                psm_output.parent.mkdir(parents=True, exist_ok=True)
                
                with open(psm_output, 'w', encoding='utf-8') as f:
                    f.write("PROPENSITY SCORE MATCHING (PSM) RESULTS\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("Causal Effect Estimate:\n")
                    f.write(f"  Value: {psm_results['causal_estimate']['value']:.4f}\n")
                    if psm_results['causal_estimate']['ci_lower']:
                        f.write(f"  95% CI: [{psm_results['causal_estimate']['ci_lower']:.4f}, {psm_results['causal_estimate']['ci_upper']:.4f}]\n")
                    f.write(f"  Interpretation: {psm_results['causal_estimate']['interpretation']}\n\n")
                    
                    f.write("Summary Statistics:\n")
                    for k, v in psm_results['summary_stats'].items():
                        f.write(f"  {k}: {v}\n")
                    
                    if psm_results.get('refutation_results'):
                        f.write("\nRefutation Tests:\n")
                        for test_name, test_result in psm_results['refutation_results'].items():
                            f.write(f"  {test_name}:\n")
                            if 'error' in test_result:
                                f.write(f"    Error: {test_result['error']}\n")
                            else:
                                for k, v in test_result.items():
                                    f.write(f"    {k}: {v}\n")
                
                logger.info(f"Saved PSM results to {psm_output}")
                
            except Exception as e:
                logger.error(f"PSM analysis failed: {e}")
        
        # 2. Difference-in-Differences (DiD)
        logger.info("\n[2/2] Running Difference-in-Differences (DiD)...")
        
        # Try to load real DiD data, otherwise generate synthetic
        did_data_path = data_dir / "causal_inference" / "did_data.csv"
        
        if did_data_path.exists():
            logger.info(f"Loading DiD data from {did_data_path}")
            did_data = pd.read_csv(did_data_path)
            logger.info(f"Loaded dataset with columns: {list(did_data.columns)}")
        else:
            logger.info("Real DiD data not found. Generating synthetic data...")
            did_data = generate_synthetic_did_data(n_groups=50, n_periods=2)
        
        did_params = causal_config.get("did", {})
        
        # Identify columns (adjust based on your data structure)
        treatment_col_did = 'treatment'
        outcome_col_did = 'outcome'
        time_col = 'period'
        control_vars_did = ['group_size', 'group_quality'] if did_params.get("include_controls", True) else None
        
        # Filter to available columns
        available_controls = []
        if control_vars_did:
            available_controls = [c for c in control_vars_did if c in did_data.columns]
            if len(available_controls) == 0:
                available_controls = None
        
        if (treatment_col_did not in did_data.columns or 
            outcome_col_did not in did_data.columns or 
            time_col not in did_data.columns):
            logger.warning("Required columns not found in DiD data. Skipping DiD analysis.")
        else:
            try:
                did_results = run_did_analysis(
                    df=did_data,
                    treatment_col=treatment_col_did,
                    outcome_col=outcome_col_did,
                    time_col=time_col,
                    control_vars=available_controls,
                    confidence_level=did_params.get("confidence_level", 0.95)
                )
                
                # Save DiD results
                did_output = outputs_dir / "reports" / "did_results.txt"
                did_output.parent.mkdir(parents=True, exist_ok=True)
                
                with open(did_output, 'w', encoding='utf-8') as f:
                    f.write("DIFFERENCE-IN-DIFFERENCES (DiD) RESULTS\n")
                    f.write("=" * 60 + "\n\n")
                    f.write("DiD Effect Estimate:\n")
                    f.write(f"  Coefficient: {did_results['did_effect']['coefficient']:.4f}\n")
                    f.write(f"  Std Error: {did_results['did_effect']['std_error']:.4f}\n")
                    f.write(f"  P-value: {did_results['did_effect']['p_value']:.4f}\n")
                    f.write(f"  95% CI: [{did_results['did_effect']['ci_lower']:.4f}, {did_results['did_effect']['ci_upper']:.4f}]\n")
                    f.write(f"  Interpretation: {did_results['did_effect']['interpretation']}\n\n")
                    
                    f.write("Model Diagnostics:\n")
                    for k, v in did_results['diagnostics'].items():
                        f.write(f"  {k}: {v}\n")
                    
                    f.write("\nSummary Statistics:\n")
                    for k, v in did_results['summary_stats'].items():
                        if k == 'outcome_by_group_time':
                            f.write(f"  {k}:\n")
                            for gk, gv in v.items():
                                f.write(f"    {gk}: {gv:.4f}\n")
                        else:
                            f.write(f"  {k}: {v}\n")
                    
                    f.write("\nAll Coefficients:\n")
                    for var, coef_info in did_results['coefficients'].items():
                        f.write(f"  {var}:\n")
                        f.write(f"    Coefficient: {coef_info['coefficient']:.4f}\n")
                        f.write(f"    P-value: {coef_info['p_value']:.4f}\n")
                
                logger.info(f"Saved DiD results to {did_output}")
                
            except Exception as e:
                logger.error(f"DiD analysis failed: {e}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("CAUSAL INFERENCE SUMMARY")
        logger.info("=" * 60)
        
        if 'psm_results' in locals():
            logger.info("\nPropensity Score Matching (PSM):")
            logger.info(f"  Causal Effect: {psm_results['causal_estimate']['value']:.4f}")
            logger.info(f"  Interpretation: {psm_results['causal_estimate']['interpretation']}")
        
        if 'did_results' in locals():
            logger.info("\nDifference-in-Differences (DiD):")
            logger.info(f"  DiD Effect: {did_results['did_effect']['coefficient']:.4f} (p={did_results['did_effect']['p_value']:.4f})")
            logger.info(f"  R²: {did_results['diagnostics']['r_squared']:.4f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)
        
    except (DataValidationError, ModelTrainingError, MissingConfigurationError) as e:
        logger.error(f"Pipeline failed with error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

