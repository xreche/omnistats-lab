"""Multi-touch attribution using Markov chains."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from collections import defaultdict

from src.utils.exceptions import DataValidationError, ModelTrainingError
from src.data.validators import validate_required_columns

logger = logging.getLogger("omnistats")


def build_markov_chains(
    customer_journeys: pd.DataFrame,
    touchpoint_col: str = 'touchpoint',
    conversion_col: str = 'conversion',
    customer_id_col: str = 'customer_id',
    order_col: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Build Markov chain transition probabilities from customer journeys.
    
    Args:
        customer_journeys: DataFrame with customer touchpoint sequences
        touchpoint_col: Name of touchpoint column
        conversion_col: Name of conversion indicator column (0/1)
        customer_id_col: Name of customer ID column
        order_col: Optional column indicating touchpoint order
    
    Returns:
        Tuple of:
        - Transition matrix: Dict[from_state][to_state] = probability
        - Removal effects: Dict[touchpoint] = removal effect
    """
    try:
        validate_required_columns(
            customer_journeys,
            [customer_id_col, touchpoint_col, conversion_col],
            "customer_journeys"
        )
        
        # Group by customer to get journeys
        journeys = customer_journeys.groupby(customer_id_col).agg({
            touchpoint_col: list,
            conversion_col: 'max'
        }).reset_index()
        
        # Build transition counts
        transition_counts = defaultdict(lambda: defaultdict(int))
        state_counts = defaultdict(int)
        
        for _, journey in journeys.iterrows():
            touchpoints = journey[touchpoint_col]
            converted = journey[conversion_col]
            
            # Add start state
            prev_state = 'start'
            state_counts[prev_state] += 1
            
            for touchpoint in touchpoints:
                current_state = str(touchpoint)
                state_counts[current_state] += 1
                
                # Count transition
                transition_counts[prev_state][current_state] += 1
                prev_state = current_state
            
            # Add conversion state
            if converted:
                transition_counts[prev_state]['conversion'] += 1
                state_counts['conversion'] += 1
            else:
                transition_counts[prev_state]['null'] += 1
                state_counts['null'] += 1
        
        # Calculate transition probabilities
        transition_matrix = {}
        for from_state, to_states in transition_counts.items():
            total = sum(to_states.values())
            transition_matrix[from_state] = {
                to_state: count / total
                for to_state, count in to_states.items()
            }
        
        logger.info(
            f"Built Markov chain with {len(transition_matrix)} states "
            f"and {sum(len(v) for v in transition_matrix.values())} transitions"
        )
        
        return transition_matrix, dict(state_counts)
        
    except Exception as e:
        error_msg = f"Error building Markov chains: {str(e)}"
        logger.error(error_msg)
        raise DataValidationError(error_msg) from e


def calculate_removal_effect(
    transition_matrix: Dict[str, Dict[str, float]],
    state_counts: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate removal effect for each touchpoint using Markov chain.
    
    Removal effect = (Total conversions - Conversions without touchpoint) / Total conversions
    
    Args:
        transition_matrix: Markov chain transition probabilities
        state_counts: Count of each state
    
    Returns:
        Dictionary mapping touchpoints to removal effects
    """
    # Get total conversion probability from start
    def get_conversion_probability(state: str, visited: set) -> float:
        """Recursively calculate conversion probability."""
        if state == 'conversion':
            return 1.0
        if state == 'null' or state in visited:
            return 0.0
        
        visited.add(state)
        prob = 0.0
        
        if state in transition_matrix:
            for next_state, transition_prob in transition_matrix[state].items():
                prob += transition_prob * get_conversion_probability(
                    next_state, visited.copy()
                )
        
        return prob
    
    # Calculate baseline conversion rate
    baseline_rate = get_conversion_probability('start', set())
    
    # Calculate removal effect for each touchpoint
    removal_effects = {}
    touchpoints = [
        state for state in transition_matrix.keys()
        if state not in ['start', 'conversion', 'null']
    ]
    
    for touchpoint in touchpoints:
        # Create modified transition matrix without this touchpoint
        modified_matrix = {}
        for from_state, to_states in transition_matrix.items():
            if from_state == touchpoint:
                continue  # Remove transitions from this touchpoint
            modified_matrix[from_state] = {
                to: prob for to, prob in to_states.items()
                if to != touchpoint  # Remove transitions to this touchpoint
            }
        
        # Recalculate conversion rate without this touchpoint
        modified_rate = get_conversion_probability('start', set())
        
        # Removal effect
        if baseline_rate > 0:
            removal_effect = (baseline_rate - modified_rate) / baseline_rate
        else:
            removal_effect = 0.0
        
        removal_effects[touchpoint] = removal_effect
    
    return removal_effects


def calculate_markov_attribution(
    customer_journeys: pd.DataFrame,
    touchpoint_col: str = 'touchpoint',
    conversion_col: str = 'conversion',
    customer_id_col: str = 'customer_id',
    total_conversions: Optional[int] = None
) -> Dict:
    """
    Calculate multi-touch attribution using Markov chain model.
    
    Args:
        customer_journeys: DataFrame with customer touchpoint sequences
        touchpoint_col: Name of touchpoint column
        conversion_col: Name of conversion indicator column
        customer_id_col: Name of customer ID column
        total_conversions: Total number of conversions (if None, calculated from data)
    
    Returns:
        Dictionary with:
        - attribution_scores: Attribution scores per touchpoint
        - removal_effects: Removal effects per touchpoint
        - transition_matrix: Markov chain transition probabilities
        - model_summary: Summary statistics
    """
    try:
        logger.info("Calculating Markov chain attribution...")
        
        # Build Markov chains
        transition_matrix, state_counts = build_markov_chains(
            customer_journeys,
            touchpoint_col,
            conversion_col,
            customer_id_col
        )
        
        # Calculate removal effects
        removal_effects = calculate_removal_effect(
            transition_matrix,
            state_counts
        )
        
        # Calculate total conversions
        if total_conversions is None:
            total_conversions = int(
                customer_journeys.groupby(customer_id_col)[conversion_col].max().sum()
            )
        
        # Calculate attribution scores
        # Attribution = Removal Effect × Total Conversions
        attribution_scores = {
            touchpoint: removal_effect * total_conversions
            for touchpoint, removal_effect in removal_effects.items()
        }
        
        # Normalize to sum to total conversions
        total_attributed = sum(attribution_scores.values())
        if total_attributed > 0:
            normalization_factor = total_conversions / total_attributed
            attribution_scores = {
                tp: score * normalization_factor
                for tp, score in attribution_scores.items()
            }
        
        # Model summary
        model_summary = {
            'total_conversions': total_conversions,
            'n_touchpoints': len(attribution_scores),
            'n_customers': customer_journeys[customer_id_col].nunique(),
            'n_journeys': len(customer_journeys),
            'baseline_conversion_rate': (
                total_conversions / customer_journeys[customer_id_col].nunique()
                if customer_journeys[customer_id_col].nunique() > 0 else 0
            )
        }
        
        results = {
            'attribution_scores': attribution_scores,
            'removal_effects': removal_effects,
            'transition_matrix': transition_matrix,
            'model_summary': model_summary
        }
        
        logger.info(
            f"Attribution calculated for {len(attribution_scores)} touchpoints. "
            f"Total conversions: {total_conversions}"
        )
        
        return results
        
    except Exception as e:
        error_msg = f"Error calculating Markov attribution: {str(e)}"
        logger.error(error_msg)
        raise ModelTrainingError(error_msg) from e

