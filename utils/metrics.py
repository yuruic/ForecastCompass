"""
Scoring metrics for evaluating probabilistic forecasts: Brier score and
Expected Calibration Error (ECE).
"""
from typing import Dict, List

import numpy as np


def calculate_brier_score(predictions: Dict[str, float], outcomes: Dict[str, int]) -> float:
    """
    Calculate Brier score for a single prediction.

    Args:
        predictions: Dictionary mapping option names to predicted probabilities (0-1)
        outcomes: Dictionary mapping option names to actual outcomes (0 or 1)

    Returns:
        Brier score (lower is better, range 0-2 for multi-class, 0-1 for binary/single)
    """
    if not predictions or not outcomes:
        raise ValueError("Predictions and outcomes cannot be empty")

    predictions = {str(k): v for k, v in predictions.items()}
    outcomes = {str(k): v for k, v in outcomes.items()}

    pred_keys = set(predictions.keys())
    outcome_keys = set(outcomes.keys())

    if pred_keys != outcome_keys:
        missing_in_pred = outcome_keys - pred_keys
        missing_in_outcome = pred_keys - outcome_keys
        raise ValueError(
            f"Mismatch between prediction and outcome keys. "
            f"Missing in predictions: {missing_in_pred}, "
            f"Missing in outcomes: {missing_in_outcome}"
        )

    score = 0.0
    for option in predictions:
        predicted_prob = predictions[option]
        actual_outcome = outcomes[option]
        score += (predicted_prob - actual_outcome) ** 2

    if len(predictions) == 1:
        return score
    else:
        return score / len(predictions)


def calculate_average_brier_score(results: List[Dict]) -> float:
    """
    Calculate average Brier score across multiple predictions.

    Args:
        results: List of result dictionaries, each containing 'brier_score'

    Returns:
        Average Brier score
    """
    if not results:
        return 0.0

    scores = [r.get('brier_score', 0.0) for r in results if 'brier_score' in r]
    if not scores:
        return 0.0

    return float(np.mean(scores))


def rank_by_brier_score(results: List[Dict], percentile: float = 0.2) -> tuple:
    """
    Rank results by Brier score and return bottom percentile (worst cases).

    Args:
        results: List of result dictionaries with 'brier_score'
        percentile: Bottom percentile to return (default 0.2 = worst 20%)

    Returns:
        (bad_cases, good_cases) - tuple of lists
    """
    valid_results = [r for r in results if 'brier_score' in r and r['brier_score'] is not None]
    if not valid_results:
        return [], []

    # Sort by brier score descending (worst first)
    sorted_results = sorted(valid_results, key=lambda x: x['brier_score'], reverse=True)

    # Get bottom percentile
    cutoff_idx = max(1, int(len(sorted_results) * percentile))
    bad_cases = sorted_results[:cutoff_idx]
    good_cases = sorted_results[cutoff_idx:]

    return bad_cases, good_cases


def compute_ece(probs: list[float], outcomes: list[int], n_bins: int = 10) -> float:
    """
    Compute the empirical Expected Calibration Error (ECE).

    ECE = (1/m) * sum_b [ m_b * |o_hat_b - p_hat_b| ]

    where for each bin b:
      - m_b    = number of samples in the bin
      - o_hat_b = mean observed outcome (true frequency)
      - p_hat_b = mean predicted probability

    Args:
        probs:    Predicted probabilities in [0, 1].
        outcomes: Binary outcomes (0 or 1).
        n_bins:   Number of equal-width bins (default 10).

    Returns:
        ECE value.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    m = len(probs)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge only for the last bin so [0,1] is fully covered.
        if i < n_bins - 1:
            mask = (probs >= lo) & (probs < hi)
        else:
            mask = (probs >= lo) & (probs <= hi)

        m_b = mask.sum()
        if m_b == 0:
            continue

        p_hat_b = probs[mask].mean()
        o_hat_b = outcomes[mask].mean()
        ece += m_b * abs(o_hat_b - p_hat_b)

    return ece / m
