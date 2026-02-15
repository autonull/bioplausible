"""
Multi-Algorithm Comparison Framework

Data structures and utilities for fair comparison of bioplausible learning algorithms.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


class ComparisonMetric(Enum):
    """Primary metrics for algorithm comparison."""

    ACCURACY = "accuracy"
    PERPLEXITY = "perplexity"
    LOSS = "loss"
    PARAM_EFFICIENCY = "param_efficiency"  # accuracy / params
    TIME_EFFICIENCY = "time_efficiency"  # accuracy / time


@dataclass
class AlgorithmRanking:
    """Ranking of a single algorithm family."""

    family: str
    rank: int
    best_value: float  # Best metric value
    avg_value: float  # Average across trials
    std_value: float  # Standard deviation
    gap_to_baseline: float  # Percentage gap to baseline
    n_trials: int
    best_trial_id: int
    pareto_count: int  # Trials on Pareto frontier


@dataclass
class ComparisonStudy:
    """Multi-algorithm comparison experiment."""

    name: str
    task: str
    dataset: str
    primary_metric: ComparisonMetric

    # Algorithms to compare
    algorithms: List[str]  # Model names or families
    baseline: str  # Reference algorithm (usually "Backprop")

    # Optuna studies per algorithm
    studies: Dict[str, str] = field(default_factory=dict)  # family → study_name

    # Comparison results
    rankings: List[AlgorithmRanking] = field(default_factory=list)

    # Winner
    winner_family: Optional[str] = None
    winner_trial_id: Optional[int] = None

    # Metadata
    created_at: Optional[str] = None
    completed_at: Optional[str] = None

    def get_ranking(self, family: str) -> Optional[AlgorithmRanking]:
        """Get ranking for a specific algorithm family."""
        for ranking in self.rankings:
            if ranking.family == family:
                return ranking
        return None

    def get_gap_to_baseline(self, family: str) -> float:
        """Calculate performance gap to baseline."""
        baseline_ranking = self.get_ranking(self.baseline)
        family_ranking = self.get_ranking(family)

        if not baseline_ranking or not family_ranking:
            return float("inf")

        # For metrics where lower is better (perplexity, loss)
        if self.primary_metric in [ComparisonMetric.PERPLEXITY, ComparisonMetric.LOSS]:
            return (
                (family_ranking.best_value - baseline_ranking.best_value)
                / baseline_ranking.best_value
                * 100
            )
        else:  # Higher is better (accuracy)
            return (
                (baseline_ranking.best_value - family_ranking.best_value)
                / baseline_ranking.best_value
                * 100
            )


def compute_algorithm_rankings(
    trials_by_family: Dict[str, List[Dict]],
    metric: ComparisonMetric = ComparisonMetric.ACCURACY,
    maximize: bool = True,
) -> List[AlgorithmRanking]:
    """
    Compute rankings for each algorithm family.

    Args:
        trials_by_family: Dict mapping family name to list of trial dicts
        metric: Metric to rank by
        maximize: Whether higher is better

    Returns:
        List of AlgorithmRanking sorted by performance
    """
    metric_key = metric.value
    if metric == ComparisonMetric.PARAM_EFFICIENCY:
        # Special case: accuracy / params
        metric_key = "accuracy"  # We'll compute efficiency manually

    rankings = []

    for family, trials in trials_by_family.items():
        if not trials:
            continue

        # Extract metric values
        if metric == ComparisonMetric.PARAM_EFFICIENCY:
            values = [
                t["accuracy"] / max(t.get("param_count", 1), 0.01) for t in trials
            ]
        elif metric == ComparisonMetric.TIME_EFFICIENCY:
            values = [
                t["accuracy"] / max(t.get("iteration_time", 1), 0.001) for t in trials
            ]
        else:
            values = [t.get(metric_key, 0) for t in trials]

        # Best and average
        best_value = max(values) if maximize else min(values)
        avg_value = np.mean(values)
        std_value = np.std(values)

        # Find best trial
        best_idx = values.index(best_value)
        best_trial_id = trials[best_idx]["trial_id"]

        rankings.append(
            AlgorithmRanking(
                family=family,
                rank=0,  # Will be assigned after sorting
                best_value=best_value,
                avg_value=avg_value,
                std_value=std_value,
                gap_to_baseline=0.0,  # Computed separately
                n_trials=len(trials),
                best_trial_id=best_trial_id,
                pareto_count=0,  # Computed separately
            )
        )

    # Sort and assign ranks
    rankings.sort(key=lambda r: r.best_value, reverse=maximize)
    for idx, ranking in enumerate(rankings, 1):
        ranking.rank = idx

    return rankings


def compute_statistical_significance(
    family_a_trials: List[Dict], family_b_trials: List[Dict], metric: str = "accuracy"
) -> Tuple[float, float]:
    """
    Test if difference between two algorithm families is statistically significant.

    Args:
        family_a_trials: Trials from first family
        family_b_trials: Trials from second family
        metric: Metric to compare

    Returns:
        (t_statistic, p_value) from Welch's t-test
    """
    values_a = [t.get(metric, 0) for t in family_a_trials]
    values_b = [t.get(metric, 0) for t in family_b_trials]

    # Welch's t-test (doesn't assume equal variance)
    statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

    return statistic, p_value


def is_bio_plausible(model_name: str) -> bool:
    """Check if a model is bio-plausible (not backprop)."""
    from bioplausible.models.registry import get_model_spec

    try:
        spec = get_model_spec(model_name)
        return spec.family != "baseline"
    except (ValueError, KeyError):
        return (
            "backprop" not in model_name.lower()
            and "baseline" not in model_name.lower()
        )


def group_trials_by_family(trials: List[Dict]) -> Dict[str, List[Dict]]:
    """Group trials by algorithm family."""
    from collections import defaultdict

    from bioplausible.models.registry import get_model_spec

    grouped = defaultdict(list)

    for trial in trials:
        model_name = trial.get("model_name", "Unknown")

        try:
            spec = get_model_spec(model_name)
            family = spec.family
        except (ValueError, KeyError):
            family = model_name.split()[0].lower()

        grouped[family].append(trial)

    return dict(grouped)


def generate_comparison_summary(
    rankings: List[AlgorithmRanking], baseline: str = "baseline"
) -> str:
    """Generate human-readable comparison summary."""
    baseline_ranking = next((r for r in rankings if r.family == baseline), None)

    if not baseline_ranking:
        return "No baseline found for comparison."

    summary = f"Algorithm Comparison Summary\n"
    summary += f"{'='*50}\n\n"

    summary += f"Baseline: {baseline} (rank #{baseline_ranking.rank})\n"
    summary += f"Best value: {baseline_ranking.best_value:.4f}\n\n"

    summary += "Bio-plausible algorithms:\n"
    for ranking in rankings:
        if ranking.family == baseline:
            continue

        gap = ranking.gap_to_baseline
        gap_str = f"+{gap:.1f}%" if gap > 0 else f"{gap:.1f}%"

        summary += f"  {ranking.rank}. {ranking.family}\n"
        summary += f"     Best: {ranking.best_value:.4f} ({gap_str} vs baseline)\n"
        summary += f"     Trials: {ranking.n_trials}, Pareto: {ranking.pareto_count}\n"

    # Find closest to baseline
    bio_plausible = [r for r in rankings if r.family != baseline]
    if bio_plausible:
        best_bio = min(bio_plausible, key=lambda r: abs(r.gap_to_baseline))
        summary += f"\n✨ Best bio-plausible: {best_bio.family}\n"
        summary += f"   Gap to baseline: {abs(best_bio.gap_to_baseline):.1f}%\n"

    return summary
