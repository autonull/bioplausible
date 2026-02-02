"""
Statistical Analysis Toolkit

Provides publication-grade statistical analysis for comparing algorithms.
Includes Cohen's d, paired t-tests, confidence intervals, and automated reporting.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


class StatisticalAnalyzer:
    """
    Performs rigorous statistical analysis on experimental results.
    """

    def cohens_d(self, x: List[float], y: List[float]) -> float:
        """
        Calculate Cohen's d effect size for two independent samples.
        d = (mean(x) - mean(y)) / pooled_std
        """
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2

        if dof < 1:
            return 0.0

        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)
        pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / dof)

        if pooled_std == 0:
            return 0.0

        return (np.mean(x) - np.mean(y)) / pooled_std

    def confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for the mean.
        """
        n = len(data)
        if n < 2:
            return (np.mean(data), np.mean(data))

        m, se = np.mean(data), stats.sem(data)
        h = se * stats.t.ppf((1 + confidence) / 2., n-1)
        return m - h, m + h

    def interpret_d(self, d: float) -> str:
        """Interpret Cohen's d magnitude."""
        d = abs(d)
        if d < 0.2: return "negligible"
        if d < 0.5: return "small"
        if d < 0.8: return "medium"
        return "large"

    def interpret_p(self, p: float) -> str:
        """Interpret p-value significance."""
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    def compare_algorithms(
        self,
        results_a: List[float],
        results_b: List[float],
        names: Tuple[str, str] = ("Algorithm A", "Algorithm B"),
        paired: bool = False
    ) -> Dict[str, Union[str, float]]:
        """
        Compare two sets of results and generate a statistical report.

        Args:
            results_a: List of scores for algorithm A
            results_b: List of scores for algorithm B
            names: Names of the algorithms
            paired: Whether samples are paired (e.g. same seeds)

        Returns:
            Dictionary containing stats and a formatted markdown report.
        """
        if len(results_a) < 2 or len(results_b) < 2:
            return {"report": "Insufficient data for statistical analysis."}

        # Descriptive stats
        mean_a, mean_b = np.mean(results_a), np.mean(results_b)
        std_a, std_b = np.std(results_a, ddof=1), np.std(results_b, ddof=1)
        ci_a = self.confidence_interval(results_a)
        ci_b = self.confidence_interval(results_b)

        # Effect size
        d = self.cohens_d(results_a, results_b)
        effect_size_desc = self.interpret_d(d)

        # Hypothesis testing
        if paired:
            # Check length equality
            min_len = min(len(results_a), len(results_b))
            t_stat, p_val = stats.ttest_rel(results_a[:min_len], results_b[:min_len])
            test_type = "Paired t-test"
        else:
            t_stat, p_val = stats.ttest_ind(results_a, results_b, equal_var=False)
            test_type = "Welch's t-test"

        significance = self.interpret_p(p_val)

        # Generate Report
        report = f"""
### Statistical Comparison: {names[0]} vs {names[1]}

| Metric | {names[0]} | {names[1]} |
|--------|------------|------------|
| Mean   | {mean_a:.4f} | {mean_b:.4f} |
| Std Dev| {std_a:.4f} | {std_b:.4f} |
| 95% CI | [{ci_a[0]:.4f}, {ci_a[1]:.4f}] | [{ci_b[0]:.4f}, {ci_b[1]:.4f}] |
| N      | {len(results_a)} | {len(results_b)} |

**Analysis**:
*   **Test**: {test_type}
*   **t-statistic**: {t_stat:.4f}
*   **p-value**: {p_val:.4e} ({significance})
*   **Effect Size (Cohen's d)**: {d:.4f} ({effect_size_desc})

**Conclusion**:
The difference is statistically **{'significant' if p_val < 0.05 else 'not significant'}** (p < 0.05).
{names[0] if mean_a > mean_b else names[1]} performs better on average with a {effect_size_desc} effect size.
"""

        return {
            "mean_a": mean_a, "mean_b": mean_b,
            "std_a": std_a, "std_b": std_b,
            "ci_a": ci_a, "ci_b": ci_b,
            "t_stat": t_stat, "p_val": p_val,
            "cohens_d": d,
            "report": report
        }
