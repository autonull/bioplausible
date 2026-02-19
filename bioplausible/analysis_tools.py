"""
Bioplausible Analysis Utilities

Statistical analysis and reporting utilities for experiment results.

Features:
- Statistical significance testing
- Effect size calculation
- Confidence intervals
- Result aggregation
- Report generation
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class StatisticalComparison:
    """Results of statistical comparison between two methods."""
    method_a: str
    method_b: str
    metric: str
    
    # Method A stats
    mean_a: float
    std_a: float
    n_a: int
    
    # Method B stats
    mean_b: float
    std_b: float
    n_b: int
    
    # Test results
    t_statistic: float
    p_value: float
    cohens_d: float
    confidence_interval: Tuple[float, float]
    
    # Interpretation
    significant: bool
    effect_size: str  # 'negligible', 'small', 'medium', 'large'
    better_method: str
    
    def summary(self) -> str:
        """Get human-readable summary."""
        sig_symbol = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (
            f"{self.method_a} vs {self.method_b} ({self.metric}):\n"
            f"  {self.method_a}: {self.mean_a:.2f} ± {self.std_a:.2f} (n={self.n_a})\n"
            f"  {self.method_b}: {self.mean_b:.2f} ± {self.std_b:.2f} (n={self.n_b})\n"
            f"  t({self.n_a + self.n_b - 2:.0f}) = {self.t_statistic:.3f}, p = {self.p_value:.4f}{sig_symbol}\n"
            f"  Cohen's d = {self.cohens_d:.3f} ({self.effect_size})\n"
            f"  95% CI: [{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}]\n"
            f"  Better: {self.better_method}"
        )


@dataclass
class AnalysisReport:
    """Complete analysis report for experiment results."""
    total_experiments: int
    models_tested: List[str]
    optimizers_tested: List[str]
    
    # Best results
    best_accuracy: float
    best_optimizer: str
    best_model: str
    
    # Statistical comparisons
    comparisons: List[StatisticalComparison]
    
    # Rankings
    optimizer_ranking: List[Tuple[str, float]]
    model_ranking: List[Tuple[str, float]]
    
    # Summary statistics
    mean_accuracy: float
    std_accuracy: float
    median_accuracy: float
    
    # Recommendations
    recommendations: List[str]
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=" * 60,
            "BIPLAUSIBLE ANALYSIS REPORT",
            "=" * 60,
            f"Total Experiments: {self.total_experiments}",
            f"Models Tested: {', '.join(self.models_tested)}",
            f"Optimizers Tested: {', '.join(self.optimizers_tested)}",
            "",
            "BEST RESULTS:",
            f"  Best Accuracy: {self.best_accuracy:.2f}%",
            f"  Best Optimizer: {self.best_optimizer}",
            f"  Best Model: {self.best_model}",
            "",
            "SUMMARY STATISTICS:",
            f"  Mean Accuracy: {self.mean_accuracy:.2f} ± {self.std_accuracy:.2f}%",
            f"  Median Accuracy: {self.median_accuracy:.2f}%",
            "",
            "OPTIMIZER RANKING:",
        ]
        
        for i, (opt, acc) in enumerate(self.optimizer_ranking, 1):
            lines.append(f"  {i}. {opt}: {acc:.2f}%")
        
        if self.comparisons:
            lines.extend(["", "STATISTICAL COMPARISONS:"])
            for comp in self.comparisons[:5]:  # Show top 5
                lines.append(f"  {comp.summary().split(chr(10))[0]}")
        
        if self.recommendations:
            lines.extend(["", "RECOMMENDATIONS:"])
            for rec in self.recommendations:
                lines.append(f"  • {rec}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


class ResultAnalyzer:
    """
    Analyze experiment results statistically.
    
    Example usage:
        analyzer = ResultAnalyzer()
        
        # Add results
        analyzer.add_results(results)
        
        # Get statistical comparison
        comp = analyzer.compare_optimizers('smep', 'smep_fast')
        print(comp.summary())
        
        # Generate full report
        report = analyzer.generate_report()
        print(report.summary())
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.results = []
        self.confidence_level = confidence_level
    
    def add_result(self, result: Any) -> None:
        """Add a single experiment result."""
        self.results.append(result)
    
    def add_results(self, results: List[Any]) -> None:
        """Add multiple experiment results."""
        self.results.extend(results)
    
    def compare_optimizers(
        self,
        optimizer_a: str,
        optimizer_b: str,
        metric: str = 'val_accuracy',
    ) -> Optional[StatisticalComparison]:
        """
        Compare two optimizers statistically.
        
        Args:
            optimizer_a: First optimizer name.
            optimizer_b: Second optimizer name.
            metric: Metric to compare.
        
        Returns:
            StatisticalComparison or None if insufficient data.
        """
        values_a = [getattr(r, metric) for r in self.results if r.optimizer_name == optimizer_a]
        values_b = [getattr(r, metric) for r in self.results if r.optimizer_name == optimizer_b]
        
        if len(values_a) < 2 or len(values_b) < 2:
            return None
        
        return self._statistical_test(
            optimizer_a, values_a,
            optimizer_b, values_b,
            metric,
        )
    
    def compare_models(
        self,
        model_a: str,
        model_b: str,
        metric: str = 'val_accuracy',
    ) -> Optional[StatisticalComparison]:
        """
        Compare two models statistically.
        
        Args:
            model_a: First model name.
            model_b: Second model name.
            metric: Metric to compare.
        
        Returns:
            StatisticalComparison or None if insufficient data.
        """
        values_a = [getattr(r, metric) for r in self.results if r.model_name == model_a]
        values_b = [getattr(r, metric) for r in self.results if r.model_name == model_b]
        
        if len(values_a) < 2 or len(values_b) < 2:
            return None
        
        return self._statistical_test(
            model_a, values_a,
            model_b, values_b,
            metric,
        )
    
    def _statistical_test(
        self,
        name_a: str,
        values_a: List[float],
        name_b: str,
        values_b: List[float],
        metric: str,
    ) -> StatisticalComparison:
        """Perform statistical test between two groups."""
        mean_a = np.mean(values_a)
        std_a = np.std(values_a, ddof=1)
        n_a = len(values_a)
        
        mean_b = np.mean(values_b)
        std_b = np.std(values_b, ddof=1)
        n_b = len(values_b)
        
        # Welch's t-test (unequal variances)
        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
        cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
        
        # Confidence interval for difference
        alpha = 1 - self.confidence_level
        df = n_a + n_b - 2
        t_crit = stats.t.ppf(1 - alpha/2, df)
        se = np.sqrt(std_a**2/n_a + std_b**2/n_b)
        diff = mean_a - mean_b
        ci = (diff - t_crit * se, diff + t_crit * se)
        
        # Interpretation
        significant = p_value < alpha
        effect_size = self._interpret_cohens_d(abs(cohens_d))
        better = name_a if mean_a > mean_b else name_b
        
        return StatisticalComparison(
            method_a=name_a,
            method_b=name_b,
            metric=metric,
            mean_a=mean_a,
            std_a=std_a,
            n_a=n_a,
            mean_b=mean_b,
            std_b=std_b,
            n_b=n_b,
            t_statistic=t_stat,
            p_value=p_value,
            cohens_d=cohens_d,
            confidence_interval=ci,
            significant=significant,
            effect_size=effect_size,
            better_method=better,
        )
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return 'negligible'
        elif d < 0.5:
            return 'small'
        elif d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def get_optimizer_ranking(self, metric: str = 'val_accuracy') -> List[Tuple[str, float]]:
        """
        Get ranking of optimizers by metric.
        
        Args:
            metric: Metric to rank by.
        
        Returns:
            List of (optimizer_name, mean_metric) sorted by metric.
        """
        optimizer_metrics = {}
        
        for r in self.results:
            opt = r.optimizer_name
            if opt not in optimizer_metrics:
                optimizer_metrics[opt] = []
            optimizer_metrics[opt].append(getattr(r, metric))
        
        rankings = [
            (opt, np.mean(values))
            for opt, values in optimizer_metrics.items()
        ]
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_model_ranking(self, metric: str = 'val_accuracy') -> List[Tuple[str, float]]:
        """
        Get ranking of models by metric.
        
        Args:
            metric: Metric to rank by.
        
        Returns:
            List of (model_name, mean_metric) sorted by metric.
        """
        model_metrics = {}
        
        for r in self.results:
            model = r.model_name
            if model not in model_metrics:
                model_metrics[model] = []
            model_metrics[model].append(getattr(r, metric))
        
        rankings = [
            (model, np.mean(values))
            for model, values in model_metrics.items()
        ]
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def generate_report(self) -> AnalysisReport:
        """
        Generate comprehensive analysis report.
        
        Returns:
            AnalysisReport with all findings.
        """
        if not self.results:
            raise ValueError("No results to analyze")
        
        # Collect unique models and optimizers
        models = list(set(r.model_name for r in self.results))
        optimizers = list(set(r.optimizer_name for r in self.results))
        
        # Find best results
        best_result = max(self.results, key=lambda r: r.val_accuracy)
        
        # Summary statistics
        all_accuracies = [r.val_accuracy for r in self.results]
        
        # Generate comparisons
        comparisons = []
        for i, opt_a in enumerate(optimizers):
            for opt_b in optimizers[i+1:]:
                comp = self.compare_optimizers(opt_a, opt_b)
                if comp:
                    comparisons.append(comp)
        
        # Sort by p-value (most significant first)
        comparisons.sort(key=lambda c: c.p_value)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            self.get_optimizer_ranking(),
            self.get_model_ranking(),
            comparisons,
        )
        
        return AnalysisReport(
            total_experiments=len(self.results),
            models_tested=models,
            optimizers_tested=optimizers,
            best_accuracy=best_result.val_accuracy,
            best_optimizer=best_result.optimizer_name,
            best_model=best_result.model_name,
            comparisons=comparisons,
            optimizer_ranking=self.get_optimizer_ranking(),
            model_ranking=self.get_model_ranking(),
            mean_accuracy=np.mean(all_accuracies),
            std_accuracy=np.std(all_accuracies, ddof=1),
            median_accuracy=np.median(all_accuracies),
            recommendations=recommendations,
        )
    
    def _generate_recommendations(
        self,
        optimizer_ranking: List[Tuple[str, float]],
        model_ranking: List[Tuple[str, float]],
        comparisons: List[StatisticalComparison],
    ) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        if optimizer_ranking:
            best_opt = optimizer_ranking[0]
            recommendations.append(
                f"Best optimizer: {best_opt[0]} ({best_opt[1]:.2f}% avg accuracy)"
            )
            
            # Check for significant differences
            for comp in comparisons[:3]:
                if comp.significant and comp.effect_size in ['medium', 'large']:
                    recommendations.append(
                        f"{comp.better_method} significantly outperforms "
                        f"{comp.method_a if comp.better_method != comp.method_a else comp.method_b} "
                        f"(p={comp.p_value:.4f}, d={comp.cohens_d:.2f})"
                    )
        
        if model_ranking:
            best_model = model_ranking[0]
            recommendations.append(
                f"Best model: {best_model[0]} ({best_model[1]:.2f}% avg accuracy)"
            )
        
        # Speed recommendation
        speed_results = [(r.optimizer_name, r.steps_per_second) for r in self.results]
        if speed_results:
            fastest = max(speed_results, key=lambda x: x[1])
            recommendations.append(
                f"Fastest optimizer: {fastest[0]} ({fastest[1]:.1f} steps/s)"
            )
        
        return recommendations


def analyze_results(results: List[Any]) -> AnalysisReport:
    """
    Convenience function to analyze results.
    
    Args:
        results: List of ExperimentResult objects.
    
    Returns:
        AnalysisReport with findings.
    """
    analyzer = ResultAnalyzer()
    analyzer.add_results(results)
    return analyzer.generate_report()


__all__ = [
    'StatisticalComparison',
    'AnalysisReport',
    'ResultAnalyzer',
    'analyze_results',
]
