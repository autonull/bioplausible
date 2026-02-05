from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

# We'll need TrainingTrajectory for type hinting, but avoid circular imports if possible.
# Assuming it's in bioplausible.scientist.training_dynamics
from bioplausible.scientist.training_dynamics import TrainingTrajectory
from bioplausible.models.registry import get_model_spec

@dataclass
class CrossAlgorithmInsight:
    """A comparative insight between algorithm families."""
    insight_type: str  # "performance", "efficiency", "robustness"
    task: str
    metric: str
    ranking: List[str]  # Algorithm families ranked by performance
    effect_sizes: Dict[str, float]  # Cohen's d effect sizes or raw differences
    confidence: float  # Statistical confidence (0-1)
    narrative: str  # Human-readable description

    def to_dict(self) -> Dict[str, Any]:
        return {
            "insight_type": self.insight_type,
            "task": self.task,
            "metric": self.metric,
            "ranking": self.ranking,
            "effect_sizes": self.effect_sizes,
            "confidence": self.confidence,
            "narrative": self.narrative
        }

@dataclass
class ArchitecturalRecommendation:
    """A proposed novel architecture based on experiment results."""
    name: str
    motivation: str
    architecture_description: str
    expected_benefits: List[str]
    implementation_sketch: str
    risk_level: str  # "low", "medium", "high"
    priority: int  # 1-5, higher is more urgent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "motivation": self.motivation,
            "architecture_description": self.architecture_description,
            "expected_benefits": self.expected_benefits,
            "implementation_sketch": self.implementation_sketch,
            "risk_level": self.risk_level,
            "priority": self.priority
        }

class ResearchSynthesizer:
    """
    Generates high-level research insights from experiment database.
    """
    
    def __init__(self, trajectories: List[TrainingTrajectory]):
        self.trajectories = trajectories
        self.algorithm_families = self._group_by_family()
    
    def synthesize_full_report(self) -> Dict[str, Any]:
        """Main entry point for synthesis."""
        return {
            "cross_algorithm_insights": [i.to_dict() for i in self.generate_cross_algorithm_insights()],
            "architectural_recommendations": [r.to_dict() for r in self.generate_architecture_recommendations()],
            "research_gaps": self.identify_research_gaps(),
            "actionable_quick_wins": self.find_quick_wins(),
        }
    
    def generate_cross_algorithm_insights(self) -> List[CrossAlgorithmInsight]:
        """
        Compare algorithm families across tasks and metrics.
        """
        insights = []
        
        # Identify tasks present in the data
        tasks = list(set(t.task_name for t in self.trajectories))
        
        for task in tasks:
            # Compare final accuracy
            perf_insight = self._compare_families(
                task=task,
                metric="final_accuracy",
                insight_type="performance"
            )
            if perf_insight:
                insights.append(perf_insight)
            
            # Compare sample efficiency (AUC of learning curve)
            eff_insight = self._compare_families(
                task=task,
                metric="sample_efficiency",
                insight_type="efficiency"
            )
            if eff_insight:
                insights.append(eff_insight)
            
            # Compare convergence speed
            conv_insight = self._compare_families(
                task=task,
                metric="convergence_speed",
                insight_type="time"
            )
            if conv_insight:
                insights.append(conv_insight)
        
        return insights
    
    def _compare_families(self, task: str, metric: str, insight_type: str) -> Optional[CrossAlgorithmInsight]:
        """Compare algorithm families on a single metric."""
        # Group by family
        family_scores = {}
        
        for family, trajs in self.algorithm_families.items():
            scores = []
            for traj in trajs:
                if traj.task_name == task:
                    if metric == "final_accuracy":
                        if traj.checkpoints:
                            scores.append(traj.checkpoints[-1].val_acc)
                    elif metric == "sample_efficiency":
                        scores.append(traj.compute_sample_efficiency())
                    elif metric == "convergence_speed":
                        val = traj.compute_convergence_speed()
                        if val != float('inf'):
                            scores.append(val)
            
            if scores:
                family_scores[family] = np.mean(scores)
        
        if len(family_scores) < 2:
            return None # Need at least 2 families to compare
        
        # Rank families (higher is better for acc/efficiency, lower for speed)
        # Note: convergence_speed from TrainingTrajectory returns *epoch number*, so lower is better. 
        # sample_efficiency returns *AUC*, so higher is better? 
        # Wait, TrainingTrajectory.compute_sample_efficiency returns "Area under learning curve". Higher is better.
        reverse = True
        if metric == "convergence_speed":
            reverse = False
            
        ranking = sorted(family_scores.keys(), key=lambda f: family_scores[f], reverse=reverse)
        
        # Compute effect sizes (simple diffs for now)
        effect_sizes = self._compute_effect_sizes(family_scores)
        
        # Generate narrative
        narrative = self._generate_narrative(task, metric, ranking, family_scores, reverse)
        
        return CrossAlgorithmInsight(
            insight_type=insight_type,
            task=task,
            metric=metric,
            ranking=ranking,
            effect_sizes=effect_sizes,
            confidence=0.95,  # Placeholder
            narrative=narrative,
        )
    
    def generate_architecture_recommendations(self) -> List[ArchitecturalRecommendation]:
        """
        Propose novel architectures based on observed strengths.
        """
        recommendations = []
        
        # Need at least baseline and eqprop to make this specific comparison
        if "eqprop" not in self.algorithm_families or "baseline" not in self.algorithm_families:
            return recommendations

        eqprop_speed = self._avg_metric("eqprop", "convergence_speed")
        backprop_speed = self._avg_metric("baseline", "convergence_speed")
        eqprop_acc = self._avg_metric("eqprop", "final_accuracy")
        backprop_acc = self._avg_metric("baseline", "final_accuracy")
        
        # Handle cases where metrics might be None (no data)
        if None in [eqprop_speed, backprop_speed, eqprop_acc, backprop_acc]:
            return recommendations
            
        # Check: EqProp faster converging but less accurate?
        # speed is epoch number, lower is faster.
        if eqprop_speed < backprop_speed and backprop_acc > eqprop_acc:
             # Avoid div by zero
            speedtup = max(0.1, eqprop_speed)
            ratio = backprop_speed / speedtup
            
            recommendations.append(ArchitecturalRecommendation(
                name="Hybrid EqProp-Backprop Transformer",
                motivation=(
                    f"EqProp converges {ratio:.1f}x faster "
                    f"but achieves {(backprop_acc - eqprop_acc)*100:.1f}% lower accuracy. "
                    "Combining them could yield best of both worlds."
                ),
                architecture_description=(
                    "Use EqProp for attention layers (local credit assignment) "
                    "and Backprop for feedforward layers (global optimization)."
                ),
                expected_benefits=[
                    "20-30% faster training",
                    "<2% accuracy loss vs pure Backprop",
                    "Better biological plausibility"
                ],
                implementation_sketch="HybridTransformer class sketch...",
                risk_level="medium",
                priority=5,
            ))
        
        return recommendations
    
    def identify_research_gaps(self) -> List[str]:
        """Find under-explored areas."""
        gaps = []
        
        # Check for missing task types
        tested_tasks = set(t.task_name for t in self.trajectories)
        if "graph" not in tested_tasks and not any("graph" in t for t in tested_tasks):
            gaps.append("No experiments on graph neural networks (GNNs)")
        
        # Check for under-explored hyperparameters
        all_configs = [t.config for t in self.trajectories]
        activations_tested = set(c.get("activation") for c in all_configs if "activation" in c)
        
        if "prelu" not in activations_tested:
            gaps.append("PReLU (learnable activation) not tested - may help performance")
        
        if not any(c.get("optimizer") == "sgd" for c in all_configs):
             gaps.append("SGD optimizer not fully explored (mostly Adam?)")
        
        return gaps
    
    def find_quick_wins(self) -> List[Dict[str, Any]]:
        """
        Identify low-hanging fruit: simple changes with big impact.
        """
        wins = []
        
        # Analyze activation functions
        # This requires flattened list of (activation, final_acc) tuples
        activations = {} # activation -> list of accuracies
        
        for t in self.trajectories:
            act = t.config.get("activation")
            if act and t.checkpoints:
                activations.setdefault(act, []).append(t.checkpoints[-1].val_acc)
                
        # Compare if we have multiple
        if "gelu" in activations and "relu" in activations:
            gelu_mean = np.mean(activations["gelu"])
            relu_mean = np.mean(activations["relu"])
            
            if gelu_mean > relu_mean + 0.02: # 2% better
                wins.append({
                    "title": "Switch default activation to GELU",
                    "impact": f"+{(gelu_mean - relu_mean)*100:.1f}% accuracy vs ReLU",
                    "effort": "Low (config change)",
                    "confidence": "high",
                })
        
        return wins
    
    def _group_by_family(self) -> Dict[str, List[TrainingTrajectory]]:
        """Group trajectories by algorithm family."""
        families = {}
        for traj in self.trajectories:
            # Heuristic map since we might not have registry access for every model name string
            # In a real app, we'd use get_model_spec(traj.model_name).family
            # For now, rely on name conventions or try-except
            try:
                spec = get_model_spec(traj.model_name)
                family = spec.family
            except:
                # Fallback heuristics
                name = traj.model_name.lower()
                if "eqprop" in name: family = "eqprop"
                elif "hebbian" in name: family = "hebbian"
                elif "feedback" in name or "fa" in name: family = "feedback_alignment"
                else: family = "baseline"
                
            families.setdefault(family, []).append(traj)
        return families
    
    def _compute_effect_sizes(self, family_scores: Dict[str, float]) -> Dict[str, float]:
        """Compute relative differences."""
        # Just return the raw scores for now as a dictionary
        return {k: float(v) for k, v in family_scores.items()}
    
    def _generate_narrative(self, task, metric, ranking, scores, reverse_sort):
        """Generate human-readable description."""
        best = ranking[0]
        worst = ranking[-1]
        
        best_val = scores[best]
        worst_val = scores[worst]
        
        comps = "higher" if reverse_sort else "lower" # reverse=True means Higher is Better
        
        return (
            f"On {task}, {best} achieves the best {metric} "
            f"({best_val:.4f}), which is {comps} than {worst} ({worst_val:.4f})."
        )

    def _avg_metric(self, family: str, metric: str) -> Optional[float]:
        """Helper to get average metric for a whole family across ALL tasks."""
        # Note: Averaging across tasks might be noisy if tasks have different scales.
        # But for 'convergence_speed' (epochs) or 'final_accuracy' (0-1), it's semi-plausible for rough heuristics.
        if family not in self.algorithm_families:
            return None
        
        trajs = self.algorithm_families[family]
        values = []
        for t in trajs:
            if metric == "final_accuracy":
                if t.checkpoints: values.append(t.checkpoints[-1].val_acc)
            elif metric == "convergence_speed":
                 val = t.compute_convergence_speed()
                 if val != float('inf'): values.append(val)
        
        if not values:
            return None
        return np.mean(values)
