"""
Cross-Domain Benchmark Suite for Phase 3 Validation.

Provides unified benchmarking across all domains:
- Vision (MNIST, CIFAR-10)
- Language Modeling (Tiny Shakespeare)
- Reinforcement Learning
- Graph
- Time Series
- Tabular
- Scientific Simulation

Integrates with KnowledgeBase for persistent storage and LeaderboardGenerator.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.domains import (GraphTask, LMTask, RLTask, ScientificTask,
                                  TabularTask, TimeSeriesTask, VisionTask)
from bioplausible.evaluation.base import BenchmarkResult
from bioplausible.knowledge import KnowledgeBase, KnowledgeEntry
from bioplausible.leaderboard.generator import (LeaderboardEntry,
                                                LeaderboardGenerator)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSuiteConfig:
    """Configuration for running the benchmark suite."""

    models: Optional[List[str]] = None
    tasks: Optional[List[str]] = None
    quick_mode: bool = False
    intermediate_mode: bool = False
    device: str = "auto"
    track_energy: bool = True
    max_batches: int = 100
    epochs: int = 5
    batch_size: int = 64
    output_dir: str = "benchmark_results"


@dataclass
class BenchmarkSuiteResult:
    """Results from running the benchmark suite."""

    config: BenchmarkSuiteConfig
    results: List[BenchmarkResult] = field(default_factory=list)
    leaderboard_entries: List[LeaderboardEntry] = field(default_factory=list)
    total_time_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "n_results": len(self.results),
            "total_time_s": self.total_time_s,
            "results": [r.to_dict() for r in self.results],
        }


class CrossDomainBenchmarkSuite:
    """
    Unified benchmark suite running across all domains.

    Integrates with KnowledgeBase to store results and with LeaderboardGenerator
    to produce public rankings.
    """

    def __init__(
        self,
        kb: Optional[KnowledgeBase] = None,
        leaderboard: Optional[LeaderboardGenerator] = None,
        output_dir: str = "benchmark_results",
    ):
        self.kb = kb or KnowledgeBase()
        self.leaderboard = leaderboard or LeaderboardGenerator(output_dir=output_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_benchmark_tasks(self) -> Dict[str, Any]:
        """Get all available benchmark tasks by domain."""
        tasks = {
            "vision": ["mnist", "fashion_mnist"],
            "lm": ["char_ngram"],
            "tabular": ["synthetic_classification"],
            "timeseries": ["synthetic_forecast"],
            "graph": ["synthetic_graph"],
            "scientific": ["synthetic_physics"],
            "rl": ["cartpole"],
        }
        return tasks

    def create_task(self, domain: str, name: str, **kwargs) -> Optional[Any]:
        """Create a domain task by name."""
        task_map = {
            "vision": VisionTask,
            "lm": LMTask,
            "rl": RLTask,
            "graph": GraphTask,
            "tabular": TabularTask,
            "timeseries": TimeSeriesTask,
            "scientific": ScientificTask,
        }
        task_cls = task_map.get(domain)
        if task_cls is None:
            logger.warning(f"Unknown domain: {domain}")
            return None
        try:
            task = task_cls(name=name, **kwargs)
            task.setup()
            return task
        except Exception as e:
            logger.warning(f"Failed to create task {domain}/{name}: {e}")
            return None

    def get_models_for_domain(self, domain: str) -> List[str]:
        """Get models compatible with a domain from registry."""
        domain_enum = {
            "vision": "vision",
            "lm": "lm",
            "rl": "rl",
            "graph": "graph",
            "tabular": "tabular",
            "timeseries": "timeseries",
            "scientific": "scientific",
        }
        domain_val = domain_enum.get(domain)

        if domain_val:
            models = Registry.query(category=ComponentCategory.MODEL, domain=domain_val)
            return [m["name"] for m in models]

        return []

    def run_model_on_task(
        self,
        model_name: str,
        task,
        epochs: int = 5,
        batch_size: int = 64,
        device: str = "cpu",
        track_energy: bool = False,
    ) -> Optional[BenchmarkResult]:
        """Run a single model on a task and return benchmark result."""
        from bioplausible.core.trainer import CoreTrainer, TrainerConfig

        try:
            config = TrainerConfig(
                model=model_name,
                task=task.name,
                epochs=epochs,
                batch_size=batch_size,
                device=device,
                track_energy=track_energy,
                val_batches=20,
            )

            trainer = CoreTrainer(config)
            trainer._setup_data()

            model = trainer.model
            if model is None:
                model = Registry.get(ComponentCategory.MODEL, model_name)()

            model = model.to(trainer.device)

            history = trainer.fit()

            if history:
                final = history[-1]
                result = BenchmarkResult(
                    model_name=model_name,
                    task_name=task.name,
                    metrics={
                        "accuracy": final.val_accuracy or 0.0,
                        "loss": final.val_loss or float("inf"),
                    },
                    params_count=sum(p.numel() for p in model.parameters()),
                    metadata={
                        "epochs": len(history),
                        "train_accuracy": final.train_accuracy,
                        "energy_proxy": final.energy_proxy,
                    },
                )
                return result

        except Exception as e:
            logger.error(f"Failed to run {model_name} on {task.name}: {e}")

        return None

    def run_suite(
        self,
        config: BenchmarkSuiteConfig,
    ) -> BenchmarkSuiteResult:
        """Run the full benchmark suite."""
        start_time = time.time()
        results: List[BenchmarkResult] = []
        entries: List[LeaderboardEntry] = []

        tasks = config.tasks or list(self.get_benchmark_tasks().keys())
        device = config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        for domain in tasks:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running benchmarks for {domain.upper()} domain")
            logger.info(f"{'=' * 60}")

            task_names = self.get_benchmark_tasks().get(domain, [])
            for task_name in task_names:
                task = self.create_task(domain, task_name)
                if task is None:
                    continue

                model_names = config.models or self.get_models_for_domain(domain)

                for model_name in model_names:
                    logger.info(f"  Testing {model_name} on {task_name}...")

                    result = self.run_model_on_task(
                        model_name=model_name,
                        task=task,
                        epochs=config.epochs,
                        batch_size=config.batch_size,
                        device=device,
                        track_energy=config.track_energy,
                    )

                    if result is not None:
                        results.append(result)

                        try:
                            meta = Registry.get_metadata(
                                ComponentCategory.MODEL, model_name
                            )
                            entry = LeaderboardEntry(
                                rank=0,
                                model=model_name,
                                propagator=meta.tags[0] if meta.tags else None,
                                optimizer="adam",
                                task=task_name,
                                accuracy=result.metrics.get("accuracy", 0.0),
                                loss=result.metrics.get("loss", float("inf")),
                                bio_plausibility_score=meta.bio_plausibility_score,
                                requires_backward=meta.requires_backward,
                                params=result.params_count or 0,
                                energy_proxy=result.metadata.get("energy_proxy"),
                            )
                            entries.append(entry)
                            self.leaderboard.add_result(entry)

                            self._store_in_kb(model_name, task_name, result)
                        except Exception as e:
                            logger.warning(f"Failed to create leaderboard entry: {e}")

        total_time = time.time() - start_time

        return BenchmarkSuiteResult(
            config=config,
            results=results,
            leaderboard_entries=entries,
            total_time_s=total_time,
        )

    def _store_in_kb(
        self, model_name: str, task_name: str, result: BenchmarkResult
    ) -> None:
        """Store benchmark result in KnowledgeBase."""
        entry = KnowledgeEntry(
            id=f"BENCH-{model_name}-{task_name}",
            topic="Benchmark",
            model_family=model_name,
            finding=f"Benchmark on {task_name}",
            details=f"Accuracy: {result.metrics.get('accuracy', 0):.4f}",
            confidence=1.0,
            tags=["benchmark", task_name, model_name],
            source="benchmark",
            metrics=result.metrics,
        )
        self.kb.add_entry(entry)

    def save_results(
        self, suite_result: BenchmarkSuiteResult, path: Optional[str] = None
    ) -> str:
        """Save benchmark results to JSON."""
        save_path = Path(path or self.output_dir / "suite_results.json")
        with open(save_path, "w") as f:
            json.dump(suite_result.to_dict(), f, indent=2, default=str)
        logger.info(f"Results saved: {save_path}")
        return str(save_path)

    def generate_leaderboard(self, path: Optional[str] = None) -> str:
        """Generate and save the leaderboard."""
        return self.leaderboard.save(path)


def run_cross_domain_benchmark(
    quick_mode: bool = True,
    models: Optional[List[str]] = None,
    output_dir: str = "benchmark_results",
) -> BenchmarkSuiteResult:
    """Convenience function to run cross-domain benchmark suite."""
    config = BenchmarkSuiteConfig(
        models=models,
        quick_mode=quick_mode,
        output_dir=output_dir,
        epochs=3 if quick_mode else 10,
        max_batches=20 if quick_mode else 100,
    )
    suite = CrossDomainBenchmarkSuite(output_dir=output_dir)
    return suite.run_suite(config)


__all__ = [
    "BenchmarkSuiteConfig",
    "BenchmarkSuiteResult",
    "CrossDomainBenchmarkSuite",
    "run_cross_domain_benchmark",
]
