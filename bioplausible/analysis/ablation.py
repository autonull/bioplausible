import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from bioplausible.config.schema import RunConfig
from bioplausible.core.trainer import run_from_runconfig as run_from_config


class AblationStudy:
    """
    Systematic parameter sensitivity study framework.
    """

    def __init__(self, base_cfg: RunConfig, dimensions: dict[str, list[Any]]):
        self.base_cfg = base_cfg
        self.dimensions = dimensions
        self.results = None

    def _generate_configs(self) -> list[tuple]:
        """Generate configurations based on Cartesian product of all dimensions."""
        keys = list(self.dimensions.keys())
        values = list(self.dimensions.values())

        configs = []
        for combo in product(*values):
            cfg = copy.deepcopy(self.base_cfg)
            params = dict(zip(keys, combo))

            # Apply overrides based on parameter name
            for k, v in params.items():
                if k == "learning_rate":
                    cfg.optimizer.lr = v
                elif k == "model_depth":
                    cfg.model.num_layers = v
                elif k == "hidden_dim":
                    cfg.model.hidden_dim = v
                elif k == "eq_steps":
                    if not cfg.model.extra:
                        cfg.model.extra = {}
                    cfg.model.extra["max_steps"] = v
                elif k == "beta":
                    cfg.optimizer.beta = v
                elif k == "sparsity_target":
                    if not cfg.model.extra:
                        cfg.model.extra = {}
                    cfg.model.extra["sparsity_target"] = v
                elif k == "data_fraction":
                    cfg.data.data_fraction = v
                elif k == "spectral_bound_gamma":
                    if not cfg.model.extra:
                        cfg.model.extra = {}
                    cfg.model.extra["spectral_bound_gamma"] = v
                else:
                    raise ValueError(f"Unknown ablation dimension: {k}")

            configs.append((params, cfg))
        return configs

    def _run_single_experiment(self, params_and_cfg: tuple) -> dict:
        params, cfg = params_and_cfg

        try:
            import warnings

            # Avoid memory efficient/looped mlp num_layers conflict
            spec = get_model_spec(cfg.model.name)
            if spec and spec.model_type in [
                "eqprop_mlp",
                "memory_efficient_mlp",
                "backprop_mlp",
                "looped_mlp",
            ]:
                if hasattr(cfg.model, "num_layers"):
                    delattr(cfg.model, "num_layers")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = run_from_config(cfg)

            score = float(res.get("final_val_accuracy", 0.0))
            return {**params, "success": True, "val_accuracy": score}
        except Exception as e:
            return {**params, "success": False, "val_accuracy": 0.0, "error": str(e)}

    def run(self, parallel_workers: int = 4) -> pd.DataFrame:
        """Run all experiments defined in the continuous parameter space."""
        configs = self._generate_configs()
        results_list = []

        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(self._run_single_experiment, c) for c in configs]
            for future in tqdm(
                as_completed(futures), total=len(configs), desc="Ablation runs"
            ):
                results_list.append(future.result())

        self.results = pd.DataFrame(results_list)
        return self.results

    def plot_sensitivity_heatmap(
        self, param1: str, param2: str, metric: str = "val_accuracy"
    ) -> plt.Figure:
        """Plot a heatmap of the sensitivity with respect to two dimensions."""
        if self.results is None or self.results.empty:
            raise ValueError("No results to plot. Call run() first.")

        if param1 not in self.results.columns or param2 not in self.results.columns:
            raise ValueError(
                f"Parameters {param1} and {param2} must be valid ablation dimensions."
            )

        pivot_table = self.results.pivot_table(
            values=metric, index=param1, columns=param2, aggfunc=np.mean
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot_table, annot=True, cmap="viridis", ax=ax, fmt=".3f")
        ax.set_title(f"Sensitivity Heatmap: {param1} vs {param2}")
        fig.tight_layout()
        return fig

    def identify_critical_hyperparams(self) -> list[str]:
        """Identify critical hyperparameters using the variance of the mean outcomes."""
        if self.results is None or self.results.empty:
            raise ValueError("No results to analyze. Call run() first.")

        variances = []
        for col in self.dimensions.keys():
            if col in self.results.columns:
                mean_per_val = self.results.groupby(col)["val_accuracy"].mean()
                if not mean_per_val.empty:
                    variances.append((col, mean_per_val.var()))

        # Sort by variance descending
        variances.sort(key=lambda x: x[1] if pd.notna(x[1]) else 0.0, reverse=True)
        return [col for col, var in variances]
