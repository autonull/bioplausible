import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import List, Tuple

def _power_law(N: np.ndarray, a: float, b: float) -> np.ndarray:
    """Power law equation: L = a * N^(-b)"""
    return a * np.power(N, -b)

def fit_power_law(param_counts: List[int], losses: List[float]) -> Tuple[float, float]:
    """
    Fit L = a * N^(-b). Return (a, b). Positive b means loss decreases with scale.
    """
    N = np.array(param_counts, dtype=float)
    L = np.array(losses, dtype=float)

    # Initial guess for optimization
    p0 = [L[0] * (N[0] ** 0.1), 0.1]
    
    try:
        # Curve fit limits: a > 0, b > -1 (allowing slight divergence, but usually > 0)
        popt, _ = curve_fit(_power_law, N, L, p0=p0, bounds=([0, -1], [np.inf, 2]))
        a, b = popt
        return a, b
    except Exception as e:
        print(f"Failed to fit power law: {e}")
        return float('nan'), float('nan')

def compute_compute_optimal(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each algorithm and parameter count, find the best configuration
    (lowest loss/highest accuracy) to form the Pareto frontier.
    
    Expected DF cols: ['model', 'param_count', 'val_loss', 'val_accuracy', ...]
    """
    if 'param_count' not in results_df.columns or 'model' not in results_df.columns:
        raise ValueError("DataFrame must contain 'model' and 'param_count' columns.")
        
    metric = 'val_loss' if 'val_loss' in results_df.columns else 'val_accuracy'
    ascending = True if metric == 'val_loss' else False
    
    # Sort by metric
    sorted_df = results_df.sort_values(by=metric, ascending=ascending)
    
    # Group by model and parameter count, then take the best
    optimal_df = sorted_df.groupby(['model', 'param_count']).first().reset_index()
    return optimal_df

def plot_scaling_curves(results_df: pd.DataFrame, metric: str = "val_loss") -> plt.Figure:
    """
    Per-algorithm scaling curves with power-law fits.
    """
    if 'param_count' not in results_df.columns or 'model' not in results_df.columns or metric not in results_df.columns:
        raise ValueError(f"DataFrame must contain 'model', 'param_count', and '{metric}' columns.")

    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = results_df['model'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for idx, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        
        # Aggregate pareto front points
        if metric == "val_loss":
            agg = model_data.groupby('param_count')[metric].min().reset_index()
        else: # e.g. val_accuracy
            agg = model_data.groupby('param_count')[metric].max().reset_index()
            
        N = agg['param_count'].values
        Y = agg[metric].values
        
        # Plot raw points
        ax.scatter(N, Y, color=colors[idx], label=f"{model} (empirical)")
        
        # Fit power law
        if len(N) >= 3 and metric == "val_loss": 
            # Power law makes sense mostly for losses directly
            a, b = fit_power_law(N.tolist(), Y.tolist())
            if not np.isnan(a) and not np.isnan(b):
                N_smooth = np.logspace(np.log10(min(N)), np.log10(max(N)), 100)
                Y_fit = _power_law(N_smooth, a, b)
                ax.plot(N_smooth, Y_fit, color=colors[idx], linestyle='--', alpha=0.7, label=f"{model} fit ($N^{{-{b:.3f}}}$)")

    ax.set_xscale("log")
    if metric == "val_loss":
        ax.set_yscale("log")
        
    ax.set_xlabel("Parameter Count (N)")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Scaling Laws: {metric.replace('_', ' ').title()} vs. Parameter Count")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    
    return fig
