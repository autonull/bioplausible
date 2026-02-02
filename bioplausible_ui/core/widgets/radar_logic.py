"""
Radar Logic Module

Encapsulates the business logic for the Radar View:
1. Projection Strategies (UMAP, PCA, t-SNE, Heuristic)
2. Visualization Mapping (Color/Size normalization and palettes)
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtGui import QBrush, QColor, QPen

# Optional ML imports
try:
    import umap
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    HAS_ML = True
except ImportError:
    HAS_ML = False


class ProjectionEngine:
    """Handles dimensionality reduction strategies."""

    @staticmethod
    def get_available_methods():
        methods = ["Heuristic"]
        if HAS_ML:
            methods.extend(["PCA", "t-SNE", "UMAP"])
        return methods

    @staticmethod
    def vectorize_trials(trials, selected_params):
        """Convert list of trial dicts to numpy array based on selected params."""
        if not trials:
            return np.array([]), []

        vectors = []
        trial_ids = []

        for t in trials:
            vec = []
            for k in selected_params:
                # Resolve key from config or attributes
                val = t.get("config", {}).get(k, t.get(k))

                # Numeric handling
                if isinstance(val, (int, float)):
                    vec.append(val)
                elif isinstance(val, str):
                    # Deterministic hash for categoricals
                    vec.append(hash(val) % 1000)
                elif isinstance(val, bool):
                    vec.append(1.0 if val else 0.0)
                else:
                    vec.append(0.0)

            vectors.append(vec)
            trial_ids.append(t.get("trial_id", t.get("trial_number", "unknown")))

        X = np.array(vectors)

        # Standard Scaler
        if len(X) > 1 and X.shape[1] > 0:
            try:
                # basic normalization if sklearn missing?
                # check HAS_ML inside method or assume simple normalization
                if HAS_ML:
                    X = StandardScaler().fit_transform(X)
                else:
                    # Simple min-max or standardization
                    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
            except:
                pass  # Fail safe

        return X, trial_ids

    @staticmethod
    def project(trials, method, selected_params):
        """
        Project trials to 2D.
        Returns: (embedding [N, 2], trial_ids, error_msg)
        """
        if not trials:
            return None, [], "No trials"

        # 1. Heuristic (No ML needed)
        if method == "Heuristic":
            embedding = np.zeros((len(trials), 2))
            ids = []
            for i, t in enumerate(trials):
                ids.append(t.get("trial_id", t.get("trial_number", "unknown")))
                conf = t.get("config", {})

                # Extract core structural/dynamic properties
                width = float(conf.get("hidden_dim", 64))
                depth = float(conf.get("num_layers", 2))
                lr = float(conf.get("lr", 0.01))

                # X: Structural Complexity
                x = np.log2(width if width > 0 else 1) * depth

                # Y: Learning Dynamics (negative log scale usually better for LR)
                y = -np.log10(lr if lr > 0 else 1e-5)

                # Add heavy jitter to separate identical configs
                jit = np.random.normal(0, 0.15, 2)
                embedding[i] = [x + jit[0], y + jit[1]]

            return embedding, ids, None

        # 2. ML Methods
        if not HAS_ML:
            return None, [], "ML libraries not installed"

        X, ids = ProjectionEngine.vectorize_trials(trials, selected_params)

        # Guard against too few samples
        if len(X) < 3:
            # Just map randomly or return zeros if < 2
            # PCA needs min n_samples
            # Fallback to heuristic for tiny datasets?
            # Or just random layout
            return np.random.rand(len(X), 2), ids, None

        try:
            embedding = None
            if method == "PCA":
                reducer = PCA(n_components=2)
                embedding = reducer.fit_transform(X)

            elif method == "t-SNE":
                perp = min(30, len(X) - 1)
                reducer = TSNE(
                    n_components=2,
                    perplexity=perp,
                    random_state=42,
                    init="pca",
                    learning_rate="auto",
                )
                embedding = reducer.fit_transform(X)

            elif method == "UMAP":
                n_neighbors = min(15, len(X) - 1)
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors, min_dist=0.1, random_state=42
                )
                embedding = reducer.fit_transform(X)

            else:
                return None, [], f"Unknown method: {method}"

            return embedding, ids, None

        except Exception as e:
            return None, [], str(e)


class VisualMapper:
    """Handles mapping of trial data to visual properties (Color, Size)."""

    @staticmethod
    def get_ranges(trials):
        """Compute min/max for all metrics."""
        if not trials:
            return {}

        metrics = ["accuracy", "final_loss", "param_count", "iteration_time"]
        ranges = {}
        for m in metrics:
            vals = [t.get(m, 0.0) for t in trials]
            if vals:
                ranges[m] = (min(vals), max(vals))
            else:
                ranges[m] = (0.0, 1.0)
        return ranges

    @staticmethod
    def get_brush(trial, mode, ranges):
        """Return QBrush for a trial based on mode."""

        def normalize(val, bounds):
            mn, mx = bounds
            if mx <= mn:
                return 0.5
            return (val - mn) / (mx - mn)

        if mode == "Model Family":
            # Hash-based categorical color
            fam = trial.get("model_name") or trial.get("model") or "Unknown"
            h = hash(fam) & 0xFFFFFF
            return pg.mkBrush(QColor(f"#{h:06x}"))

        elif mode == "Accuracy":
            val = trial.get("accuracy", 0.0)
            hue = 0.33 * val  # 0=Red, 0.33=Green
            c = QColor.fromHslF(hue, 0.85, 0.5)
            return pg.mkBrush(c)

        elif mode == "Loss":
            val = trial.get("final_loss") or trial.get("loss") or 0.0
            mn, mx = ranges.get("final_loss", (0, 1))
            norm = normalize(val, (0, 2.0))
            hue = 0.66 * (1.0 - min(norm, 1.0))
            c = QColor.fromHslF(hue, 0.8, 0.5)
            return pg.mkBrush(c)

        elif mode == "Param Count":
            val = trial.get("param_count") or trial.get("params_count") or 0.0
            mn, mx = ranges.get("param_count", (0, 1))
            norm = normalize(val, (mn, mx))
            hue = 0.15 + (0.65 * (1.0 - norm))
            c = QColor.fromHslF(hue, 0.9, 0.5)
            return pg.mkBrush(c)

        elif mode == "Time":
            val = trial.get("iteration_time") or trial.get("time") or 0.0
            mn, mx = ranges.get("iteration_time", (0, 1))
            norm = normalize(val, (mn, mx))
            hue = 0.5 + (0.4 * norm)
            c = QColor.fromHslF(hue, 0.8, 0.5)
            return pg.mkBrush(c)

        return pg.mkBrush("c")  # Default

    @staticmethod
    def get_size(trial, mode, ranges):
        """Return size (int)."""
        base_size = 12

        if mode == "Uniform":
            return base_size

        mn, mx = ranges.get("accuracy", (0, 1))

        if mode == "Accuracy":
            val = trial.get("accuracy", 0.0)
            return 8 + (val * 14)

        elif mode == "Param Count":
            val = trial.get("param_count") or trial.get("params_count") or 0.0
            mn, mx = ranges.get("param_count", (0, 1))
            if mx <= mn:
                norm = 0.5
            else:
                norm = (val - mn) / (mx - mn)
            return 8 + (norm * 14)

        return base_size
