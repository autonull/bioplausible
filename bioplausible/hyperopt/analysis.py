"""
Analysis tools for Hyperparameter Optimization results.
Includes encoding and dimensionality reduction.
"""

import logging
from typing import Any, Dict, List, Union

import numpy as np

logger = logging.getLogger("HyperoptAnalysis")

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not found. Analysis features disabled.")


def flatten_config(config: Dict[str, Any], prefix="") -> Dict[str, Any]:
    """Recursively flatten dictionary."""
    items = []
    for k, v in config.items():
        new_key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key + ".").items())
        else:
            items.append((new_key, v))
    return dict(items)


def encode_configs(configs: List[Dict[str, Any]]) -> np.ndarray:
    """
    Convert a list of configuration dictionaries into a numerical matrix.
    Handles numerical and categorical data.
    """
    if not HAS_SKLEARN or not configs:
        return np.array([])

    # 1. Flatten all configs
    flat_configs = [flatten_config(c) for c in configs]

    # 2. Collect all keys (union)
    all_keys = sorted(set().union(*(d.keys() for d in flat_configs)))

    # 3. Determine type for each key
    # Simple heuristic: if any value is str, treat as categorical. Else numerical.
    # Also handle missing values (fill with mean for num, 'missing' for cat)

    num_keys = []
    cat_keys = []

    data_by_key = {k: [] for k in all_keys}

    for c in flat_configs:
        for k in all_keys:
            data_by_key[k].append(c.get(k, None))

    for k in all_keys:
        values = [v for v in data_by_key[k] if v is not None]
        if not values:
            continue  # Skip empty columns

        if any(isinstance(v, str) for v in values):
            cat_keys.append(k)
        else:
            num_keys.append(k)

    # 4. Construct feature lists
    X_num = []
    X_cat = []

    for i in range(len(configs)):
        row_num = []
        row_cat = []

        for k in num_keys:
            val = flat_configs[i].get(k, None)
            row_num.append(val if val is not None else np.nan)

        for k in cat_keys:
            val = flat_configs[i].get(k, "missing")
            row_cat.append(str(val))

        X_num.append(row_num)
        X_cat.append(row_cat)

    X_num = np.array(X_num, dtype=float)
    X_cat = np.array(X_cat, dtype=object)

    # 5. Pipeline
    transformers = []

    if num_keys:
        # Fill NaNs with mean
        # Since SimpleImputer is another import, let's just do it manually for X_num
        col_means = np.nanmean(X_num, axis=0)
        inds = np.where(np.isnan(X_num))
        X_num[inds] = np.take(col_means, inds[1])

        transformers.append(("num", StandardScaler(), list(range(len(num_keys)))))

    if cat_keys:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(range(len(num_keys), len(num_keys) + len(cat_keys))),
            )
        )

    # Combine
    # We construct a combined matrix first
    X_combined = []
    if num_keys and cat_keys:
        X_combined = np.hstack([X_num, X_cat])
    elif num_keys:
        X_combined = X_num
    elif cat_keys:
        X_combined = X_cat

    if len(X_combined) == 0:
        return np.array([])

    # ColumnTransformer requires indices or names.
    # It's easier to apply encoders separately and hstack

    features = []

    if num_keys:
        scaler = StandardScaler()
        features.append(scaler.fit_transform(X_num))

    if cat_keys:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        features.append(encoder.fit_transform(X_cat))

    if features:
        return np.hstack(features)

    return np.array([])


def reduce_dimensions(features: np.ndarray, method="pca", n_components=2) -> np.ndarray:
    """Reduce dimensionality of feature matrix."""
    if not HAS_SKLEARN or features.size == 0:
        return np.array([])

    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        # TSNE requires more samples than perplexity
        perp = min(30, max(5, features.shape[0] // 2))
        reducer = TSNE(
            n_components=n_components, perplexity=perp, init="pca", learning_rate="auto"
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    try:
        return reducer.fit_transform(features)
    except Exception as e:
        logger.error(f"Reduction failed: {e}")
        return np.zeros((features.shape[0], n_components))
