import json
import logging
import sqlite3
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_text

logger = logging.getLogger(__name__)


class KnowledgebaseMetamodel:
    """
    Acts as a surrogate analytical model on top of empirical experiment records.
    Constructs human-readable symbolic rules predicting success/failure cases
    and clusters algorithms based on hyperparameter sensitivity.
    """

    def __init__(self):
        self.df = None
        self.fitted = False

    def fit(self, db_path: str):
        """
        Pull experiment configurations and outcomes from SQLite DB into a DataFrame.
        """
        try:
            conn = sqlite3.connect(db_path)

            # Use failures table as our mock "experiments" table if real ones
            # aren't available for parsing
            # In a full system, we'd query a `trials` table. Here we demonstrate
            # parsing `failures` configs just as a structural proxy
            query = "SELECT model_name, task_name, failure_type, config FROM failures"
            raw_data = pd.read_sql_query(query, conn)
            conn.close()

            records = []
            for _, row in raw_data.iterrows():
                try:
                    cfg = json.loads(row["config"])
                    # Flatten into a feature row
                    record = {
                        "model": row["model_name"],
                        "task": row["task_name"],
                        "outcome": row["failure_type"],
                        # Arbitrary mock metrics parsing
                        "lr": cfg.get("lr", 0.0),
                        "hidden_dim": cfg.get("hidden_dim", 0.0),
                        "num_layers": cfg.get("num_layers", 0.0),
                        "max_steps": (
                            cfg.get("extra", {}).get("max_steps", 0.0)
                            if "extra" in cfg
                            else cfg.get("max_steps", 0.0)
                        ),
                    }
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Failed to parse config row for metamodel: {e}")

            self.df = pd.DataFrame(records)
            self.fitted = True
            logger.info(f"Metamodel successfully fitted to {len(self.df)} records.")

        except Exception as e:
            logger.error(f"Metamodel fit failed: {e}")
            self.fitted = False

    def extract_symbolic_rules(
        self, target_metric: str = "outcome", focus_model: str = "eqprop_mlp"
    ) -> List[str]:
        """
        Produce human-readable symbolic heuristics using a lightweight
        Decision Tree surrogate.
        """
        if not self.fitted or self.df is None or self.df.empty:
            return ["No data available to extract rules."]

        # Filter for the relevant model
        model_df = self.df[self.df["model"] == focus_model].copy()

        if len(model_df) < 2:
            return [f"Not enough data for model {focus_model} to extract rules."]

        # Prepare Features (X)
        features = ["lr", "hidden_dim", "num_layers", "max_steps"]
        X = model_df[features].fillna(0.0)

        # Prepare Targets (Y) - Binary encode string outcomes to demonstrate splits
        # 1.0 represents divergence/failure, 0.0 represents success
        Y = np.where(model_df["outcome"] == "settling_divergence", 1.0, 0.0)

        if len(np.unique(Y)) < 2:
            return [
                (
                    f"For {focus_model}: Outcome is uniform; "
                    "cannot extract discriminative rules."
                )
            ]

        # Fit a shallow decision tree to ensure human readability
        tree = DecisionTreeRegressor(max_depth=3, min_samples_leaf=1)
        tree.fit(X, Y)

        rules_text = export_text(tree, feature_names=features)

        # Format the sklearn output into heuristics
        formatted_rules = []
        formatted_rules.append(
            f"Heuristics for {focus_model} avoiding Settling Divergence:"
        )

        for line in rules_text.split("\n"):
            if line.strip():
                formatted_rules.append(
                    "  "
                    + line.strip()
                    .replace("value: [0.0]", "--> Success")
                    .replace("value: [1.0]", "--> Divergence")
                )

        return formatted_rules

    def compute_algorithm_similarity(self) -> pd.DataFrame:
        """
        Cluster models based on their sensitivity across hyperparameter sweeps.
        Returns a cosine-similarity matrix DataFrame.
        """
        if not self.fitted or self.df is None or self.df.empty:
            return pd.DataFrame()

        # Create a behavioral fingerprint for each model
        # We group by model and average the hyperparameters to see which models
        # tend to cluster in similar configurations
        features = ["lr", "hidden_dim", "num_layers", "max_steps"]
        grouped = self.df.groupby("model")[features].mean().fillna(0.0)

        if len(grouped) < 2:
            return pd.DataFrame()

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(grouped)

        sim_matrix = cosine_similarity(scaled_features)

        return pd.DataFrame(sim_matrix, index=grouped.index, columns=grouped.index)
