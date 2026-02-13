"""
Research Synthesizer.

Synthesizes high-level insights from experimental results, generating
strategic analysis, failure pattern detection, and actionable recommendations
to guide future research directions.
"""

import json
import sqlite3
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class ResearchSynthesizer:
    """
    Synthesizes research insights from experimental results.

    Generates high-level strategic analysis and actionable recommendations.

    Attributes:
        db_path (str): Path to the results database.
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the ResearchSynthesizer.

        Args:
            db_path (str): Path to the SQLite database.
        """
        self.db_path = db_path

    def _load_convergence_data(self, conn: sqlite3.Connection) -> pd.DataFrame:
        """
        Load detailed convergence data linked to trials.

        Args:
            conn (sqlite3.Connection): Database connection.

        Returns:
            pd.DataFrame: Convergence data.
        """
        try:
            query = """
            SELECT 
                traj.trial_id,
                MAX(CASE WHEN ua.key = 'model_name' THEN ua.value_json END) as model_name,
                MAX(CASE WHEN ua.key = 'task_name' THEN ua.value_json END) as task_name,
                ckpt.epoch,
                ckpt.train_loss,
                ckpt.val_acc,
                ckpt.train_acc,
                ckpt.perplexity,
                ckpt.samples_seen
            FROM training_checkpoints ckpt
            JOIN training_trajectories traj ON ckpt.trajectory_id = traj.id
            JOIN trials t ON traj.trial_id = t.trial_id
            LEFT JOIN trial_user_attributes ua ON t.trial_id = ua.trial_id
            WHERE t.state = 'COMPLETE'
            GROUP BY traj.trial_id, ckpt.epoch
            ORDER BY traj.trial_id, ckpt.epoch
            """

            df = pd.read_sql(query, conn)

            # Clean up JSON strings
            for col in ["model_name", "task_name"]:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: (
                            json.loads(x)
                            if isinstance(x, str) and x.startswith('"')
                            else x
                        )
                    )

            return df
        except Exception as e:
            print(f"⚠️ Error loading convergence data: {e}")
            return pd.DataFrame()

    def synthesize_full_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive research insights.

        Returns:
            Dict[str, Any]: Structured insights dictionary.
        """
        try:
            conn = sqlite3.connect(self.db_path)

            # Load Data with full metadata
            trials_df = self._get_trials_df(conn)

            # Load Convergence Data
            convergence_df = self._load_convergence_data(conn)

            try:
                failures_query = "SELECT * FROM failures"
                failures_df = pd.read_sql(failures_query, conn)
            except Exception:
                failures_df = pd.DataFrame()

            insights = {
                "cross_algorithm_insights": self._analyze_cross_algo(trials_df),
                "task_specific_winners": self._analyze_by_task(trials_df),
                "efficiency_analysis": self._analyze_efficiency(
                    trials_df, convergence_df
                ),
                "ablation_analysis": self._analyze_ablations(trials_df),
                "statistical_significance": self._analyze_significance(trials_df),
                "failure_analysis": self._analyze_failures(failures_df),
                "quick_wins": self._find_quick_wins(trials_df, failures_df),
                "research_gaps": self._identify_gaps(trials_df),
            }
            conn.close()
            return insights
        except Exception as e:
            traceback.print_exc()
            return {"error": str(e)}

    def _get_trials_df(self, conn: sqlite3.Connection) -> pd.DataFrame:
        """
        Query and denormalize Optuna trials with full hyperparameters.

        Args:
            conn (sqlite3.Connection): Database connection.

        Returns:
            pd.DataFrame: DataFrame containing trial data.
        """
        query = """
        SELECT
            t.trial_id,
            t.state,
            s.study_name,
            v.value as accuracy,
            MAX(CASE WHEN ua.key = 'model_name' THEN ua.value_json END) as model_name,
            MAX(CASE WHEN ua.key = 'task_name' THEN ua.value_json END) as task_name,
            MAX(CASE WHEN ua.key = 'param_count' THEN ua.value_json END) as param_count,
            MAX(CASE WHEN ua.key = 'num_epochs' THEN ua.value_json END) as num_epochs,
            MAX(CASE WHEN ua.key = 'tier' THEN ua.value_json END) as tier,
            hl.param_count as param_count_actual
        FROM trials t
        LEFT JOIN studies s ON t.study_id = s.study_id
        LEFT JOIN trial_values v ON t.trial_id = v.trial_id
        LEFT JOIN trial_user_attributes ua ON t.trial_id = ua.trial_id
        LEFT JOIN hyperopt_logs hl ON t.trial_id = hl.trial_id
        WHERE t.state = 'COMPLETE'
        GROUP BY t.trial_id
        ORDER BY accuracy DESC
        """
        df = pd.read_sql(query, conn)

        # Fetch hyperparameters
        params_query = "SELECT trial_id, param_name, param_value FROM trial_params"
        params_df = pd.read_sql(params_query, conn)
        if not params_df.empty:
            params_pivot = params_df.pivot(
                index="trial_id", columns="param_name", values="param_value"
            )
            df = df.join(params_pivot, on="trial_id")

        # JSON deserialization
        for col in ["model_name", "task_name", "tier"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if x and isinstance(x, str) else x
                )

        # Metadata rescue
        known_tasks = [
            "tiny_shakespeare",
            "char_ngram",
            "fashion_mnist",
            "mnist",
            "cifar10",
            "cartpole",
            "pendulum",
        ]

        def rescue_metadata(row: pd.Series) -> pd.Series:
            if not row.get("model_name") and row.get("study_name"):
                for task in known_tasks:
                    if f"_{task}_" in row["study_name"]:
                        parts = row["study_name"].split(f"_{task}_")
                        if len(parts) >= 2:
                            row["model_name"] = parts[0]
                            row["task_name"] = task
                            tier_cand = parts[-1]
                            if tier_cand in ["smoke", "shallow", "standard", "deep"]:
                                row["tier"] = tier_cand
            # Estimate epochs if missing
            if not row.get("num_epochs") or row["num_epochs"] == 0:
                row["num_epochs"] = 10

            # Estimate param count
            p_actual = row.get("param_count_actual")
            if p_actual is not None and p_actual > 0:
                if p_actual < 500:
                    row["param_count"] = int(p_actual * 1_000_000)
                else:
                    row["param_count"] = int(p_actual)
            else:
                p = row.get("param_count")
                if p is None or p == 0:
                    row["param_count"] = self._estimate_param_count(row)
                elif isinstance(p, (int, float)) and p < 500:
                    row["param_count"] = int(p * 1_000_000)
                else:
                    try:
                        p_val = float(p)
                        if p_val < 500:
                            row["param_count"] = int(p_val * 1_000_000)
                        else:
                            row["param_count"] = int(p_val)
                    except (ValueError, TypeError):
                        row["param_count"] = self._estimate_param_count(row)

            return row

        df = df.apply(rescue_metadata, axis=1)
        return df

    def _estimate_param_count(self, row: pd.Series) -> int:
        """Estimate parameter count based on hyperparameters if missing."""
        h = row.get("hidden_dim", 32)
        l = row.get("num_layers", 1)
        try:
            h_int = int(h) if pd.notnull(h) else 32
            l_int = int(l) if pd.notnull(l) else 1
        except (ValueError, TypeError):
            h_int = 32
            l_int = 1

        return l_int * (h_int * h_int) + (h_int * 10)

    def _analyze_ablations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyze results from ablation studies."""
        if df.empty:
            return []

        ablations = []
        try:
            # Helper to check if trial is ablation
            def is_ablation(row: pd.Series) -> bool:
                if "config" in row and isinstance(row["config"], dict):
                    return row["config"].get("is_ablation", False)
                return False

            ablation_trials = df[df.apply(is_ablation, axis=1)]

            for _, trial in ablation_trials.iterrows():
                config = trial["config"]
                parent_id = config.get("verified_trial_id") or config.get(
                    "verification_of_trial_id"
                )
                param = config.get("ablation_param", "Unknown")
                val = config.get(param, "Unknown")

                if parent_id:
                    parent = df[df["trial_id"] == parent_id]
                    if not parent.empty:
                        parent_acc = parent.iloc[0]["accuracy"]
                        my_acc = trial["accuracy"]
                        delta = my_acc - parent_acc

                        ablations.append(
                            {
                                "model": trial.get("model_name", "Unknown"),
                                "task": trial.get("task_name", "Unknown"),
                                "ablation_param": param,
                                "ablation_value": val,
                                "accuracy": my_acc,
                                "baseline_accuracy": parent_acc,
                                "delta": delta,
                                "significant": abs(delta) > 0.02,  # 2% threshold
                            }
                        )
        except Exception as e:
            return [{"error": f"Ablation analysis error: {e}"}]

        return ablations

    def _analyze_significance(
        self, df: pd.DataFrame
    ) -> List[Dict[str, Union[str, float]]]:
        """Perform statistical significance tests between top models."""
        if df.empty or "model_name" not in df.columns:
            return []

        try:
            from scipy import stats

            model_accs = {}
            for model in df["model_name"].dropna().unique():
                accs = df[df["model_name"] == model]["accuracy"].dropna().tolist()
                if len(accs) >= 3:
                    model_accs[model] = accs

            if len(model_accs) < 2:
                return [
                    {
                        "error": "Insufficient data for significance testing (need >= 2 models with >= 3 trials)."
                    }
                ]

            results: List[Dict[str, Union[str, float]]] = []
            models = sorted(model_accs.keys())
            for i, m1 in enumerate(models):
                for j, m2 in enumerate(models):
                    if i >= j:
                        continue

                    t_stat, p_val = stats.ttest_ind(
                        model_accs[m1], model_accs[m2], equal_var=False
                    )

                    mean1 = sum(model_accs[m1]) / len(model_accs[m1])
                    mean2 = sum(model_accs[m2]) / len(model_accs[m2])
                    diff = mean1 - mean2

                    if p_val < 0.05:
                        winner = m1 if diff > 0 else m2
                        loser = m2 if diff > 0 else m1
                        results.append(
                            {
                                "winner": winner,
                                "loser": loser,
                                "p_value": float(p_val),
                                "mean_diff": abs(diff),
                                "confidence": "High" if p_val < 0.01 else "Moderate",
                            }
                        )

            results.sort(key=lambda x: x["p_value"])  # type: ignore
            return results

        except ImportError:
            return [{"error": "SciPy not installed, skipping statistical tests."}]
        except Exception as e:
            return [{"error": f"Significance analysis error: {e}"}]

    def _analyze_cross_algo(self, df: pd.DataFrame) -> Union[str, Dict[str, Any]]:
        """Cross-algorithm performance comparison."""
        if df.empty or "model_name" not in df.columns:
            return "No model data available."

        try:
            summary = (
                df.groupby("model_name")
                .agg({"accuracy": ["mean", "max", "std"], "trial_id": "count"})
                .round(4)
            )

            summary.columns = [
                "_".join(col).strip() for col in summary.columns.values  # type: ignore
            ]
            summary = summary.rename(columns={"trial_id_count": "num_trials"})
            summary = summary.sort_values("accuracy_max", ascending=False)

            rankings = []
            for model, row in summary.iterrows():
                rankings.append(
                    {
                        "model": model,
                        "best_accuracy": float(row["accuracy_max"]),
                        "mean_accuracy": float(row["accuracy_mean"]),
                        "std": float(row.get("accuracy_std", 0)),
                        "trials": int(row["num_trials"]),
                    }
                )

            return {"rankings": rankings, "summary_table": summary.to_dict()}
        except Exception as e:
            return f"Analysis failed: {e}"

    def _analyze_by_task(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """Task-specific winners."""
        if df.empty or "task_name" not in df.columns:
            return {}

        task_winners = {}
        for task in df["task_name"].dropna().unique():
            task_df = (
                df[df["task_name"] == task]
                .sort_values("accuracy", ascending=False)
                .head(3)
            )
            task_winners[task] = [
                {
                    "model": row["model_name"] if row.get("model_name") else "Unknown",
                    "accuracy": float(row["accuracy"]) if row.get("accuracy") else 0.0,
                    "params": int(row.get("param_count", 0) or 0),
                }
                for _, row in task_df.iterrows()
            ]
        return task_winners

    def _analyze_efficiency(
        self, df: pd.DataFrame, convergence_df: pd.DataFrame
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze parameter efficiency (Acc/Param) and epoch efficiency (Acc/Epoch)."""
        if df.empty:
            return {}

        analysis = {}

        if "param_count" in df.columns:
            df_valid = df[df["param_count"] > 0].copy()
            if not df_valid.empty:
                df_valid["param_efficiency"] = df_valid["accuracy"] / (
                    df_valid["param_count"] / 1e6
                )
                top_param = df_valid.nlargest(5, "param_efficiency")[
                    ["model_name", "accuracy", "param_count", "param_efficiency"]
                ]
                analysis["top_param_efficient"] = top_param.to_dict("records")

        if convergence_df is not None and not convergence_df.empty:
            trial_epochs = (
                convergence_df.groupby("trial_id")["epoch"].max().reset_index()
            )
            trial_epochs.columns = ["trial_id", "actual_epochs"]  # type: ignore

            if "samples_seen" in convergence_df.columns:
                trial_samples = (
                    convergence_df.groupby("trial_id")["samples_seen"]
                    .max()
                    .reset_index()
                )
                trial_samples.columns = ["trial_id", "total_samples"]  # type: ignore
            else:
                trial_samples = pd.DataFrame(columns=["trial_id", "total_samples"])

            fast_convergence = []
            for trial_id in convergence_df["trial_id"].unique():
                t_data = convergence_df[convergence_df["trial_id"] == trial_id]
                final_acc = t_data["val_acc"].max()
                target = final_acc * 0.9
                reached = t_data[t_data["val_acc"] >= target]["epoch"].min()
                if pd.isna(reached):
                    reached = t_data["epoch"].max()
                fast_convergence.append({"trial_id": trial_id, "epochs_to_90": reached})

            df_fast = pd.DataFrame(fast_convergence)

            df_epoch = df.merge(trial_epochs, on="trial_id", how="left")
            if not trial_samples.empty:
                df_epoch = df_epoch.merge(trial_samples, on="trial_id", how="left")
            df_epoch = df_epoch.merge(df_fast, on="trial_id", how="left")

            df_epoch["actual_epochs"] = df_epoch["actual_epochs"].fillna(
                df_epoch.get("num_epochs", 10)
            )

            df_epoch["epoch_efficiency"] = df_epoch["accuracy"] / df_epoch[
                "actual_epochs"
            ].replace(0, 1)

            top_epoch = df_epoch.nlargest(5, "epoch_efficiency")[
                [
                    "model_name",
                    "task_name",
                    "accuracy",
                    "actual_epochs",
                    "epoch_efficiency",
                ]
            ]
            analysis["top_epoch_efficient"] = top_epoch.to_dict("records")

            if "total_samples" in df_epoch.columns:
                df_epoch["sample_efficiency"] = df_epoch["accuracy"] / (
                    df_epoch["total_samples"] / 1e6
                ).replace(0, 0.001)
                top_sample = df_epoch.nlargest(5, "sample_efficiency")[
                    [
                        "model_name",
                        "task_name",
                        "accuracy",
                        "total_samples",
                        "sample_efficiency",
                    ]
                ]
                analysis["top_sample_efficient"] = top_sample.to_dict("records")

            learners = df_epoch[df_epoch["accuracy"] > 0.5].copy()
            if not learners.empty:
                top_fast = learners.sort_values(
                    ["epochs_to_90", "accuracy"], ascending=[True, False]
                ).head(5)
                analysis["fastest_learners"] = top_fast[
                    ["model_name", "task_name", "accuracy", "epochs_to_90"]
                ].to_dict("records")

        elif "num_epochs" in df.columns:
            df_valid = df[df["num_epochs"] > 0].copy()
            if not df_valid.empty:
                df_valid["epoch_efficiency"] = (
                    df_valid["accuracy"] / df_valid["num_epochs"]
                )
                top_epoch = df_valid.nlargest(5, "epoch_efficiency")[
                    [
                        "model_name",
                        "task_name",
                        "accuracy",
                        "num_epochs",
                        "epoch_efficiency",
                    ]
                ]
                analysis["top_epoch_efficient"] = top_epoch.to_dict("records")

        return analysis

    def _analyze_failures(self, df: pd.DataFrame) -> Union[str, Dict[str, Any]]:
        """Failure pattern analysis."""
        if df.empty:
            return "No failures recorded."

        try:
            if "failure_type" in df.columns:
                counts = df["failure_type"].value_counts().to_dict()

                patterns = []
                if any("nan" in str(k).lower() for k in counts.keys()):
                    patterns.append(
                        "NaN instability detected (likely exploding gradients or high LR)"
                    )
                if any("timeout" in str(k).lower() for k in counts.keys()):
                    patterns.append(
                        "Timeout issues (consider reducing model depth or using checkpointing)"
                    )

                return {"counts": counts, "patterns": patterns}
            return "Failures exist but missing failure_type."
        except Exception as e:
            return f"Failure analysis failed: {e}"

    def _find_quick_wins(
        self, trials: pd.DataFrame, failures: pd.DataFrame
    ) -> List[str]:
        """Actionable recommendations."""
        suggestions = []

        if not failures.empty:
            if "failure_type" in failures.columns:
                nan_fails = failures[
                    failures["failure_type"]
                    .astype(str)
                    .str.contains("nan", case=False, na=False)
                ]
                if len(nan_fails) > 5:
                    suggestions.append(
                        f"🔥 {len(nan_fails)} NaN failures detected. Recommendation: Lower learning rates globally or add gradient clipping."
                    )

            if "model_name" in failures.columns and not trials.empty:
                # Calculate failure rate per model
                fail_counts = failures["model_name"].value_counts()
                success_counts = trials["model_name"].value_counts()

                for model in fail_counts.index:
                    f_count = fail_counts[model]
                    s_count = success_counts.get(model, 0)
                    total = f_count + s_count

                    if total >= 5:
                        rate = f_count / total
                        if rate > 0.5:
                            suggestions.append(
                                f"⚠️ Model '{model}' has a {rate:.0%} failure rate ({f_count}/{total}). Consider debugging initialization or disabling."
                            )

        if not trials.empty and "tier" in trials.columns:
            tier_counts = trials["tier"].value_counts()
            if tier_counts.get("smoke", 0) > tier_counts.get("shallow", 0) * 2:
                suggestions.append(
                    "💡 Heavy smoke testing detected. Consider promoting successful configs to shallow/standard tiers."
                )

        if not trials.empty and "model_name" in trials.columns:
            model_counts = trials["model_name"].value_counts()
            underexplored = [m for m, c in model_counts.items() if c < 5]
            if underexplored:
                suggestions.append(
                    f"📊 Underexplored models: {', '.join(underexplored[:3])}. Allocate more trials for statistical significance."
                )

        return suggestions

    def _identify_gaps(self, df: pd.DataFrame) -> List[str]:
        """Identify research gaps and unexplored areas."""
        gaps = []

        if not df.empty and "task_name" in df.columns:
            explored_tasks = set(df["task_name"].dropna().unique())
            all_tasks = {
                "mnist",
                "cifar10",
                "char_ngram",
                "cartpole",
                "pendulum",
                "tiny_shakespeare",
            }
            missing = all_tasks - explored_tasks
            if missing:
                gaps.append(f"Unexplored tasks: {', '.join(missing)}")

        if not df.empty and "model_name" in df.columns:
            models = set(df["model_name"].dropna().unique())
            if "GNN" not in str(models):
                gaps.append("No Graph Neural Network experiments detected")
            if "Transformer" not in str(models):
                gaps.append("No Transformer architecture experiments detected")

        return gaps
