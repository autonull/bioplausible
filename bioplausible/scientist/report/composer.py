"""
Report Composer.

Composes modular reports from experiment data, combining visualizations,
summary statistics, leaderboards, and detailed analysis into a single document.
"""

import json
import sqlite3
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from bioplausible.scientist.report.analysis import BayesianRanker, MLAnalyzer
from bioplausible.scientist.report.latex import LatexGenerator
from bioplausible.visualization import ResultVisualizer


class ReportComposer:
    """
    Composes modular reports from experiment data.

    Attributes:
        db_path (str): Path to the SQLite database.
        output_dir (Path): Directory for report output.
        conn (sqlite3.Connection): Database connection.
        visualizer (ResultVisualizer): Visualization tool.
        ml_analyzer (MLAnalyzer): Machine learning analysis tool.
        bayesian_ranker (BayesianRanker): Probabilistic ranking tool.
        latex_generator (LatexGenerator): LaTeX report generator.
    """

    def __init__(self, db_path: str, output_dir: str) -> None:
        """
        Initialize the ReportComposer.

        Args:
            db_path (str): Path to the database file.
            output_dir (str): Directory where reports will be saved.
        """
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.visualizer = ResultVisualizer(self.output_dir / "images")

        # New analysis components
        self.ml_analyzer = MLAnalyzer(self.output_dir / "images")
        self.bayesian_ranker = BayesianRanker()
        self.latex_generator = LatexGenerator()

    def generate_report(self) -> None:
        """Generate all report sections."""
        manifest: Dict[str, Any] = {
            "title": "Scientist++ Experiment Report",
            "sections": [],
            "images": [],
            "analysis": {},
        }

        # Load data once
        df = self._get_trials_df()
        logs = self._get_decision_logs()

        # 0. Generate Visualizations
        self._generate_visualizations(df, manifest)

        # 1. Summary
        summary_path = self.output_dir / "01_summary.md"
        self._write_summary(summary_path, df)
        manifest["sections"].append(str(summary_path.name))

        # 2. Leaderboards
        leaderboard_path = self.output_dir / "03_leaderboards.md"
        self._write_leaderboards(leaderboard_path, df)
        manifest["sections"].append(str(leaderboard_path.name))

        # 3. ML Analysis & Bayesian Ranking
        if not df.empty:
            data_list = df.to_dict(orient="records")
            flat_data = []
            for d in data_list:
                item = d.copy()
                if isinstance(item.get("config"), dict):
                    for k, v in item["config"].items():
                        if k not in item:
                            item[k] = v
                if "model" not in item and "model_name" in item:
                    item["model"] = item["model_name"]
                if "task" not in item and "task_name" in item:
                    item["task"] = item["task_name"]
                flat_data.append(item)

            # ML Analysis
            insights, robustness = self.ml_analyzer.run_analysis(flat_data)
            manifest["analysis"]["ml_insights"] = insights
            manifest["analysis"]["robustness"] = robustness

            # Bayesian Ranking
            agg_data = self._aggregate_for_ranking(flat_data)
            ranking_table = self.bayesian_ranker.rank_models(agg_data)
            manifest["analysis"]["bayesian_ranking"] = ranking_table

            # LaTeX Generation
            self.latex_generator.generate_report(agg_data, logs, self.output_dir)

        # 4. Create Full Report
        self._compose_full_report(manifest)

        # 5. Manifest
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _get_decision_logs(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch recent decision logs."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master"
                " WHERE type='table' AND name='decision_logs'"
            )
            if not cursor.fetchone():
                return []

            query = (
                "SELECT timestamp, event_type, description"
                " FROM decision_logs ORDER BY id DESC LIMIT ?"
            )
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            logs = []
            for r in rows:
                logs.append({"date_str": r[0], "event_type": r[1], "description": r[2]})
            return logs
        except Exception as e:
            print(f"Error fetching logs: {e}")
            return []

    def _aggregate_for_ranking(
        self, data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simple aggregation for Bayesian ranking."""
        from collections import defaultdict

        model_task_stats = defaultdict(list)
        for d in data:
            model = d.get("model_name") or d.get("model")
            task = d.get("task_name") or d.get("task")
            if model:
                model_task_stats[(model, task or "unknown")].append(d["accuracy"])

        agg = []
        for (model, task), accs in model_task_stats.items():
            agg.append(
                {
                    "model": model,
                    "task": task,
                    "accuracy": float(np.mean(accs)),
                    "accuracy_std": float(np.std(accs)) if len(accs) > 1 else 0.0,
                    "count": len(accs),
                }
            )
        return agg

    def _get_trials_df(
        self, task_filter: Optional[str] = None, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Query and denormalize Optuna trials into a DataFrame.
        """
        query = """
        SELECT
            t.trial_id,
            t.state,
            s.study_name,
            v.value as accuracy,
            MAX(CASE WHEN ua.key = 'model_name'
                THEN ua.value_json END) as model_name,
            MAX(CASE WHEN ua.key = 'task_name'
                THEN ua.value_json END) as task_name,
            MAX(CASE WHEN ua.key = 'tier'
                THEN ua.value_json END) as tier_value,
            MAX(CASE WHEN ua.key = 'param_count'
                THEN ua.value_json END) as param_count_attr,
            MAX(CASE WHEN ua.key = 'iteration_time'
                THEN ua.value_json END) as iteration_time_attr,
            MAX(CASE WHEN ua.key = 'config'
                THEN ua.value_json END) as config,
            MAX(CASE WHEN ua.key = 'noise_score'
                THEN ua.value_json END) as noise_score,
            MAX(CASE WHEN ua.key = 'perturbation_score'
                THEN ua.value_json END) as perturbation_score,
            MAX(CASE WHEN ua.key = 'ood_score'
                THEN ua.value_json END) as ood_score,
            MAX(CASE WHEN ua.key = 'adversarial_fgsm'
                THEN ua.value_json END) as adversarial_fgsm,
            MAX(CASE WHEN ua.key = 'adversarial_pgd'
                THEN ua.value_json END) as adversarial_pgd,
            MAX(CASE WHEN ua.key = 'robustness_score'
                THEN ua.value_json END) as robustness_score,
            hl.param_count as param_count_actual,
            hl.iteration_time as iteration_time_actual
        FROM trials t
        LEFT JOIN studies s ON t.study_id = s.study_id
        LEFT JOIN trial_values v ON t.trial_id = v.trial_id
        LEFT JOIN trial_user_attributes ua ON t.trial_id = ua.trial_id
        LEFT JOIN hyperopt_logs hl ON t.trial_id = hl.trial_id
        WHERE t.state = 'COMPLETE'
        GROUP BY t.trial_id
        """

        query += " ORDER BY accuracy DESC"

        try:
            df = pd.read_sql(query, self.conn)

            # Fetch Hyperparameters (trial_params)
            params_query = "SELECT trial_id, param_name, param_value FROM trial_params"
            params_df = pd.read_sql(params_query, self.conn)

            if not params_df.empty:
                params_pivot = params_df.pivot(
                    index="trial_id", columns="param_name", values="param_value"
                )
                df = df.join(params_pivot, on="trial_id")

            for col in ["model_name", "task_name", "config", "tier_value"]:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: json.loads(x) if x and isinstance(x, str) else x
                    )

            if "tier_value" in df.columns:
                df["tier"] = df["tier_value"]

            known_tasks = [
                "tiny_shakespeare",
                "char_ngram",
                "fashion_mnist",
                "mnist",
                "cifar10",
                "cartpole",
                "pendulum",
            ]

            def parse_metadata(row: pd.Series) -> pd.Series:
                if not row["model_name"]:
                    s = row["study_name"]
                    if s:
                        for task in known_tasks:
                            token = f"_{task}_"
                            if token in s:
                                parts = s.split(token)
                                if len(parts) >= 2:
                                    row["model_name"] = parts[0]
                                    row["task_name"] = task
                                    if "tier" not in row or not row["tier"]:
                                        tier_cand = parts[-1]
                                        if tier_cand in [
                                            "smoke",
                                            "shallow",
                                            "standard",
                                            "deep",
                                        ]:
                                            row["config"] = {"tier": tier_cand}
                                            row["tier"] = tier_cand

                p_actual = row.get("param_count_actual")
                if p_actual is not None and p_actual > 0:
                    if p_actual < 500:
                        row["param_count"] = int(p_actual * 1_000_000)
                    else:
                        row["param_count"] = int(p_actual)
                else:
                    p = row.get("param_count_attr")
                    if p is None or p == 0:
                        h = row.get("hidden_dim", 32)
                        n_layers = row.get("num_layers", 1)
                        h = h if pd.notnull(h) else 32
                        n_layers = n_layers if pd.notnull(n_layers) else 1
                        row["param_count"] = n_layers * (h * h) + (h * 10)
                    else:
                        row["param_count"] = p

                it_actual = row.get("iteration_time_actual")
                if it_actual is not None and it_actual > 0:
                    row["iteration_time"] = it_actual
                elif row.get("iteration_time_attr"):
                    row["iteration_time"] = row.get("iteration_time_attr")

                return row

            df = df.apply(parse_metadata, axis=1)

            if task_filter:
                df = df[df["task_name"] == task_filter]

            if limit:
                df = df.head(limit)

            return df
        except Exception as e:
            print(f"Error querying trials: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    def _load_convergence_data(self) -> pd.DataFrame:
        """Load per-epoch checkpoint data for convergence analysis."""
        try:
            query = """
            SELECT
                traj.trial_id,
                traj.model_name,
                traj.task_name,
                ckpt.epoch,
                ckpt.val_acc,
                ckpt.samples_seen
            FROM training_checkpoints ckpt
            JOIN training_trajectories traj ON ckpt.trajectory_id = traj.id
            ORDER BY traj.trial_id, ckpt.epoch
            """

            df = pd.read_sql(query, self.conn)

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

    def _generate_visualizations(
        self, df: pd.DataFrame, manifest: Dict[str, Any]
    ) -> None:
        """Generate ALL plots using ResultVisualizer."""
        if df.empty:
            return

        data = df.to_dict(orient="records")

        for d in data:
            if d.get("iteration_time") is None:
                d["iteration_time"] = 0
            if d.get("accuracy") is None:
                d["accuracy"] = 0.0
            if d.get("param_count") is None:
                d["param_count"] = 0

            if d.get("config") and isinstance(d["config"], dict):
                for k, v in d["config"].items():
                    if k not in d:
                        d[k] = v

            if "lr" in d and "learning_rate" not in d:
                d["learning_rate"] = d["lr"]

            if "model_name" in d and "model" not in d:
                d["model"] = d["model_name"]
            if "task_name" in d and "task" not in d:
                d["task"] = d["task_name"]

            if d.get("param_count"):
                try:
                    d["params"] = float(d["param_count"]) / 1_000_000.0
                except TypeError, ValueError:
                    d["params"] = 0.0
            else:
                d["params"] = 0.0

        path = self.visualizer.plot_pareto_frontier(data)
        manifest["images"].append(
            {"title": "Pareto Efficiency Frontier", "path": Path(path).name}
        )

        path = self.visualizer.plot_convergence_speed(data)
        if path:
            manifest["images"].append(
                {"title": "Convergence Speed", "path": Path(path).name}
            )

        path = self.visualizer.plot_tier_progress(data)
        manifest["images"].append(
            {"title": "Progress by Tier", "path": Path(path).name}
        )

        if len(data) > 3:
            paths = self.visualizer.plot_hyperparam_correlations(data)
            for p in paths:
                p_obj = Path(p)
                try:
                    rel_path = p_obj.relative_to(self.output_dir)
                    parts = rel_path.parts
                    param = p_obj.stem.replace("impact_", "").replace("_", " ").title()

                    if len(parts) >= 4:
                        task = parts[-3]
                        tier = parts[-2]
                        title = f"Impact of {param}: {task} ({tier})"
                    else:
                        title = f"Impact of {param}"

                    manifest["images"].append({"title": title, "path": str(rel_path)})
                except ValueError:
                    manifest["images"].append(
                        {"title": "Impact Plot", "path": p_obj.name}
                    )

        tasks = df["task_name"].dropna().unique()
        for task in tasks:
            path = self.visualizer.plot_leaderboard(data, task, metric="accuracy")
            if path:
                manifest["images"].append(
                    {
                        "title": f"Leaderboard (Accuracy): {task}",
                        "path": Path(path).name,
                    }
                )

            path = self.visualizer.plot_leaderboard(
                data, task, metric="compound_efficiency"
            )
            if path:
                manifest["images"].append(
                    {
                        "title": f"Leaderboard (Compound Efficiency): {task}",
                        "path": Path(path).name,
                    }
                )

            path = self.visualizer.plot_leaderboard(data, task, metric="efficiency")
            if path:
                manifest["images"].append(
                    {
                        "title": f"Leaderboard (Efficiency): {task}",
                        "path": Path(path).name,
                    }
                )

        self._generate_significance_matrix(df, data, manifest)

        conv_df = self._load_convergence_data()
        if not conv_df.empty:
            trajectories = []
            for trial_id, group in conv_df.groupby("trial_id"):
                checkpoints = []
                for _, row in group.iterrows():
                    checkpoints.append(
                        SimpleNamespace(
                            epoch=row["epoch"],
                            val_acc=row["val_acc"],
                            samples_seen=row.get("samples_seen", 0),
                        )
                    )

                if group.empty:
                    continue

                traj = SimpleNamespace(
                    model_name=group.iloc[0]["model_name"],
                    task_name=group.iloc[0]["task_name"],
                    checkpoints=checkpoints,
                )
                trajectories.append(traj)

            paths = self.visualizer.plot_convergence_curves(trajectories)
            for p in paths:
                manifest["images"].append(
                    {
                        "title": (
                            f"Convergence: "
                            f"{Path(p).stem.replace('convergence_curves_', '')}"
                        ),
                        "path": Path(p).name,
                    }
                )

            paths = self.visualizer.plot_sample_complexity(trajectories)
            for p in paths:
                manifest["images"].append(
                    {
                        "title": (
                            f"Sample Complexity: "
                            f"{Path(p).stem.replace('sample_complexity_', '')}"
                        ),
                        "path": Path(p).name,
                    }
                )

    def _generate_significance_matrix(
        self, df: pd.DataFrame, data: List[Dict[str, Any]], manifest: Dict[str, Any]
    ) -> None:
        """Generate statistical significance matrix."""
        try:
            from scipy import stats

            models = df["model_name"].dropna().unique().tolist()
            if len(models) < 2:
                return

            model_accs = {}
            for model in models:
                accs = df[df["model_name"] == model]["accuracy"].dropna().tolist()
                if len(accs) >= 2:
                    model_accs[model] = accs

            if len(model_accs) < 2:
                return

            labels = sorted(model_accs.keys())
            n = len(labels)
            p_matrix = np.zeros((n, n))

            for i, m1 in enumerate(labels):
                for j, m2 in enumerate(labels):
                    if i == j:
                        p_matrix[i, j] = 1.0
                    else:
                        _, p_val = stats.ttest_ind(model_accs[m1], model_accs[m2])
                        p_matrix[i, j] = p_val

            path = self.visualizer.plot_significance_matrix(p_matrix, labels)
            manifest["images"].append(
                {"title": "Statistical Significance Matrix", "path": Path(path).name}
            )

        except ImportError:
            print("⚠️  scipy not available, skipping significance matrix")
        except Exception as e:
            print(f"⚠️  Error generating significance matrix: {e}")

    def _write_summary(self, path: Path, df: pd.DataFrame) -> None:
        """Write executive summary."""
        try:
            best = df.iloc[0] if not df.empty else None
            total_trials = len(df)
        except Exception as e:
            print(f"Report generation error in summary: {e}")
            best = None
            total_trials = 0

        content = "# Executive Summary\n\n"
        if best is not None:
            content += f"**Best Model**: {best['model_name']}\n"
            content += f"**Accuracy**: {best['accuracy']:.2%}\n"
            content += f"**Task**: {best.get('task_name', 'N/A')}\n"
            content += f"**Config**: `{best.get('config', '{}')}`\n\n"
        else:
            content += "No completed trials found.\n\n"

        content += "## Overview\n"
        content += f"Total Trials: {total_trials}\n"

        with open(path, "w") as f:
            f.write(content)

    def _write_leaderboards(self, path: Path, df: pd.DataFrame) -> None:
        """Write leaderboards per task."""
        content = "# Leaderboards (Data)\n\n"

        try:
            if df.empty:
                tasks = []
            else:
                tasks = df["task_name"].unique().tolist()

            for task in tasks:
                if not task:
                    continue
                content += f"## Task: {task}\n\n"
                task_df = df[df["task_name"] == task].head(10)

                cols = ["model_name", "accuracy", "param_count", "iteration_time"]
                display_cols = [c for c in cols if c in task_df.columns]

                if not task_df.empty:
                    content += task_df[display_cols].to_markdown(index=False)
                    content += "\n\n"
                else:
                    content += "No results.\n\n"
        except Exception as e:
            content += f"Error generating leaderboards: {e}\n"

        with open(path, "w") as f:
            f.write(content)

    def _compose_full_report(self, manifest: Dict[str, Any]) -> None:
        """Concatenate all sections and embed images."""
        full_path = self.output_dir / "FULL_REPORT.md"
        with open(full_path, "w") as outfile:
            outfile.write(f"# {manifest['title']}\n\n")

            synthesis_path = self.output_dir / "synthesis/SYNTHESIS.md"
            if synthesis_path.exists():
                with open(synthesis_path, "r") as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n\n---\n\n")

            outfile.write("## Key Visualizations\n\n")

            for img in manifest["images"]:
                name = img["path"]
                title = img["title"]
                outfile.write(f"### {title}\n\n")
                outfile.write(f"![{title}](images/{name})\n\n")

            outfile.write("---\n\n")

            if "analysis" in manifest:
                analysis = manifest["analysis"]

                if "bayesian_ranking" in analysis:
                    outfile.write("## Probabilistic Ranking\n\n")
                    outfile.write(analysis["bayesian_ranking"])
                    outfile.write("\n\n---\n\n")

                if "ml_insights" in analysis:
                    outfile.write("## Machine Learning Analysis\n\n")
                    outfile.write(analysis["ml_insights"])
                    outfile.write("\n\n")

                if "robustness" in analysis:
                    outfile.write("## Robustness Analysis\n\n")
                    outfile.write(analysis["robustness"])
                    outfile.write("\n\n---\n\n")

            for section in manifest["sections"]:
                section_path = self.output_dir / section
                if section_path.exists():
                    with open(section_path, "r") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n---\n\n")

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None  # type: ignore

    def __enter__(self) -> "ReportComposer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
