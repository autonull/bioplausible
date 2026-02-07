
import os
import json
import sqlite3
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

from bioplausible.visualization import ResultVisualizer


class ReportComposer:
    """
    Composes modular reports from experiment data.
    """

    def __init__(self, db_path: str, output_dir: str):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.visualizer = ResultVisualizer(self.output_dir / "images")

    def generate_report(self):
        """Generate all report sections."""
        manifest = {
            "title": "Scientist++ Experiment Report",
            "sections": [],
            "images": []
        }

        # Load data once
        df = self._get_trials_df()

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

        # 3. Create Full Report
        self._compose_full_report(manifest)

        # 4. Manifest
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    def _get_trials_df(self, task_filter=None, limit=None):
        """
        Query and denormalize Optuna trials into a DataFrame.
        """
        query = """
        SELECT
            t.trial_id,
            t.state,
            s.study_name,
            v.value as accuracy,
            MAX(CASE WHEN ua.key = 'model_name' THEN ua.value_json END) as model_name,
            MAX(CASE WHEN ua.key = 'task_name' THEN ua.value_json END) as task_name,
            MAX(CASE WHEN ua.key = 'tier' THEN ua.value_json END) as tier_value,
            MAX(CASE WHEN ua.key = 'param_count' THEN ua.value_json END) as param_count,
            MAX(CASE WHEN ua.key = 'iteration_time' THEN ua.value_json END) as iteration_time,
            MAX(CASE WHEN ua.key = 'config' THEN ua.value_json END) as config
        FROM trials t
        LEFT JOIN studies s ON t.study_id = s.study_id
        LEFT JOIN trial_values v ON t.trial_id = v.trial_id
        LEFT JOIN trial_user_attributes ua ON t.trial_id = ua.trial_id
        WHERE t.state = 'COMPLETE'
        GROUP BY t.trial_id
        """

        # Determine sorting by metrics (default accuracy DESC)
        query += " ORDER BY accuracy DESC"

        try:
            df = pd.read_sql(query, self.conn)

            # Fetch Hyperparameters (trial_params)
            params_query = "SELECT trial_id, param_name, param_value FROM trial_params"
            params_df = pd.read_sql(params_query, self.conn)

            # Pivot params: trial_id -> [lr, hidden, ...]
            if not params_df.empty:
                params_pivot = params_df.pivot(
                    index="trial_id", columns="param_name", values="param_value")
                # Merge into main DF
                df = df.join(params_pivot, on="trial_id")

            # Clean up JSON strings from Optuna attributes
            for col in ['model_name', 'task_name', 'config', 'tier_value']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(
                        x) if x and isinstance(x, str) else x)

            # Assign tier if available
            if 'tier_value' in df.columns:
                df['tier'] = df['tier_value']

            # Metadata Rescue Logic
            known_tasks = ['tiny_shakespeare', 'char_ngram',
                           'fashion_mnist', 'mnist', 'cifar10', 'cartpole', 'pendulum']

            def parse_metadata(row):
                # 1. Recover Model/Task
                if not row['model_name']:
                    s = row['study_name']
                    if s:
                        for task in known_tasks:
                            token = f"_{task}_"
                            if token in s:
                                parts = s.split(token)
                                if len(parts) >= 2:
                                    row['model_name'] = parts[0]
                                    row['task_name'] = task
                                    if 'tier' not in row or not row['tier']:
                                        tier_cand = parts[-1]
                                        if tier_cand in ['smoke', 'shallow', 'standard', 'deep']:
                                            row['config'] = {"tier": tier_cand}
                                            # We generally want tier as a column too
                                            row['tier'] = tier_cand

                # 2. Recover/Estimating param_count
                p = row.get('param_count')
                if p is None or p == 0:
                    # HEURISTIC ESTIMATION based on hidden_dim and num_layers
                    # If columns exist from join
                    h = row.get('hidden_dim', 32)
                    l = row.get('num_layers', 1)
                    # Generic MLP-ish estimate: l * h^2
                    # Inputs/Outputs add a bit but dominant term is usually internal
                    # This is just for visualization scaling
                    # Default h=32, l=1 -> 1000 params
                    # Handle if h or l are NaN
                    h = h if pd.notnull(h) else 32
                    l = l if pd.notnull(l) else 1
                    row['param_count'] = l * (h * h) + (h * 10)  # rough proxy

                return row

            df = df.apply(parse_metadata, axis=1)

            if task_filter:
                df = df[df['task_name'] == task_filter]

            if limit:
                df = df.head(limit)

            return df
        except Exception as e:
            print(f"Error querying trials: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _load_convergence_data(self):
        """Load per-epoch checkpoint data for convergence analysis."""
        try:
            query = """
            SELECT 
                t.trial_id,
                MAX(CASE WHEN ua.key = 'model_name' THEN ua.value_json END) as model_name,
                MAX(CASE WHEN ua.key = 'task_name' THEN ua.value_json END) as task_name,
                ckpt.epoch,
                ckpt.val_acc,
                ckpt.samples_seen
            FROM training_checkpoints ckpt
            JOIN trials t ON ckpt.trial_id = t.trial_id
            LEFT JOIN trial_user_attributes ua ON t.trial_id = ua.trial_id
            WHERE t.state = 'COMPLETE'
            GROUP BY t.trial_id, ckpt.epoch
            ORDER BY t.trial_id, ckpt.epoch
            """
            
            df = pd.read_sql(query, self.conn)
            
            # Clean up JSON strings
            for col in ['model_name', 'task_name']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if x and isinstance(x, str) else x)
            
            return df
        except Exception as e:
            print(f"⚠️ Error loading convergence data: {e}")
            return pd.DataFrame()

    def _generate_visualizations(self, df: pd.DataFrame, manifest: Dict):
        """Generate ALL plots using ResultVisualizer - complete restoration."""
        if df.empty:
            return

        # Convert DF to list of dicts for visualizer
        data = df.to_dict(orient="records")

        # Data sanitization and enrichment
        for d in data:
            if d.get("iteration_time") is None:
                d["iteration_time"] = 0
            if d.get("accuracy") is None:
                d["accuracy"] = 0.0
            if d.get("param_count") is None:
                d["param_count"] = 0

            # Flatten config into the dict for hyperparam plots
            if d.get("config") and isinstance(d["config"], dict):
                for k, v in d["config"].items():
                    if k not in d:
                        d[k] = v

            # Map Aliases for Visualization
            if "lr" in d and "learning_rate" not in d:
                d["learning_rate"] = d["lr"]

            # Ensure 'model', 'task', and 'tier' keys for visualizer compatibility
            if "model_name" in d and "model" not in d:
                d["model"] = d["model_name"]
            if "task_name" in d and "task" not in d:
                d["task"] = d["task_name"]

            # Update 'params' key for visualizer (legacy key)
            if d.get("param_count"):
                d["params"] = d["param_count"] / 1_000_000.0  # Convert to Millions
            else:
                d["params"] = 0.0

        # ===== CORE VISUALIZATIONS =====

        # 1. Pareto Frontier (Efficiency)
        path = self.visualizer.plot_pareto_frontier(data)
        manifest["images"].append(
            {"title": "Pareto Efficiency Frontier", "path": Path(path).name})

        # 2. Convergence Speed
        path = self.visualizer.plot_convergence_speed(data)
        if path:  # Only if data available
            manifest["images"].append(
                {"title": "Convergence Speed", "path": Path(path).name})

        # 3. Tier Progress
        path = self.visualizer.plot_tier_progress(data)
        manifest["images"].append(
            {"title": "Progress by Tier", "path": Path(path).name})

        # 4. Hyperparameter Impact Analysis
        if len(data) > 3:
            paths = self.visualizer.plot_hyperparam_correlations(data)
            for p in paths:
                p_obj = Path(p)
                try:
                    rel_path = p_obj.relative_to(self.output_dir)

                    # Extract metadata from path structure: images/task/tier/file.png
                    parts = rel_path.parts
                    param = p_obj.stem.replace('impact_', '').replace('_', ' ').title()

                    if len(parts) >= 4:
                        # images/task/tier/file
                        task = parts[-3]
                        tier = parts[-2]
                        title = f"Impact of {param}: {task} ({tier})"
                    else:
                        title = f"Impact of {param}"

                    manifest["images"].append({"title": title, "path": str(rel_path)})
                except ValueError:
                    manifest["images"].append(
                        {"title": f"Impact Plot", "path": p_obj.name})

        # ===== LEADERBOARDS (ACCURACY + EFFICIENCY) =====
        tasks = df["task_name"].dropna().unique()
        for task in tasks:
            # Accuracy Leaderboard
            path = self.visualizer.plot_leaderboard(data, task, metric="accuracy")
            if path:
                manifest["images"].append(
                    {"title": f"Leaderboard (Accuracy): {task}", "path": Path(path).name})

            # Efficiency Leaderboard (Accuracy / Params)
            path = self.visualizer.plot_leaderboard(data, task, metric="efficiency")
            if path:
                manifest["images"].append(
                    {"title": f"Leaderboard (Efficiency): {task}", "path": Path(path).name})

        # ===== STATISTICAL SIGNIFICANCE MATRIX =====
        # Compute pairwise t-test p-values between models
        self._generate_significance_matrix(df, data, manifest)

        # ===== CONVERGENCE & EFFICIENCY PLOTS (NEW) =====
        conv_df = self._load_convergence_data()
        if not conv_df.empty:
            from types import SimpleNamespace
            trajectories = []
            for trial_id, group in conv_df.groupby('trial_id'):
                checkpoints = []
                for _, row in group.iterrows():
                    checkpoints.append(SimpleNamespace(
                        epoch=row['epoch'],
                        val_acc=row['val_acc'],
                        samples_seen=row.get('samples_seen', 0)
                    ))
                
                # Check if group is empty or missing data
                if group.empty:
                    continue

                traj = SimpleNamespace(
                    model_name=group.iloc[0]['model_name'],
                    task_name=group.iloc[0]['task_name'],
                    checkpoints=checkpoints
                )
                trajectories.append(traj)

            # 6. Convergence Curves
            paths = self.visualizer.plot_convergence_curves(trajectories)
            for p in paths:
                manifest["images"].append(
                    {"title": f"Convergence: {Path(p).stem.replace('convergence_curves_', '')}", 
                     "path": Path(p).name})

            # 7. Sample Complexity
            paths = self.visualizer.plot_sample_complexity(trajectories)
            for p in paths:
                manifest["images"].append(
                    {"title": f"Sample Complexity: {Path(p).stem.replace('sample_complexity_', '')}", 
                     "path": Path(p).name})

    def _generate_significance_matrix(self, df: pd.DataFrame, data: List[Dict], manifest: Dict):
        """Generate statistical significance matrix (pairwise t-tests between models)."""
        try:
            from scipy import stats
            import numpy as np

            # Group by model
            models = df["model_name"].dropna().unique().tolist()
            if len(models) < 2:
                return  # Need at least 2 models for comparison

            # Build accuracy lists per model
            model_accs = {}
            for model in models:
                accs = df[df["model_name"] == model]["accuracy"].dropna().tolist()
                if len(accs) >= 2:  # Need at least 2 samples for t-test
                    model_accs[model] = accs

            if len(model_accs) < 2:
                return

            # Compute pairwise p-values
            labels = sorted(model_accs.keys())
            n = len(labels)
            p_matrix = np.zeros((n, n))

            for i, m1 in enumerate(labels):
                for j, m2 in enumerate(labels):
                    if i == j:
                        p_matrix[i, j] = 1.0  # Same model
                    else:
                        # Two-sample t-test
                        _, p_val = stats.ttest_ind(model_accs[m1], model_accs[m2])
                        p_matrix[i, j] = p_val

            # Generate heatmap
            path = self.visualizer.plot_significance_matrix(p_matrix, labels)
            manifest["images"].append(
                {"title": "Statistical Significance Matrix", "path": Path(path).name})

        except ImportError:
            print("⚠️  scipy not available, skipping significance matrix")
        except Exception as e:
            print(f"⚠️  Error generating significance matrix: {e}")

    def _write_summary(self, path: Path, df: pd.DataFrame):
        """Write executive summary."""
        try:
            best = df.iloc[0] if not df.empty else None
            total_trials = len(df)
        except Exception as e:
            print(f"Report generation error in summary: {e}")
            best = None
            total_trials = 0

        content = f"# Executive Summary\n\n"
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

    def _write_leaderboards(self, path: Path, df: pd.DataFrame):
        """Write leaderboards per task (Markdown tables)."""
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

                # Select display columns
                cols = ["model_name", "accuracy", "param_count", "iteration_time"]
                display_cols = [c for c in cols if c in task_df.columns]

                if not task_df.empty:
                    # Basic markdown table
                    content += task_df[display_cols].to_markdown(index=False)
                    content += "\n\n"
                else:
                    content += "No results.\n\n"
        except Exception as e:
            content += f"Error generating leaderboards: {e}\n"

        with open(path, "w") as f:
            f.write(content)

    def _compose_full_report(self, manifest: Dict):
        """Concatenate all sections and embed images."""
        full_path = self.output_dir / "FULL_REPORT.md"
        with open(full_path, "w") as outfile:
            outfile.write(f"# {manifest['title']}\n\n")

            # Embed Key Visualizations at the top
            outfile.write("## Key Visualizations\n\n")
            # Filter for specific key plots to show first
            priority_plots = ["Pareto", "Progress"]

            for img in manifest["images"]:
                name = img["path"]
                title = img["title"]

                # Use standard markdown image syntax
                # Since FULL_REPORT is in the same dir as images/ subdirectory
                outfile.write(f"### {title}\n\n")
                outfile.write(f"![{title}](images/{name})\n\n")

            outfile.write("---\n\n")

            for section in manifest["sections"]:
                section_path = self.output_dir / section
                if section_path.exists():
                    with open(section_path, "r") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n---\n\n")

    def close(self):
        self.conn.close()
