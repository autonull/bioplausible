
import os
import json
import sqlite3
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

class ReportComposer:
    """
    Composes modular reports from experiment data.
    """
    
    def __init__(self, db_path: str, output_dir: str):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        
    def generate_report(self):
        """Generate all report sections."""
        manifest = {
            "title": "Scientist++ Experiment Report",
            "sections": []
        }
        
        # 1. Summary
        summary_path = self.output_dir / "01_summary.md"
        self._write_summary(summary_path)
        manifest["sections"].append(str(summary_path.name))
        
        # 2. Leaderboards
        leaderboard_path = self.output_dir / "03_leaderboards.md"
        self._write_leaderboards(leaderboard_path)
        manifest["sections"].append(str(leaderboard_path.name))
        
        # 3. Create Full Report
        self._compose_full_report(manifest)
        
        # 4. Manifest
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            
    def _write_summary(self, path: Path):
        """Write executive summary."""
        # Query best model
        try:
            df = pd.read_sql("SELECT * FROM trials WHERE state='completed' ORDER BY accuracy DESC LIMIT 5", self.conn)
            best = df.iloc[0] if not df.empty else None
            
            total_trials = pd.read_sql("SELECT COUNT(*) as count FROM trials", self.conn).iloc[0]["count"]
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
            
    def _write_leaderboards(self, path: Path):
        """Write leaderboards per task."""
        content = "# Leaderboards\n\n"
        
        try:
            tasks_df = pd.read_sql("SELECT DISTINCT task_name FROM trials", self.conn)
            tasks = tasks_df["task_name"].tolist() if not tasks_df.empty else []
            
            for task in tasks:
                if not task: continue
                content += f"## Task: {task}\n\n"
                df = pd.read_sql(f"SELECT model_name, accuracy, param_count, iteration_time FROM trials WHERE task_name='{task}' AND state='completed' ORDER BY accuracy DESC LIMIT 10", self.conn)
                
                if not df.empty:
                    # Basic markdown table
                    content += df.to_markdown(index=False)
                    content += "\n\n"
                else:
                    content += "No results.\n\n"
        except Exception as e:
            content += f"Error generating leaderboards: {e}\n"
                
        with open(path, "w") as f:
            f.write(content)

    def _compose_full_report(self, manifest: Dict):
        """Concatenate all sections."""
        full_path = self.output_dir / "FULL_REPORT.md"
        with open(full_path, "w") as outfile:
            outfile.write(f"# {manifest['title']}\n\n")
            
            for section in manifest["sections"]:
                section_path = self.output_dir / section
                if section_path.exists():
                    with open(section_path, "r") as infile:
                        outfile.write(infile.read())
                        outfile.write("\n\n---\n\n")
                        
    def close(self):
        self.conn.close()
