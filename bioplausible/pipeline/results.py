import glob
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional


class ResultsManager:
    """Manages persistence of training results."""

    BASE_DIR = "results/runs"

    def __init__(self, base_dir=None):
        if base_dir:
            self.BASE_DIR = base_dir
        os.makedirs(self.BASE_DIR, exist_ok=True)

    def save_run(self, run_id: str, config: Dict[str, Any], metrics: Dict[str, Any]):
        """Save a training run."""
        run_dir = os.path.join(self.BASE_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)

        data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
        }

        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(data, f, indent=2)

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all saved runs."""
        runs = []
        for meta_path in glob.glob(os.path.join(self.BASE_DIR, "*/metadata.json")):
            try:
                with open(meta_path, "r") as f:
                    runs.append(json.load(f))
            except Exception as e:
                print(f"Error loading run metadata {meta_path}: {e}")

        # Sort by timestamp descending
        runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return runs

    def load_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Load specific run data."""
        path = os.path.join(self.BASE_DIR, run_id, "metadata.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return None

    def delete_run(self, run_id: str):
        """Delete a run."""
        run_dir = os.path.join(self.BASE_DIR, run_id)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)

    def export_run(self, run_id: str, zip_path: str):
        """Export run to zip file."""
        run_dir = os.path.join(self.BASE_DIR, run_id)
        if not os.path.exists(run_dir):
            raise FileNotFoundError(f"Run {run_id} not found.")

        # shutil.make_archive adds extension automatically if not present in format
        # but we want exact control.
        base_name = zip_path
        if base_name.endswith(".zip"):
            base_name = base_name[:-4]

        shutil.make_archive(base_name, "zip", run_dir)

    def import_run(self, zip_path: str) -> str:
        """Import run from zip file. Returns run_id."""
        import zipfile

        # Verify it's a valid run zip (simple check)
        with zipfile.ZipFile(zip_path, "r") as z:
            if "metadata.json" not in z.namelist():
                raise ValueError("Invalid run package: metadata.json missing.")

            with z.open("metadata.json") as f:
                meta = json.load(f)
                run_id = meta.get("run_id")

        if not run_id:
            raise ValueError("Could not determine run_id.")

        target_dir = os.path.join(self.BASE_DIR, run_id)
        # Unpack
        shutil.unpack_archive(zip_path, target_dir)
        return run_id
