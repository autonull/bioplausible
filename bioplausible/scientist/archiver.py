"""
Artifact Archiver for Bio-Plausible Experiments

Handles saving model checkpoints, configurations, and logs to a structured
archive format (ZIP) for reproducibility and industrial-grade tracking.
"""

import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("Archiver")

ARTIFACTS_DIR = Path("artifacts")


class ExperimentArchiver:
    """
    Manages the creation and storage of experiment artifacts.
    """

    def __init__(self, base_dir: Path = ARTIFACTS_DIR):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def archive_trial(
        self,
        trial_id: int,
        model: torch.nn.Module,
        config: Dict[str, Any],
        metrics: Dict[str, Any],
        extra_files: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Creates a ZIP archive for a specific trial.

        Args:
            trial_id: Unique trial ID.
            model: The PyTorch model to save.
            config: Configuration dictionary.
            metrics: Final metrics dictionary.
            extra_files: Optional dictionary mapping filename -> content (str).

        Returns:
            Path to the created ZIP file, or None if failed.
        """
        try:
            trial_name = f"trial_{trial_id}_{config.get('model', 'unknown')}"
            trial_dir = self.base_dir / trial_name
            trial_dir.mkdir(exist_ok=True)

            # 1. Save Model Checkpoint
            checkpoint_path = trial_dir / "model.pt"
            torch.save(model.state_dict(), checkpoint_path)

            # 2. Save Config
            with open(trial_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            # 3. Save Metrics
            with open(trial_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # 4. Extra Files (e.g., reproduction script snippet)
            if extra_files:
                for fname, content in extra_files.items():
                    with open(trial_dir / fname, "w") as f:
                        f.write(content)

            # 5. Create ZIP
            zip_path = self.base_dir / f"{trial_name}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in trial_dir.iterdir():
                    zf.write(file, file.name)

            # Cleanup directory
            shutil.rmtree(trial_dir)

            logger.info(f"Archived trial {trial_id} to {zip_path}")
            return str(zip_path)

        except Exception as e:
            logger.error(f"Failed to archive trial {trial_id}: {e}")
            return None
