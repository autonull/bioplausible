"""
Reproducibility Framework for EquiTile
=======================================

Tools for ensuring reproducible research:
- Seed management
- Configuration logging
- Result versioning
- Environment capture

Example
-------
>>> from bioplausible.models.equitile.utils.reproducibility import ReproducibilityTracker
>>> tracker = ReproducibilityTracker(seed=42)
>>> tracker.log_config(config)
>>> tracker.save_results(results)
"""

import hashlib
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class EnvironmentInfo:
    """Captured environment information."""
    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    gpu_info: List[Dict[str, Any]]
    os_name: str
    cpu_count: int
    timestamp: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    command_line: str


@dataclass
class ExperimentConfig:
    """Experiment configuration for reproducibility."""
    seed: int
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
    hardware_config: Dict[str, Any]


class ReproducibilityTracker:
    """Track and ensure reproducibility of experiments.
    
    Parameters
    ----------
    seed : int
        Random seed for all operations
    results_dir : str
        Directory to save results
    """
    
    def __init__(self, seed: int = 42, results_dir: str = "results") -> None:
        self.seed = seed
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set all seeds
        self._set_seeds()
        
        # Capture environment
        self.env_info = self._capture_environment()
        
        # Experiment tracking
        self.experiment_id = self._generate_experiment_id()
        self.config_log: List[Dict[str, Any]] = []
    
    def _set_seeds(self) -> None:
        """Set all random seeds."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _capture_environment(self) -> EnvironmentInfo:
        """Capture current environment information."""
        # Git info
        git_commit = None
        git_branch = None
        try:
            import subprocess
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            git_branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode('ascii').strip()
        except Exception:
            pass
        
        # GPU info
        gpu_info = []
        cuda_version = None
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "name": props.name,
                    "memory_gb": props.total_memory / 1e9,
                    "compute_capability": f"{props.major}.{props.minor}",
                })
        
        return EnvironmentInfo(
            python_version=sys.version,
            torch_version=torch.__version__,
            cuda_version=cuda_version,
            gpu_info=gpu_info,
            os_name=os.name,
            cpu_count=os.cpu_count() or 0,
            timestamp=datetime.now().isoformat(),
            git_commit=git_commit,
            git_branch=git_branch,
            command_line=" ".join(sys.argv),
        )
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{timestamp}_{self.seed}_{os.getpid()}"
        hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"exp_{timestamp}_{hash_id}"
    
    def log_config(self, config: Any, name: str = "config") -> None:
        """Log configuration for reproducibility.
        
        Parameters
        ----------
        config : Any
            Configuration object (dataclass, dict, etc.)
        name : str
            Configuration name
        """
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        elif hasattr(config, '__dataclass_fields__'):
            config_dict = asdict(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = vars(config)
        
        self.config_log.append({
            "name": name,
            "config": config_dict,
            "timestamp": datetime.now().isoformat(),
        })
    
    def save_results(
        self,
        results: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save results with full reproducibility information.
        
        Parameters
        ----------
        results : dict
            Experimental results
        metrics : dict, optional
            Key metrics to extract
        
        Returns
        -------
        Path
            Path to saved results file
        """
        # Create results bundle
        bundle = {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "environment": asdict(self.env_info),
            "configs": self.config_log,
            "results": results,
            "metrics": metrics or {},
            "saved_at": datetime.now().isoformat(),
        }
        
        # Save to file
        filepath = self.results_dir / f"{self.experiment_id}.json"
        with open(filepath, 'w') as f:
            json.dump(bundle, f, indent=2, default=str)
        
        # Also save as latest
        latest_path = self.results_dir / "latest.json"
        with open(latest_path, 'w') as f:
            json.dump(bundle, f, indent=2, default=str)
        
        print(f"Results saved to {filepath}")
        return filepath
    
    def load_results(self, experiment_id: str) -> Dict[str, Any]:
        """Load results from a previous experiment.
        
        Parameters
        ----------
        experiment_id : str
            Experiment ID to load
        
        Returns
        -------
        dict
            Experiment results bundle
        """
        filepath = self.results_dir / f"{experiment_id}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Experiment {experiment_id} not found")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def verify_reproducibility(self, results_path: Path) -> Dict[str, bool]:
        """Verify if results can be reproduced.
        
        Parameters
        ----------
        results_path : Path
            Path to results file
        
        Returns
        -------
        dict
            Verification results
        """
        with open(results_path, 'r') as f:
            bundle = json.load(f)
        
        verification = {
            "seed_present": "seed" in bundle,
            "environment_captured": "environment" in bundle,
            "config_logged": len(bundle.get("configs", [])) > 0,
            "git_commit_present": bundle.get("environment", {}).get("git_commit") is not None,
            "torch_version_match": bundle.get("environment", {}).get("torch_version") == torch.__version__,
        }
        
        return verification
    
    def get_experiment_summary(self) -> str:
        """Get summary of current experiment."""
        lines = [
            f"Experiment ID: {self.experiment_id}",
            f"Seed: {self.seed}",
            f"Results dir: {self.results_dir}",
            "",
            "Environment:",
            f"  Python: {self.env_info.python_version.split()[0]}",
            f"  PyTorch: {self.env_info.torch_version}",
            f"  CUDA: {self.env_info.cuda_version or 'N/A'}",
            f"  GPU: {self.env_info.gpu_info[0]['name'] if self.env_info.gpu_info else 'N/A'}",
            "",
            f"Configs logged: {len(self.config_log)}",
        ]
        
        if self.env_info.git_commit:
            lines.append(f"Git: {self.env_info.git_commit[:8]} ({self.env_info.git_branch})")
        
        return "\n".join(lines)


# =============================================================================
# Configuration Utilities
# =============================================================================

@dataclass
class ReproducibleConfig:
    """Base configuration with reproducibility support."""
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ReproducibleConfig':
        """Load configuration from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def get_hash(self) -> str:
        """Get hash of configuration for versioning."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]


# =============================================================================
# Factory Functions
# =============================================================================

def create_tracker(
    seed: int = 42,
    results_dir: str = "results",
) -> ReproducibilityTracker:
    """Create reproducibility tracker.
    
    Parameters
    ----------
    seed : int
        Random seed
    results_dir : str
        Results directory
    
    Returns
    -------
    ReproducibilityTracker
        Tracker instance
    """
    return ReproducibilityTracker(seed=seed, results_dir=results_dir)


def set_reproducible_mode(seed: int = 42) -> None:
    """Set all seeds for reproducible execution.
    
    Parameters
    ----------
    seed : int
        Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
