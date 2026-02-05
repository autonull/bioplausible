from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json

class ReportSection(ABC):
    """
    Abstract base class for a section of the Scientist++ report.
    Each section is responsible for generating its own content 
    based on the provided experiment data.
    """
    
    def __init__(self, data: Dict[str, Any]):
        """
        Args:
            data: Dictionary containing all experiment data 
                  (trajectory, config, metrics, model info).
        """
        self.data = data
        
    @property
    @abstractmethod
    def section_id(self) -> str:
        """Unique identifier for the section (e.g., 'performance', 'dynamics')."""
        pass
    
    @property
    @abstractmethod
    def title(self) -> str:
        """Human-readable title for the section."""
        pass
        
    @property
    @abstractmethod
    def priority(self) -> int:
        """Ordering priority (lower comes first)."""
        pass

    @abstractmethod
    def generate_markdown(self) -> str:
        """Generate the markdown content for this section."""
        pass
        
    @abstractmethod
    def generate_json(self) -> Dict[str, Any]:
        """Generate structured data for this section (for API/UI consumption)."""
        pass

class ConfigSection(ReportSection):
    """Displays the hyperparameter configuration."""
    
    @property
    def section_id(self) -> str:
        return "config"
        
    @property
    def title(self) -> str:
        return "Experimental Configuration"
        
    @property
    def priority(self) -> int:
        return 10
        
    def generate_markdown(self) -> str:
        config = self.data.get("config", {})
        md = f"## {self.title}\n\n"
        md += "| Hyperparameter | Value |\n"
        md += "| :--- | :--- |\n"
        
        for k, v in sorted(config.items()):
            # Format floats nicely
            if isinstance(v, float):
                val_str = f"{v:.4g}"
            else:
                val_str = str(v)
            md += f"| `{k}` | {val_str} |\n"
            
        return md
        
    def generate_json(self) -> Dict[str, Any]:
        return {
            "section": self.section_id,
            "data": self.data.get("config", {})
        }

class PerformanceSection(ReportSection):
    """Summarizes final performance metrics."""
    
    @property
    def section_id(self) -> str:
        return "performance"
        
    @property
    def title(self) -> str:
        return "Performance Summary"
        
    @property
    def priority(self) -> int:
        return 20
        
    def generate_markdown(self) -> str:
        # Extract metrics
        # Expecting data to contain 'metrics' or 'trajectory'
        metrics = self.data.get("metrics", {})
        
        # If trajectory is present, use last checkpoint for more detail
        trajectory = self.data.get("trajectory")
        if trajectory and hasattr(trajectory, "checkpoints") and trajectory.checkpoints:
            last_ckpt = trajectory.checkpoints[-1]
            metrics.update({
                "final_train_acc": last_ckpt.train_acc,
                "final_val_acc": last_ckpt.val_acc,
                "final_train_loss": last_ckpt.train_loss,
                "final_val_loss": last_ckpt.val_loss,
            })
            
        md = f"## {self.title}\n\n"
        
        # Key Metrics Table
        md += "### Key Metrics\n\n"
        md += "| Metric | Result |\n"
        md += "| :--- | :--- |\n"
        
        if "accuracy" in metrics:
            md += f"| **Validation Accuracy** | **{metrics.get('accuracy', 0.0):.2%}** |\n"
        if "final_train_acc" in metrics:
            md += f"| Training Accuracy | {metrics.get('final_train_acc', 0.0):.2%} |\n"
        if "loss" in metrics:
            md += f"| Final Loss | {metrics.get('loss', 0.0):.4f} |\n"
        if "perplexity" in metrics and metrics['perplexity'] > 0:
            md += f"| Perplexity | {metrics['perplexity']:.2f} |\n"
            
        # Comparison logic could go here later (e.g. vs baseline)
        
        return md
        
    def generate_json(self) -> Dict[str, Any]:
        return {
            "section": self.section_id,
            "data": self.data.get("metrics", {})
        }

class DynamicsSection(ReportSection):
    """Analyzes training dynamics (stability, overfitting, convergence)."""
    
    @property
    def section_id(self) -> str:
        return "dynamics"
        
    @property
    def title(self) -> str:
        return "Training Dynamics Analysis"
        
    @property
    def priority(self) -> int:
        return 30
        
    def generate_markdown(self) -> str:
        trajectory = self.data.get("trajectory")
        
        md = f"## {self.title}\n\n"
        
        if not trajectory:
            md += "*No training trajectory data available.*\n"
            return md
            
        # Convergence
        conv_epoch = getattr(trajectory, "convergence_epoch", None)
        converged = getattr(trajectory, "converged", False)
        
        if converged:
            md += f"- **Convergence**: Converged at epoch **{conv_epoch}**.\n"
        else:
            md += "- **Convergence**: Model did **NOT** converge within the epoch budget.\n"
            
        # Overfitting
        overfitting = getattr(trajectory, "overfitting_detected", False)
        if overfitting:
             md += "- **Overfitting**: ⚠️ **Detected**. Validation performance lagged significantly behind training.\n"
        else:
             md += "- **Overfitting**: Not detected. Generalization gap remained stable.\n"
             
        # Stability
        unstable = getattr(trajectory, "unstable", False)
        if unstable:
             md += "- **Stability**: ⚠️ **Unstable**. High variance in loss observed.\n"
        else:
             md += "- **Stability**: Training was stable.\n"
             
        return md
        
    def generate_json(self) -> Dict[str, Any]:
        traj = self.data.get("trajectory")
        if not traj: 
            return {"section": self.section_id, "data": None}
            
        return {
            "section": self.section_id,
            "data": {
                "converged": getattr(traj, "converged", False),
                "convergence_epoch": getattr(traj, "convergence_epoch", None),
                "overfitting": getattr(traj, "overfitting_detected", False),
                "unstable": getattr(traj, "unstable", False)
            }
        }
