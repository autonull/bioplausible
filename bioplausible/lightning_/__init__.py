"""
Bioplausible PyTorch Lightning Integration

Provides LightningModule, HPO, callbacks, and strategies
for biologically plausible learning algorithms.
"""

from bioplausible.lightning_.callbacks import BioPrecisionCallback
from bioplausible.lightning_.callbacks import BioPredictionWriter
from bioplausible.lightning_.callbacks import EnergyConvergenceCallback
from bioplausible.lightning_.experiment import run_pl_trial
from bioplausible.lightning_.experiment import run_pl_trial_with_wandb
from bioplausible.lightning_.hpo import BioOptunaPruner
from bioplausible.lightning_.hpo import BioRayTuneSearch
from bioplausible.lightning_.module import BioLightningModule
from bioplausible.lightning_.nas import run_nas_search
from bioplausible.lightning_.strategies import BioPrecisionMixin
from bioplausible.lightning_.strategies import build_trainer

__all__ = [
    "BioLightningModule",
    "BioOptunaPruner",
    "BioRayTuneSearch",
    "BioPrecisionCallback",
    "EnergyConvergenceCallback",
    "BioPredictionWriter",
    "BioPrecisionMixin",
    "build_trainer",
    "run_pl_trial",
    "run_pl_trial_with_wandb",
    "run_nas_search",
]
