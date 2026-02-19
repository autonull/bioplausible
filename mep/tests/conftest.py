import pytest
import torch
import numpy as np
import random

@pytest.fixture(autouse=True)
def seed_everything():
    """Set random seeds for reproducibility."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@pytest.fixture(scope="session")
def device():
    """Return the device to run tests on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
