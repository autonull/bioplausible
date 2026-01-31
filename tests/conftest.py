import sys
import types
from unittest.mock import MagicMock

# Helper to create mock modules
def mock_module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

# Only mock if not installed
try:
    import torch
except ImportError:
    torch = mock_module("torch")
    # Mark as package so import torch.submodule works if not in sys.modules (but we put them there)
    # torch.__path__ = []

    torch.nn = mock_module("torch.nn")
    torch.nn.functional = mock_module("torch.nn.functional")
    torch.optim = mock_module("torch.optim")
    torch.utils = mock_module("torch.utils")
    torch.utils.data = mock_module("torch.utils.data")
    torch.nn.utils = mock_module("torch.nn.utils")
    torch.nn.utils.parametrizations = mock_module("torch.nn.utils.parametrizations")
    torch.cuda = mock_module("torch.cuda")
    torch.backends = mock_module("torch.backends")
    torch.backends.cudnn = mock_module("torch.backends.cudnn")
    torch.autograd = mock_module("torch.autograd")

    # Define MockNNModule
    class MockNNModule:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            return MagicMock()
        def to(self, device):
            return self
        def eval(self):
            return self
        def train(self):
            return self
        def parameters(self):
            return []
        def state_dict(self):
            return {}
        def load_state_dict(self, d):
            pass

    torch.nn.Module = MockNNModule
    torch.nn.utils.parametrizations.spectral_norm = MagicMock(side_effect=lambda x: x)
    torch.nn.Parameter = MagicMock()

    # Autograd
    class MockFunction:
        @staticmethod
        def apply(*args, **kwargs):
            return MagicMock()
    torch.autograd.Function = MockFunction

    # Utils
    class MockDataLoader:
        def __init__(self, *args, **kwargs): pass
        def __iter__(self):
            return iter([])
    torch.utils.data.DataLoader = MockDataLoader
    torch.utils.data.Dataset = MagicMock
    torch.utils.data.TensorDataset = MagicMock
    torch.utils.data.Subset = MagicMock
    torch.utils.data.random_split = MagicMock

    # Tensor
    class MockTensor(MagicMock):
        pass

    torch.Tensor = MockTensor
    torch.float32 = "float32"
    torch.long = "long"
    torch.device = lambda x: x
    torch.no_grad = MagicMock()
    torch.manual_seed = MagicMock()
    torch.cat = MagicMock()
    torch.stack = MagicMock()
    torch.zeros = MagicMock()
    torch.ones = MagicMock()
    torch.randn = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)
    torch.backends.cudnn.benchmark = False
    torch.save = MagicMock()
    torch.load = MagicMock()

try:
    import torchvision
except ImportError:
    torchvision = mock_module("torchvision")
    torchvision.transforms = mock_module("torchvision.transforms")
    torchvision.datasets = mock_module("torchvision.datasets")
    torchvision.utils = mock_module("torchvision.utils")

try:
    import gymnasium
except ImportError:
    gymnasium = mock_module("gymnasium")
    gymnasium.spaces = mock_module("gymnasium.spaces")

# bioplausible.acceleration checks for cupy
sys.modules["cupy"] = MagicMock()
