"""
EqProp Kernel: Pure NumPy/CuPy Equilibrium Propagation

Standalone implementation without PyTorch autograd. Can use CuPy for GPU
acceleration or fall back to NumPy for CPU/portability.

Key advantages:
- No computation graph overhead
- O(1) memory training via contrastive Hebbian
- Direct portability to HLS/Verilog for FPGA

Usage:
    from bioplausible.kernel import EqPropKernel
    kernel = EqPropKernel(784, 256, 10, use_gpu=True)
    kernel.train_step(x_batch, y_batch)
"""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Robust CUDA_PATH detection logic
# Priority:
# 1. Environment variable (explicit override)
# 2. torch.utils.cpp_extension.CUDA_HOME
# 3. nvcc location
# 4. Standard system paths / ldconfig / pip packages


def _find_cuda_path() -> Optional[str]:
    """Finds the CUDA installation path using a simplified 4-source fallback."""

    # 1. Environment variable
    if "CUDA_PATH" in os.environ and os.path.exists(os.environ["CUDA_PATH"]):
        return os.environ["CUDA_PATH"]

    # 2. Ask PyTorch (best bet for compatibility)
    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME and os.path.exists(CUDA_HOME):
            return CUDA_HOME
    except ImportError, Exception:
        pass

    # 3. Look for nvcc
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        # Resolve symlinks:
        # /usr/bin/nvcc -> /etc/alternatives/nvcc -> /usr/local/cuda/bin/nvcc
        try:
            real_nvcc_path = os.path.realpath(nvcc_path)
            # /usr/local/cuda/bin/nvcc -> /usr/local/cuda
            cuda_root = os.path.dirname(os.path.dirname(real_nvcc_path))
            if os.path.exists(os.path.join(cuda_root, "bin", "nvcc")):
                return cuda_root
        except Exception:
            pass

    # 4. Fallbacks (Pip, Library Search, Standard Paths)

    # 4a. Pip-installed nvidia-cuda-nvcc (common in PyTorch 2.x)
    try:
        import nvidia.cuda_nvcc

        nvcc_pkg_path = os.path.dirname(nvidia.cuda_nvcc.__file__)
        if os.path.exists(os.path.join(nvcc_pkg_path, "bin", "nvcc")):
            return nvcc_pkg_path
    except ImportError:
        pass

    # 4b. Standard System Paths
    common_paths = [
        "/usr/local/cuda",
        "/opt/cuda",
        "/usr/lib/cuda",
        "/usr/lib/nvidia-cuda-toolkit",
    ]
    # Add versioned paths (CUDA 12.x and 11.x)
    cuda_versions = [
        "12.8",
        "12.6",
        "12.5",
        "12.4",
        "12.3",
        "12.2",
        "12.1",
        "12.0",
        "11.8",
        "11.7",
    ]
    for ver in cuda_versions:
        common_paths.append(f"/usr/local/cuda-{ver}")

    for path in common_paths:
        if os.path.exists(path) and os.path.isdir(path):
            if os.path.exists(os.path.join(path, "bin", "nvcc")) or os.path.exists(
                os.path.join(path, "include", "cuda.h")
            ):
                return path

    # 4c. Library Search (LD_LIBRARY_PATH, ldconfig) via ctypes
    try:
        from ctypes.util import find_library

        cudart = find_library("cudart")
        if cudart:
            # If absolute path, use it
            if os.path.isabs(cudart) and os.path.exists(cudart):
                cuda_root = os.path.dirname(os.path.dirname(cudart))
                if os.path.exists(cuda_root):
                    return cuda_root
            else:
                # Check LD_LIBRARY_PATH
                ld_path = os.environ.get("LD_LIBRARY_PATH", "")
                for lib_dir in ld_path.split(os.pathsep):
                    if not lib_dir:
                        continue
                    potential_path = os.path.join(lib_dir, cudart)
                    if os.path.exists(potential_path):
                        cuda_root = os.path.dirname(os.path.dirname(potential_path))
                        if os.path.exists(cuda_root):
                            return cuda_root
                        break
    except Exception:
        pass

    # 5. Fallback for Windows or unusual Linux setups
    if os.name == "nt":
        # Check Program Files
        pg_files = os.environ.get("ProgramFiles", "C:\\Program Files")
        nvidia_gpu = os.path.join(pg_files, "NVIDIA GPU Computing Toolkit", "CUDA")
        if os.path.exists(nvidia_gpu):
            # Return highest version
            versions = sorted(os.listdir(nvidia_gpu), reverse=True)
            if versions:
                return os.path.join(nvidia_gpu, versions[0])

    return None


_detected_cuda_path = _find_cuda_path()
if _detected_cuda_path:
    os.environ["CUDA_PATH"] = _detected_cuda_path
    # Also ensure it is in PATH for nvcc if not already
    bin_path = os.path.join(_detected_cuda_path, "bin")
    if os.path.exists(bin_path) and bin_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")

# Try to import CuPy for GPU
try:
    import cupy as cp

    # Verify it actually works (catches CUDA_PATH errors)
    try:
        if hasattr(cp, "cuda") and cp.cuda.is_available():
            with cp.cuda.Device(0):
                _ = cp.array([1.0])
                _ = cp.random.rand(1)  # Trigger random generator init
            HAS_CUPY = True
        else:
            HAS_CUPY = False
            cp = None
    except Exception:
        HAS_CUPY = False
        cp = None

except ImportError:
    cp = None
    HAS_CUPY = False
except Exception:  # Capture other potential import errors
    cp = None
    HAS_CUPY = False

# Try to import Triton kernels
try:
    from bioplausible.models.triton_kernel import TritonEqPropOps

    HAS_TRITON_OPS = True
except ImportError:
    TritonEqPropOps = None
    HAS_TRITON_OPS = False


def get_backend(use_gpu: bool) -> Any:
    """Return appropriate array library (CuPy or NumPy)."""
    if use_gpu and HAS_CUPY:
        return cp
    return np


def to_numpy(arr: Any) -> np.ndarray:
    """Convert array to NumPy (handles both NumPy and CuPy arrays)."""
    if HAS_CUPY:
        try:
            if hasattr(arr, "__class__") and arr.__class__.__module__.startswith(
                "cupy"
            ):
                return cp.asnumpy(arr)
        except Exception:
            pass
    return arr


def softmax(x: np.ndarray, xp: Any = np) -> np.ndarray:
    """Stable softmax."""
    x_max = xp.max(x, axis=-1, keepdims=True)
    exp_x = xp.exp(x - x_max)
    return exp_x / xp.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy(logits: np.ndarray, targets: np.ndarray, xp: Any = np) -> float:
    """Cross-entropy loss from logits."""
    batch_size = logits.shape[0]
    probs = softmax(logits, xp)
    probs = xp.clip(probs, 1e-10, 1.0)
    log_probs = xp.log(probs)
    loss = -xp.sum(log_probs[xp.arange(batch_size), targets]) / batch_size
    return loss


def tanh_deriv(x: np.ndarray, xp: Any = np) -> np.ndarray:
    """Derivative of tanh: 1 - tanh(x)^2"""
    return 1 - xp.tanh(x) ** 2


def spectral_normalize(
    W: np.ndarray, num_iters: int = 1, u: Optional[np.ndarray] = None, xp: Any = np
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """Power iteration spectral normalization.

    Normalizes W by its largest singular value (spectral norm).
    This ensures the operator norm ‖W‖ ≈ 1, maintaining Lipschitz < 1.

    Args:
        W: Weight matrix [out_dim, in_dim]
        num_iters: Power iteration steps (1 is usually enough)
        u: Previous u vector for warm start
        xp: Array module (np or cp)

    Returns:
        W_normalized: Normalized weight matrix
        u_new: Updated u vector for next call
        sigma: Estimated spectral norm
    """
    out_dim, in_dim = W.shape

    u = _initialize_u_vector(u, out_dim, W.dtype, xp)

    for _ in range(num_iters):
        v = _compute_v_vector(W, u, xp)
        u = _compute_u_vector(W, v, xp)

    sigma = _compute_spectral_norm(W, u, v)
    W_normalized = W / (_add_epsilon(sigma))

    return W_normalized, u, sigma


def _add_epsilon(value: float, epsilon: float = 1e-12) -> float:
    """Add small epsilon to prevent division by zero."""
    return value + epsilon


def _initialize_u_vector(
    u: Optional[np.ndarray], out_dim: int, dtype: np.dtype, xp: Any
) -> np.ndarray:
    """Initialize or validate the u vector for power iteration."""
    if u is None:
        u = xp.random.randn(out_dim).astype(dtype)
    return u / xp.linalg.norm(u)


def _compute_v_vector(W: np.ndarray, u: np.ndarray, xp: Any) -> np.ndarray:
    """Compute v vector in power iteration: v = W.T @ u, normalized."""
    v = W.T @ u
    norm = xp.linalg.norm(v)
    return v / _add_epsilon(norm)


def _compute_u_vector(W: np.ndarray, v: np.ndarray, xp: Any) -> np.ndarray:
    """Compute u vector in power iteration: u = W @ v, normalized."""
    u = W @ v
    norm = xp.linalg.norm(u)
    return u / _add_epsilon(norm)


def _compute_spectral_norm(W: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Compute the spectral norm (largest singular value) of W."""
    return u @ W @ v


class EqPropKernel:
    """Pure NumPy/CuPy Equilibrium Propagation kernel.

    Implements:
    - Forward pass to equilibrium
    - Free and nudged phases
    - Contrastive Hebbian weight updates
    - Spectral normalization for stability
    - Adam optimizer

    Example:
        >>> kernel = EqPropKernel(784, 256, 10, use_gpu=True)
        >>> for x_batch, y_batch in data_loader:
        ...     metrics = kernel.train_step(x_batch, y_batch)
        ...     print(f"Loss: {metrics['loss']:.4f}")
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        gamma: float = 0.5,
        beta: float = 0.22,
        max_steps: int = 10,
        epsilon: float = 1e-3,
        lr: float = 0.001,
        use_spectral_norm: bool = True,
        use_gpu: bool = False,
        adaptive_epsilon: bool = True,
        architecture: str = "layered",  # "layered" or "rnn"
    ) -> None:
        """Initialize EqProp kernel."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.beta = beta
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.lr = lr
        self.use_spectral_norm = use_spectral_norm
        self.use_gpu = use_gpu and HAS_CUPY
        self.adaptive_epsilon = adaptive_epsilon
        self.architecture = architecture

        self.xp = get_backend(self.use_gpu)

        # Initialize weights
        scale = 0.5
        if self.architecture == "layered":
            self.weights = {
                "embed": self._init_weight(input_dim, hidden_dim, scale),
                "W1": self._init_weight(hidden_dim, hidden_dim * 4, scale),
                "W2": self._init_weight(hidden_dim * 4, hidden_dim, scale),
                "head": self._init_weight(hidden_dim, output_dim, scale),
            }
            self.biases = {
                "embed": self.xp.zeros(hidden_dim, dtype=np.float32),
                "W1": self.xp.zeros(hidden_dim * 4, dtype=np.float32),
                "W2": self.xp.zeros(hidden_dim, dtype=np.float32),
                "head": self.xp.zeros(output_dim, dtype=np.float32),
            }
            self.sn_state: Dict[str, Optional[np.ndarray]] = {
                "W1_u": None,
                "W2_u": None,
            }
        elif self.architecture == "rnn":
            self.weights = {
                "W_in": self._init_weight(input_dim, hidden_dim, scale),
                "W_rec": self._init_weight(hidden_dim, hidden_dim, scale),
                "W_out": self._init_weight(hidden_dim, output_dim, scale),
            }
            self.biases = {
                "W_in": self.xp.zeros(hidden_dim, dtype=np.float32),
                "W_rec": self.xp.zeros(hidden_dim, dtype=np.float32),
                "W_out": self.xp.zeros(output_dim, dtype=np.float32),
            }
            self.sn_state: Dict[str, Optional[np.ndarray]] = {"W_rec_u": None}
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        # Adam state
        self.adam_state = {
            "m": {k: self.xp.zeros_like(v) for k, v in self.weights.items()},
            "v": {k: self.xp.zeros_like(v) for k, v in self.weights.items()},
            "t": 0,
        }

    def _init_weight(self, in_dim: int, out_dim: int, scale: float = 0.5) -> np.ndarray:
        """Initialize weight matrix with Xavier-like initialization."""
        xp = self.xp
        std = scale * np.sqrt(2.0 / (in_dim + out_dim))
        W = xp.random.randn(out_dim, in_dim).astype(np.float32) * std
        return W

    def _get_normalized_weights(self) -> Dict[str, np.ndarray]:
        """Get spectral-normalized weights."""
        if not self.use_spectral_norm:
            return self.weights.copy()

        weights = self.weights.copy()

        if self.architecture == "layered":
            weights["W1"] = self._normalize_weight("W1", "W1_u")
            weights["W2"] = self._normalize_weight("W2", "W2_u")
        elif self.architecture == "rnn":
            weights["W_rec"] = self._normalize_weight("W_rec", "W_rec_u")

        return weights

    def _should_normalize_weight(self, weight_key: str) -> bool:
        """Check if a weight should be normalized."""
        if not self.use_spectral_norm:
            return False
        if self.architecture == "layered":
            return weight_key in ["W1", "W2"]
        elif self.architecture == "rnn":
            return weight_key in ["W_rec"]
        return False

    def _normalize_weight(self, weight_key: str, sn_state_key: str) -> np.ndarray:
        """Normalize a specific weight matrix using spectral normalization."""
        weight = self.weights[weight_key]
        u_state = self.sn_state[sn_state_key]

        normalized_weight, new_u_state, _ = spectral_normalize(
            weight, u=u_state, xp=self.xp
        )

        self.sn_state[sn_state_key] = new_u_state
        return normalized_weight

    def forward_step(
        self,
        h: np.ndarray,
        x_emb: np.ndarray,
        weights: Dict[str, np.ndarray],
        return_activations: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Single equilibrium step."""
        xp = self.xp

        if self.architecture == "layered":
            h_mean = xp.mean(h, axis=-1, keepdims=True)
            h_std = xp.std(h, axis=-1, keepdims=True) + 1e-5
            h_norm = (h - h_mean) / h_std

            ffn_hidden = xp.tanh(h_norm @ weights["W1"].T + self.biases["W1"])
            ffn_out = ffn_hidden @ weights["W2"].T + self.biases["W2"]

            if (
                HAS_TRITON_OPS
                and self.use_gpu
                and HAS_CUPY
                and isinstance(h, cp.ndarray)
            ):
                h_next = TritonEqPropOps.step_linear_cupy(
                    h, ffn_out + x_emb, self.gamma
                )
            else:
                h_next = (1 - self.gamma) * h + self.gamma * (ffn_out + x_emb)

            if return_activations:
                activations = {
                    "h_norm": h_norm,
                    "ffn_hidden": ffn_hidden,
                    "h": h,
                    "h_next": h_next,
                }
                return h_next, activations
            return h_next, None

        elif self.architecture == "rnn":
            # RNN: h = (1-gamma)h + gamma * tanh(W_rec h + x_emb)
            # x_emb here is W_in @ x + b_in
            pre_act = h @ weights["W_rec"].T + self.biases["W_rec"] + x_emb

            if (
                HAS_TRITON_OPS
                and self.use_gpu
                and HAS_CUPY
                and isinstance(h, cp.ndarray)
            ):
                # Using Triton kernel which fuses (1-a)h + a*tanh(pre_act)
                h_next = TritonEqPropOps.step_cupy(h, pre_act, self.gamma)
            else:
                h_next = (1 - self.gamma) * h + self.gamma * xp.tanh(pre_act)

            if return_activations:
                activations = {
                    "h": h,
                    "h_next": h_next,
                }
                return h_next, activations
            return h_next, None
        return h, None

    def solve_equilibrium(
        self,
        x: np.ndarray,
        nudge_grad: Optional[np.ndarray] = None,
        store_trajectory: bool = False,
    ) -> Tuple[np.ndarray, List[Dict[str, np.ndarray]], Dict[str, Any]]:
        """Find equilibrium state h* via fixed-point iteration."""
        xp = self.xp
        batch_size = x.shape[0]

        x = self._prepare_input(x)
        x_emb = self._compute_embedded_input(x)
        weights = self._get_normalized_weights()
        h = xp.zeros((batch_size, self.hidden_dim), dtype=np.float32)

        activations_log = []
        last_activations = None

        for t in range(self.max_steps):
            h_prev = h

            h, activations = self._perform_equilibrium_step(
                h, x_emb, weights, nudge_grad, return_activations=True
            )
            last_activations = activations

            if store_trajectory:
                activations_log.append(activations)

            if self._check_convergence(h, h_prev, t):
                if not store_trajectory:
                    activations_log = [last_activations]
                return h, activations_log, {"steps": t + 1, "converged": True}

        if not store_trajectory:
            activations_log = [last_activations]

        return h, activations_log, {"steps": self.max_steps, "converged": False}

    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """Prepare input for processing on the appropriate device."""
        if self.use_gpu and not isinstance(x, self.xp.ndarray):
            return self.xp.asarray(x)
        return x

    def _compute_embedded_input(self, x: np.ndarray) -> np.ndarray:
        """Compute embedded input representation."""
        if self.architecture == "layered":
            return x @ self.weights["embed"].T + self.biases["embed"]
        elif self.architecture == "rnn":
            return x @ self.weights["W_in"].T + self.biases["W_in"]
        return x

    def _perform_equilibrium_step(
        self,
        h: np.ndarray,
        x_emb: np.ndarray,
        weights: Dict[str, np.ndarray],
        nudge_grad: Optional[np.ndarray],
        return_activations: bool = True,
    ) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """Perform a single equilibrium step, applying nudge if provided."""
        h, activations = self.forward_step(
            h, x_emb, weights, return_activations=return_activations
        )

        if nudge_grad is not None:
            h = h - self.beta * nudge_grad

        return h, activations

    def _check_convergence(self, h: np.ndarray, h_prev: np.ndarray, step: int) -> bool:
        """Check if the equilibrium has converged."""
        # OPTIMIZATION: Use max norm (simpler, faster)
        # Original: diff = self.xp.max(self.xp.linalg.norm(h - h_prev, axis=1))
        diff = self.xp.abs(h - h_prev).max()
        threshold = self._get_convergence_threshold(step)
        return diff < threshold

    def _get_convergence_threshold(self, step: int) -> float:
        """Get the convergence threshold based on the current step."""
        multiplier = 2.0 if self.adaptive_epsilon and step > 5 else 1.0
        return self.epsilon * multiplier

    def compute_output(self, h: np.ndarray) -> np.ndarray:
        """Compute output logits from hidden state."""
        if self.architecture == "layered":
            return h @ self.weights["head"].T + self.biases["head"]
        elif self.architecture == "rnn":
            return h @ self.weights["W_out"].T + self.biases["W_out"]
        return h

    def compute_hebbian_update(
        self,
        act_free: Dict[str, np.ndarray],
        act_nudged: Dict[str, np.ndarray],
        x: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute contrastive Hebbian weight updates."""
        batch_size = act_free["h"].shape[0]
        grads = {}

        if self.architecture == "layered":
            grad_free_W2 = act_free["h_next"].T @ act_free["ffn_hidden"] / batch_size
            grad_nudged_W2 = (
                act_nudged["h_next"].T @ act_nudged["ffn_hidden"] / batch_size
            )
            grads["W2"] = (1.0 / self.beta) * (grad_nudged_W2 - grad_free_W2)

            grad_free_W1 = act_free["ffn_hidden"].T @ act_free["h_norm"] / batch_size
            grad_nudged_W1 = (
                act_nudged["ffn_hidden"].T @ act_nudged["h_norm"] / batch_size
            )
            grads["W1"] = (1.0 / self.beta) * (grad_nudged_W1 - grad_free_W1)

        elif self.architecture == "rnn":
            grad_free_rec = act_free["h_next"].T @ act_free["h"] / batch_size
            grad_nudged_rec = act_nudged["h_next"].T @ act_nudged["h"] / batch_size
            grads["W_rec"] = (1.0 / self.beta) * (grad_nudged_rec - grad_free_rec)

            if x is not None:
                grad_free_in = act_free["h_next"].T @ x / batch_size
                grad_nudged_in = act_nudged["h_next"].T @ x / batch_size
                grads["W_in"] = (1.0 / self.beta) * (grad_nudged_in - grad_free_in)

        return grads

    def adam_update(
        self,
        grads: Dict[str, np.ndarray],
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Apply Adam optimizer update."""
        self.adam_state["t"] += 1
        t = self.adam_state["t"]

        for key in grads:
            if key not in self.weights:
                continue

            g = grads[key]
            self.adam_state["m"][key] = (
                beta1 * self.adam_state["m"][key] + (1 - beta1) * g
            )
            self.adam_state["v"][key] = beta2 * self.adam_state["v"][key] + (
                1 - beta2
            ) * (g**2)

            m_hat = self.adam_state["m"][key] / (1 - beta1**t)
            v_hat = self.adam_state["v"][key] / (1 - beta2**t)

            self.weights[key] -= self.lr * m_hat / (self.xp.sqrt(v_hat) + eps)

    def train_step(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Full EqProp training step."""
        xp = self.xp

        # Prepare inputs
        x, y = self._prepare_inputs(x, y)

        # Free Phase
        h_free, act_log_free, info_free = self.solve_equilibrium(x)

        # Compute gradients for nudging
        logits, d_logits, nudge_grad = self._compute_gradients_for_nudging(
            h_free, y, xp
        )

        # Nudged Phase
        h_nudged, act_log_nudged, info_nudged = self.solve_equilibrium(x, nudge_grad)

        # Compute Updates
        grads = self.compute_hebbian_update(act_log_free[-1], act_log_nudged[-1], x)

        if self.architecture == "layered":
            grads["head"] = d_logits.T @ h_free / self._get_batch_size(x)
        elif self.architecture == "rnn":
            grads["W_out"] = d_logits.T @ h_free / self._get_batch_size(x)

        self.adam_update(grads)

        # Compute metrics
        metrics = self._compute_training_metrics(logits, y, info_free, info_nudged, xp)

        return metrics

    def _prepare_inputs(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare input tensors for processing."""
        xp = self.xp

        if not isinstance(x, np.ndarray) and not (
            HAS_CUPY and cp is not None and isinstance(x, cp.ndarray)
        ):
            x = np.asarray(x)
        if self.use_gpu:
            x = xp.asarray(x)
            y = xp.asarray(y)

        return x, y

    def _compute_gradients_for_nudging(
        self, h_free: np.ndarray, y: np.ndarray, xp: Any
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients needed for the nudging phase."""
        logits = self.compute_output(h_free)
        probs = softmax(logits, xp)
        batch_size = self._get_batch_size(logits)

        one_hot = xp.zeros_like(probs)
        one_hot[xp.arange(batch_size), y] = 1.0
        d_logits = probs - one_hot

        if self.architecture == "layered":
            nudge_grad = d_logits @ self.weights["head"]
        elif self.architecture == "rnn":
            nudge_grad = d_logits @ self.weights["W_out"]

        return logits, d_logits, nudge_grad

    def _get_batch_size(self, tensor: np.ndarray) -> int:
        """Get the batch size from a tensor."""
        return tensor.shape[0]

    def _compute_training_metrics(
        self,
        logits: np.ndarray,
        y: np.ndarray,
        info_free: Dict[str, Any],
        info_nudged: Dict[str, Any],
        xp: Any,
    ) -> Dict[str, float]:
        """Compute training metrics."""
        loss = cross_entropy(logits, y, xp)
        preds = xp.argmax(logits, axis=1)
        accuracy = xp.mean(preds == y)

        return {
            "loss": float(to_numpy(loss)),
            "accuracy": float(to_numpy(accuracy)),
            "free_steps": info_free["steps"],
            "nudged_steps": info_nudged["steps"],
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Run inference on input."""
        xp = self.xp
        if self.use_gpu:
            x = xp.asarray(x)

        h_star, _, _ = self.solve_equilibrium(x)
        logits = self.compute_output(h_star)
        return to_numpy(xp.argmax(logits, axis=1))

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate accuracy on a dataset."""
        xp = self.xp

        # Prepare inputs
        x, y = self._prepare_inputs(x, y)

        h_star, _, _ = self.solve_equilibrium(x)
        logits = self.compute_output(h_star)

        # Metrics
        loss = cross_entropy(logits, y, xp)
        preds = xp.argmax(logits, axis=1)
        accuracy = xp.mean(preds == y)

        return {
            "loss": float(to_numpy(loss)),
            "accuracy": float(to_numpy(accuracy)),
        }


class EqPropKernelBPTT:
    """
    NumPy/CuPy kernel that replicates PyTorch's BPTT through equilibrium iterations.

    This is O(steps) memory but gives IDENTICAL gradients to PyTorch.
    Now with optional GPU acceleration via CuPy.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_steps: int = 30,
        lr: float = 0.01,
        use_gpu: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_steps = max_steps
        self.lr = lr
        self.use_gpu = use_gpu and HAS_CUPY

        # Get array backend
        self.xp = get_backend(self.use_gpu)

        # Xavier initialization with gain=0.5
        scale = 0.5
        xp = self.xp
        self.W_in = (
            xp.random.randn(hidden_dim, input_dim).astype(xp.float32)
            * scale
            * xp.sqrt(2.0 / input_dim)
        )
        self.W_rec = (
            xp.random.randn(hidden_dim, hidden_dim).astype(xp.float32)
            * scale
            * xp.sqrt(2.0 / hidden_dim)
        )
        self.W_out = (
            xp.random.randn(output_dim, hidden_dim).astype(xp.float32)
            * scale
            * xp.sqrt(2.0 / hidden_dim)
        )

        self.b_in = xp.zeros(hidden_dim, dtype=xp.float32)
        self.b_rec = xp.zeros(hidden_dim, dtype=xp.float32)
        self.b_out = xp.zeros(output_dim, dtype=xp.float32)

    def forward(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """Forward pass storing trajectory for BPTT."""
        xp = self.xp

        # Convert input to GPU if needed
        if self.use_gpu and not isinstance(x, xp.ndarray):
            x = xp.asarray(x)

        batch_size = x.shape[0]

        # Compute x_proj once
        x_proj = x @ self.W_in.T + self.b_in

        # Initialize h
        h = xp.zeros((batch_size, self.hidden_dim), dtype=xp.float32)

        # Store trajectory (pre-activations) for backprop
        trajectory = []  # List of (pre_act, h) pairs

        for _ in range(self.max_steps):
            pre_act = x_proj + h @ self.W_rec.T + self.b_rec

            if (
                HAS_TRITON_OPS
                and self.use_gpu
                and HAS_CUPY
                and isinstance(h, cp.ndarray)
            ):
                # Use Triton kernel for tanh update: (1-a)h + a*tanh(pre_act) with a=1.0
                h = TritonEqPropOps.step_cupy(h, pre_act, alpha=1.0)
            else:
                h = xp.tanh(pre_act)

            trajectory.append((pre_act.copy(), h.copy()))

        # Output
        logits = h @ self.W_out.T + self.b_out

        return logits, trajectory

    def backward(
        self,
        x: np.ndarray,
        trajectory: List[Tuple[np.ndarray, np.ndarray]],
        d_logits: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Backprop through time - exactly matches PyTorch.

        Returns gradients for all parameters.
        """
        xp = self.xp

        if self.use_gpu and not isinstance(x, xp.ndarray):
            x = xp.asarray(x)

        batch_size = x.shape[0]

        # Gradient w.r.t. output layer
        h_final = trajectory[-1][1]
        dW_out = d_logits.T @ h_final / batch_size
        db_out = d_logits.mean(axis=0)

        # Gradient w.r.t. final hidden state
        dh = d_logits @ self.W_out  # [batch, hidden]

        # Initialize gradient accumulators
        dW_rec = xp.zeros_like(self.W_rec)
        dW_in = xp.zeros_like(self.W_in)
        db_rec = xp.zeros_like(self.b_rec)

        # BPTT: backprop through all timesteps
        for t in reversed(range(self.max_steps)):
            pre_act, h = trajectory[t]

            # Gradient through tanh
            dtanh = dh * tanh_deriv(pre_act, xp)  # [batch, hidden]

            # Accumulate gradients
            if t > 0:
                h_prev = trajectory[t - 1][1]
            else:
                h_prev = xp.zeros_like(h)

            dW_rec += dtanh.T @ h_prev / batch_size
            dW_in += dtanh.T @ x / batch_size
            db_rec += dtanh.mean(axis=0)

            # Gradient to previous hidden state
            dh = dtanh @ self.W_rec

        return {
            "dW_out": dW_out,
            "db_out": db_out,
            "dW_rec": dW_rec,
            "db_rec": db_rec,
            "dW_in": dW_in,
        }

    def train_step(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Complete training step with BPTT."""
        xp = self.xp

        # Convert input to GPU if needed
        if self.use_gpu:
            if not isinstance(x, xp.ndarray):
                x = xp.asarray(x)
            if not isinstance(y, xp.ndarray):
                y = xp.asarray(y)

        batch_size = x.shape[0]

        # Forward
        logits, trajectory = self.forward(x)

        # Loss gradient
        probs = softmax(logits, xp)
        one_hot = xp.zeros_like(probs)
        one_hot[xp.arange(batch_size), y] = 1.0
        d_logits = probs - one_hot

        # Backward
        grads = self.backward(x, trajectory, d_logits)

        # Update
        self.W_out -= self.lr * grads["dW_out"]
        self.W_rec -= self.lr * grads["dW_rec"]
        self.W_in -= self.lr * grads["dW_in"]
        self.b_out -= self.lr * grads["db_out"]
        self.b_rec -= self.lr * grads["db_rec"]

        # Metrics
        loss = cross_entropy(logits, y, xp)
        preds = xp.argmax(logits, axis=1)
        acc = xp.mean(preds == y)

        return {"loss": float(to_numpy(loss)), "accuracy": float(to_numpy(acc))}

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate accuracy."""
        xp = self.xp

        if self.use_gpu:
            if not isinstance(x, xp.ndarray):
                x = xp.asarray(x)
            if not isinstance(y, xp.ndarray):
                y = xp.asarray(y)

        logits, _ = self.forward(x)
        preds = xp.argmax(logits, axis=1)
        acc = xp.mean(preds == y)
        loss = cross_entropy(logits, y, xp)
        return {"accuracy": float(to_numpy(acc)), "loss": float(to_numpy(loss))}


def compare_memory_autograd_vs_kernel(hidden_dim: int, depth: int) -> Dict[str, float]:
    """Compare memory usage."""
    kernel_activation = 32 * hidden_dim * 4
    autograd_activation = 32 * hidden_dim * depth * 4
    return {
        "kernel_activation_mb": kernel_activation / 1e6,
        "autograd_activation_mb": autograd_activation / 1e6,
        "ratio": autograd_activation / kernel_activation,
    }


__all__ = [
    "EqPropKernel",
    "EqPropKernelBPTT",
    "HAS_CUPY",
    "get_backend",
    "to_numpy",
    "spectral_normalize",
    "compare_memory_autograd_vs_kernel",
]
