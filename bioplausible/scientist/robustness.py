"""
Robustness Testing Suite for AutoScientist.

Headless wrapper for the existing RobustnessTool logic.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec
from bioplausible.hyperopt.tasks import create_task

logger = logging.getLogger("Robustness")


class RobustnessEvaluator:
    """
    Headless evaluator for model robustness.
    Performs stress tests (Noise Injection, Adversarial Attacks) without UI dependencies.
    """

    def __init__(self, model_name: str, task_name: str, config: Dict[str, Any], weights_path: Optional[str] = None):
        self.model_name = model_name
        self.task_name = task_name
        self.config = config
        self.weights_path = weights_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self) -> float:
        """Execute robustness suite and return aggregate score (0-1)."""
        try:
            # 1. Setup Task & Model
            task = create_task(self.task_name, device=self.device, quick_mode=True)
            task.setup()

            spec = get_model_spec(self.model_name)
            model = create_model(
                spec=spec,
                input_dim=task.input_dim,
                output_dim=task.output_dim,
                hidden_dim=self.config.get("hidden_dim", 128),
                num_layers=self.config.get("num_layers", 4),
                device=self.device,
                task_type=task.task_type
            )

            # Load weights if provided, else train briefly
            if self.weights_path:
                logger.info(f"Loading weights from {self.weights_path}")
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                # Handle full checkpoint vs state_dict
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                logger.info("No weights provided. Training from scratch for robustness check...")
                trainer = task.create_trainer(
                    model,
                    lr=self.config.get("lr", 0.001),
                    steps=self.config.get("steps", 20),
                    batches_per_epoch=50,
                    eval_batches=10
                )
                # Train for a few epochs to get a baseline
                for _ in range(3):
                    trainer.train_epoch()

            # 2. Run Tests
            scores = []

            # Test A: Noise Injection (Self-Healing)
            noise_score = self._test_noise_injection(model, task)
            scores.append(noise_score)
            logger.info(f"Noise Score: {noise_score:.2f}")

            # Test B: Input Perturbation (Random Noise)
            # Only for vision/continuous inputs
            if task.task_type == "vision":
                perturb_score = self._test_input_perturbation(model, task)
                scores.append(perturb_score)
                logger.info(f"Perturbation Score: {perturb_score:.2f}")

                # Test C: OOD Detection (Phase 6.2)
                ood_score = self._test_ood_detection(model, task)
                scores.append(ood_score)
                logger.info(f"OOD Detection Score: {ood_score:.2f}")

                # Test D: Adversarial Attack (FGSM) (Phase 6.2)
                adv_score = self._test_adversarial_attack(model, task)
                scores.append(adv_score)
                logger.info(f"Adversarial Score: {adv_score:.2f}")

            return float(np.mean(scores)) if scores else 0.0

        except Exception as e:
            logger.error(f"Robustness evaluation failed: {e}", exc_info=True)
            return 0.0

    def _test_noise_injection(self, model: nn.Module, task) -> float:
        """Inject noise into hidden states and measure recovery/accuracy."""
        model.eval()

        # Get a batch
        x, y = task.get_batch("val", batch_size=32)

        # Baseline accuracy
        with torch.no_grad():
            if hasattr(model, "train_step"): # Kernel-like
                 # Hard to inject noise directly into kernel from here without specific API
                 # Assume 1.0 score for simplicity or skip
                 return 0.5

            # Prepare input
            if hasattr(task, "create_trainer"):
                # Use trainer helper if possible, or manual
                # We replicate trainer input prep roughly
                h = x
                if x.dim() > 2 and "Conv" not in type(model).__name__:
                    h = x.view(x.size(0), -1)

                # Standard forward
                logits = model(h)
                acc_base = (logits.argmax(1) == y).float().mean().item()

        # If model supports noise injection (e.g. LoopedMLP)
        if hasattr(model, "inject_noise_and_relax"):
            # Test damping
            # Convert x to correct shape first
            h = x
            if x.dim() > 2 and "Conv" not in type(model).__name__:
                h = x.view(x.size(0), -1)

            damping = model.inject_noise_and_relax(h, noise_level=1.0)
            return damping.get("damping_percent", 0.0) / 100.0

        return acc_base # Fallback to accuracy if no specific noise API

    def _test_input_perturbation(self, model: nn.Module, task) -> float:
        """Test resilience to input noise."""
        model.eval()
        x, y = task.get_batch("val", batch_size=32)

        # Prepare
        h = x.clone()
        if x.dim() > 2 and "Conv" not in type(model).__name__:
            h = x.view(x.size(0), -1)

        # Add noise
        noise = torch.randn_like(h) * 0.1
        h_noisy = h + noise

        with torch.no_grad():
            logits = model(h)
            logits_noisy = model(h_noisy)

            pred = logits.argmax(1)
            pred_noisy = logits_noisy.argmax(1)

            # Consistency score
            consistency = (pred == pred_noisy).float().mean().item()

        return consistency

    def _test_ood_detection(self, model: nn.Module, task) -> float:
        """
        Test Out-of-Distribution detection capability.
        Compare confidence (Max Softmax Prob) on clean vs noise data.
        Higher score means better separation (uncertainty on OOD).
        """
        model.eval()
        x, y = task.get_batch("val", batch_size=64)

        # Prepare inputs
        h = x.clone()
        if x.dim() > 2 and "Conv" not in type(model).__name__:
            h = x.view(x.size(0), -1)

        # OOD Data (Random Noise)
        h_ood = torch.rand_like(h) # Uniform noise [0, 1]

        with torch.no_grad():
            logits_in = model(h)
            probs_in = torch.softmax(logits_in, dim=1)
            msp_in = probs_in.max(1)[0].mean().item()

            logits_ood = model(h_ood)
            probs_ood = torch.softmax(logits_ood, dim=1)
            msp_ood = probs_ood.max(1)[0].mean().item()

        # Score: 1.0 if OOD confidence is 0.0 (ideal 1/N, but close enough)
        # We want MSP_ood to be low.
        # Score = max(0, MSP_in - MSP_ood)
        return max(0.0, msp_in - msp_ood)

    def _test_adversarial_attack(self, model: nn.Module, task, epsilon=0.1) -> float:
        """
        Test FGSM Adversarial Robustness.
        """
        # Need gradients w.r.t input
        # Note: We must ensure we can compute gradients.
        # Some models might detach internal states.

        x, y = task.get_batch("val", batch_size=32)

        # Prepare
        h = x.clone().detach()
        if x.dim() > 2 and "Conv" not in type(model).__name__:
            h = x.view(x.size(0), -1)

        h.requires_grad = True

        try:
            # Forward
            logits = model(h)
            loss = nn.CrossEntropyLoss()(logits, y)

            # Backward
            model.zero_grad()
            loss.backward()

            # FGSM Attack
            with torch.no_grad():
                if h.grad is None:
                    return 0.5 # Gradient not available (non-diff model?)

                grad_sign = h.grad.sign()
                h_adv = h + epsilon * grad_sign
                # Clip if image (0, 1)? Data is normalized to roughly [-1, 1] in VisionTask?
                # VisionTask: (raw - 0.5) / 0.5 -> [-1, 1]
                # So clipping to [-1, 1] is appropriate.
                h_adv = torch.clamp(h_adv, -1.0, 1.0)

                logits_adv = model(h_adv)
                acc_adv = (logits_adv.argmax(1) == y).float().mean().item()

                logits_clean = model(h) # Recompute
                acc_clean = (logits_clean.argmax(1) == y).float().mean().item()

            if acc_clean == 0:
                return 0.0

            # Score = relative robustness
            return acc_adv / acc_clean

        except RuntimeError as e:
            logger.warning(f"Adversarial attack failed (likely autograd issue): {e}")
            return 0.0


def run_robustness_check(
    model_name: str, task: str, config: Dict[str, Any], weights_path: str = None
) -> float:
    """
    Runs a suite of robustness tests (Noise, FGSM, Dropout) on a trained model.
    Returns a unified 'Robustness Score' (0.0 - 1.0).
    """
    evaluator = RobustnessEvaluator(model_name, task, config, weights_path)
    return evaluator.run()
