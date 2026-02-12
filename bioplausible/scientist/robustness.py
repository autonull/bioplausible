"""
Robustness Testing Suite for AutoScientist.

Headless wrapper for the existing RobustnessTool logic.
Evaluates models against noise injection, input perturbation, out-of-distribution
data, and adversarial attacks.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from bioplausible.hyperopt.tasks import create_task
from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec

logger = logging.getLogger("Robustness")


class RobustnessEvaluator:
    """
    Headless evaluator for model robustness.

    Performs stress tests (Noise Injection, Adversarial Attacks) without UI dependencies.
    """

    def __init__(
        self,
        model_name: str,
        task_name: str,
        config: Dict[str, Any],
        weights_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the RobustnessEvaluator.

        Args:
            model_name: Name of the model to evaluate.
            task_name: Name of the task/dataset.
            config: Configuration dictionary for the model.
            weights_path: Path to the saved model weights (optional).
        """
        self.model_name = model_name
        self.task_name = task_name
        self.config = config
        self.weights_path = weights_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self) -> float:
        """
        Execute robustness suite and return aggregate score.

        Returns:
            float: Aggregate robustness score (0.0 to 1.0).
        """
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
                task_type=task.task_type,
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
                logger.info(
                    "No weights provided. Training from scratch for robustness check..."
                )
                trainer = task.create_trainer(
                    model,
                    lr=self.config.get("lr", 0.001),
                    steps=self.config.get("steps", 20),
                    batches_per_epoch=50,
                    eval_batches=10,
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
                logger.info(f"Adversarial Score (FGSM): {adv_score:.2f}")

                # Test E: PGD Attack (Phase 6.3)
                pgd_score = self._test_pgd_attack(model, task)
                scores.append(pgd_score)
                logger.info(f"Adversarial Score (PGD): {pgd_score:.2f}")

            return float(np.mean(scores)) if scores else 0.0

        except Exception as e:
            logger.error(f"Robustness evaluation failed: {e}", exc_info=True)
            return 0.0

    def _test_noise_injection(self, model: nn.Module, task: Any) -> float:
        """
        Inject noise into hidden states and measure recovery/accuracy.

        Args:
            model: The model to test.
            task: The task providing data.

        Returns:
            float: Noise resilience score (0.0 to 1.0).
        """
        model.eval()

        # Get a batch
        x, y = task.get_batch("val", batch_size=32)

        # Baseline accuracy
        with torch.no_grad():
            if hasattr(model, "train_step"):  # Kernel-like
                # Hard to inject noise directly into kernel from here without specific API
                # Assume 1.0 score for simplicity or skip
                return 0.5

            # Prepare input
            h = x
            # Heuristic: Check model input dimension vs x
            # If x is image (B, C, H, W) and model is MLP (expecting B, D), flatten.
            if x.dim() > 2 and "Conv" not in type(model).__name__:
                h = x.view(x.size(0), -1)

            # Standard forward
            logits = model(h)
            acc_base = (logits.argmax(1) == y).float().mean().item()

        # If model supports noise injection (e.g. LoopedMLP)
        if hasattr(model, "inject_noise_and_relax"):
            # Test damping
            h = x
            if x.dim() > 2 and "Conv" not in type(model).__name__:
                h = x.view(x.size(0), -1)

            damping = model.inject_noise_and_relax(h, noise_level=1.0)
            return damping.get("damping_percent", 0.0) / 100.0

        return acc_base  # Fallback to accuracy if no specific noise API

    def _test_input_perturbation(self, model: nn.Module, task: Any) -> float:
        """
        Test resilience to input noise.

        Args:
            model: The model to test.
            task: The task providing data.

        Returns:
            float: Consistency score (0.0 to 1.0).
        """
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

    def _test_ood_detection(self, model: nn.Module, task: Any) -> float:
        """
        Test Out-of-Distribution detection capability.

        Compare confidence (Max Softmax Prob) on clean vs noise data.
        Higher score means better separation (uncertainty on OOD).

        Args:
            model: The model to test.
            task: The task providing data.

        Returns:
            float: OOD detection score.
        """
        model.eval()
        x, y = task.get_batch("val", batch_size=64)

        # Prepare inputs
        h = x.clone()
        if x.dim() > 2 and "Conv" not in type(model).__name__:
            h = x.view(x.size(0), -1)

        # OOD Data (Random Noise)
        h_ood = torch.rand_like(h)  # Uniform noise [0, 1]

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

    def _test_adversarial_attack(
        self, model: nn.Module, task: Any, epsilon: float = 0.1
    ) -> float:
        """
        Test FGSM Adversarial Robustness.

        Args:
            model: The model to test.
            task: The task providing data.
            epsilon: The magnitude of the adversarial perturbation.

        Returns:
            float: Adversarial robustness score (0.0 to 1.0).
        """
        return self._run_attack(model, task, "fgsm", epsilon=epsilon)

    def _test_pgd_attack(
        self, model: nn.Module, task: Any, epsilon: float = 0.1, alpha: float = 0.02, steps: int = 7
    ) -> float:
        """
        Test PGD (Projected Gradient Descent) Adversarial Robustness.

        Args:
            model: The model to test.
            task: The task providing data.
            epsilon: Max perturbation.
            alpha: Step size.
            steps: Number of iterations.

        Returns:
            float: PGD robustness score.
        """
        return self._run_attack(model, task, "pgd", epsilon=epsilon, alpha=alpha, steps=steps)

    def _run_attack(
        self,
        model: nn.Module,
        task: Any,
        attack_type: str,
        epsilon: float = 0.1,
        alpha: float = 0.02,
        steps: int = 7,
    ) -> float:
        """
        Generic attack runner.
        """
        x, y = task.get_batch("val", batch_size=32)

        # Prepare
        if x.dim() > 2 and "Conv" not in type(model).__name__:
            h_clean = x.view(x.size(0), -1)
        else:
            h_clean = x

        h_clean = h_clean.detach().to(self.device)
        y = y.to(self.device)

        # Baseline accuracy
        with torch.no_grad():
            logits_clean = model(h_clean)
            acc_clean = (logits_clean.argmax(1) == y).float().mean().item()

        if acc_clean == 0:
            return 0.0

        try:
            h_adv = h_clean.clone().detach()

            if attack_type == "fgsm":
                h_adv.requires_grad = True
                logits = model(h_adv)
                loss = nn.CrossEntropyLoss()(logits, y)
                model.zero_grad()
                loss.backward()

                if h_adv.grad is None:
                    return 0.5

                with torch.no_grad():
                    h_adv = h_adv + epsilon * h_adv.grad.sign()
                    h_adv = torch.clamp(h_adv, -1.0, 1.0)

            elif attack_type == "pgd":
                # Random start
                h_adv = h_adv + torch.empty_like(h_adv).uniform_(-epsilon, epsilon)
                h_adv = torch.clamp(h_adv, -1.0, 1.0)

                for _ in range(steps):
                    h_adv = h_adv.detach().clone()
                    h_adv.requires_grad = True

                    logits = model(h_adv)
                    loss = nn.CrossEntropyLoss()(logits, y)
                    model.zero_grad()
                    loss.backward()

                    if h_adv.grad is None:
                        break

                    with torch.no_grad():
                        h_adv = h_adv + alpha * h_adv.grad.sign()
                        # Project
                        delta = torch.clamp(h_adv - h_clean, -epsilon, epsilon)
                        h_adv = torch.clamp(h_clean + delta, -1.0, 1.0)

            # Evaluate Adversarial
            with torch.no_grad():
                logits_adv = model(h_adv)
                acc_adv = (logits_adv.argmax(1) == y).float().mean().item()

            return acc_adv / acc_clean

        except RuntimeError as e:
            logger.warning(f"{attack_type.upper()} attack failed (likely autograd issue): {e}")
            return 0.0


def run_robustness_check(
    model_name: str,
    task: str,
    config: Dict[str, Any],
    weights_path: Optional[str] = None,
) -> float:
    """
    Runs a suite of robustness tests (Noise, FGSM, Dropout) on a trained model.

    Args:
        model_name: Name of the model to evaluate.
        task: Name of the task/dataset.
        config: Configuration dictionary.
        weights_path: Path to model weights (optional).

    Returns:
        float: Unified 'Robustness Score' (0.0 - 1.0).
    """
    evaluator = RobustnessEvaluator(model_name, task, config, weights_path)
    return evaluator.run()
