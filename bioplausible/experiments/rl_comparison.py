import argparse
import logging
import os
import time
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from bioplausible.models import BackpropMLP, LoopedMLP
from bioplausible.training.rl import RLTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_and_evaluate(
    name: str,
    trainer: RLTrainer,
    episodes: int,
    log_interval: int = 50,
) -> Dict[str, Any]:
    """
    Train and evaluate a model using the provided trainer.

    Args:
        name (str): Name of the experiment/model.
        trainer (RLTrainer): The RL trainer instance.
        episodes (int): Number of training episodes.
        log_interval (int): Interval for logging progress.

    Returns:
        Dict[str, Any]: Dictionary containing rewards and training time.
    """
    logger.info(f"--- Training {name} ---")
    start_time = time.time()
    rewards: List[float] = []

    for ep in range(episodes):
        metrics = trainer.train_episode()
        rewards.append(metrics["reward"])

        if (ep + 1) % log_interval == 0:
            avg_reward = np.mean(rewards[-log_interval:])
            logger.info(f"Episode {ep + 1}: Avg Reward {avg_reward:.1f}")

    training_time = time.time() - start_time
    final_eval = trainer.evaluate()

    logger.info(f"Final Eval: {final_eval:.2f}")
    logger.info(f"Time: {training_time:.2f}s")

    return {"rewards": rewards, "time": training_time, "final_eval": final_eval}


def main() -> None:
    parser = argparse.ArgumentParser(description="RL Comparison: BPTT vs EqProp")
    parser.add_argument(
        "--env", type=str, default="CartPole-v1", help="Gym environment"
    )
    parser.add_argument(
        "--episodes", type=int, default=500, help="Number of training episodes"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--steps", type=int, default=20, help="Equilibrium steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    # Environment info
    temp_env = gym.make(args.env)

    # Handle different observation space types (e.g. Box vs Discrete)
    if hasattr(temp_env.observation_space, "shape"):
        input_dim = temp_env.observation_space.shape[0]
    else:
        # Fallback or error handling
        input_dim = temp_env.observation_space.n

    # Handle different action space types
    if hasattr(temp_env.action_space, "n"):
        output_dim = temp_env.action_space.n
    elif hasattr(temp_env.action_space, "shape"):
        output_dim = temp_env.action_space.shape[0]
    else:
        raise ValueError("Unsupported action space type")

    temp_env.close()

    logger.info(f"Environment: {args.env}")
    logger.info(f"Input Dim: {input_dim}, Output Dim: {output_dim}")
    logger.info(f"Episodes: {args.episodes}")

    results: Dict[str, Any] = {}

    # 1. Standard Backprop MLP
    model_bp = BackpropMLP(input_dim, args.hidden, output_dim)
    trainer_bp = RLTrainer(model_bp, args.env, args.device, args.lr, seed=args.seed)
    results["bptt_mlp"] = train_and_evaluate(
        "Standard Backprop MLP", trainer_bp, args.episodes
    )

    # 2. LoopedMLP with BPTT (Unrolled)
    model_looped_bptt = LoopedMLP(
        input_dim,
        args.hidden,
        output_dim,
        max_steps=args.steps,
        gradient_method="bptt",
        use_spectral_norm=True,
    )
    trainer_looped_bptt = RLTrainer(
        model_looped_bptt, args.env, args.device, args.lr, seed=args.seed
    )
    results["looped_bptt"] = train_and_evaluate(
        "LoopedMLP (BPTT Mode)", trainer_looped_bptt, args.episodes
    )

    # 3. LoopedMLP with Equilibrium Prop (Implicit Diff)
    model_eq = LoopedMLP(
        input_dim,
        args.hidden,
        output_dim,
        max_steps=args.steps,
        gradient_method="equilibrium",
        use_spectral_norm=True,
    )
    trainer_eq = RLTrainer(model_eq, args.env, args.device, args.lr, seed=args.seed)
    results["eq_prop"] = train_and_evaluate(
        "LoopedMLP (Equilibrium Mode)", trainer_eq, args.episodes
    )

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Model':<30} | {'Max Reward':<10} | {'Time (s)':<10}")
    print("-" * 55)
    for name, res in results.items():
        # Calculate max reward using a moving window of 50 episodes
        rewards = res["rewards"]
        window_size = 50
        max_r = 0.0
        if len(rewards) >= window_size:
            max_r = max(
                [
                    np.mean(rewards[i : i + window_size])
                    for i in range(0, len(rewards) - window_size + 1, window_size)
                ]
            )
        else:
            max_r = np.mean(rewards)

        print(f"{name:<30} | {max_r:<10.1f} | {res['time']:<10.2f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    torch.save(results, "results/rl_comparison.pt")
    logger.info("Results saved to results/rl_comparison.pt")


if __name__ == "__main__":
    main()
