import argparse
import os
import time

import numpy as np
import torch

from bioplausible.models import BackpropMLP
from bioplausible.models.looped_mlp import LoopedMLP
from bioplausible.rl.trainer import RLTrainer


def main():
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
    import gymnasium as gym

    temp_env = gym.make(args.env)
    input_dim = temp_env.observation_space.shape[0]
    output_dim = temp_env.action_space.n
    temp_env.close()

    print(f"Environment: {args.env}")
    print(f"Input Dim: {input_dim}, Output Dim: {output_dim}")
    print(f"Episodes: {args.episodes}")

    results = {}

    # 1. Standard Backprop MLP
    print("\n--- Training Standard Backprop MLP ---")
    model_bp = BackpropMLP(input_dim, args.hidden, output_dim)
    trainer_bp = RLTrainer(model_bp, args.env, args.device, args.lr, seed=args.seed)

    start_time = time.time()
    rewards_bp = []
    for ep in range(args.episodes):
        metrics = trainer_bp.train_episode()
        rewards_bp.append(metrics["reward"])
        if ep % 50 == 0:
            avg_reward = np.mean(rewards_bp[-50:])
            print(f"Episode {ep}: Avg Reward {avg_reward:.1f}")

    time_bp = time.time() - start_time
    print(f"Final Eval: {trainer_bp.evaluate()}")
    print(f"Time: {time_bp:.2f}s")
    results["bptt_mlp"] = {"rewards": rewards_bp, "time": time_bp}

    # 2. LoopedMLP with BPTT (Unrolled)
    print("\n--- Training LoopedMLP (BPTT Mode) ---")
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

    start_time = time.time()
    rewards_looped_bptt = []
    for ep in range(args.episodes):
        metrics = trainer_looped_bptt.train_episode()
        rewards_looped_bptt.append(metrics["reward"])
        if ep % 50 == 0:
            avg_reward = np.mean(rewards_looped_bptt[-50:])
            print(f"Episode {ep}: Avg Reward {avg_reward:.1f}")

    time_looped_bptt = time.time() - start_time
    print(f"Final Eval: {trainer_looped_bptt.evaluate()}")
    print(f"Time: {time_looped_bptt:.2f}s")
    results["looped_bptt"] = {"rewards": rewards_looped_bptt, "time": time_looped_bptt}

    # 3. LoopedMLP with Equilibrium Prop (Implicit Diff)
    print("\n--- Training LoopedMLP (Equilibrium Mode) ---")
    model_eq = LoopedMLP(
        input_dim,
        args.hidden,
        output_dim,
        max_steps=args.steps,
        gradient_method="equilibrium",
        use_spectral_norm=True,
    )
    trainer_eq = RLTrainer(model_eq, args.env, args.device, args.lr, seed=args.seed)

    start_time = time.time()
    rewards_eq = []
    for ep in range(args.episodes):
        metrics = trainer_eq.train_episode()
        rewards_eq.append(metrics["reward"])
        if ep % 50 == 0:
            avg_reward = np.mean(rewards_eq[-50:])
            print(f"Episode {ep}: Avg Reward {avg_reward:.1f}")

    time_eq = time.time() - start_time
    print(f"Final Eval: {trainer_eq.evaluate()}")
    print(f"Time: {time_eq:.2f}s")
    results["eq_prop"] = {"rewards": rewards_eq, "time": time_eq}

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Model':<20} | {'Max Reward':<10} | {'Time (s)':<10}")
    print("-" * 45)
    for name, res in results.items():
        max_r = max(
            [
                np.mean(res["rewards"][i : i + 50])
                for i in range(0, len(res["rewards"]), 50)
            ]
        )
        print(f"{name:<20} | {max_r:<10.1f} | {res['time']:<10.2f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    torch.save(results, "results/rl_comparison.pt")
    print("Results saved to results/rl_comparison.pt")


if __name__ == "__main__":
    main()
