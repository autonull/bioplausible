#!/usr/bin/env python3
"""
Track 39: EqProp Diffusion (Diffusion via Energy Minimization).

Hypothesis: Diffusion can be formulated as equilibrium energy minimization.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import torch.optim as optim
from torchvision import datasets, transforms

# Add root to path (./) so we can import bioplausible
# parent -> experiments
# parent.parent -> bioplausible
# parent.parent.parent -> root
sys.path.append(str(Path(__file__).parent.parent.parent))

from bioplausible.models.eqprop_diffusion import EqPropDiffusion  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true", help="Quick validation mode")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("TRACK 39: EqProp Diffusion (MNIST)")
    print("=" * 60)

    # Data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    if args.quick:
        # Initial subset for quick smoke test
        indices = torch.arange(1000)
        train_dataset = torch.utils.data.Subset(train_dataset, indices)
        print("⚠️ Quick mode: using 1000 samples")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )

    # Model
    model = EqPropDiffusion(img_channels=1, hidden_channels=32 if args.quick else 64)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Noise schedule
    T = 1000
    beta = torch.linspace(1e-4, 0.02, T, device=device)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    print(f"Training for {args.epochs} epochs on {device}...")
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            current_batch_size = x.size(0)

            # Sample timestep
            t = torch.randint(0, T, (current_batch_size,), device=device)

            # Add noise
            noise = torch.randn_like(x)
            # x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1)

            x_noisy = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise

            # Denoise via equilibrium (approximate x_0)
            # For training simple DDPM, we predict x_0 directly.

            # Normalize t to [0,1] for embedding
            t_norm = t.float() / T

            # Forward pass: predict cleaned image
            # Note: The denoise_step in model is iterative inference.
            # For training, we use the direct network output for training stability.

            # EqPropDiffusion.denoiser is a ConvEqProp which has internal equilibrium.
            t_emb = t_norm.view(current_batch_size, 1, 1, 1).expand(
                current_batch_size, 1, 28, 28
            )
            x_input = torch.cat([x_noisy, t_emb], dim=1)

            h_flat = model.denoiser(x_input)
            x_pred = h_flat.view_as(x)

            # Simple MSE loss on x_0 prediction
            loss = ((x_pred - x) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"  Epoch {epoch+1}/{args.epochs}:"
            f" Loss = {total_loss / len(train_loader):.4f}"
        )

    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time:.1f}s")

    # Validation / Generation Demo
    print("\nGenerating consistency check samples...")
    model.eval()
    with torch.no_grad():
        # Take a few validation images, add noise, and try to denoise
        x_val, _ = next(iter(train_loader))
        x_val = x_val[:4].to(device)

        # Add moderate noise (t=500)
        t_val = torch.ones(4, dtype=torch.long, device=device) * 500
        noise = torch.randn_like(x_val)
        sqrt_ab = torch.sqrt(alpha_bar[500])
        sqrt_omab = torch.sqrt(1 - alpha_bar[500])

        x_noisy_val = sqrt_ab * x_val + sqrt_omab * noise

        # Denoise
        t_norm_val = t_val.float() / T

        # Use iterative refinement here
        x_recon = model.denoise_step(x_noisy_val, t_norm_val, steps=20)

        recon_error = ((x_recon - x_val) ** 2).mean().item()
        print(f"  Reconstruction MSE (t=500): {recon_error:.4f}")

        passed = recon_error < 0.1  # Loose threshold for proof of concept

    # Save results
    save_dir = Path("results/track_39")
    save_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "epochs": args.epochs,
        "final_loss": total_loss / len(train_loader),
        "recon_mse": recon_error,
        "passed": passed,
    }

    with open(save_dir / "diffusion_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
