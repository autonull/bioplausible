#!/usr/bin/env python3
"""
TorEqProp Comprehensive Verification Suite (CLI)

Entry point for the installed `eqprop-verify` command.
"""

import argparse
import os
import sys

from bioplausible.validation import Verifier


def check_system_command():
    """Run system diagnostics."""
    print("=" * 60)
    print("Bioplausible System Check")
    print("=" * 60)

    # Python
    print(f"Python: {sys.version.split()[0]}")

    # PyTorch
    try:
        import torch

        print(f"PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
    except ImportError:
        print("PyTorch: ❌ Not found")

    # Triton
    try:
        from bioplausible.models.triton_kernel import TritonEqPropOps

        if TritonEqPropOps.is_available():
            print("Triton: ✅ Available")
        else:
            print("Triton: ⚠️ Installed but not available (requires CUDA)")
    except ImportError:
        print("Triton: ❌ Not installed")
    except Exception as e:
        print(f"Triton: ❌ Error ({e})")

    # CuPy
    try:
        from bioplausible.kernel import HAS_CUPY, cp

        if HAS_CUPY:
            try:
                # Check if it actually works (if GPU is present)
                if cp.cuda.is_available():
                    print("CuPy: ✅ Available and working")
                else:
                    print("CuPy: ⚠️ Installed but no GPU detected")
            except Exception as e:
                print(f"CuPy: ⚠️ Installed but check failed ({e})")
        else:
            print("CuPy: ❌ Not installed or CUDA_PATH missing")

        # Check CUDA_PATH
        cuda_path = os.environ.get("CUDA_PATH", "Not Set")
        print(f"  CUDA_PATH: {cuda_path}")

    except Exception as e:
        print(f"CuPy Check Error: {e}")

    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="TorEqProp Comprehensive Verification Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check-system",
        action="store_true",
        help="Check system compatibility (CUDA, Triton, CuPy)",
    )
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick mode (~2 min, smoke test)"
    )
    parser.add_argument(
        "--intermediate",
        "-i",
        action="store_true",
        help="Intermediate mode (~1 hr, directional validation)",
    )
    parser.add_argument(
        "--track", "-t", type=int, nargs="+", help="Run specific track(s)"
    )
    parser.add_argument("--list", "-l", action="store_true", help="List all tracks")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override number of seeds for robustness checks (redundancy)",
    )
    parser.add_argument("--export", action="store_true", help="Export raw data to CSV")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory to save verification results (defaults to ./results)",
    )

    args = parser.parse_args()

    if args.check_system:
        check_system_command()
        return

    verifier = Verifier(
        quick_mode=args.quick,
        intermediate_mode=args.intermediate,
        seed=args.seed,
        n_seeds_override=args.seeds,
        export_data=args.export,
        output_dir=args.output_dir,
    )

    if args.list:
        verifier.list_tracks()
    else:
        verifier.run_tracks(args.track)


if __name__ == "__main__":
    main()
