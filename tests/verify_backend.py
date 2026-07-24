import os
import pathlib
import sys

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def verify_backend():
    print("=" * 60)
    print("Backend Verification")
    print("=" * 60)

    # 0. Check Torch import
    print("\n[Checking Torch]...")
    try:
        import torch

        print(f"Torch version: {torch.__version__}")
        print(f"Torch file: {torch.__file__}")
    except ImportError as e:
        print(f"❌ Failed to import torch: {e}")
        # Continue to see what else fails

    # 1. Check Environment Variables
    cuda_path = os.environ.get("CUDA_PATH")
    print(f"\nCUDA_PATH (env): {cuda_path}")

    # 2. Import kernel (triggers auto-detection)
    print("\n[Importing bioplausible.acceleration.kernels]...")
    try:
        from bioplausible.acceleration import kernels as kernel

        print("Successfully imported bioplausible.acceleration.kernels")
    except ImportError as e:
        print(f"Failed to import bioplausible.acceleration.kernels: {e}")
        # If this fails, we can't continue checking kernel properties
        return

    # Check detected CUDA_PATH in environment after import
    cuda_path_after = os.environ.get("CUDA_PATH")
    print(f"CUDA_PATH (after import): {cuda_path_after}")

    if cuda_path_after:
        if pathlib.Path(cuda_path_after).exists():
            print(f"✅ CUDA_PATH exists: {cuda_path_after}")
        else:
            print(f"❌ CUDA_PATH does not exist: {cuda_path_after}")
    else:
        print("⚠️ CUDA_PATH not detected by kernel module")

    # 3. Check CuPy
    print("\n[Checking CuPy]...")
    print(f"HAS_CUPY: {kernel.HAS_CUPY}")
    if kernel.HAS_CUPY:
        import cupy

        print(f"CuPy version: {cupy.__version__}")
        try:
            # Try a simple operation
            cupy.array([1, 2, 3])
            print("✅ CuPy basic operation successful")
        except Exception as e:
            print(f"❌ CuPy operation failed: {e}")

    # 4. Check Triton
    print("\n[Checking Triton]...")
    try:
        from bioplausible.acceleration.triton_kernels import HAS_TRITON, TritonEqPropOps

        print(f"HAS_TRITON: {HAS_TRITON}")
        print(f"TritonEqPropOps.is_available(): {TritonEqPropOps.is_available()}")

        if TritonEqPropOps.is_available():
            print("✅ Triton is available")
        else:
            if not HAS_TRITON:
                print("⚠️ Triton import failed")
            elif not os.environ.get("CUDA_PATH"):
                print("⚠️ CUDA might be missing or torch.cuda.is_available() is False")

            print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

    except ImportError as e:
        print(f"Failed to import triton_kernel: {e}")


if __name__ == "__main__":
    verify_backend()
