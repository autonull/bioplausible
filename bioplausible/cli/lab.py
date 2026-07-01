"""
CLI Lab for Verification and Inspection
"""

import argparse

import torch

from bioplausible.hyperopt.tasks import create_task
from bioplausible.models.factory import create_model
from bioplausible.models.registry import get_model_spec


def inspect_model(args):
    print(f"🔬 Inspecting Model: {args.model}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create Task
    task = create_task(args.task, device=device)
    task.setup()
    print(f"Task: {args.task}, Input: {task.input_dim}, Output: {task.output_dim}")

    # Create Model
    spec = get_model_spec(args.model)
    model = create_model(
        spec, task.input_dim, task.output_dim, device=device, task_type=task.task_type
    )

    print(f"Model Created: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Run Dummy Forward
    print("\nRunning Verification Inference...")
    x, y = task.get_batch("val")
    model.eval()
    with torch.no_grad():
        # Prepare input same as trainer but quick manual
        # Trainer logic handles embedding etc. Here we assume direct
        # or we use trainer helper?
        # Ideally we reuse trainer logic, but let's do a simple check.
        if hasattr(model, "embed") and args.task == "lm":
            # Manual embed if needed, or rely on model forward handling it if integrated
            pass

        # Basic check: just call forward. If it fails, verification catches it.
        # This is "headless lab" - verifying if model runs at all.
        try:
            # We create a dummy trainer just to use its prepare_input
            # logic/forward wrapper?
            # Or just try/except raw forward

            # Simple heuristic
            if x.dim() > 2 and "Conv" not in args.model:
                x = x.view(x.size(0), -1)

            x = x.to(device)
            out = model(x)
            print(f"✓ Forward pass successful. Output shape: {out.shape}")

        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback

            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Bioplausible Lab CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    inspect = subparsers.add_parser("inspect", help="Inspect a model architecture")
    inspect.add_argument("--model", required=True, help="Model name")
    inspect.add_argument("--task", default="vision", help="Task type")

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
