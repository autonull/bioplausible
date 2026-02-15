"""
CLI Runner for Bioplausible Experiments
"""

import argparse

from bioplausible.hyperopt import create_optuna_space, create_study
from bioplausible.hyperopt.eval_tiers import PatientLevel, get_evaluation_config
from bioplausible.hyperopt.experiment import run_single_trial_task
from bioplausible.models.registry import MODEL_REGISTRY
from bioplausible.pipeline.config import TrainingConfig
from bioplausible.pipeline.session import TrainingSession


def run_training(args):
    """Run a single training session."""
    print(f"🚀 Starting Headless Training: {args.model} on {args.task}")

    config = TrainingConfig(
        task=args.task,
        # Default dataset based on task if not provided? Or require it.
        dataset=(
            args.dataset
            if args.dataset
            else ("mnist" if args.task == "vision" else "tinyshakespeare")
        ),
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hyperparams={},  # Could parse extra args here
    )

    session = TrainingSession(config)

    try:
        from tqdm import tqdm

        pbar = tqdm(range(args.epochs), desc="Epochs")

        iterator = session.start()
        for event in iterator:
            if hasattr(event, "epoch"):
                pbar.update(1)
                pbar.set_postfix(event.metrics)

        pbar.close()
        print("✅ Training Complete")

    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted")
        session.stop()


def run_search(args):
    """Run a hyperparameter search (discovery protocol)."""
    tier_name = args.tier.lower()
    try:
        tier = PatientLevel(tier_name)
    except ValueError:
        print(
            f"❌ Invalid tier: {tier_name}. Available: {[t.value for t in PatientLevel]}"
        )
        return

    config = get_evaluation_config(tier)
    print(f"🧪 Starting {tier.name} Discovery Run")
    print(f"   Models: {args.models}")
    print(f"   Config: {config.epochs} epochs, {config.n_trials} trials")

    from bioplausible.models.registry import get_model_spec, list_model_names

    if args.models.lower() == "all":
        models = list_model_names()
    else:
        models = args.models.split(",")
        models = [m.strip() for m in models if m.strip()]

    for model in models:
        # Check compatibility
        try:
            spec = get_model_spec(model)
            if spec.task_compat and args.task not in spec.task_compat:
                # Normalize task name check just in case (e.g. cifar10 -> vision?)
                # For now assume explicit match.
                # Special case: vision covers mnist/cifar
                is_compat = False
                if args.task in spec.task_compat:
                    is_compat = True
                elif args.task in ["mnist", "cifar10"] and "vision" in spec.task_compat:
                    is_compat = True
                elif (
                    args.task in ["tiny_shakespeare", "wikitext"]
                    and "lm" in spec.task_compat
                ):
                    is_compat = True

                if not is_compat:
                    print(
                        f"⚠️  Skipping {model}: Incompatible with task '{args.task}' (Needs {spec.task_compat})"
                    )
                    continue
        except ValueError:
            pass  # Unknown model, let it try/fail naturally later

        print(f"\n🔍 Exploring {model}...")

        study_name = f"{model}_{args.task}_{tier.value}"
        study = create_study(
            model_names=[model],
            n_objectives=2,
            storage="sqlite:///bioplausible.db",
            study_name=study_name,
            use_pruning=config.use_pruning,
            sampler_name="tpe",
        )

        def objective(trial):
            # 1. Sample Hyperparameters
            trial_config = create_optuna_space(trial, model)

            # 2. Apply Tier Constraints
            trial_config["epochs"] = config.epochs
            trial_config["batch_size"] = config.batch_size

            # 3. Tag Trial
            trial.set_user_attr("tier", tier.value)
            trial.set_user_attr("model_family", model)

            # 4. Run Trial
            metrics = run_single_trial_task(
                task=args.task,
                model_name=model,
                config=trial_config,
                storage_path="bioplausible.db",
                # job_id removed used to be trial._trial_id
                quick_mode=(tier == PatientLevel.SMOKE),
                verbose=False,
            )

            if metrics:
                acc = metrics.get("accuracy", 0.0)
                loss = metrics.get("loss", float("inf"))
                print(f"   Trial {trial.number}: Acc={acc:.4f} | Params={trial_config}")
                return acc, loss
            else:
                import optuna

                raise optuna.TrialPruned()

        try:
            study.optimize(objective, n_trials=config.n_trials)
        except KeyboardInterrupt:
            print("\n🛑 Search Interrupted")
            break
        except Exception as e:
            print(f"❌ Error optimizing {model}: {e}")


def list_models(args):
    print("Available Models:")
    for name in sorted(MODEL_REGISTRY.keys()):
        print(f" - {name}")


def main():
    parser = argparse.ArgumentParser(description="Bioplausible Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run a training session")
    train_parser.add_argument("--model", required=True, help="Model name")
    train_parser.add_argument(
        "--task", default="vision", choices=["vision", "lm", "rl"], help="Task type"
    )
    train_parser.add_argument("--dataset", help="Dataset name")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Run a discovery search (Tiered)"
    )
    search_parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated model names (e.g. 'EqPropMLP,BackpropMLP')",
    )
    search_parser.add_argument("--task", default="vision", help="Task type")
    search_parser.add_argument(
        "--tier",
        default="smoke",
        choices=["smoke", "shallow", "standard", "deep"],
        help="Discovery Tier",
    )

    # List command
    subparsers.add_parser("list", help="List available models")

    args = parser.parse_args()

    if args.command == "train":
        run_training(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "list":
        list_models(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
