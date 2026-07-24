"""
CLI Runner for Bioplausible Experiments
"""

import argparse

from bioplausible.core.registry import ComponentCategory, Registry
from bioplausible.core.trainer import CoreTrainer, TrainerConfig
from bioplausible.hyperopt import create_optuna_space, create_study
from bioplausible.hyperopt.eval_tiers import PatientLevel, get_evaluation_config
from bioplausible.hyperopt.experiment import run_single_trial_task


def run_training(args):
    """Run a single training session or training from YAML config."""
    if args.config:
        # Training from YAML config (as specified in TODO.md)
        run_from_yaml(args)
        return

    print(f"🚀 Starting Headless Training: {args.model} on {args.task}")

    config = TrainerConfig(
        model=args.model,
        task=(
            args.dataset
            if args.dataset
            else ("mnist" if args.task == "vision" else "tinyshakespeare")
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer_kwargs={"lr": args.lr},
    )

    trainer = CoreTrainer(config)

    try:
        from tqdm import tqdm

        pbar = tqdm(range(args.epochs), desc="Epochs")

        metrics = trainer.fit()
        for epoch_metric in metrics:
            pbar.update(1)
            pbar.set_postfix({"loss": epoch_metric.loss, "acc": epoch_metric.accuracy})

        pbar.close()
        print("✅ Training Complete")

    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted")


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

    if args.models.lower() == "all":
        models = list(Registry._components.get(ComponentCategory.MODEL, {}).keys())
    else:
        models = args.models.split(",")
        models = [m.strip() for m in models if m.strip()]

    for model in models:
        # Check compatibility
        try:
            meta = Registry.get_metadata(ComponentCategory.MODEL, model)
            # Use domain list for compatibility check
            domain_names = [d.value for d in meta.domains]
            if domain_names and args.task not in domain_names:
                # Normalize task name check just in case (e.g. cifar10 -> vision?)
                # For now assume explicit match.
                # Special case: vision covers mnist/cifar
                is_compat = False
                if (
                    args.task in domain_names
                    or (args.task in ["mnist", "cifar10"] and "vision" in domain_names)
                    or (
                        args.task in ["tiny_shakespeare", "wikitext"]
                        and "lm" in domain_names
                    )
                ):
                    is_compat = True

                if not is_compat:
                    print(
                        f"⚠️  Skipping {model}: Incompatible with task "
                        f"'{args.task}' (Needs {domain_names})"
                    )
                    continue
        except Exception:
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


def run_core_train(args):
    """Run a single training session using CoreTrainer (new unified interface)."""
    from bioplausible.core.trainer import CoreTrainer, TrainerConfig

    config = TrainerConfig(
        model=args.model,
        model_kwargs={"hidden_dim": args.hidden_dim} if args.hidden_dim else {},
        optimizer=args.optimizer,
        optimizer_kwargs={"lr": args.lr} if args.lr else {},
        task=args.task,
        epochs=args.epochs,
        batch_size=args.batch_size,
        track_energy=not args.no_track_energy,
        device=args.device,
    )

    trainer = CoreTrainer(config)
    history = trainer.fit()

    if history:
        final = history[-1]
        print(
            f"\nResults: Train Acc={final.train_accuracy:.4f}, "
            f"Val Acc={final.val_accuracy:.4f}"
        )


def run_from_yaml(args):
    """Run training from a YAML config file."""
    from bioplausible.core.trainer import CoreTrainer

    trainer = CoreTrainer.from_yaml(args.config)
    history = trainer.fit()

    if history:
        final = history[-1]
        print(
            f"\nResults: Train Acc={final.train_accuracy:.4f}, "
            f"Val Acc={final.val_accuracy:.4f}"
        )


def list_models(args):
    from bioplausible.core.registry import ComponentCategory, Registry

    models = Registry.list(ComponentCategory.MODEL)
    model_names = models.get("model", [])
    print("Available Models (Zoo Registry):")
    for name in sorted(model_names):
        meta = Registry.get_metadata(ComponentCategory.MODEL, name)
        score = meta.bio_plausibility_score
        domains = ", ".join(d.value for d in meta.domains)
        print(f"  {name:25s} bio={score:.1f}  domains=[{domains}]")


def run_benchmark(args):
    """Run cross-domain benchmark suite."""
    from bioplausible.evaluation.cross_domain import CrossDomainBenchmarkSuite

    print("🔬 Cross-Domain Benchmark Suite")

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    from bioplausible.evaluation.cross_domain import BenchmarkSuiteConfig

    config = BenchmarkSuiteConfig(
        models=models,
        quick_mode=args.quick,
        intermediate_mode=args.intermediate,
        output_dir=args.output_dir,
        epochs=3 if args.quick else (10 if args.intermediate else 20),
        batch_size=64,
        track_energy=True,
    )

    suite = CrossDomainBenchmarkSuite(output_dir=args.output_dir)
    result = suite.run_suite(config)

    print("\nBenchmark Results:")
    print(f"   Total time: {result.total_time_s:.1f}s")
    print(f"   Results: {len(result.results)} benchmarks")

    if result.results:
        for r in result.results[:5]:
            print(f"   - {r.model_name} on {r.task_name}: {r.metrics}")

    suite.save_results(result)
    suite.generate_leaderboard()
    print(f"\n📁 Results saved to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Bioplausible Experiment Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Train command - supports model args or --config for YAML config
    train_parser = subparsers.add_parser(
        "train", help="Run training session or from YAML config"
    )
    train_parser.add_argument("--config", help="Path to YAML config file")
    train_parser.add_argument(
        "--model", help="Model name (required if not using --config)"
    )
    train_parser.add_argument(
        "--task", default="vision", choices=["vision", "lm", "rl"], help="Task type"
    )
    train_parser.add_argument("--dataset", help="Dataset name")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    # Core train command (new unified API)
    core_parser = subparsers.add_parser(
        "core-train", help="Train using CoreTrainer (new)"
    )
    core_parser.add_argument(
        "--model", default="MLP", help="Model name from Zoo registry"
    )
    core_parser.add_argument("--task", default="mnist", help="Task/dataset name")
    core_parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    core_parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    core_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    core_parser.add_argument("--optimizer", default="adam", help="Optimizer name")
    core_parser.add_argument(
        "--hidden-dim", type=int, default=256, help="Hidden dimension"
    )
    core_parser.add_argument(
        "--device", default="auto", help="Device (auto, cpu, cuda)"
    )
    core_parser.add_argument(
        "--no-track-energy", action="store_true", help="Disable energy tracking"
    )

    # Config file command
    config_parser = subparsers.add_parser(
        "from-config", help="Train from YAML config file"
    )
    config_parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )

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

    # Benchmark command (cross-domain)
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run cross-domain benchmark suite"
    )
    benchmark_parser.add_argument(
        "--models",
        help="Comma-separated model names (default: all registered)",
    )
    benchmark_parser.add_argument(
        "--domains",
        help="Comma-separated domains (default: all)",
    )
    benchmark_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (3 epochs, smoke test)",
    )
    benchmark_parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Intermediate mode (10 epochs)",
    )
    benchmark_parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training(args)
    elif args.command == "core-train":
        run_core_train(args)
    elif args.command == "from-config":
        run_from_yaml(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "list":
        list_models(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
