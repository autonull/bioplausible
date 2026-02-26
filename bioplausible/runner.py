import os
import json
import torch
from omegaconf import OmegaConf

from bioplausible.config_schema import RunConfig
from bioplausible.models import create_model
from bioplausible.optimizers import create_optimizer
from bioplausible.training.supervised import SupervisedTrainer
from bioplausible.hyperopt.tasks import create_task

def run_from_config(cfg: RunConfig) -> dict:
    """
    The universal entry point.
    """
    # 1. Seed everything
    torch.manual_seed(cfg.seed)
    
    device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device)
    
    # 2. Resolve task
    task = create_task(cfg.data.task, device=device)
    task.setup()
    
    # 3. Create model
    # Convert DictConfig to dict if needed
    extra_kwargs = OmegaConf.to_container(cfg.model.extra) if hasattr(cfg.model.extra, "_is_dict") else cfg.model.extra
    kwargs = {
        "input_dim": task.input_dim,
        "hidden_dim": cfg.model.hidden_dim,
        "output_dim": task.output_dim,
    }
    if hasattr(cfg.model, "num_layers"):
        kwargs["num_layers"] = cfg.model.num_layers
    kwargs.update(extra_kwargs)
        
    model = create_model(
        cfg.model.name,
        **kwargs
    )
    model = model.to(device)
    model = model.to(device)

    # 4. Create optimizer
    opt_kwargs = {
        "lr": cfg.optimizer.lr,
        "weight_decay": cfg.optimizer.weight_decay,
    }
    # Pass through MEP-specific args if present
    if cfg.optimizer.name.startswith("mep") or cfg.optimizer.name in ["smep", "sdmep", "local_ep", "natural_ep", "muon_backprop"]:
        if hasattr(cfg.optimizer, "beta"): opt_kwargs["beta"] = cfg.optimizer.beta
        if hasattr(cfg.optimizer, "settle_steps"): opt_kwargs["settle_steps"] = cfg.optimizer.settle_steps
        if hasattr(cfg.optimizer, "mode"): opt_kwargs["mode"] = cfg.optimizer.mode
        
    optimizer = create_optimizer(model, cfg.optimizer.name, **opt_kwargs)

    # 5. Build Trainer using Task Factory
    ablation_tags = OmegaConf.to_container(cfg.ablation_tags) if hasattr(cfg.ablation_tags, "_is_dict") else cfg.ablation_tags

    # Use task-specific trainer creation logic
    trainer = task.create_trainer(
        model=model,
        optimizer=optimizer,
        epochs=cfg.trainer.epochs,
        batches_per_epoch=cfg.trainer.batches_per_epoch,
        grad_clip=cfg.trainer.grad_clip,
        use_compile=cfg.trainer.use_compile,
        track_energy=cfg.trainer.track_energy,
        ablation_tags=ablation_tags,
        output_dir=cfg.output_dir,
        device=device
    )

    # 6. Run training
    results = []

    # RLTrainer handles loop differently (fit() instead of train_epoch())?
    # SupervisedTrainer has train_epoch(). RLTrainer might not.
    # Check if trainer has train_epoch or fit

    if hasattr(trainer, "train_epoch"):
        for epoch in range(cfg.trainer.epochs):
            epoch_metrics = trainer.train_epoch()
            results.append(epoch_metrics)
    elif hasattr(trainer, "run"): # RLTrainer usually has run()
        history = trainer.run()
        # Adapt history to list of metrics
        if isinstance(history, dict) and "rewards" in history:
             # Convert RL history dict to list of epoch-like dicts
             for i, r in enumerate(history["rewards"]):
                 results.append({"epoch": i, "reward": r, "val_accuracy": r}) # Proxy reward as accuracy for unified metric
    else:
        # Fallback to fit()
        history = trainer.fit(train_loader=None, epochs=cfg.trainer.epochs)
        # Parse history
        pass
    
    # 7. Return metrics and save
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    return {"history": results, "final_val_accuracy": results[-1].get("val_accuracy", 0.0) if results else 0.0}
