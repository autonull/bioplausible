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
    if cfg.optimizer.name in ["mep", "mep_sgd", "mep_adam"]:
        opt_kwargs["beta"] = cfg.optimizer.beta
        opt_kwargs["settle_steps"] = cfg.optimizer.settle_steps
        opt_kwargs["mode"] = cfg.optimizer.mode
        
    optimizer = create_optimizer(model, cfg.optimizer.name, **opt_kwargs)

    # 5. Build Trainer
    ablation_tags = OmegaConf.to_container(cfg.ablation_tags) if hasattr(cfg.ablation_tags, "_is_dict") else cfg.ablation_tags
    trainer = SupervisedTrainer(
        model=model,
        optimizer=optimizer,
        task=task,
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
    for epoch in range(cfg.trainer.epochs):
        epoch_metrics = trainer.train_epoch()
        results.append(epoch_metrics)
    
    # 7. Return metrics and save
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    return {"history": results, "final_val_accuracy": results[-1].get("val_accuracy", 0.0) if results else 0.0}
