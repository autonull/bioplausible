import copy
import pathlib
from itertools import product

from omegaconf import OmegaConf

from bioplausible.config.schema import RunConfig
from bioplausible.core.trainer import run_from_runconfig as run_from_config


def run_reduced_sweep():
    # Load base config
    config_path = "configs/sweep_phase1.yaml"
    if not pathlib.Path(config_path).exists():
        print(f"Error: {config_path} not found.")
        return

    # Load as raw dict to handle custom structures
    conf = OmegaConf.load(config_path)

    # Extract search space
    search_space = conf.get("search_space", {})
    models = search_space.get("model", [])
    data_tasks = search_space.get("data", [])
    optimizers = search_space.get("optimizer", [])

    # Reduce the search space for demonstration
    reduced_models = []
    for m in models:
        # Keep only the smallest hidden_dim and num_layers for speed
        reduced_m = copy.deepcopy(m)
        if "hidden_dim" in reduced_m and isinstance(reduced_m.hidden_dim, list):
            reduced_m.hidden_dim = [reduced_m.hidden_dim[0]]
        if "num_layers" in reduced_m and isinstance(reduced_m.num_layers, list):
            reduced_m.num_layers = [reduced_m.num_layers[0]]
        if (
            "extra" in reduced_m
            and "max_steps" in reduced_m.extra
            and isinstance(reduced_m.extra["max_steps"], list)
        ):
            reduced_m.extra["max_steps"] = [reduced_m.extra["max_steps"][0]]

        reduced_models.append(reduced_m)

    reduced_data = []
    for d in data_tasks:
        if d.task == "mnist":
            reduced_d = copy.deepcopy(d)
            if "data_fraction" in reduced_d and isinstance(
                reduced_d.data_fraction, list
            ):
                # Only run with 10% data for speed
                reduced_d.data_fraction = [0.1]
            reduced_data.append(reduced_d)

    reduced_optimizers = []
    for o in optimizers:
        if o.name == "adam":
            reduced_o = copy.deepcopy(o)
            if "lr" in reduced_o and isinstance(reduced_o.lr, (list, tuple)):
                pass  # keep as is
            reduced_optimizers.append(reduced_o)

    print("Running reduced sweep (Phase 1 Ignition)...")

    # Create output directory
    output_dir = "results/phase1_reduced"
    pathlib.Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Flatten and generate combinations
    for m_cfg in reduced_models:
        for d_cfg in reduced_data:
            for o_cfg in reduced_optimizers:
                # Resolve list parameters inside m_cfg, d_cfg, o_cfg
                m_hdims = m_cfg.get("hidden_dim", [256])
                m_layers = m_cfg.get("num_layers", [2])

                for hdim, nlayers in product(m_hdims, m_layers):
                    # Construct RunConfig
                    rcfg = OmegaConf.structured(RunConfig)
                    rcfg.seed = 42
                    rcfg.device = "auto"
                    rcfg.output_dir = output_dir

                    # Set trainer (reduced for speed)
                    rcfg.trainer.epochs = 1
                    rcfg.trainer.batches_per_epoch = 10
                    rcfg.trainer.track_energy = True

                    # Set Data
                    rcfg.data.task = d_cfg.task
                    rcfg.data.data_fraction = d_cfg.get("data_fraction", [1.0])[0]

                    # Set Model
                    model_name = m_cfg.name
                    # Handle smep_mlp which is a preset
                    if model_name == "smep_mlp":
                        rcfg.model.name = m_cfg.get("model_ref", "looped_mlp").replace(
                            "eqprop_mlp", "looped_mlp"
                        )
                        rcfg.optimizer.name = "smep"
                    else:
                        rcfg.model.name = model_name.replace("eqprop_mlp", "looped_mlp")
                        rcfg.optimizer.name = o_cfg.name

                    lr_val = o_cfg.get("lr", 0.001)
                    if hasattr(lr_val, "__iter__") and not isinstance(lr_val, str):
                        rcfg.optimizer.lr = float(lr_val[0])
                    else:
                        rcfg.optimizer.lr = float(lr_val)

                    rcfg.model.hidden_dim = hdim
                    rcfg.model.num_layers = nlayers

                    if "extra" in m_cfg:
                        extra = {}
                        for k, v in m_cfg.extra.items():
                            if hasattr(v, "__iter__") and not isinstance(v, str):
                                extra[k] = v[0]
                            else:
                                extra[k] = v
                        rcfg.model.extra = extra

                    # Add ablation tags
                    rcfg.ablation_tags = {
                        "model": model_name,
                        "hidden_dim": hdim,
                        "num_layers": nlayers,
                        "task": d_cfg.task,
                        "optimizer": rcfg.optimizer.name,
                    }

                    print(
                        f"\\n--> Evaluating: {model_name} on {d_cfg.task} (h:{hdim}, l:{nlayers})"
                    )
                    try:
                        run_from_config(rcfg)
                        print("    Success.")
                    except Exception as e:
                        import traceback

                        print(f"    Failed: {e}")
                        traceback.print_exc()


if __name__ == "__main__":
    run_reduced_sweep()
