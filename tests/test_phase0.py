import os
import pytest
import torch
import warnings
from omegaconf import OmegaConf

from bioplausible.config_schema import RunConfig
from bioplausible.runner import run_from_config
from bioplausible.models import create_model, list_models
from bioplausible.models.registry import get_model_spec
from bioplausible.energy import EnergyTracker

# --- 1. Config Loading ---
def test_config_load():
    cfg = OmegaConf.create({
        "seed": 42,
        "device": "cpu",
        "output_dir": "test_results",
        "data": {
            "task": "mnist",
            "batch_size": 32,
            "augment": False
        },
        "model": {
            "name": "backprop_mlp",
            "hidden_dim": 64,
            "num_layers": 2
        },
        "optimizer": {
            "name": "adam",
            "lr": 0.001
        },
        "trainer": {
            "epochs": 1,
            "batches_per_epoch": 10,
            "track_energy": True
        }
    })

    # Validate against schema
    conf = OmegaConf.merge(OmegaConf.structured(RunConfig), cfg)
    assert conf.seed == 42
    assert conf.data.task == "mnist"

# --- 2. Forward-Forward Model ---
def test_forward_forward_train_step():
    from bioplausible.models.forward_forward import ForwardForwardNet

    model = ForwardForwardNet(input_dim=10, hidden_dim=20, output_dim=2, num_layers=2)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    # Test forward
    out = model(x)
    assert out.shape == (4, 2)

    # Test train_step
    metrics = model.train_step(x, y)
    assert "loss" in metrics
    assert "accuracy" in metrics

    # Check requires_backward metadata
    spec = get_model_spec("forward_forward")
    assert not spec.requires_backward

# --- 3. PEPITA Model ---
def test_pepita_train_step():
    from bioplausible.models.pepita import PEPITA

    model = PEPITA(input_dim=10, hidden_dim=20, output_dim=2, num_layers=2)
    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    # Test forward
    out = model(x)
    assert out.shape == (4, 2)

    # Test train_step
    metrics = model.train_step(x, y)
    assert "loss" in metrics
    assert "accuracy" in metrics

    # Check requires_backward metadata
    spec = get_model_spec("pepita")
    assert not spec.requires_backward

# --- 4. Energy Tracking ---
def test_energy_tracking():
    model = torch.nn.Linear(10, 2)
    x = torch.randn(4, 10)

    with EnergyTracker(model, requires_backward=True) as et:
        out = model(x)
        out.sum().backward()

    prof = et.profile
    assert prof is not None
    assert prof.forward_flops > 0
    assert prof.backward_flops > 0
    assert prof.energy_proxy > 0
    assert prof.requires_backward

    # Test backward-free model
    model_nobwd = torch.nn.Linear(10, 2)
    with EnergyTracker(model_nobwd, requires_backward=False) as et_nobwd:
        out = model_nobwd(x)
        # No backward pass here in reality for FF/PEPITA,
        # but tracker just calculates proxy based on flag

    prof_nobwd = et_nobwd.profile
    assert prof_nobwd.backward_flops == 0
    assert not prof_nobwd.requires_backward
    assert prof_nobwd.energy_proxy < prof.energy_proxy # Should be roughly half (ignoring sparsity)

# --- 5. Run from Config (Integration) ---
@pytest.mark.skipif(not torch.cuda.is_available() and False, reason="Run CPU test if needed")
def test_integration_run():
    # Use CharNGram for speed/no-download
    cfg = OmegaConf.create({
        "seed": 42,
        "device": "cpu",
        "output_dir": "/tmp/bioplausible_test_run",
        "data": {
            "task": "char_ngram",
            "batch_size": 16,
        },
        "model": {
            "name": "backprop_mlp",
            "hidden_dim": 32,
            "num_layers": 1,
            # For CharNGram (ctx=3), input dim after flattening is 3
            # But BackpropMLP will init with input_dim from task.
            # CharNGram doesn't set _input_dim in init, sets it to None.
            # We fixed BackpropMLP to default to 1 if None.
            # But here we want 3.
            # Let's override or ensure task sets input_dim
        },
        "optimizer": {
            "name": "adam",
            "lr": 0.01
        },
        "trainer": {
            "epochs": 1,
            "batches_per_epoch": 5,
            "track_energy": True,
            "use_compile": False # slower for tiny tests
        }
    })

    conf = OmegaConf.merge(OmegaConf.structured(RunConfig), cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = run_from_config(conf)

    assert "history" in res
    assert len(res["history"]) == 1
    assert "loss" in res["history"][0]
    assert "energy_proxy" in res["history"][0]
