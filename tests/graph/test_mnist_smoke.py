"""Slow MNIST smoke test — marks as @pytest.mark.slow."""

import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms

from bioplausible.graph import Edge
from bioplausible.graph import InferenceSGD
from bioplausible.graph import Linear
from bioplausible.graph import ReLU
from bioplausible.graph import TaskMap
from bioplausible.graph import graph
from bioplausible.graph import initialize_params
from bioplausible.graph import train_backprop
from bioplausible.graph import train_pcn


@pytest.mark.slow
def test_backprop_mnist_smoke():
    """Backprop trains 1 epoch on 1000 MNIST samples without error."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(torch.flatten),
        ]
    )
    train_set = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    subset = Subset(train_set, range(1000))
    loader = DataLoader(subset, batch_size=64, shuffle=True)

    device = torch.device("cpu")
    inp = Linear(shape=(784, 256), name="input")
    act = ReLU(name="hidden")
    out = Linear(shape=(256, 10), name="output")
    g = graph(
        nodes=[inp, act, out],
        edges=[
            Edge(source=inp, target=act.slot("input")),
            Edge(source=act, target=out.slot("input")),
        ],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(),
    )
    params = initialize_params(g)
    for node_p in params.values():
        for p in node_p.values():
            p.data = p.to(device)

    results = train_backprop(g, params, loader, epochs=1, lr=0.001, device=device)
    assert results["train_loss"] < 5.0


@pytest.mark.slow
def test_pcn_mnist_smoke():
    """PC trains 1 epoch on 1000 MNIST samples without error."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(torch.flatten),
        ]
    )
    train_set = datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    )
    subset = Subset(train_set, range(1000))
    loader = DataLoader(subset, batch_size=64, shuffle=True)

    device = torch.device("cpu")
    inp = Linear(shape=(784, 256), name="input")
    act = ReLU(name="hidden")
    out = Linear(shape=(256, 10), name="output")
    g = graph(
        nodes=[inp, act, out],
        edges=[
            Edge(source=inp, target=act.slot("input")),
            Edge(source=act, target=out.slot("input")),
        ],
        task_map=TaskMap(x=inp, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )
    params = initialize_params(g)
    for node_p in params.values():
        for p in node_p.values():
            p.data = p.to(device)

    results = train_pcn(
        g,
        params,
        loader,
        epochs=1,
        lr=0.001,
        device=device,
        infer_steps=5,
        eta_infer=0.05,
    )
    assert "train_loss" in results
