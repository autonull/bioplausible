import torch
import torch.nn as nn

from bioplausible.models.registry import register_model

try:
    from bioplausible.hyperopt.graph_task import GraphTask
    from bioplausible.hyperopt.tabular_task import TabularTask
    from bioplausible.knowledge.metamodel import KnowledgebaseMetamodel
    from bioplausible.knowledge.seed import KnowledgeBase
    from bioplausible.models.forward_forward import ForwardForwardNet
    from bioplausible.models.graph_eqprop import GraphEqProp
    from bioplausible.models.pepita import PEPITA
    from bioplausible.models.spiking_stdp import SpikingSTDP
    from bioplausible.models.target_prop import DifferenceTargetProp
    from bioplausible.models.three_factor import ThreeFactorHebbian
except ImportError as e:
    print(f"ImportError: {e}")
    exit(1)


def verify_models():
    input_dim = 10
    hidden_dim = 20
    output_dim = 5
    batch_size = 4

    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, output_dim, (batch_size,))

    models = [
        ("ForwardForward", ForwardForwardNet(input_dim, hidden_dim, output_dim)),
        ("PEPITA", PEPITA(input_dim, hidden_dim, output_dim)),
        (
            "DifferenceTargetProp",
            DifferenceTargetProp(input_dim, hidden_dim, output_dim),
        ),
        ("ThreeFactorHebbian", ThreeFactorHebbian(input_dim, hidden_dim, output_dim)),
        ("SpikingSTDP", SpikingSTDP(input_dim, hidden_dim, output_dim)),
    ]

    for name, model in models:
        try:
            out = model(x)
            assert out.shape == (batch_size, output_dim)
            metrics = model.train_step(x, y)
            assert "loss" in metrics
            print(f"✓ {name} verification passed")
        except Exception as e:
            print(f"✗ {name} verification failed: {e}")


def verify_knowledgebase():
    try:
        kb = KnowledgeBase(storage_path="test_kb.json", load_seed=True)
        assert len(kb.findings) > 0
        mm = KnowledgebaseMetamodel()
        mm.fit("bioplausible.db")  # Dummy DB
        print("✓ KnowledgeBase verification passed")
    except Exception as e:
        print(f"✗ KnowledgeBase verification failed: {e}")


if __name__ == "__main__":
    verify_models()
    verify_knowledgebase()
