import warnings

import pytest
import torch

from bioplausible.hyperopt.tasks import create_task


# --- 1. Tabular Task ---
def test_tabular_task_setup():
    # Tabular uses sklearn, which should be installed
    try:
        task = create_task("breast_cancer", device="cpu", quick_mode=True)
        task.setup()

        x, y = task.get_batch("train", batch_size=4)
        assert x.shape[0] == 4
        assert y.shape[0] == 4
        assert task.input_dim > 0
        assert task.output_dim == 2
        print("TabularTask setup successful.")
    except ImportError as e:
        pytest.skip(f"Tabular dependencies missing: {e}")


# --- 2. Graph Task ---
def test_graph_task_setup():
    # Graph uses torch_geometric
    try:
        import torch_geometric

        task = create_task("cora", device="cpu", quick_mode=True)
        task.setup()

        # Graph get_batch returns full graph usually or node indices
        # GraphTask implementation in bioplausible returns (data, y)
        # where data is the graph object.
        x, y = task.get_batch("train", batch_size=4)

        assert hasattr(x, "edge_index")
        assert y.shape == x.y.shape
        assert task.input_dim > 0
        assert task.output_dim > 0
        print("GraphTask setup successful.")

    except ImportError:
        pytest.skip("torch_geometric not installed. Skipping graph test.")
    except Exception as e:
        pytest.fail(f"Graph task failed: {e}")
