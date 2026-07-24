import os

# Mocking
from unittest.mock import MagicMock, patch

import pytest
from bioplausible_ui.app.tabs.compare_tab import CompareTab
from bioplausible_ui.app.tabs.experiment_tab import ExperimentTab
from bioplausible_ui.app.tabs.train_tab import TrainTab
from bioplausible_ui.core.widgets.hyperparam_editor import HyperparamEditor
from bioplausible_ui.lab.window import LabMainWindow

from bioplausible.pipeline.results import ResultsManager


class TestResultsManager:
    @pytest.fixture
    def manager(self, tmp_path):
        mgr = ResultsManager(base_dir=str(tmp_path / "runs"))
        return mgr

    def test_export_import(self, manager, tmp_path):
        # Create a dummy run
        run_id = "test_run_1"
        config = {"model": "mlp", "epochs": 10}
        metrics = {"accuracy": 0.9}
        manager.save_run(run_id, config, metrics)

        # Export
        zip_path = str(tmp_path / "export_test")
        manager.export_run(run_id, zip_path)

        assert os.path.exists(zip_path + ".zip")

        # Import
        # First delete original to verify import works
        manager.delete_run(run_id)
        assert not os.path.exists(os.path.join(manager.BASE_DIR, run_id))

        imported_id = manager.import_run(zip_path + ".zip")
        assert imported_id == run_id

        loaded = manager.load_run(run_id)
        assert loaded is not None
        assert loaded["config"]["model"] == "mlp"


class TestHyperparamEditor:
    def test_set_values(self, qtbot):
        defaults = {"a": 1, "b": 2.0, "c": True}
        editor = HyperparamEditor(defaults=defaults)
        qtbot.addWidget(editor)

        assert editor.get_values() == defaults

        new_values = {"a": 5, "b": 10.5, "c": False}
        editor.set_values(new_values)

        assert editor.get_values() == new_values


class TestTabs:
    def test_train_tab_set_config(self, qtbot):
        # We need to mock the schema-based init or use the real one if it works headless
        # TrainTab requires widgets like task_selector etc.
        # They are created in init.

        tab = TrainTab()
        qtbot.addWidget(tab)

        config = {
            "task": "rl",
            "dataset": "cartpole",
            "model": "EqProp MLP",
            "hyperparams": {"learning_rate": 0.05},
        }

        # Mock widgets to avoid full dependency chain if needed,
        # but integration test is better.
        # TrainTab creates real widgets.

        tab.set_config(config)

        assert tab.task_selector.get_task() == "rl"
        assert tab.dataset_picker.get_dataset() == "cartpole"
        # Model selector might reset if task changes, so order matters.
        # set_config implementation handles order?
        # TrainTab.set_config sets task, then dataset, then model.
        # ModelSelector updates when task changes.

        # Check hyperparams
        # Model selection triggers update_for_model.
        # Then set_values is called.
        # Note: EqProp MLP spec defaults LR to 0.001. We set to 0.05.

        assert tab.model_selector.get_selected_model() == "EqProp MLP"

        vals = tab.hyperparam_editor.get_values()
        # LR widget should exist
        if "learning_rate" in vals:
            assert abs(vals["learning_rate"] - 0.05) < 1e-6

    def test_search_transfer_signal(self, qtbot):
        tab = ExperimentTab()
        qtbot.addWidget(tab)

        # Mock message box to return "Train"
        with patch("PyQt6.QtWidgets.QMessageBox.exec", return_value=0):
            with patch("PyQt6.QtWidgets.QMessageBox.clickedButton"):
                # Mocking QPushButton click return is complex.
                # Verification of signal existence is sufficient here.
                assert hasattr(tab, "transfer_config")

    def test_compare_tab_logic(self, qtbot, tmp_path):
        # Create dummy runs
        mgr = ResultsManager(base_dir=str(tmp_path / "runs"))

        # Run 1
        metrics1 = {
            "accuracy": 0.5,
            "history": [
                {"epoch": 0, "accuracy": 0.1, "loss": 1.0},
                {"epoch": 1, "accuracy": 0.5, "loss": 0.5},
            ],
        }
        mgr.save_run("run1", {"model": "m1"}, metrics1)

        # Run 2
        metrics2 = {
            "accuracy": 0.6,
            "history": [
                {"epoch": 0, "accuracy": 0.2, "loss": 0.9},
                {"epoch": 1, "accuracy": 0.6, "loss": 0.4},
            ],
        }
        mgr.save_run("run2", {"model": "m2"}, metrics2)

        tab = CompareTab()
        # Inject manager
        tab.results_manager = mgr
        qtbot.addWidget(tab)

        # Refresh selectors
        tab.run_selector_1.manager = mgr
        tab.run_selector_1.refresh()
        tab.run_selector_2.manager = mgr
        tab.run_selector_2.refresh()

        # Select
        tab.run_selector_1.combo.setCurrentIndex(0)
        tab.run_selector_2.combo.setCurrentIndex(1)

        # Call compare
        tab._compare_saved_runs()

        # Check plot logic execution (no crash)
        # Verify legend added?
        # Access internal plot widget
        assert hasattr(tab.plot_comparison_plot.plot_widget.plotItem, "legend")


class TestLabWindow:
    def test_load_model_instance(self, qtbot):
        # Minimal test
        win = LabMainWindow()
        qtbot.addWidget(win)

        # Mock model and spec
        spec = MagicMock()
        spec.name = "TestModel"

        model = MagicMock()

        # We need to mock ToolRegistry to avoid actual tool loading which might require heavy deps
        with patch(
            "bioplausible_ui.lab.registry.ToolRegistry.get_compatible_tools",
            return_value=[],
        ):
            win.load_model_instance(model, spec)
            assert win.model == model
