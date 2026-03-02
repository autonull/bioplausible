import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from bioplausible.hyperopt import PatientLevel
from bioplausible.models.base import BioModel, ModelConfig
from bioplausible.models.registry import ModelSpec, ModelRegistry, MODEL_REGISTRY
from bioplausible.scientist.core import AutoScientist
from bioplausible.scientist.report.orchestrator import ReportOrchestrator
from bioplausible.scientist.task import ExperimentTask


class MockBioModel(BioModel):
    """A tiny mock model that executes instantly and returns high performance."""
    algorithm_name = "MockBioAlgo"

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        # Just a tiny linear layer to have parameters
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x, **kwargs):
        return self.fc(x)

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        # Fake training step that returns high accuracy
        return {"loss": 0.05, "accuracy": 0.98, "val_acc": 0.98, "train_acc": 0.98, "val_loss": 0.05, "time": 0.001, "energy_proxy": 10.0}

class MockBaselineModel(BioModel):
    """Another tiny mock model that returns baseline performance."""
    algorithm_name = "Backprop Baseline"

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x, **kwargs):
        return self.fc(x)

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        # Fake training step that returns lower accuracy
        return {"loss": 0.3, "accuracy": 0.85, "val_acc": 0.85, "train_acc": 0.85, "val_loss": 0.3, "time": 0.005, "energy_proxy": 50.0}


class TestMockAnalysisIntegration(unittest.TestCase):
    """
    True end-to-end integration test that actually runs the mock models
    and experiment runner to verify the analysis pipeline.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)

        self.db_path = str(self.temp_dir_path / "mock_bioplausible.db")
        self.report_dir = self.temp_dir_path / "reports"
        self.report_dir.mkdir()

        # Register mock models
        ModelRegistry.register("MockBioAlgo")(MockBioModel)
        ModelRegistry.register("Backprop Baseline")(MockBaselineModel)

        self.mock_specs = [
            ModelSpec(
                name="MockBioAlgo",
                description="Fast mock algorithm",
                model_type="MockBioAlgo",
                task_compat=["digits"],
                family="eqprop"
            ),
            ModelSpec(
                name="Backprop Baseline",
                description="Fast mock baseline",
                model_type="Backprop Baseline",
                task_compat=["digits"],
                family="baseline"
            )
        ]

        # Patch the global MODEL_REGISTRY
        self.registry_patcher = patch('bioplausible.models.registry.MODEL_REGISTRY', self.mock_specs)
        self.registry_patcher.start()

        # Also patch get_model_spec so the runner can find our mock specs
        def mock_get_model_spec(name):
            target = name.lower().replace(" ", "").replace("_", "")
            for spec in self.mock_specs:
                if spec.name.lower().replace(" ", "").replace("_", "") == target:
                    return spec
            raise ValueError(f"Unknown mock model: {name}")

        self.get_spec_patcher = patch('bioplausible.hyperopt.experiment.get_model_spec', mock_get_model_spec)
        self.get_spec_patcher.start()

        # Same for scientist runner
        self.get_spec_patcher2 = patch('bioplausible.scientist.report.latex.get_model_spec', mock_get_model_spec)
        self.get_spec_patcher2.start()

    def tearDown(self):
        self.registry_patcher.stop()
        self.get_spec_patcher.stop()
        self.get_spec_patcher2.stop()
        self.temp_dir.cleanup()

    def test_end_to_end_mock_analysis(self):
        # We need to run AutoScientist but control its tasks so it finishes quickly.
        # We'll patch `generate_candidates` to only return our two tasks,
        # and `plan_next` to return them one by one, then None to stop.

        task1 = ExperimentTask(
            model_name="MockBioAlgo",
            task_name="digits",
            tier=PatientLevel.SMOKE, # Fastest
            study_name="MockBioAlgo_digits_smoke",
            priority=100.0
        )

        task2 = ExperimentTask(
            model_name="Backprop Baseline",
            task_name="digits",
            tier=PatientLevel.SMOKE,
            study_name="Backprop_Baseline_digits_smoke",
            priority=90.0
        )

        # Provide a sequence: task1, task2, then None (which causes the loop to pause/wait,
        # so we'll patch _handle_no_task to break the loop instead).
        tasks = [task1, task2]

        def mock_plan_next():
            if tasks:
                return tasks.pop(0)
            return None

        from bioplausible.config import GLOBAL_CONFIG
        GLOBAL_CONFIG.quick_mode = True

        # In a real environment, _process_task delegates to run_single_trial_task,
        # which uses DB_PATH from core.py unless explicitly passed.
        # AutoScientist's loop runs standard optuna trials automatically if the DB matches.
        with patch("bioplausible.scientist.core.DB_PATH", self.db_path):
            scientist = AutoScientist(db_path=self.db_path, num_workers=1)

            # We want Optuna to create real trials, so we should NOT use fixed_config for these
            # test tasks. If we use fixed_config, _process_task bypasses Optuna's study.ask()
            # and study.tell(), leaving no Optuna trials for the reporter.
            # So, we modify the tasks to omit fixed_config, ensuring Optuna sampling happens.
            task1.fixed_config = None
            task2.fixed_config = None

            with patch.object(scientist.strategy, 'plan_next', side_effect=mock_plan_next):
                original_handle_no_task = scientist._handle_no_task

                def stop_loop_on_no_task(task):
                    if task is None:
                        scientist.running = False
                        return False
                    return original_handle_no_task(task)

                with patch.object(scientist, '_handle_no_task', side_effect=stop_loop_on_no_task):
                    with patch("time.sleep"):
                        scientist.run()

        # Now generate reports based on the actual trials run by the system
        orchestrator = ReportOrchestrator(self.db_path, str(self.report_dir))
        orchestrator.generate_reports()

        # Verify outputs were created
        run_dirs = [d for d in self.report_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        self.assertEqual(len(run_dirs), 1, "Expected exactly one run directory to be generated.")

        report_path = run_dirs[0]

        full_report_md = report_path / "FULL_REPORT.md"
        synthesis_json = report_path / "synthesis" / "research_synthesis.json"

        self.assertTrue(full_report_md.exists(), "FULL_REPORT.md should exist.")
        self.assertTrue(synthesis_json.exists(), "research_synthesis.json should exist.")

        # Verify the synthesis JSON contains our models
        with open(synthesis_json, "r") as f:
            synthesis_data = json.load(f)

        insights = synthesis_data.get("cross_algorithm_insights", {})
        rankings = insights.get("rankings", []) if isinstance(insights, dict) else []

        if len(rankings) < 2:
            import pprint
            print("\n--- SYNTHESIS DATA DEBUGINFO ---")
            pprint.pprint(synthesis_data)
            print("----------------------------------\n")

        self.assertTrue(len(rankings) >= 2, f"Should have at least 2 models in rankings. Found {len(rankings)}")

        ranked_models = [r["model"] for r in rankings]
        self.assertIn("MockBioAlgo", ranked_models)
        self.assertIn("Backprop Baseline", ranked_models)

        # Verify accurate synthesis logic
        bio_rank = next(r for r in rankings if r["model"] == "MockBioAlgo")
        baseline_rank = next(r for r in rankings if r["model"] == "Backprop Baseline")

        # Assert accuracy mapping correctly parses standard values
        self.assertAlmostEqual(bio_rank["best_accuracy"], 0.98, places=2)
        self.assertAlmostEqual(baseline_rank["best_accuracy"], 0.85, places=2)

        # Verify Backprop Gap Analysis identifies Bio-Algorithm advantage
        gap_analysis = synthesis_data.get("backprop_gap_analysis", {})
        self.assertIn("summary", gap_analysis)
        self.assertEqual(gap_analysis["summary"].get("bio_wins_on_tasks"), 1)
        self.assertIn("winning_models", gap_analysis)
        winning_models = gap_analysis["winning_models"]
        self.assertTrue(len(winning_models) > 0)
        self.assertEqual(winning_models[0]["model"], "MockBioAlgo")
        self.assertAlmostEqual(winning_models[0]["avg_advantage"], 0.13, places=2) # 0.98 - 0.85

        # Verify Task-Specific Winners
        task_winners = synthesis_data.get("task_specific_winners", {})
        self.assertIn("digits", task_winners)
        self.assertEqual(task_winners["digits"][0]["model"], "MockBioAlgo")
        self.assertAlmostEqual(task_winners["digits"][0]["accuracy"], 0.98, places=2)

        # Verify Markdown Generation contents
        synthesis_md = report_path / "synthesis" / "SYNTHESIS.md"
        self.assertTrue(synthesis_md.exists(), "SYNTHESIS.md should exist.")

        with open(synthesis_md, "r") as f:
            md_content = f.read()

        # Assert correct markdown elements are generated and reflect synthetic scores
        self.assertIn("## 🏆 Cross-Algorithm Performance Rankings", md_content)
        self.assertIn("MockBioAlgo", md_content)
        self.assertIn("98.00%", md_content)
        self.assertIn("85.00%", md_content)
        self.assertIn("## 🎯 Backprop Baseline Comparison", md_content)
        self.assertIn("+13.00%", md_content)
        self.assertIn("## 📊 Task-Specific Winners", md_content)

if __name__ == '__main__':
    unittest.main()
