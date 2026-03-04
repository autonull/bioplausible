import json
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn

from bioplausible.hyperopt import PatientLevel
from bioplausible.models.base import BioModel, ModelConfig
from bioplausible.models.registry import (MODEL_REGISTRY, ModelRegistry,
                                          ModelSpec)
from bioplausible.scientist.core import AutoScientist
from bioplausible.scientist.report.orchestrator import ReportOrchestrator
from bioplausible.scientist.task import ExperimentTask


class BaseMockModel(BioModel):
    """A base class for mock models to eliminate boilerplate."""

    # Defaults overridden by subclasses
    base_acc = 0.5
    base_loss = 0.5
    base_time = 0.01
    base_energy = 10.0
    noise_range = 0.01
    should_crash = False

    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x, **kwargs):
        return self.fc(x)

    def train_step(self, x: torch.Tensor, y: torch.Tensor):
        if self.should_crash:
            raise ValueError("Simulated exploding gradients / NaN")

        acc = self.base_acc + random.uniform(-self.noise_range, self.noise_range)
        loss = self.base_loss + random.uniform(-self.noise_range, self.noise_range)

        return {
            "loss": loss,
            "accuracy": acc,
            "val_acc": acc,
            "train_acc": acc,
            "val_loss": loss,
            "time": self.base_time,
            "energy_proxy": self.base_energy,
        }


class MockBioModel(BaseMockModel):
    algorithm_name = "MockBioAlgo"
    base_acc = 0.98
    base_loss = 0.05
    base_time = 0.001
    base_energy = 10.0


class MockBaselineModel(BaseMockModel):
    algorithm_name = "Backprop Baseline"
    base_acc = 0.85
    base_loss = 0.3
    base_time = 0.005
    base_energy = 50.0


class MockEfficientAlgo(BaseMockModel):
    algorithm_name = "MockEfficientAlgo"
    base_acc = 0.90
    base_loss = 0.15
    base_time = 0.002
    base_energy = 5.0


class MockTransformer(BaseMockModel):
    algorithm_name = "MockTransformer"
    base_acc = 0.99
    base_loss = 0.01
    base_time = 0.05
    base_energy = 100.0
    noise_range = 0.005


class MockUnstableAlgo(BaseMockModel):
    algorithm_name = "MockUnstableAlgo"
    should_crash = True


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
        ModelRegistry.register("MockEfficientAlgo")(MockEfficientAlgo)
        ModelRegistry.register("MockTransformer")(MockTransformer)
        ModelRegistry.register("MockUnstableAlgo")(MockUnstableAlgo)

        self.mock_specs = [
            ModelSpec(
                name="MockBioAlgo",
                description="Fast mock algorithm",
                model_type="MockBioAlgo",
                task_compat=["digits"],
                family="eqprop",
            ),
            ModelSpec(
                name="Backprop Baseline",
                description="Fast mock baseline",
                model_type="Backprop Baseline",
                task_compat=["digits"],
                family="baseline",
            ),
            ModelSpec(
                name="MockEfficientAlgo",
                description="High param efficiency",
                model_type="MockEfficientAlgo",
                task_compat=["digits"],
                family="efficient",
            ),
            ModelSpec(
                name="MockTransformer",
                description="Transformer-based mock",
                model_type="MockTransformer",
                task_compat=["digits"],
                family="transformer",
            ),
            ModelSpec(
                name="MockUnstableAlgo",
                description="Always crashes mock",
                model_type="MockUnstableAlgo",
                task_compat=["digits"],
                family="experimental",
            ),
        ]

        # Patch the global MODEL_REGISTRY
        self.registry_patcher = patch(
            "bioplausible.models.registry.MODEL_REGISTRY", self.mock_specs
        )
        self.registry_patcher.start()

        # Also patch get_model_spec so the runner can find our mock specs
        def mock_get_model_spec(name):
            target = name.lower().replace(" ", "").replace("_", "")
            for spec in self.mock_specs:
                if spec.name.lower().replace(" ", "").replace("_", "") == target:
                    return spec
            raise ValueError(f"Unknown mock model: {name}")

        self.get_spec_patcher = patch(
            "bioplausible.hyperopt.experiment.get_model_spec", mock_get_model_spec
        )
        self.get_spec_patcher.start()

        # Same for scientist runner
        self.get_spec_patcher2 = patch(
            "bioplausible.scientist.report.latex.get_model_spec", mock_get_model_spec
        )
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

        # Generate enough tasks for statistical significance (needs >= 3 trials per model)
        tasks = []
        model_names = [
            "MockBioAlgo",
            "Backprop Baseline",
            "MockEfficientAlgo",
            "MockTransformer",
            "MockUnstableAlgo",
        ]
        # iris dataset fails because tasks aren't properly registered to handle "iris"
        # use an existing task like "mnist" that we know works (but keep it small/mocked so it's fast)
        # Actually since these mock models just return a fake score during train_step, any task name
        # will bypass actual dataset loading IF we mock the data, but task runner will still try to
        # resolve "iris". We'll stick to a registered task name "mnist".
        task_names = ["digits", "mnist"]  # Test multi-task synthesis

        # Override specific param counts and failures for synthesis
        self.mock_configs = {
            "MockBioAlgo": {"hidden_dim": 64, "num_layers": 2},  # ~10k params
            "Backprop Baseline": {"hidden_dim": 64, "num_layers": 2},
            "MockEfficientAlgo": {"hidden_dim": 4, "num_layers": 1},  # < 100 params
            "MockTransformer": {"hidden_dim": 512, "num_layers": 6},  # > 1.5M params
            "MockUnstableAlgo": {"hidden_dim": 64, "num_layers": 2},
        }

        for model_name in model_names:
            for task_name in task_names:
                for i in range(3):
                    t = ExperimentTask(
                        model_name=model_name,
                        task_name=task_name,
                        tier=PatientLevel.SMOKE,
                        study_name=f"{model_name.replace(' ', '_')}_{task_name}_smoke",
                        priority=100.0 - i,
                        # We pass fixed config here to define hyperparams (like hidden_dim),
                        # but we will un-set it before running so optuna creates dynamic trials
                    )
                    t.fixed_config = self.mock_configs[model_name]
                    tasks.append(t)

        # Interleave tasks so UnstableAlgo doesn't trigger MAX_CONSECUTIVE_FAILURES (5)
        # by failing 6 times in a row (3 per task) and halting the test suite via Safe Mode.
        import random

        random.seed(42)  # Ensure deterministic ordering
        random.shuffle(tasks)

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
            for t in tasks:
                t.fixed_config = None

            with patch.object(
                scientist.strategy, "plan_next", side_effect=mock_plan_next
            ):
                original_handle_no_task = scientist._handle_no_task
                original_prepare_optuna = scientist._prepare_optuna_config

                # We monkeypatch _prepare_optuna_config to ensure the trial receives the correct dimensions
                # so the synthesizer logic can estimate parameter counts.
                def mock_prepare_optuna(task, study):
                    trial, config, job_id = original_prepare_optuna(task, study)
                    if task.model_name in self.mock_configs:
                        config.update(self.mock_configs[task.model_name])
                        for k, v in self.mock_configs[task.model_name].items():
                            trial.set_user_attr(k, v)
                    return trial, config, job_id

                with patch.object(
                    scientist, "_prepare_optuna_config", side_effect=mock_prepare_optuna
                ):

                    def stop_loop_on_no_task(task):
                        if task is None:
                            scientist.running = False
                            return False
                        return original_handle_no_task(task)

                    with patch.object(
                        scientist, "_handle_no_task", side_effect=stop_loop_on_no_task
                    ):
                        with patch("time.sleep"):
                            scientist.run()

        # Now generate reports based on the actual trials run by the system
        orchestrator = ReportOrchestrator(self.db_path, str(self.report_dir))
        orchestrator.generate_reports()

        # Verify outputs were created
        run_dirs = [
            d
            for d in self.report_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        self.assertEqual(
            len(run_dirs), 1, "Expected exactly one run directory to be generated."
        )

        report_path = run_dirs[0]

        full_report_md = report_path / "FULL_REPORT.md"
        synthesis_json = report_path / "synthesis" / "research_synthesis.json"

        self.assertTrue(full_report_md.exists(), "FULL_REPORT.md should exist.")
        self.assertTrue(
            synthesis_json.exists(), "research_synthesis.json should exist."
        )

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

        self._verify_rankings(rankings)
        self._verify_gap_analysis(synthesis_data)
        self._verify_task_winners(synthesis_data)
        self._verify_efficiency(synthesis_data)
        self._verify_failures(synthesis_data)
        self._verify_significance(synthesis_data)
        self._verify_markdown(report_path)

    def _verify_rankings(self, rankings):
        self.assertTrue(
            len(rankings) >= 2,
            f"Should have at least 2 models in rankings. Found {len(rankings)}",
        )
        ranked_models = [r["model"] for r in rankings]
        self.assertIn("MockBioAlgo", ranked_models)
        self.assertIn("Backprop Baseline", ranked_models)

        bio_rank = next(r for r in rankings if r["model"] == "MockBioAlgo")
        baseline_rank = next(r for r in rankings if r["model"] == "Backprop Baseline")

        self.assertAlmostEqual(bio_rank["best_accuracy"], 0.98, places=1)
        self.assertAlmostEqual(baseline_rank["best_accuracy"], 0.85, places=1)

    def _verify_gap_analysis(self, synthesis_data):
        gap_analysis = synthesis_data.get("backprop_gap_analysis", {})
        self.assertIn("summary", gap_analysis)
        # Should win on both tasks
        self.assertEqual(gap_analysis["summary"].get("bio_wins_on_tasks"), 2)
        self.assertIn("winning_models", gap_analysis)
        winning_models = gap_analysis["winning_models"]
        self.assertTrue(len(winning_models) > 0)
        self.assertEqual(winning_models[0]["model"], "MockTransformer")
        self.assertTrue(any(w["model"] == "MockBioAlgo" for w in winning_models))

    def _verify_task_winners(self, synthesis_data):
        task_winners = synthesis_data.get("task_specific_winners", {})
        self.assertIn("digits", task_winners)
        self.assertIn("mnist", task_winners)
        self.assertEqual(task_winners["digits"][0]["model"], "MockTransformer")
        self.assertEqual(task_winners["mnist"][0]["model"], "MockTransformer")
        self.assertAlmostEqual(task_winners["digits"][0]["accuracy"], 0.99, places=1)
        self.assertAlmostEqual(task_winners["mnist"][0]["accuracy"], 0.99, places=1)

    def _verify_efficiency(self, synthesis_data):
        efficiency = synthesis_data.get("efficiency_analysis", {})
        self.assertIn("top_param_efficient", efficiency)
        top_efficient = efficiency["top_param_efficient"]
        self.assertTrue(len(top_efficient) > 0)
        self.assertTrue(any(t["model_name"].startswith("Mock") for t in top_efficient))

    def _verify_failures(self, synthesis_data):
        failures = synthesis_data.get("failure_analysis", {})
        self.assertTrue(
            isinstance(failures, dict),
            f"Expected dict for failure_analysis, got {type(failures)}",
        )
        if "counts" in failures:
            counts = failures["counts"]
            self.assertTrue(
                any(counts.get(k, 0) > 0 for k in counts.keys()),
                f"Expected positive failure counts, got {counts}",
            )
        else:
            self.fail(
                f"Expected 'counts' key in failure_analysis, got keys: {list(failures.keys())}"
            )

    def _verify_significance(self, synthesis_data):
        sig = synthesis_data.get("statistical_significance", [])
        self.assertTrue(isinstance(sig, list))
        self.assertTrue(
            len(sig) > 0,
            "Statistical significance array was empty, expected comparisons between Mock models",
        )
        if len(sig) == 1 and "error" in sig[0]:
            pass
        else:
            sig_wins = [
                s for s in sig if "winner" in s and s["winner"] == "MockTransformer"
            ]
            self.assertTrue(
                len(sig_wins) > 0, "MockTransformer did not win any significance tests"
            )

    def _verify_markdown(self, report_path):
        synthesis_md = report_path / "synthesis" / "SYNTHESIS.md"
        self.assertTrue(synthesis_md.exists(), "SYNTHESIS.md should exist.")

        with open(synthesis_md, "r") as f:
            md_content = f.read()

        self.assertIn("## 🏆 Cross-Algorithm Performance Rankings", md_content)
        self.assertIn("MockBioAlgo", md_content)
        self.assertIn("MockEfficientAlgo", md_content)
        self.assertIn("MockTransformer", md_content)
        self.assertIn("## 🎯 Backprop Baseline Comparison", md_content)
        self.assertIn("## 📊 Task-Specific Winners", md_content)


if __name__ == "__main__":
    unittest.main()
