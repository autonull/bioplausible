
import unittest
import json
from dataclasses import dataclass
from bioplausible.scientist.report.composer import ReportComposer
# Sections imported implicitly by Composer, but we test output

class TestReportGeneration(unittest.TestCase):

    def setUp(self):
        # Mock Data with structure similar to actual training output
        self.mock_data = {
            "model_name": "TestModel",
            "task_name": "TestTask",
            "trial_id": 123,
            "config": {
                "lr": 0.01,
                "optimizer": "adam",
                "layers": 3
            },
            "metrics": {
                "accuracy": 0.85, 
                "loss": 0.4,
                "perplexity": 12.5
            },
            # Mocking a Trajectory object structure
            "trajectory": type("obj", (object,), {
                "checkpoints": [
                    type("ckpt", (object,), {
                        "train_acc": 0.88, "val_acc": 0.85, 
                        "train_loss": 0.35, "val_loss": 0.4
                    })
                ],
                "convergence_epoch": 10,
                "converged": True,
                "overfitting_detected": False,
                "unstable": False
            })()
        }

    def test_markdown_compilation(self):
        """Test Markdown report generation."""
        composer = ReportComposer(self.mock_data)
        md = composer.compile_markdown()
        
        self.assertIn("# Scientist++ Experiment Report", md)
        self.assertIn("TestModel", md)
        # Check for sections
        self.assertIn("Experimental Configuration", md)
        self.assertIn("Performance Summary", md)
        self.assertIn("Training Dynamics Analysis", md)
        
        # Check for data values
        self.assertIn("0.01", md)
        self.assertIn("85.00%", md)
        self.assertIn("Converged at epoch **10**", md)

    def test_json_compilation(self):
        """Test JSON report generation."""
        composer = ReportComposer(self.mock_data)
        json_out = composer.compile_json()
        
        data = json.loads(json_out)
        self.assertEqual(data["meta"]["model_name"], "TestModel")
        
        # Should have 3 sections by default
        self.assertEqual(len(data["sections"]), 3)
        
        # Verify sections exist by ID
        section_ids = [s["section"] for s in data["sections"]]
        self.assertIn("config", section_ids)
        self.assertIn("performance", section_ids)
        self.assertIn("dynamics", section_ids)
        
        # Check specific values
        config_section = next(s for s in data["sections"] if s["section"] == "config")
        self.assertEqual(config_section["data"]["lr"], 0.01)

    def test_save_reports(self):
        """Test saving reports to disk."""
        import tempfile
        import shutil
        import os
        from pathlib import Path
        
        temp_dir = tempfile.mkdtemp()
        try:
            composer = ReportComposer(self.mock_data)
            composer.save_reports(temp_dir)
            
            # Check files exist
            files = os.listdir(temp_dir)
            self.assertIn("report.md", files)
            self.assertIn("report.json", files)
            self.assertIn("manifest.json", files)
            
            # Verify manifest content
            with open(Path(temp_dir) / "manifest.json", "r") as f:
                manifest = json.load(f)
                self.assertEqual(manifest["report_version"], "2.0")
                self.assertEqual(len(manifest["files"]), 2)
                self.assertIn("config", manifest["sections"])
                
        finally:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    unittest.main()
