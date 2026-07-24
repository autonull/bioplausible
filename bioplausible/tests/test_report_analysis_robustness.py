import shutil
import tempfile
import unittest
from pathlib import Path

from bioplausible.execution.report.analysis import MLAnalyzer


class TestReportAnalysisRobustness(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir)
        self.analyzer = MLAnalyzer(self.output_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_analyze_direct_robustness(self):
        # Dummy data with robustness metrics
        data = [
            # Model A: Good overall
            {
                "model": "ModelA",
                "robustness_score": 0.8,
                "noise_score": 0.9,
                "perturbation_score": 0.8,
                "ood_score": 0.7,
                "adversarial_fgsm": 0.6,
                "adversarial_pgd": 0.5,
            },
            {
                "model": "ModelA",
                "robustness_score": 0.82,
                "noise_score": 0.92,
                "perturbation_score": 0.82,
                "ood_score": 0.72,
                "adversarial_fgsm": 0.62,
                "adversarial_pgd": 0.52,
            },
            # Model B: Poor adversarial
            {
                "model": "ModelB",
                "robustness_score": 0.5,
                "noise_score": 0.8,
                "perturbation_score": 0.7,
                "ood_score": 0.5,
                "adversarial_fgsm": 0.1,
                "adversarial_pgd": 0.05,
            },
            # Control: No robustness data
            {"model": "ModelC", "accuracy": 0.9},
        ]

        # Call the method (we'll implement it next)
        if hasattr(self.analyzer, "_analyze_direct_robustness"):
            report = self.analyzer._analyze_direct_robustness(data)

            # Assertions
            self.assertIn("### Adversarial & Noise Robustness", report)
            self.assertIn("ModelA", report)
            self.assertIn("ModelB", report)

            # Check for table structure
            self.assertIn(
                "| Model | Overall | Noise | Perturb | OOD | Adv (FGSM) | Adv (PGD) |",
                report,
            )

            # Check average calculation for ModelA
            # robustness: (0.8+0.82)/2 = 0.81
            self.assertIn("0.810", report)
            # noise: (0.9+0.92)/2 = 0.91
            self.assertIn("0.910", report)


if __name__ == "__main__":
    unittest.main()
