import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from bioplausible.p2p.node import Coordinator


class TestCoordinator(unittest.TestCase):
    def setUp(self):
        self.coordinator = Coordinator(port=0)  # Ephemeral port

    def test_job_generation(self):
        """Test that initial jobs are populated."""
        self.coordinator._populate_initial_jobs()
        self.assertGreater(len(self.coordinator.job_queue), 0)

        # Check job structure
        job = self.coordinator.job_queue[0]
        self.assertIn("job_id", job)
        self.assertIn("config", job)
        self.assertIn("requirements", job)

    def test_gpu_prioritization(self):
        """Test that GPU nodes get GPU jobs preferentially."""

        # Add a GPU job and a CPU job
        gpu_job = {
            "job_id": 1,
            "model_name": "ModernConvEqProp",
            "requirements": {"gpu": True},
        }
        cpu_job = {
            "job_id": 2,
            "model_name": "EqProp MLP",
            "requirements": {"gpu": False},
        }

        # Clear default jobs
        self.coordinator.job_queue = []

        # Scenario 1: Queue order [CPU, GPU]. Worker has GPU.
        # Should pick GPU job (index 1) over CPU job (index 0)
        self.coordinator.job_queue = [cpu_job, gpu_job]
        self.coordinator.node_capabilities["gpu_node"] = {"cuda": True}

        # Mock _populate_initial_jobs to prevent auto-refill during test
        with patch.object(
            self.coordinator, "_populate_initial_jobs", return_value=None
        ):
            job = self.coordinator.get_job("gpu_node")
            self.assertEqual(job["job_id"], 1, "GPU node should pick GPU job first")

            # Remaining job should be CPU
            self.assertEqual(len(self.coordinator.job_queue), 1)
            self.assertEqual(self.coordinator.job_queue[0]["job_id"], 2)

    def test_capability_filtering(self):
        """Test that CPU nodes cannot take GPU jobs."""
        gpu_job = {"job_id": 1, "requirements": {"gpu": True}}
        cpu_job = {"job_id": 2, "requirements": {"gpu": False}}

        self.coordinator.job_queue = [gpu_job, cpu_job]
        self.coordinator.node_capabilities["cpu_node"] = {"cuda": False}

        job = self.coordinator.get_job("cpu_node")
        self.assertEqual(job["job_id"], 2, "CPU node must skip GPU job")


if __name__ == "__main__":
    unittest.main()
