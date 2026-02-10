import json
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from bioplausible.p2p.node import Coordinator, Worker


class TestP2P(unittest.TestCase):
    def setUp(self):
        # Start Coordinator on a random high port
        import socket

        sock = socket.socket()
        sock.bind(("", 0))
        self.port = sock.getsockname()[1]
        sock.close()

        self.coordinator = Coordinator(port=self.port)
        self.coordinator.start()

        self.url = f"http://localhost:{self.port}"
        self.worker = Worker(self.url, client_id="test_worker")

    def tearDown(self):
        self.worker.stop()
        self.coordinator.stop()

    def test_registration(self):
        # Manually register
        resp = self.worker._post("/register", {"client_id": "test_worker"})
        self.assertEqual(resp["status"], "registered")
        self.assertIn("test_worker", self.coordinator.nodes)

    @patch("bioplausible.p2p.node.run_single_trial_task")
    def test_full_cycle(self, mock_run_task):
        # Setup Mock
        mock_run_task.return_value = {
            "accuracy": 0.95,
            "loss": 0.1,
            "perplexity": 1.0,
            "time": 1.0,
        }

        # Start Worker in thread
        self.worker.start_loop()

        # Wait a bit for cycle to complete
        time.sleep(2)

        # Verify
        self.worker.stop()

        # Check Coordinator
        self.assertGreater(self.coordinator.jobs_completed, 0)

        # Check Worker
        self.assertGreater(self.worker.jobs_done, 0)
        self.assertGreater(self.worker.points, 0)

    def test_status_endpoint(self):
        resp = self.worker._get("/status")
        self.assertEqual(resp["status"], "online")


if __name__ == "__main__":
    unittest.main()
