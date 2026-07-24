import time
import unittest

import requests

from bioplausible.p2p.node import Coordinator
from bioplausible.p2p.node import Worker


class TestP2PIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start Coordinator on a random port (or fixed for simplicity)
        cls.coord_port = 8001
        cls.coord = Coordinator(host="127.0.0.1", port=cls.coord_port)
        cls.coord.start()
        # Give it a moment to bind
        time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        cls.coord.stop()

    def test_worker_registration_and_job(self):
        coord_url = f"http://127.0.0.1:{self.coord_port}"

        # Manually register via API to test endpoint
        resp = requests.post(
            f"{coord_url}/register",
            json={"client_id": "test_client", "capabilities": {}},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "registered")

        # Start Worker
        # We mock run_job to avoid actually training models during test
        original_run_job = Worker._run_job

        def mock_run_job(self, job_id, task, model_name, config):
            print(f"MOCK running job {job_id}")
            return {"accuracy": 0.99, "loss": 0.1}

        Worker._run_job = mock_run_job

        worker = Worker(coord_url, client_id="worker_1")
        worker.start_loop()

        # Wait for worker to fetch and submit
        # The coordinator populates jobs automatically

        # Wait for worker to register
        timeout = 10
        start = time.time()
        while "worker_1" not in self.coord.nodes and time.time() - start < timeout:
            time.sleep(0.5)

        timeout = 10
        start = time.time()
        job_done = False

        while time.time() - start < timeout:
            if worker.jobs_done > 0:
                job_done = True
                break
            time.sleep(0.5)

        worker.stop()
        Worker._run_job = original_run_job

        self.assertTrue(job_done, "Worker should have completed at least one job")
        self.assertGreater(worker.points, 0)

        # Check coordinator state
        self.assertIn("worker_1", self.coord.nodes)
        self.assertGreater(self.coord.jobs_completed, 0)


if __name__ == "__main__":
    unittest.main()
