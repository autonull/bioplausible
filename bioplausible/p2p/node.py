import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from bioplausible.hyperopt.experiment import run_single_trial_task
from bioplausible.hyperopt.search_space import get_search_space
from bioplausible.p2p.state import load_state, save_state

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("P2PNode")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class CoordinatorHandler(BaseHTTPRequestHandler):
    def _send_response(self, data: Dict[str, Any], status=200):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            # Simple dashboard
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = f"""
            <html>
            <head><title>BioPlausible P2P Coordinator</title></head>
            <body>
                <h1>Coordinator Status</h1>
                <p>Status: Online</p>
                <p>Connected Nodes: {len(self.server.coordinator.nodes)}</p>
                <p>Jobs Completed: {self.server.coordinator.jobs_completed}</p>
                <p>Queue Length: {len(self.server.coordinator.job_queue)}</p>
                <h3>Nodes</h3>
                <pre>{json.dumps(
                    self.server.coordinator.node_capabilities, indent=2
                )}</pre>
            </body>
            </html>
            """
            self.wfile.write(html.encode("utf-8"))
        elif self.path == "/status":
            self._send_response(
                {
                    "status": "online",
                    "nodes": len(self.server.coordinator.nodes),
                    "jobs_completed": self.server.coordinator.jobs_completed,
                    "node_capabilities": self.server.coordinator.node_capabilities,
                }
            )
        elif self.path == "/health":
            self._send_response({"status": "healthy"})
        elif self.path.startswith("/get_job"):
            # Parse query params (simple)
            # /get_job?client_id=xyz
            import urllib.parse

            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            client_id = params.get("client_id", [None])[0]

            job = self.server.coordinator.get_job(client_id)
            if job:
                self._send_response(job)
            else:
                self._send_response({"status": "no_jobs"})
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/submit_result":
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                result = json.loads(post_data.decode("utf-8"))
                self.server.coordinator.submit_result(result)
                self._send_response({"status": "accepted"})
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
            except Exception as e:
                logger.error(f"Error processing result: {e}")
                self.send_error(500)
        elif self.path == "/register":
            try:
                content_length = int(self.headers["Content-Length"])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode("utf-8"))
                client_id = data.get("client_id", "unknown")
                capabilities = data.get("capabilities", {})
                self.server.coordinator.register_node(client_id, capabilities)
                self._send_response({"status": "registered"})
            except json.JSONDecodeError:
                self.send_error(400, "Invalid JSON")
            except Exception as e:
                logger.error(f"Error registering node: {e}")
                self.send_error(500)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress default logging


class Coordinator:
    def __init__(self, host="0.0.0.0", port=8000):
        self.host = host
        self.port = port
        self.server = None
        self.thread = None
        self.running = False
        self.nodes = set()
        self.node_capabilities = {}
        self.jobs_completed = 0
        self.job_counter = 0
        self.lock = threading.Lock()

        # Simple job queue
        self.job_queue = []
        self._populate_initial_jobs()

    def _populate_initial_jobs(self):
        # Dynamically generate jobs using search space sampling
        tasks = ["shakespeare", "mnist"]
        # Basic models to seed the network
        models = [
            "EqProp MLP",
            "Backprop Baseline",
            "Direct Feedback Alignment",
            "EquiTile",
        ]

        # Add some advanced ones occasionally
        if self.job_counter % 5 == 0:
            models.append("EqProp Transformer (Attention Only)")
            models.append("ModernConvEqProp")
            models.append("EquiTile EP")
            models.append("LM EquiTile")

        # Populate queue if running low
        import random

        while len(self.job_queue) < 10:
            task = random.choice(tasks)
            model_name = random.choice(models)

            requirements = {}
            if model_name in [
                "ModernConvEqProp",
                "EqProp Transformer (Attention Only)",
                "LM EquiTile",
            ]:
                requirements["gpu"] = True

            try:
                space = get_search_space(model_name)
                config = space.sample()

                # Apply sensible defaults for P2P/Network load
                config["epochs"] = 1
                if "steps" in config:
                    # Vary steps to explore trade-off
                    config["steps"] = random.choice([5, 10, 15, 20])

                # Explore nudge strength
                if "beta" not in config:
                    config["beta"] = random.choice([0.1, 0.22, 0.5])

                self.job_queue.append(
                    {
                        "job_id": self.job_counter,
                        "task": task,
                        "model_name": model_name,
                        "config": config,
                        "requirements": requirements,
                    }
                )
                self.job_counter += 1
            except Exception as e:
                logger.error(f"Failed to generate job for {model_name}: {e}")

    def start(self):
        if self.running:
            return
        self.server = ThreadingHTTPServer((self.host, self.port), CoordinatorHandler)
        self.server.coordinator = self
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        self.running = True
        logger.info(f"Coordinator started on {self.host}:{self.port}")

    def stop(self):
        if self.server:
            # Shutdown in separate thread to avoid deadlocks if called from
            # request handler
            t = threading.Thread(target=self.server.shutdown)
            t.start()
            t.join(timeout=2)
            self.server.server_close()
        self.running = False
        logger.info("Coordinator stopped")

    def get_job(self, client_id: str = None) -> Optional[Dict]:
        with self.lock:
            if len(self.job_queue) < 5:
                self._populate_initial_jobs()  # Replenish if running low

            # Check capabilities if client_id is known
            client_caps = self.node_capabilities.get(client_id, {})
            has_gpu = client_caps.get("cuda", False)

            # Find a suitable job
            best_job_idx = -1

            for i, job in enumerate(self.job_queue):
                requirements = job.get("requirements", {})
                requires_gpu = requirements.get("gpu", False)

                # Filter incompatible jobs
                if requires_gpu and not has_gpu:
                    continue

                # Prioritize:
                # 1. If worker has GPU, prefer GPU jobs (to not waste GPU on CPU tasks)
                # 2. Otherwise FCFS
                if has_gpu and requires_gpu:
                    best_job_idx = i
                    break  # Ideal match

                # If worker has GPU but job is CPU, keep looking for a GPU job
                # But if we haven't found anything else, this is a candidate
                if best_job_idx == -1:
                    best_job_idx = i

            if best_job_idx != -1:
                return self.job_queue.pop(best_job_idx)

            return None

    def submit_result(self, result: Dict):
        with self.lock:
            self.jobs_completed += 1
        job_id = result.get("job_id")
        acc = result.get("accuracy", 0.0)
        logger.info(f"Job {job_id} completed. Acc: {acc:.4f}")
        # Here we would feed back into the evolutionary algo

    def register_node(self, client_id: str, capabilities: Dict = None):
        with self.lock:
            self.nodes.add(client_id)
            if capabilities:
                self.node_capabilities[client_id] = capabilities


class Worker:
    def __init__(self, coordinator_url: str, client_id: str = None):
        self.coordinator_url = coordinator_url.rstrip("/")
        self.client_id = client_id or str(uuid.uuid4())[:8]
        self.running = False

        # Load state
        state = load_state()
        self.points = state.get("points", 0)
        self.jobs_done = state.get("jobs_done", 0)

        self.current_status = "Idle"

        # Signals/Callbacks
        self.on_status_change = None  # func(status, points, jobs)
        self.on_log = None  # func(msg)

    def log(self, msg):
        logger.info(msg)
        if self.on_log:
            self.on_log(msg)

    def start_loop(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        self.log(
            f"Worker {self.client_id} started. Connecting to {self.coordinator_url}..."
        )

        # Collect capabilities
        import torch

        from bioplausible.models.triton_kernel import TritonEqPropOps

        caps = {
            "cuda": torch.cuda.is_available(),
            "triton": TritonEqPropOps.is_available(),
            "device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
            ),
        }

        # Register with retry
        registered = False
        while not registered and self.running:
            try:
                self._post(
                    "/register", {"client_id": self.client_id, "capabilities": caps}
                )
                registered = True
            except Exception as e:
                self.log(f"Failed to register: {e}. Retrying in 2s...")
                time.sleep(2)

        while self.running:
            try:
                # 1. Get Job
                self._update_status("Fetching Job...")
                job = self._get("/get_job?client_id=" + self.client_id)

                if job and "job_id" in job:
                    job_id = job["job_id"]
                    task = job.get("task", "shakespeare")
                    model_name = job.get("model_name")
                    config = job.get("config")

                    self.log(f"Got Job {job_id}: {model_name} on {task}")
                    self._update_status(f"Running Job {job_id}...")

                    # 2. Run Job
                    result_metrics = self._run_job(job_id, task, model_name, config)

                    if result_metrics:
                        # 3. Submit Result
                        self._update_status("Submitting Result...")
                        payload = {
                            "job_id": job_id,
                            "client_id": self.client_id,
                            **result_metrics,
                        }
                        self._post("/submit_result", payload)

                        self.points += 10  # Gamification
                        self.jobs_done += 1
                        save_state(self.points, self.jobs_done)  # Persist
                        self.log(f"Job {job_id} submitted! Points: {self.points}")
                    else:
                        self.log(f"Job {job_id} failed locally.")

                else:
                    self._update_status("Idle (No jobs)")
                    time.sleep(5)

            except URLError:
                self._update_status("Connection Failed")
                self.log("Cannot connect to coordinator. Retrying in 10s...")
                time.sleep(10)
            except Exception as e:
                self.log(f"Worker Error: {e}")
                time.sleep(5)

            time.sleep(1)  # Breath

        self._update_status("Stopped")

    def _run_job(self, job_id, task, model_name, config) -> Optional[Dict]:
        # Remote jobs are stored in the main DB so they appear in visualizations
        # Use default storage path: "results/hyperopt.db"
        return run_single_trial_task(
            task=task,
            model_name=model_name,
            config=config,
            storage_path="results/hyperopt.db",
            job_id=job_id,
        )

    def _update_status(self, status):
        self.current_status = status
        if self.on_status_change:
            self.on_status_change(status, self.points, self.jobs_done)

    def _get(self, endpoint) -> Dict:
        url = f"{self.coordinator_url}{endpoint}"
        req = Request(url)
        with urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())

    def _post(self, endpoint, data: Dict) -> Dict:
        url = f"{self.coordinator_url}{endpoint}"
        req = Request(
            url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
