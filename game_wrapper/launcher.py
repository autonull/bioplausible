import threading
import time
import sys
import os
import subprocess
import json

# We'll use subprocess to run the training in a separate process 
# to ensure the game loop never stutters due to the GIL.
# We will invoke a helper script.

class JobLauncher:
    def __init__(self, python_path=sys.executable):
        self.python_path = python_path
        self.active_jobs = [] # List of (Popen, model_name, start_time)
        self.log_file = "results/launcher.log"
        # Ensure log dir exists
        os.makedirs("results", exist_ok=True)

    def launch(self, model_name, config):
        worker_script = os.path.join(os.path.dirname(__file__), "worker_process.py")
        env = os.environ.copy()
        config_str = json.dumps(config)
        
        # Log launch
        with open(self.log_file, "a") as f:
            f.write(f"[LAUNCH] {time.ctime()}: {model_name} | {config_str}\n")
        
        try:
            # Unix-specific: setsid to allow group killing if needed
            proc = subprocess.Popen(
                [self.python_path, worker_script, model_name, config_str],
                cwd=os.getcwd(),
                env=env,
                stdout=subprocess.DEVNULL, # Redirect to avoid buffer filling
                stderr=subprocess.DEVNULL, # Ideally pipe to log, but keep simple
                start_new_session=True 
            )
            self.active_jobs.append((proc, model_name, time.time()))
        except Exception as e:
            with open(self.log_file, "a") as f:
                f.write(f"[ERROR] Launch failed: {e}\n")

    def prune(self):
        """Remove finished jobs and update count."""
        living = []
        for proc, name, start in self.active_jobs:
            ret = proc.poll()
            if ret is None:
                living.append((proc, name, start))
            else:
                # Job finished
                duration = time.time() - start
                status = "SUCCESS" if ret == 0 else f"FAIL({ret})"
                with open(self.log_file, "a") as f:
                    f.write(f"[FINISH] {name}: {status} in {duration:.1f}s\n")
        self.active_jobs = living
        return len(self.active_jobs)

    def cleanup(self):
        with open(self.log_file, "a") as f:
            f.write(f"[CLEANUP] Killing {len(self.active_jobs)} active jobs.\n")
        for proc, _, _ in self.active_jobs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    # Give it a second
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                       proc.kill()
                except Exception:
                    pass
        self.active_jobs = []
