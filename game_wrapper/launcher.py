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
    def __init__(self, python_path=sys.executable, script_path="launch_studio.py"):
        self.python_path = python_path
        self.script_path = script_path
        self.active_jobs = []

    def launch(self, model_name, config):
        """
        Launch a new training job.
        """
        # We need a script that takes arguments or json
        # Let's assume we can pass a JSON string as an arg to a generalized launcher
        
        cmd = [
            self.python_path, 
            "-m", "bioplausible.cli_smoke_models_extended", # Using smoke test for fast feedback? 
            # Or we should make a dedicated 'single_shot.py'
            # let's write a dedicated one in the main thread setup
        ]
        
        # ACTUALLY, let's just write a 'worker.py' in the game directory that imports the heavy stuff
        # and runs one job, then exits.
        
        worker_script = os.path.join(os.path.dirname(__file__), "worker_process.py")
        
        env = os.environ.copy()
        
        # Pass config via env vars or just json arg
        config_str = json.dumps(config)
        
        print(f"Launching {model_name} with {config_str}")
        
        proc = subprocess.Popen(
            [self.python_path, worker_script, model_name, config_str],
            cwd=os.getcwd(),
            env=env
        )
        self.active_jobs.append(proc)
        
    def cleanup(self):
        for p in self.active_jobs:
            if p.poll() is None:
                p.terminate()
