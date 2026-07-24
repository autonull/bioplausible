"""
P2P Evolutionary Controller.

Manages the autonomous discovery loop using the DHT.
"""

import hashlib
import json
import logging
import os
import random
import threading
import time
from typing import Any
from typing import Dict

from bioplausible.hyperopt.experiment import run_single_trial_task
from bioplausible.hyperopt.search_space import SEARCH_SPACES
from bioplausible.hyperopt.search_space import get_search_space
from bioplausible.p2p.dht import DHTNode
from bioplausible.p2p.state import load_state
from bioplausible.p2p.state import save_state

logger = logging.getLogger("P2PEvolution")


def get_config_hash(config: Dict) -> str:
    """Generate a hash for a configuration."""

    def _default(obj):
        if hasattr(obj, "item"):  # numpy scalar
            return obj.item()
        if hasattr(obj, "tolist"):  # numpy array
            return obj.tolist()
        return str(obj)

    # sort keys to ensure determinism
    s = json.dumps(config, sort_keys=True, default=_default)
    return hashlib.md5(s.encode()).hexdigest()


class P2PEvolution:
    def __init__(
        self,
        bootstrap_ip: str = None,
        bootstrap_port: int = 8468,
        discovery_mode: str = "quick",
        constraints: Dict[str, Any] = None,
        task: str = "shakespeare",
    ):
        self.bootstrap_nodes = [(bootstrap_ip, bootstrap_port)] if bootstrap_ip else []
        self.dht = None
        self.discovery_mode = discovery_mode
        self.constraints = constraints or {}
        self.task = task

        self.running = False
        self.thread = None

        # State
        self.local_best_config = None
        self.local_best_score = -float("inf")
        self.manual_queue = []  # Queue for manually injected genomes

        state = load_state()
        self.points = state.get("points", 0)
        self.jobs_done = state.get("jobs_done", 0)

        self.current_status = "Stopped"

        # Signals
        self.on_status_change = None
        self.on_log = None

    def _log(self, msg):
        logger.info(msg)
        if self.on_log:
            self.on_log(msg)

    def _update_status(self, status):
        self.current_status = status
        if self.on_status_change:
            self.on_status_change(status, self.points, self.jobs_done)

    def start(self, auto_nice=True):
        if self.running:
            return

        if auto_nice and hasattr(os, "nice"):
            try:
                os.nice(10)  # Lower priority
                self._log("Process priority lowered (Nice +10)")
            except Exception as e:
                self._log(f"Could not lower priority: {e}")

        # Start DHT
        try:
            # Try to bind to a port, with retries
            base_port = 8468 + random.randint(0, 1000)
            for i in range(10):
                try:
                    local_port = base_port + i
                    self.dht = DHTNode(
                        port=local_port, bootstrap_nodes=self.bootstrap_nodes
                    )
                    self.dht.start()
                    self._log(f"DHT started on port {local_port}")
                    break
                except Exception as e:
                    self._log(f"Port {local_port} busy/failed, retrying... ({e})")
                    if i == 9:
                        raise  # Rethrow on last attempt
                    time.sleep(0.5)

        except Exception as e:
            self._log(f"Failed to start DHT after retries: {e}")
            return

        self.running = True
        self.thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.thread.start()
        self._update_status("Starting P2P Mesh...")

    def stop(self):
        self.running = False
        if self.dht:
            self.dht.stop()
        if self.thread:
            self.thread.join(timeout=2)
        self._update_status("Stopped")

    def inject_genome(self, config: Dict):
        """Inject a manually designed genome into the evaluation queue."""
        self._log("💉 Injecting manual genome for evaluation...")
        self.manual_queue.append(config)

    def _verify_model(self, record: Dict) -> bool:
        """
        Spot-check a model from the network to ensure accuracy is valid.
        Returns True if valid (or close enough), False otherwise.
        """
        config = record.get("config", {})
        claimed_score = record.get("score", 0.0)
        model_name = config.get("model_name", "EqProp MLP")

        self._log(f"🕵️ Verifying global best... (Claimed: {claimed_score:.4f})")
        self._update_status("Verifying Global Best...")

        # Run a quick evaluation
        # We might use fewer epochs for speed, but ideally we match the claimed effort
        # For now, just run same settings as discovery mode
        metrics = run_single_trial_task(
            task=self.task,
            model_name=model_name,
            config=config,
            storage_path=None,  # Don't pollute main DB with verifications
            quick_mode=(self.discovery_mode == "quick"),
        )

        if not metrics:
            self._log("Verification failed: Could not run model.")
            return False

        real_score = metrics.get("accuracy", 0.0)

        # Allow some variance (e.g. 5%)
        tolerance = 0.05
        if abs(real_score - claimed_score) > tolerance and real_score < claimed_score:
            self._log(
                f"❌ Verification FAILED! (Real: {real_score:.4f}"
                f" vs Claimed: {claimed_score:.4f})"
            )
            return False

        self._log(f"✅ Verification PASSED (Real: {real_score:.4f})")
        return True

    def _evolution_loop(self):
        self._log("Joined P2P Mesh network.")

        while self.running:
            try:
                # 1. Fetch Global Best
                self._update_status("Syncing with Mesh...")
                best_record = self.dht.get_best_model(self.task)

                global_best_config = {}
                global_best_score = -float("inf")
                global_model_name = "EqProp MLP"  # Default starting point
                global_gen = 0
                parent_hash = None

                if best_record:
                    # PROOF OF ACCURACY CHECK
                    # 10% chance to verify if we haven't seen this hash before
                    # For simplicity, just random check
                    if random.random() < 0.1:
                        if not self._verify_model(best_record):
                            self._log("Ignoring invalid global best.")
                            best_record = None  # Discard it for this iteration

                if best_record:
                    global_best_config = best_record.get("config", {})
                    global_best_score = best_record.get("score", -float("inf"))
                    global_model_name = global_best_config.get(
                        "model_name", global_model_name
                    )
                    global_gen = global_best_config.get("generation", 0)
                    parent_hash = get_config_hash(global_best_config)

                    self._log(
                        f"Found global best: {global_best_score:.4f} (Gen {global_gen})"
                    )
                else:
                    self._log("No global best found. Will seed new...")

                # 2. Decide Strategy (Manual, New Arch, Crossover, or Mutate)
                action = "mutate"
                rnd = random.random()

                # Check manual queue first
                if self.manual_queue:
                    action = "manual"
                # Chance to switch architecture entirely (exploration)
                elif rnd < 0.05:
                    action = "new_arch"
                # Chance to crossover if we have a local best compatible with global
                elif (
                    self.local_best_config
                    and best_record
                    and self.local_best_config.get("model_name") == global_model_name
                    and rnd < 0.35
                ):
                    action = "crossover"
                else:
                    action = "mutate"

                # 3. Prepare Genome
                target_config = {}
                target_model_name = global_model_name
                next_gen = global_gen

                if action == "manual":
                    self._update_status("Evaluating Manual Design...")
                    target_config = self.manual_queue.pop(0)
                    target_model_name = target_config.get(
                        "model_name", global_model_name
                    )
                    # Treat as a new branch or continuation depending on
                    # if parent_id is set manually
                    # If not set, we can assume it's a new line or a fork of global
                    if "generation" not in target_config:
                        target_config["generation"] = global_gen + 1
                    next_gen = target_config["generation"]
                    parent_hash = target_config.get("parent_id")  # Might be None

                elif action == "new_arch":
                    self._update_status("Exploring New Architecture...")
                    # Pick random model from registry spaces
                    available_models = list(SEARCH_SPACES.keys())
                    target_model_name = random.choice(available_models)
                    space = get_search_space(target_model_name)
                    target_config = space.sample()
                    target_config["model_name"] = target_model_name
                    next_gen = 0  # Reset generation for new species
                    parent_hash = None
                    self._log(f"Selected new architecture: {target_model_name}")

                elif action == "crossover":
                    self._update_status("Crossing Over Genomes...")
                    space = get_search_space(global_model_name)
                    target_config = space.crossover(
                        global_best_config, self.local_best_config
                    )
                    target_config["model_name"] = global_model_name  # Persist name
                    # Add small mutation to avoid stagnation
                    target_config = space.mutate(target_config, mutation_rate=0.1)
                    target_model_name = global_model_name
                    # Take max generation of parents + 1
                    local_gen = self.local_best_config.get("generation", 0)
                    next_gen = max(global_gen, local_gen) + 1

                else:  # Mutate
                    self._update_status("Mutating Genome...")
                    # Decide which parent to mutate
                    # Favor global best, but sometimes use local best or random restart
                    parent_config = global_best_config
                    parent_model = global_model_name
                    parent_gen = global_gen

                    if not best_record:  # Bootstrap
                        space = get_search_space(global_model_name)
                        parent_config = space.sample()
                        parent_config["model_name"] = global_model_name
                        parent_gen = 0
                        parent_hash = None
                    elif self.local_best_config and random.random() < 0.3:
                        parent_config = self.local_best_config
                        parent_model = parent_config.get("model_name", "EqProp MLP")
                        parent_gen = parent_config.get("generation", 0)
                        parent_hash = get_config_hash(parent_config)

                    space = get_search_space(parent_model)
                    target_config = space.mutate(parent_config)
                    target_config["model_name"] = parent_model
                    target_model_name = parent_model
                    next_gen = parent_gen + 1

                # Update Lineage Info
                target_config["generation"] = next_gen
                if parent_hash:
                    target_config["parent_id"] = parent_hash

                # Re-fetch space with constraints applied for final
                # verification/mutation context
                space = get_search_space(target_model_name)
                if self.constraints:
                    space = space.apply_constraints(self.constraints)
                    # Mutate again with constrained space to ensure we are in bounds
                    # Rate 0 just clamps if implemented or we can just assume
                    # mutate clamps
                    target_config = space.mutate(target_config, mutation_rate=0.0)

                    # Manually clamp common keys if space.mutate doesn't enforce
                    # stricter bounds on existing values
                    if (
                        "max_hidden" in self.constraints
                        and "hidden_dim" in target_config
                    ):
                        target_config["hidden_dim"] = min(
                            target_config["hidden_dim"], self.constraints["max_hidden"]
                        )
                    if (
                        "max_layers" in self.constraints
                        and "num_layers" in target_config
                    ):
                        target_config["num_layers"] = min(
                            target_config["num_layers"], self.constraints["max_layers"]
                        )
                    if "max_steps" in self.constraints and "steps" in target_config:
                        target_config["steps"] = min(
                            target_config["steps"], self.constraints["max_steps"]
                        )

                # Apply Mode Settings (Quick vs Deep)
                if self.discovery_mode == "quick":
                    target_config["epochs"] = 1
                    if "steps" in target_config:
                        target_config["steps"] = min(target_config["steps"], 15)
                elif self.discovery_mode == "deep":
                    target_config["epochs"] = 5
                    # Allow larger steps

                # 4. Evaluate
                self._update_status(f"Evaluating: {target_model_name} (Gen {next_gen})")

                # Use Worker's logic to run job locally
                job_id = random.randint(1000, 9999)

                metrics = run_single_trial_task(
                    task=self.task,
                    model_name=target_model_name,
                    config=target_config,
                    storage_path="results/hyperopt.db",
                    job_id=job_id,
                    quick_mode=(self.discovery_mode == "quick"),
                )

                if metrics:
                    acc = metrics.get("accuracy", 0.0)
                    self.jobs_done += 1
                    self.points += 5
                    save_state(self.points, self.jobs_done)

                    self._log(
                        f"Eval complete: {acc:.4f}"
                        f" (Global Best: {global_best_score:.4f})"
                    )

                    # Update Local Best
                    if acc > self.local_best_score:
                        self.local_best_score = acc
                        self.local_best_config = target_config
                        self._log(f"New Local Best! ({acc:.4f})")

                    # Publish if Global Best
                    if acc > global_best_score:
                        self._update_status("Publishing Discovery...")
                        # Ensure model_name inside config
                        target_config["model_name"] = target_model_name
                        self.dht.publish_best_model(self.task, target_config, acc)
                        self.points += 50
                        save_state(self.points, self.jobs_done)
                        self._log(f"🎉 New Global Best Discovered! ({acc:.4f})")

                else:
                    self._log("Evaluation failed.")

            except Exception as e:
                self._log(f"Evolution Loop Error: {e}")
                import traceback

                traceback.print_exc()
                time.sleep(5)

            # Rest
            if self.running:
                self._update_status("Resting...")
                time.sleep(2)
