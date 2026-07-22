"""Core pipeline orchestration for ASI-Evolve.

The pipeline wires together the manager, researcher, engineer, and analyzer
agents, then executes the evolutionary loop in sequential or parallel mode.
"""

import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from ..cognition import Cognition
from ..database import Database
from ..utils import BestSnapshotManager
from ..utils.config import load_config
from ..utils.llm import create_llm_client
from ..utils.logger import init_logger
from ..utils.prompt import PromptManager
from ..utils.structures import Node
from .analyzer import Analyzer
from .engineer import Engineer
from .manager import Manager
from .researcher import Researcher


class Pipeline:
    """Coordinate a resumable ASI-Evolve experiment.

    The pipeline is responsible for loading configuration, initializing shared
    services, running each evolution step, and keeping enough state on disk to
    resume an interrupted experiment.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        if experiment_name is None:
            from ..utils.config import load_config as _load_config

            temp_config = _load_config(config_path=config_path)
            experiment_name = temp_config.get("experiment_name", "default")

        self.experiment_name = experiment_name

        self.config = load_config(
            config_path=config_path, experiment_name=experiment_name
        )
        self.config["experiment_name"] = experiment_name

        base_dir = Path(__file__).parent.parent / "experiments"
        self.experiment_dir = base_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.steps_dir = self.experiment_dir / "steps"
        self.steps_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.experiment_dir / "pipeline_state.json"

        log_config = self.config.get("logging", {})
        wandb_config = log_config.get("wandb", {})
        if wandb_config:
            wandb_config = wandb_config.copy()
            wandb_config["run_name"] = self.experiment_name
            wandb_config["config"] = self.config

        self.logger = init_logger(
            name="evolve",
            log_dir=self.experiment_dir / "logs",
            level=log_config.get("level", "INFO"),
            console=log_config.get("console", True),
            wandb_config=wandb_config,
        )

        self.llm = create_llm_client(self.config)

        prompt_dir = self.experiment_dir / "prompts"
        self.prompt_manager = PromptManager(prompt_dir)

        db_config = self.config.get("database", {})
        sampling_config = db_config.get("sampling", {})
        algorithm = sampling_config.get("algorithm", "ucb1")

        sampling_kwargs = {}
        if algorithm == "ucb1":
            sampling_kwargs["c"] = sampling_config.get("ucb1_c", 1.414)
        elif algorithm.startswith("island"):
            island_config = sampling_config.get(
                algorithm, sampling_config.get("island", {})
            )
            sampling_kwargs = {
                "num_islands": island_config.get("num_islands", 5),
                "migration_interval": island_config.get("migration_interval", 10),
                "migration_rate": island_config.get("migration_rate", 0.1),
                "exploration_ratio": island_config.get("exploration_ratio", 0.2),
                "exploitation_ratio": island_config.get("exploitation_ratio", 0.3),
                "feature_dimensions": island_config.get("feature_dimensions", []),
                "feature_bins": island_config.get("feature_bins", 10),
            }

        self.database = Database(
            storage_dir=self.experiment_dir
            / db_config.get("storage_dir", "database_data"),
            embedding_model=db_config.get("embedding", {}).get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dim=db_config.get("embedding", {}).get("dimension", 384),
            sampling_algorithm=algorithm,
            sampling_kwargs=sampling_kwargs,
            max_size=db_config.get("max_size"),
        )

        cog_config = self.config.get("cognition", {})
        self.cognition = Cognition(
            storage_dir=self.experiment_dir
            / cog_config.get("storage_dir", "cognition_data"),
            embedding_model=cog_config.get("embedding", {}).get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embedding_dim=cog_config.get("embedding", {}).get("dimension", 384),
            retrieval_top_k=cog_config.get("retrieval", {}).get("top_k", 5),
            score_threshold=cog_config.get("retrieval", {}).get("score_threshold", 0.5),
        )

        pipeline_config = self.config.get("pipeline", {})
        agents_config = pipeline_config.get("agents", {})

        self.use_manager = agents_config.get("manager", False)
        self.use_researcher = agents_config.get("researcher", True)
        self.use_engineer = agents_config.get("engineer", True)
        self.use_analyzer = agents_config.get("analyzer", True)

        self.researcher_config = pipeline_config.get("researcher", {})

        self.manager = (
            Manager(self.llm, self.prompt_manager) if self.use_manager else None
        )
        self.researcher = (
            Researcher(self.llm, self.prompt_manager, self.researcher_config)
            if self.use_researcher
            else None
        )
        self.engineer = (
            Engineer(self.llm, self.prompt_manager) if self.use_engineer else None
        )
        self.analyzer = (
            Analyzer(self.llm, self.prompt_manager) if self.use_analyzer else None
        )

        self.max_retries = pipeline_config.get("max_retries", {})

        judge_config = pipeline_config.get("judge", {})
        self.judge_enabled = judge_config.get("enabled", False)
        self.judge_ratio = judge_config.get("ratio", 0.2)

        parallel_config = pipeline_config.get("parallel", {})
        self.num_workers = parallel_config.get("num_workers", 1)
        self.step_lock = Lock()

        self.engineer_timeout = pipeline_config.get("engineer_timeout", 3600)

        self.sample_n = pipeline_config.get("sample_n", 3)

        self.step = 0
        self.manager_initialized = False
        self._load_state()

        self.is_resume = self.step > 0 or len(self.database) > 0
        if self.is_resume:
            self.logger.info(
                f"Resuming experiment '{self.experiment_name}' from step {self.step} "
                f"(database: {len(self.database)} nodes, cognition: {len(self.cognition)} items)"
            )
        else:
            self.logger.info(f"Starting new experiment: {self.experiment_name}")

        self.initial_node_created = False

        self.best_snapshot = BestSnapshotManager(self.steps_dir, logger=self.logger)
        self.best_snapshot.init_from_nodes(self.database.get_all())

    def _load_state(self):
        """Restore pipeline progress from disk or infer it from existing data."""
        import json

        if not self.state_file.exists():
            if len(self.database) > 0:
                max_id = max(n.id for n in self.database.get_all() if n.id is not None)
                self.step = max_id + 1
                prompt_dir = self.experiment_dir / "prompts"
                if prompt_dir.exists() and any(prompt_dir.glob("*.jinja2")):
                    self.manager_initialized = True
            return

        with open(self.state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

        self.step = state.get("step", 0)
        self.manager_initialized = state.get("manager_initialized", False)

    def _save_state(self):
        """Persist the current pipeline progress to the experiment directory."""
        state = {
            "step": self.step,
            "manager_initialized": self.manager_initialized,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def run_step(
        self,
        task_description: Optional[str] = None,
        eval_script: Optional[str] = None,
        sample_n: Optional[int] = None,
    ) -> Optional[Node]:
        """Run one evolutionary step and return the resulting node, if any."""
        with self.step_lock:
            self.step += 1
            current_step = self.step
            self._save_state()

        self.logger.info(f"=== Step {current_step} ===")

        if sample_n is None:
            sample_n = self.sample_n

        try:
            if task_description is None:
                input_file = self.experiment_dir / "input.md"
                if input_file.exists():
                    task_description = input_file.read_text(encoding="utf-8")
                else:
                    self.logger.error("No task description provided")
                    return None

            if self.use_manager and not self.manager_initialized:
                self._run_manager(task_description)
                self.manager_initialized = True
                self._save_state()
                self.prompt_manager = PromptManager(self.experiment_dir / "prompts")
                if self.researcher:
                    self.researcher.prompt_manager = self.prompt_manager
                if self.analyzer:
                    self.analyzer.prompt_manager = self.prompt_manager

            context_nodes = self.database.sample(sample_n)
            parent_ids = [n.id for n in context_nodes if n.id is not None]
            self.logger.info(f"Sampled {len(context_nodes)} context nodes")

            cognition_items = []
            if context_nodes:
                for node in context_nodes:
                    if node.analysis:
                        items = self.cognition.search(node.analysis, top_k=2)
                        cognition_items.extend(items)
                    else:
                        items = self.cognition.search(node.motivation, top_k=2)
                        cognition_items.extend(items)
            self.logger.info(f"Retrieved {len(cognition_items)} cognition items")

            if not self.researcher:
                self.logger.error("Researcher not enabled")
                return None

            step_dir = self.steps_dir / f"step_{current_step}"
            step_dir.mkdir(parents=True, exist_ok=True)

            if self.researcher:
                self.researcher.set_step_dir(step_dir)
            if self.analyzer:
                self.analyzer.set_step_dir(step_dir)
            if self.engineer:
                self.engineer.set_step_dir(step_dir)

            base_code = None
            if (
                self.researcher_config.get("diff_based_evolution", True)
                and context_nodes
            ):
                base_code = context_nodes[0].code
                self.logger.info(f"Using base code from: {context_nodes[0].name}")

            try:
                researcher_result = self.researcher.run(
                    task_description=task_description,
                    context_nodes=context_nodes,
                    cognition_items=cognition_items,
                    base_code=base_code,
                )
            except Exception as e:
                self.logger.error(f"Researcher failed: {type(e).__name__}: {e}")
                self.logger.error(traceback.format_exc())
                return None

            node = Node(
                name=researcher_result.get("name", f"node_{current_step}"),
                created_at=datetime.now().isoformat(),
                parent=parent_ids,
                motivation=researcher_result.get("motivation", ""),
                code=researcher_result.get("code", ""),
            )

            engineer_result = {}

            if self.engineer and (eval_script or self.judge_enabled):
                try:
                    engineer_result = self.engineer.run(
                        code=node.code,
                        experiment_dir=step_dir,
                        eval_script=eval_script,
                        timeout=self.engineer_timeout,
                        task_description=task_description,
                        judge_enabled=self.judge_enabled,
                        judge_ratio=self.judge_ratio,
                    )

                    node.results = {
                        k: v for k, v in engineer_result.items() if k != "temp"
                    }

                    node.score = engineer_result.get("score", 0.0)
                    node.meta_info["runtime"] = engineer_result.get("runtime")
                    node.meta_info["success"] = engineer_result.get("success")
                    node.meta_info["eval_score"] = engineer_result.get(
                        "eval_score", 0.0
                    )
                    if self.judge_enabled:
                        node.meta_info["judge_score"] = engineer_result.get(
                            "judge_score"
                        )

                    if not engineer_result.get("success"):
                        node.meta_info["error"] = engineer_result.get("error")

                except Exception as e:
                    self.logger.error(f"Engineer failed: {type(e).__name__}: {e}")
                    self.logger.error(traceback.format_exc())
                    node.meta_info["success"] = False
                    node.meta_info["error"] = str(e)
                    node.score = 0.0
                    engineer_result = {}

            if self.analyzer:
                try:
                    best_sampled_node = None
                    if context_nodes:
                        best_sampled_node = max(context_nodes, key=lambda n: n.score)
                        self.logger.info(
                            f"Best sampled node for comparison: {best_sampled_node.name} (score={best_sampled_node.score:.4f})"
                        )

                    analyzer_result = self.analyzer.run(
                        code=node.code,
                        results=engineer_result,
                        task_description=task_description,
                        best_sampled_node=best_sampled_node,
                    )
                    node.analysis = analyzer_result.get("analysis", "")
                except Exception as e:
                    self.logger.error(f"Analyzer failed: {type(e).__name__}: {e}")
                    self.logger.error(traceback.format_exc())
                    node.analysis = f"Analysis failed: {e}"
            else:
                if "temp" in engineer_result:
                    node.results["temp"] = engineer_result["temp"]

            node_id = self.database.add(node)
            self.logger.info(
                f"Added node {node_id}: {node.name} (score={node.score:.4f})"
            )

            self.logger.log_node(node, current_step, database=self.database)

            self.best_snapshot.update_if_better(
                node,
                step_name=f"step_{current_step}",
                source_step_dir=step_dir,
            )

            return node

        except Exception as e:
            self.logger.error(f"Step {current_step} failed with unexpected error:")
            self.logger.error(f"{type(e).__name__}: {e}")
            self.logger.error(traceback.format_exc())
            return None

    def _run_manager(self, task_description: str):
        self.logger.info("[Manager] Generating prompts...")

        eval_file = self.experiment_dir / "eval_criteria.md"
        eval_criteria = ""
        if eval_file.exists():
            eval_criteria = eval_file.read_text(encoding="utf-8")

        self.manager.run(
            task_description=task_description,
            eval_criteria=eval_criteria,
            prompt_dir=self.experiment_dir / "prompts",
        )

    def run(
        self,
        max_steps: int = 10,
        task_description: Optional[str] = None,
        eval_script: Optional[str] = None,
        sample_n: Optional[int] = None,
    ):
        """Run the pipeline for ``max_steps`` in sequential or parallel mode."""
        if sample_n is None:
            sample_n = self.sample_n

        if not self.is_resume and not self.initial_node_created:
            self._create_initial_node(task_description, eval_script)

        if self.num_workers == 1:
            self._run_sequential(max_steps, task_description, eval_script, sample_n)
        else:
            self._run_parallel(max_steps, task_description, eval_script, sample_n)

    def _create_initial_node(
        self,
        task_description: Optional[str],
        eval_script: Optional[str],
    ) -> None:
        """Evaluate and register an ``initial_program`` seed before evolution."""
        initial_program_file = self.experiment_dir / "initial_program"
        if not initial_program_file.exists():
            return

        self.logger.info(
            "Found initial_program, creating initial node before evolution steps"
        )

        if task_description is None:
            input_file = self.experiment_dir / "input.md"
            if input_file.exists():
                task_description = input_file.read_text(encoding="utf-8")
            else:
                task_description = ""

        initial_code = initial_program_file.read_text(encoding="utf-8")

        step_dir = self.steps_dir / "step_0_initial"
        step_dir.mkdir(parents=True, exist_ok=True)

        if self.researcher:
            self.researcher.set_step_dir(step_dir)
        if self.analyzer:
            self.analyzer.set_step_dir(step_dir)
        if self.engineer:
            self.engineer.set_step_dir(step_dir)

        node = Node(
            name="initial_program",
            created_at=datetime.now().isoformat(),
            parent=[],
            motivation="Initial program provided by user",
            code=initial_code,
        )

        engineer_result: Dict[str, Any] = {}

        if self.engineer and (eval_script or self.judge_enabled):
            try:
                engineer_result = self.engineer.run(
                    code=node.code,
                    experiment_dir=step_dir,
                    eval_script=eval_script,
                    timeout=self.engineer_timeout,
                    task_description=task_description or "",
                    judge_enabled=self.judge_enabled,
                    judge_ratio=self.judge_ratio,
                )

                node.results = {k: v for k, v in engineer_result.items() if k != "temp"}

                node.score = engineer_result.get("score", 0.0)
                node.meta_info["runtime"] = engineer_result.get("runtime")
                node.meta_info["success"] = engineer_result.get("success")
                node.meta_info["eval_score"] = engineer_result.get("eval_score", 0.0)
                if self.judge_enabled:
                    node.meta_info["judge_score"] = engineer_result.get("judge_score")

                if not engineer_result.get("success"):
                    node.meta_info["error"] = engineer_result.get("error")

            except Exception as e:
                self.logger.error(f"Initial Engineer failed: {type(e).__name__}: {e}")
                self.logger.error(traceback.format_exc())
                node.meta_info["success"] = False
                node.meta_info["error"] = str(e)
                node.score = 0.0
                engineer_result = {}

        if self.analyzer:
            try:
                analyzer_result = self.analyzer.run(
                    code=node.code,
                    results=engineer_result,
                    task_description=task_description or "",
                )
                node.analysis = analyzer_result.get("analysis", "")
            except Exception as e:
                self.logger.error(f"Initial Analyzer failed: {type(e).__name__}: {e}")
                self.logger.error(traceback.format_exc())
                node.analysis = f"Analysis failed: {e}"

        node_id = self.database.add(node)
        self.logger.info(
            f"Added initial node {node_id}: {node.name} (score={node.score:.4f})"
        )

        self.logger.log_node(node, 0, database=self.database)

        self.best_snapshot.update_if_better(
            node,
            step_name="step_0_initial",
            source_step_dir=step_dir,
        )

        self.initial_node_created = True

    def _run_sequential(
        self,
        max_steps: int,
        task_description: Optional[str],
        eval_script: Optional[str],
        sample_n: int,
    ):
        """Execute evolution steps one after another in the current process."""
        self.logger.info(f"Starting sequential pipeline for {max_steps} steps")

        for _ in range(max_steps):
            node = self.run_step(
                task_description=task_description,
                eval_script=eval_script,
                sample_n=sample_n,
            )

            if node is None:
                self.logger.warning("Step failed, continuing to next step...")

        self.logger.info("Pipeline completed")
        self.logger.finish()

    def _run_parallel(
        self,
        max_steps: int,
        task_description: Optional[str],
        eval_script: Optional[str],
        sample_n: int,
    ):
        """Execute evolution steps across the configured worker pool."""
        self.logger.info(
            f"Starting parallel pipeline with {self.num_workers} workers for {max_steps} steps"
        )

        completed_steps = 0
        failed_steps = 0

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for _ in range(max_steps):
                future = executor.submit(
                    self.run_step,
                    task_description=task_description,
                    eval_script=eval_script,
                    sample_n=sample_n,
                )
                futures.append(future)

            for future in as_completed(futures):
                try:
                    node = future.result()
                    if node is not None:
                        completed_steps += 1
                    else:
                        failed_steps += 1
                        self.logger.warning(
                            "Step failed, worker will continue with next task..."
                        )
                except Exception as e:
                    failed_steps += 1
                    self.logger.error(
                        f"Worker encountered unexpected error: {type(e).__name__}: {e}"
                    )
                    self.logger.error(traceback.format_exc())

        self.logger.info(
            f"Parallel pipeline completed: {completed_steps} successful, {failed_steps} failed"
        )
        self.logger.finish()

    def get_best_node(self) -> Optional[Node]:
        nodes = self.database.get_all()
        if not nodes:
            return None
        return max(nodes, key=lambda n: n.score)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "total_steps": self.step,
            "total_nodes": len(self.database),
            "total_cognition": len(self.cognition),
            "manager_initialized": self.manager_initialized,
            "llm_stats": self.logger.get_stats(),
        }
