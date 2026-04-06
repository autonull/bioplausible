"""Logging utilities for ASI-Evolve."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class EvolveLogger:
    """
    Logger wrapper with console, file, and optional Weights & Biases support.
    """

    def __init__(
        self,
        name: str = "evolve",
        log_dir: Optional[Path] = None,
        level: str = "INFO",
        console: bool = True,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.log_dir = Path(log_dir) if log_dir else None
        self.level = getattr(logging, level.upper(), logging.INFO)
        self.console = console
        self.wandb_config = wandb_config

        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self.log_dir / "evolve.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        self.wandb_run = None
        if wandb_config and wandb_config.get("enabled") and WANDB_AVAILABLE:
            self._init_wandb(wandb_config)

        self.stats = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_calls": 0,
            "total_time": 0.0,
        }

    def _init_wandb(self, config: Dict[str, Any]):
        """Initialize Weights & Biases logging."""
        import os

        if config.get("offline", False):
            os.environ["WANDB_MODE"] = "offline"

        try:
            self.wandb_run = wandb.init(
                project=config.get("project", "evolve"),
                entity=config.get("entity"),
                name=config.get("run_name"),
                config=config.get("config", {}),
                dir=str(self.log_dir) if self.log_dir else None,
                resume="allow",
            )
            self.logger.info(f"WandB initialized: {self.wandb_run.name} (mode: {wandb.run.settings.mode})")
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")

    def info(self, msg: str):
        self.logger.info(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def log_llm_call(self, call_info: Dict[str, Any]):
        """Record aggregate statistics for one LLM call."""
        usage = call_info.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        self.stats["total_calls"] += 1
        self.stats["prompt_tokens"] += prompt_tokens
        self.stats["completion_tokens"] += completion_tokens
        self.stats["total_tokens"] += prompt_tokens + completion_tokens
        self.stats["total_time"] += call_info.get("call_time", 0)

        self.debug(
            f"LLM Call: model={call_info.get('model')}, "
            f"tokens={prompt_tokens}+{completion_tokens}, "
            f"time={call_info.get('call_time', 0):.2f}s"
        )

    def log_experiment_step(self, step: int, metrics: Dict[str, Any]):
        """Record one logical experiment step."""
        self.info(f"Step {step}: {metrics}")

        if self.wandb_run:
            wandb.log({"pipeline/step": step, **metrics})

    def log_node(self, node: "Node", step: int, database: Optional[Any] = None):
        """
        Record a newly added node and its numeric metrics.
        """
        self.info(f"New node: {node.name} (score={node.score:.4f})")

        if self.wandb_run:
            log_data = {
                "pipeline/step": step,
                "node/score": node.score,
                "node/code_length": len(node.code) if node.code else 0,
                "llm/total_calls": self.stats["total_calls"],
                "llm/total_tokens": self.stats["total_tokens"],
                "llm/prompt_tokens": self.stats["prompt_tokens"],
                "llm/completion_tokens": self.stats["completion_tokens"],
            }

            if database is not None:
                all_nodes = database.get_all()
                if all_nodes:
                    max_score = max(n.score for n in all_nodes)
                    log_data["best/max_score"] = max_score
                else:
                    log_data["best/max_score"] = node.score

            if node.results:
                self._extract_metrics(node.results, "results", log_data)

            if node.meta_info:
                self._extract_metrics(node.meta_info, "meta", log_data)

            wandb.log(log_data)

    def _extract_metrics(
        self,
        data: Any,
        prefix: str,
        output: Dict[str, Any],
        max_depth: int = 3,
        _depth: int = 0,
    ):
        """
        Recursively extract numeric metrics from nested structures.
        """
        if _depth >= max_depth:
            return

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 200:
                    continue

                new_prefix = f"{prefix}/{key}"
                self._extract_metrics(value, new_prefix, output, max_depth, _depth + 1)

        elif isinstance(data, (int, float)) and not isinstance(data, bool):
            output[prefix] = data

        elif isinstance(data, bool):
            output[prefix] = 1 if data else 0

        elif isinstance(data, list) and len(data) > 0 and all(isinstance(x, (int, float)) for x in data):
            output[f"{prefix}/mean"] = sum(data) / len(data)
            output[f"{prefix}/max"] = max(data)
            output[f"{prefix}/min"] = min(data)

    def get_stats(self) -> Dict[str, Any]:
        """Return accumulated logging statistics."""
        return self.stats.copy()

    def finish(self):
        """Flush and close any logging backends."""
        self.info(f"Total stats: {self.stats}")
        if self.wandb_run:
            wandb.finish()


_logger: Optional[EvolveLogger] = None


def get_logger() -> EvolveLogger:
    """Return the process-wide logger instance."""
    global _logger
    if _logger is None:
        _logger = EvolveLogger()
    return _logger


def init_logger(
    name: str = "evolve",
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    console: bool = True,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> EvolveLogger:
    """Initialize the process-wide logger instance."""
    global _logger
    _logger = EvolveLogger(
        name=name,
        log_dir=log_dir,
        level=level,
        console=console,
        wandb_config=wandb_config,
    )
    return _logger
