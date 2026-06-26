"""Shared base class for pipeline agents."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from ..utils.llm import LLMClient
from ..utils.prompt import PromptManager
from ..utils.logger import get_logger


class BaseAgent(ABC):
    """Base implementation shared by all agent roles."""

    def __init__(
        self,
        llm: LLMClient,
        prompt_manager: PromptManager,
        name: str = "agent",
    ):
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.name = name
        self.logger = get_logger()
        self.step_dir = None

    def set_step_dir(self, step_dir):
        """
        Set the current step directory for agent-local logs.

        Args:
            step_dir: Path to the active step directory.
        """
        from pathlib import Path
        self.step_dir = Path(step_dir) if step_dir else None
        if self.step_dir:
            log_dir = self.step_dir / "llm_logs"
            self.llm.set_log_dir(log_dir)

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent.

        Returns:
            Agent-specific output.
        """
        pass

    def get_prompt(self, template_name: str, **context) -> str:
        """Render a prompt template from the configured prompt manager."""
        if self.prompt_manager.has_template(template_name):
            return self.prompt_manager.render(template_name, **context)
        raise ValueError(f"No prompt template found for: {template_name}")
