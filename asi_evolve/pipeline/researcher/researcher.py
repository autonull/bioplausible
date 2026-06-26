"""Researcher agent implementation."""

from typing import Any, Dict, List, Optional

from ..base import BaseAgent
from ...utils.structures import Node, CognitionItem
from ...utils.diff import extract_diffs, apply_diff, format_diff_summary


class Researcher(BaseAgent):
    """
    Generate the next candidate program from prior nodes and retrieved knowledge.

    Two modes are supported:
    1. `diff_based_evolution`: edit a sampled parent program incrementally.
    2. `full_rewrite`: generate a full candidate from scratch.
    """

    def __init__(self, llm, prompt_manager, config: Optional[Dict] = None):
        super().__init__(llm, prompt_manager, name="researcher")

        self.config = config or {}
        self.diff_based = self.config.get("diff_based_evolution", True)
        self.diff_pattern = self.config.get(
            "diff_pattern",
            r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        )
        self.max_code_length = self.config.get("max_code_length", 10000)

    def run(
        self,
        task_description: str,
        context_nodes: List[Node],
        cognition_items: List[CognitionItem],
        base_code: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        self.logger.info(
            f"[Researcher] Starting with {len(context_nodes)} context nodes, "
            f"mode={'diff' if self.diff_based else 'full_rewrite'}"
        )

        if self.diff_based and not base_code and context_nodes:
            base_code = context_nodes[0].code
            self.logger.info(f"[Researcher] Using first context node as base: {context_nodes[0].name}")

        if self.diff_based and base_code:
            result = self._generate_diff(
                task_description, context_nodes, cognition_items, base_code
            )
        else:
            prompt = self.get_prompt(
                "researcher",
                task_description=task_description,
                context_nodes=[n.to_dict() for n in context_nodes],
                cognition_items=[c.to_dict() for c in cognition_items],
                base_code=None,
                diff_based=False,
            )
            result = self._generate_full(prompt)

        if len(result["code"]) > self.max_code_length:
            self.logger.warning(
                f"[Researcher] Generated code exceeds max length "
                f"({len(result['code'])} > {self.max_code_length})"
            )
            result["code"] = result["code"][:self.max_code_length]

        self.logger.info(f"[Researcher] Generated: {result.get('name', 'unnamed')}")

        return result

    def _generate_diff(
        self,
        task_description: str,
        context_nodes: List[Node],
        cognition_items: List[CognitionItem],
        base_code: str,
    ) -> Dict[str, Any]:
        """
        Generate code in diff mode.

        Returns:
            Dictionary containing `name`, `motivation`, `code`, and `changes`.
        """
        prompt = self.get_prompt(
            "researcher",
            task_description=task_description,
            context_nodes=[n.to_dict() for n in context_nodes],
            cognition_items=[c.to_dict() for c in cognition_items],
            base_code=base_code,
            diff_based=True,
        )

        response = self.llm.generate(prompt, call_name="researcher_diff")
        response_text = response.content if hasattr(response, "content") else str(response)

        pattern = self.diff_pattern.replace("\\n", "\n") if isinstance(self.diff_pattern, str) else self.diff_pattern
        diff_blocks = extract_diffs(response_text, pattern)

        if not diff_blocks:
            self.logger.error(f"[Researcher] No diff blocks found. Full response ({len(response_text)} chars):\n{response_text}")
            full_prompt = self.get_prompt(
                "researcher",
                task_description=task_description,
                context_nodes=[n.to_dict() for n in context_nodes],
                cognition_items=[c.to_dict() for c in cognition_items],
                base_code=None,
                diff_based=False,
            )
            return self._generate_full(full_prompt)

        try:
            new_code = apply_diff(base_code, response_text, pattern)
            changes_summary = format_diff_summary(diff_blocks)

            import re

            name = "diff_modification"
            motivation = changes_summary

            name_match = re.search(r"<name>(.*?)</name>", response_text, re.DOTALL)
            if name_match:
                name = name_match.group(1).strip()

            motivation_match = re.search(r"<motivation>(.*?)</motivation>", response_text, re.DOTALL)
            if motivation_match:
                motivation = motivation_match.group(1).strip()

            return {
                "name": name,
                "motivation": motivation,
                "code": new_code,
                "changes": changes_summary,
            }

        except ValueError as e:
            self.logger.error(f"[Researcher] Failed to apply diff: {e}")
            full_prompt = self.get_prompt(
                "researcher",
                task_description=task_description,
                context_nodes=[n.to_dict() for n in context_nodes],
                cognition_items=[c.to_dict() for c in cognition_items],
                base_code=None,
                diff_based=False,
            )
            return self._generate_full(full_prompt)

    def _generate_full(self, prompt: str) -> Dict[str, Any]:
        """
        Generate code through a full rewrite.
        """
        try:
            result = self.llm.extract_tags(prompt, call_name="researcher_full")
        except ValueError:
            response = self.llm.generate(prompt, call_name="researcher_full_debug")
            response_text = response.content if hasattr(response, "content") else str(response)
            self.logger.error(f"[Researcher] Full rewrite tag extraction failed. Full response ({len(response_text)} chars):\n{response_text}")
            raise

        return {
            "name": result.get("name", ""),
            "motivation": result.get("motivation", ""),
            "code": result.get("code", ""),
        }
