"""Analyzer agent implementation."""

import json
from typing import Any, Dict

from ..base import BaseAgent


class Analyzer(BaseAgent):
    """
    Turn structured experiment outcomes into reusable natural-language lessons.
    """

    def __init__(self, llm, prompt_manager):
        super().__init__(llm, prompt_manager, name="analyzer")

    def run(
        self,
        code: str,
        results: Dict[str, Any],
        task_description: str,
        best_sampled_node=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Analyze one experiment result and return an `analysis` string.
        """
        self.logger.info("[Analyzer] Starting analysis")

        results_str = json.dumps(results, indent=2, ensure_ascii=False)

        best_node_info = None
        if best_sampled_node:
            best_results_str = json.dumps(best_sampled_node.results, indent=2, ensure_ascii=False)
            best_node_info = {
                "name": best_sampled_node.name,
                "score": best_sampled_node.score,
                "motivation": best_sampled_node.motivation,
                "code": best_sampled_node.code,
                "results": best_results_str,
                "analysis": best_sampled_node.analysis,
            }

        prompt = self.get_prompt(
            "analyzer",
            code=code,
            results=results_str,
            task_description=task_description,
            best_sampled_node=best_node_info,
        )

        result = self.llm.extract_tags(prompt, call_name="analyzer")

        analysis = result.get("analysis", "")

        self.logger.info("[Analyzer] Analysis completed")

        return {
            "analysis": analysis,
        }
