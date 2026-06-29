"""Prompt template management."""

from pathlib import Path
from typing import Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template


class PromptManager:
    """
    Load and render prompt templates from experiment-specific and default locations.
    """

    def __init__(self, prompt_dir: Optional[Path] = None):
        """
        Args:
            prompt_dir: Usually `experiments/<name>/prompts/`.
        """
        self.prompt_dir = Path(prompt_dir) if prompt_dir else None
        self.env = None
        self.templates: Dict[str, Template] = {}

        self.default_prompt_dir = Path(__file__).parent / "prompts"

        if self.prompt_dir and self.prompt_dir.exists():
            self.env = Environment(
                loader=FileSystemLoader(str(self.prompt_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )

    def get_template(self, name: str) -> Optional[Template]:
        """
        Return a template by name.

        Priority:
        1. Experiment-level templates
        2. Default templates in `utils/prompts/`
        """
        if name in self.templates:
            return self.templates[name]

        if self.env:
            try:
                template = self.env.get_template(f"{name}.jinja2")
                self.templates[name] = template
                return template
            except Exception:
                pass

        default_template_path = self.default_prompt_dir / f"{name}.jinja2"
        if default_template_path.exists():
            try:
                template_content = default_template_path.read_text(encoding="utf-8")
                template = Template(template_content)
                self.templates[name] = template
                return template
            except Exception:
                pass

        return None

    def get_default_template(self, name: str) -> Optional[Template]:
        """
        Return a template only from the default prompt directory.

        This is used by diff mode so the framework can enforce a stable
        SEARCH/REPLACE format even when the experiment provides custom content.
        """
        default_template_path = self.default_prompt_dir / f"{name}.jinja2"
        if not default_template_path.exists():
            return None
        try:
            template_content = default_template_path.read_text(encoding="utf-8")
            return Template(template_content)
        except Exception:
            return None

    def render(self, name: str, **context) -> str:
        """
        Render a template by name.
        """
        if name == "researcher" and context.get("diff_based", False):
            user_prompt = self._render_user_template("researcher_diff", context)
            if user_prompt:
                context["user_prompt"] = user_prompt

            template = self.get_default_template("researcher_diff")
            if template:
                return template.render(**context)

            raise ValueError("Default template 'researcher_diff' not found")

        template = self.get_template(name)
        if template:
            return template.render(**context)

        raise ValueError(f"Template '{name}' not found")

    def _render_user_template(self, name: str, context: Dict) -> Optional[str]:
        """
        Render a user-provided template if it exists.
        """
        if not self.prompt_dir or not self.prompt_dir.exists():
            return None

        template_path = self.prompt_dir / f"{name}.jinja2"
        if not template_path.exists():
            return None

        try:
            template_content = template_path.read_text(encoding="utf-8")
            template = Template(template_content)
            return template.render(**context).strip()
        except Exception:
            return None

    def has_template(self, name: str) -> bool:
        """Return whether a template exists in either source."""
        if name in self.templates:
            return True

        if self.prompt_dir:
            template_path = self.prompt_dir / f"{name}.jinja2"
            if template_path.exists():
                return True

        default_template_path = self.default_prompt_dir / f"{name}.jinja2"
        return default_template_path.exists()

    def list_templates(self) -> list:
        """List all available template names."""
        templates = set()

        if self.prompt_dir and self.prompt_dir.exists():
            templates.update(f.stem for f in self.prompt_dir.glob("*.jinja2"))

        if self.default_prompt_dir.exists():
            templates.update(f.stem for f in self.default_prompt_dir.glob("*.jinja2"))

        return list(templates)

    def save_template(self, name: str, content: str):
        """
        Save a template into the experiment prompt directory.
        """
        if not self.prompt_dir:
            raise ValueError("No prompt directory configured")

        self.prompt_dir.mkdir(parents=True, exist_ok=True)
        template_path = self.prompt_dir / f"{name}.jinja2"
        template_path.write_text(content, encoding="utf-8")

        if name in self.templates:
            del self.templates[name]
