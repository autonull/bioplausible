from typing import List, Dict, Any, Type
import json
from .sections import ReportSection, ConfigSection, PerformanceSection, DynamicsSection

class ReportComposer:
    """
    Composes a complete research report from multiple sections.
    """
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.sections: List[ReportSection] = []
        
        # Default sections - can be customized via add_section
        self.add_section(ConfigSection)
        self.add_section(PerformanceSection)
        self.add_section(DynamicsSection)
        
    def add_section(self, section_cls: Type[ReportSection]):
        """Instantiate and add a section."""
        section = section_cls(self.data)
        self.sections.append(section)
        
    def compile_markdown(self) -> str:
        """Generate the full markdown report."""
        # Sort by priority
        sorted_sections = sorted(self.sections, key=lambda s: s.priority)
        
        report = f"# Scientist++ Experiment Report: {self.data.get('model_name', 'Unknown Model')}\n"
        report += f"**Task**: {self.data.get('task_name', 'Unknown')}\n"
        report += f"**Trial ID**: {self.data.get('trial_id', 'N/A')}\n\n"
        
        report += "---\n\n"
        
        for section in sorted_sections:
            report += section.generate_markdown()
            report += "\n---\n\n"
            
        return report
        
    def compile_json(self) -> str:
        """Generate the full JSON report structure."""
        return json.dumps(self._generate_json_dict(), indent=2)

    def _generate_json_dict(self) -> Dict[str, Any]:
        """Internal helper to generate the dictionary."""
        sorted_sections = sorted(self.sections, key=lambda s: s.priority)
        
        return {
            "meta": {
                "model_name": self.data.get("model_name"),
                "task_name": self.data.get("task_name"),
                "trial_id": self.data.get("trial_id")
            },
            "sections": [s.generate_json() for s in sorted_sections]
        }

    def save_reports(self, output_dir: str):
        """
        Save the compiled report files to the specified directory.
        Generates:
        - report.md: Full markdown report
        - report.json: Full JSON data
        - manifest.json: Metadata about the report
        """
        import os
        from datetime import datetime
        from pathlib import Path
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Save Markdown
        md_content = self.compile_markdown()
        with open(out_path / "report.md", "w") as f:
            f.write(md_content)
            
        # 2. Save JSON Data
        json_content = self.compile_json()
        with open(out_path / "report.json", "w") as f:
            f.write(json_content)
            
        # 3. Generate and Save Manifest
        manifest = {
            "report_version": "2.0",
            "generated_at": datetime.now().isoformat(),
            "meta": self.data.get("meta", {
                "model_name": self.data.get("model_name"),
                "task_name": self.data.get("task_name"),
                "trial_id": self.data.get("trial_id")
            }),
            "files": [
                {"name": "report.md", "type": "markdown", "description": "Full human-readable report"},
                {"name": "report.json", "type": "json", "description": "Machine-readable experiment data"}
            ],
            # List sections available in the breakdown
            "sections": [s.section_id for s in self.sections] 
        }
        
        with open(out_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
