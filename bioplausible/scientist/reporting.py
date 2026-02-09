"""
ScientistReporter: Generates publication-quality reports from experiment data.
Now delegates to bioplausible.scientist.report.composer.ReportComposer.
"""

import logging
from bioplausible.scientist.report.composer import ReportComposer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reporter")


class ScientistReporter:
    """
    Generates analysis reports from the experiment database.
    Now wraps ReportComposer for modular generation.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path

    def generate_report(self, output_dir: str):
        """
        Main entry point. Generates Markdown and Images via ReportComposer.
        """
        logger.info(f"Generating report from {self.db_path} to {output_dir}...")
        try:
            composer = ReportComposer(self.db_path, output_dir)
            composer.generate_report()
            composer.close()
            logger.info("Report generation complete.")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
