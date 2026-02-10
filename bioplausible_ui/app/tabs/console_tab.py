import sys

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QMessageBox

from bioplausible_ui.app.schemas.console import CONSOLE_TAB_SCHEMA
from bioplausible_ui.core.base import BaseTab


class StreamRedirector(QObject):
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass


class ConsoleTab(BaseTab):
    """Console tab - UI auto-generated from schema."""

    SCHEMA = CONSOLE_TAB_SCHEMA

    def _post_init(self):
        # Redirect stdout/stderr
        self.stdout_redirector = StreamRedirector()
        self.stderr_redirector = StreamRedirector()

        self.stdout_redirector.text_written.connect(self._on_stdout)
        self.stderr_redirector.text_written.connect(self._on_stderr)

        # Save originals
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Override
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stderr_redirector

    def _on_stdout(self, text):
        # We also print to original to not lose terminal output completely
        self.original_stdout.write(text)
        if text.strip():  # Avoid empty newlines flooding
            self.log_output.log(text.strip())

    def _on_stderr(self, text):
        self.original_stderr.write(text)
        if text.strip():
            # Could style red? LogOutput uses text_edit.append which is simple.
            # We can use html for color.
            self.log_output.text_edit.append(f"<font color='red'>{text.strip()}</font>")

    def _run_diagnostics(self):
        print("Running system diagnostics...")
        print(f"Python version: {sys.version}")
        print("Diagnostics complete.")

    def _clear_logs(self):
        self.log_output.text_edit.clear()

    def _save_logs(self):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Logs", "bioplausible.log", "Log Files (*.log)"
        )
        if fname:
            try:
                with open(fname, "w") as f:
                    f.write(self.log_output.text_edit.toPlainText())
                QMessageBox.information(self, "Success", f"Logs saved to {fname}")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))

    def closeEvent(self, event):
        # Restore
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        super().closeEvent(event)
