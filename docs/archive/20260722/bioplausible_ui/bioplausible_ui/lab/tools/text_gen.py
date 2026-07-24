from bioplausible_ui.lab.registry import ToolRegistry
from bioplausible_ui.lab.tools.base import BaseTool
from bioplausible_ui.lab.tools.generation import UniversalGenerator
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
)


class GenerationWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, generator, prompt, temperature, max_tokens=100, parent=None):
        super().__init__(parent)
        self.generator = generator
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self):
        try:
            text = self.generator.generate(
                prompt=self.prompt,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            self.finished.emit(text)
        except Exception as e:
            self.error.emit(str(e))


@ToolRegistry.register("Text Generation", requires=["text_gen"])
class TextGenerationTool(BaseTool):
    ICON = "✨"

    def init_ui(self):
        super().init_ui()

        self.generator = None
        self.gen_worker = None

        # Controls
        controls = QGroupBox("Controls")
        ctrl_layout = QVBoxLayout(controls)

        # Prompt
        ctrl_layout.addWidget(QLabel("Prompt:"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Enter prompt...")
        self.prompt_input.setMaximumHeight(60)
        ctrl_layout.addWidget(self.prompt_input)

        # Temperature
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(1, 20)
        self.temp_slider.setValue(10)
        temp_layout.addWidget(self.temp_slider)
        self.temp_label = QLabel("1.0")
        self.temp_label.setFixedWidth(30)
        temp_layout.addWidget(self.temp_label)
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v/10:.1f}")
        )
        ctrl_layout.addLayout(temp_layout)

        # Generate Button
        self.gen_btn = QPushButton("Generate")
        self.gen_btn.clicked.connect(self._generate)
        ctrl_layout.addWidget(self.gen_btn)

        self.layout.addWidget(controls)

        # Output
        out_group = QGroupBox("Output")
        out_layout = QVBoxLayout(out_group)
        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        out_layout.addWidget(self.output_text)
        self.layout.addWidget(out_group)

    def _generate(self):
        if self.model is None:
            self.output_text.setText("No model loaded.")
            return

        # Initialize generator if needed
        if self.generator is None or self.generator.model is not self.model:
            try:
                vocab_size = 95
                if hasattr(self.model, "vocab_size"):
                    vocab_size = self.model.vocab_size
                elif hasattr(self.model, "lm_head"):
                    vocab_size = self.model.lm_head.out_features

                device = next(self.model.parameters()).device
                self.generator = UniversalGenerator(
                    self.model, vocab_size=vocab_size, device=str(device)
                )
            except Exception as e:
                self.output_text.setText(f"Error initializing generator: {e}")
                return

        prompt = self.prompt_input.toPlainText() or " "
        temp = self.temp_slider.value() / 10.0

        self.gen_btn.setEnabled(False)
        self.output_text.setText("Generating...")

        self.gen_worker = GenerationWorker(self.generator, prompt, temp)
        self.gen_worker.finished.connect(self._on_finished)
        self.gen_worker.error.connect(self._on_error)
        self.gen_worker.start()

    def _on_finished(self, text):
        self.output_text.setText(text)
        self.gen_btn.setEnabled(True)

    def _on_error(self, err):
        self.output_text.setText(f"Error: {err}")
        self.gen_btn.setEnabled(True)

    def refresh(self):
        self.generator = None  # Reset generator when model changes
