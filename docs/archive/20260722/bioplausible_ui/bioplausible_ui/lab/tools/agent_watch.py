import gymnasium as gym
import torch
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QComboBox, QGroupBox, QLabel, QPushButton, QVBoxLayout

from bioplausible_ui.lab.registry import ToolRegistry
from bioplausible_ui.lab.tools.base import BaseTool


class PlaybackWorker(QThread):
    finished = pyqtSignal(float, list)  # total_reward, frames
    error = pyqtSignal(str)

    def __init__(self, model, env_name, parent=None):
        super().__init__(parent)
        self.model = model
        self.env_name = env_name

    def run(self):
        try:
            env = gym.make(self.env_name, render_mode="rgb_array")
            state, _ = env.reset()
            done = False
            truncated = False
            total_reward = 0
            steps = 0
            frames = []

            while not (done or truncated) and steps < 500:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

                # Action
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0)
                    if hasattr(self.model, "device"):
                        state_t = state_t.to(self.model.device)
                    elif next(self.model.parameters()).is_cuda:
                        state_t = state_t.cuda()

                    q_values = self.model(state_t)
                    action = q_values.argmax().item()

                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1

            env.close()
            self.finished.emit(total_reward, frames)
        except Exception as e:
            self.error.emit(str(e))


@ToolRegistry.register("Agent Watch", requires=["agent_watch"])
class AgentWatchTool(BaseTool):
    ICON = "👁️"

    def init_ui(self):
        super().init_ui()

        # Controls
        controls = QGroupBox("Configuration")
        ctrl_layout = QVBoxLayout(controls)

        self.env_combo = QComboBox()
        self.env_combo.addItems(["CartPole-v1", "Acrobot-v1", "MountainCar-v0"])
        ctrl_layout.addWidget(QLabel("Environment:"))
        ctrl_layout.addWidget(self.env_combo)

        self.watch_btn = QPushButton("Watch Episode")
        self.watch_btn.clicked.connect(self._watch)
        ctrl_layout.addWidget(self.watch_btn)

        self.layout.addWidget(controls)

        # Display
        display_group = QGroupBox("View")
        display_layout = QVBoxLayout(display_group)

        self.image_label = QLabel("Click Watch to start")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: black;")
        display_layout.addWidget(self.image_label)

        self.status_label = QLabel("")
        display_layout.addWidget(self.status_label)

        self.layout.addWidget(display_group)
        self.layout.addStretch()

        self.frames = []
        self.current_frame = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)

    def _watch(self):
        if self.model is None:
            self.status_label.setText("No model loaded.")
            return

        self.watch_btn.setEnabled(False)
        self.status_label.setText("Running simulation...")

        env_name = self.env_combo.currentText()
        self.worker = PlaybackWorker(self.model, env_name)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, reward, frames):
        self.watch_btn.setEnabled(True)
        self.status_label.setText(f"Finished! Reward: {reward:.1f}")
        self.frames = frames
        self.current_frame = 0
        if frames:
            self.timer.start(50)  # 20 FPS
        else:
            self.status_label.setText("No frames captured.")

    def _on_error(self, err):
        self.watch_btn.setEnabled(True)
        self.status_label.setText(f"Error: {err}")

    def _next_frame(self):
        if not self.frames:
            return

        frame = self.frames[self.current_frame]
        h, w, c = frame.shape
        qimg = QImage(frame.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        )

        self.current_frame = (self.current_frame + 1) % len(self.frames)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)
