import numpy as np
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox

from bioplausible_ui.lab.registry import ToolRegistry
from bioplausible_ui.lab.tools.base import BaseTool

try:
    import pyqtgraph as pg

    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


@ToolRegistry.register("microscope", requires=["dynamics"])
class MicroscopeTool(BaseTool):
    ICON = "🔬"

    def init_ui(self):
        super().init_ui()
        self.layout.addWidget(QLabel("Live Network Dynamics"))

        # Controls
        ctrl_layout = QHBoxLayout()
        self.micro_steps_spin = QSpinBox()
        self.micro_steps_spin.setRange(10, 200)
        self.micro_steps_spin.setValue(50)
        ctrl_layout.addWidget(QLabel("Eq Steps:"))
        ctrl_layout.addWidget(self.micro_steps_spin)

        self.micro_layer_spin = QSpinBox()
        self.micro_layer_spin.setRange(0, 50)
        self.micro_layer_spin.setValue(0)
        ctrl_layout.addWidget(QLabel("Layer:"))
        ctrl_layout.addWidget(self.micro_layer_spin)

        self.layout.addLayout(ctrl_layout)

        self.micro_run_btn = QPushButton("▶ Run Analysis")
        self.micro_run_btn.clicked.connect(self._run_microscope_analysis)
        self.layout.addWidget(self.micro_run_btn)

        self.stability_label = QLabel("Stability: UNKNOWN")
        self.stability_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stability_label.setStyleSheet(
            "font-weight: bold; background-color: #333; padding: 5px; border-radius: 4px;"
        )
        self.layout.addWidget(self.stability_label)

        if HAS_PYQTGRAPH:
            self.micro_conv_plot = pg.PlotWidget(title="Convergence (||Δh||)")
            self.micro_conv_plot.setLabel("left", "Delta Norm")
            self.micro_conv_plot.setLabel("bottom", "Step")
            self.micro_conv_plot.setLogMode(x=False, y=True)
            self.micro_conv_curve = self.micro_conv_plot.plot(pen="y")
            self.layout.addWidget(self.micro_conv_plot)

            # Heatmap
            self.layout.addWidget(QLabel("Layer Activity Heatmap:"))
            self.micro_heat_view = pg.ImageView()
            self.micro_heat_view.ui.histogram.hide()
            self.micro_heat_view.ui.roiBtn.hide()
            self.micro_heat_view.ui.menuBtn.hide()
            self.layout.addWidget(self.micro_heat_view)
        else:
            self.layout.addWidget(QLabel("PyQtGraph required for plotting"))

    def refresh(self):
        if self.model:
            # Could auto-run?
            pass

    def _run_microscope_analysis(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "No model loaded.")
            return

        try:
            steps = self.micro_steps_spin.value()
            target_layer = self.micro_layer_spin.value()

            model = self.model

            # Determine input shape
            if hasattr(model, "input_dim"):
                input_shape = (1, model.input_dim)
            elif hasattr(model, "embed"):
                input_shape = (1, 128)  # Fake sequence
                x = torch.randint(0, model.embed.num_embeddings, input_shape)
            else:
                input_shape = (1, 784)

            if "x" not in locals():
                x = torch.randn(*input_shape)
                if hasattr(model, "device"):
                    x = x.to(model.device)
                else:
                    # Check param device
                    try:
                        device = next(model.parameters()).device
                        x = x.to(device)
                    except:
                        pass

            model.eval()
            with torch.no_grad():
                kwargs = {}
                import inspect

                sig = inspect.signature(model.forward)
                if "return_dynamics" in sig.parameters:
                    kwargs["return_dynamics"] = True
                if "return_trajectory" in sig.parameters:
                    kwargs["return_trajectory"] = True
                if "steps" in sig.parameters:
                    kwargs["steps"] = steps

                # Preprocess input if needed
                if hasattr(model, "has_embed") and model.has_embed:
                    h = model.embed(x)
                    if "LoopedMLP" in model.__class__.__name__:
                        h = h.mean(dim=1)
                    out = model(h, **kwargs)
                else:
                    out = model(x, **kwargs)

                if isinstance(out, tuple):
                    dynamics = out[1]
                else:
                    if hasattr(model, "dynamics"):
                        dynamics = model.dynamics
                    else:
                        # Fallback for models not supporting dynamics return
                        QMessageBox.information(
                            self, "Info", "Model does not return dynamics."
                        )
                        return

            deltas = dynamics.get("deltas", [])
            traj = dynamics.get("trajectory", [])

            if not deltas:
                return

            if traj:
                heat_data = []
                for t_step in traj:
                    if isinstance(t_step, (list, tuple)):
                        if target_layer < len(t_step):
                            state = t_step[target_layer]
                        else:
                            state = t_step[-1]
                    else:
                        state = t_step

                    if isinstance(state, tuple):  # (pre_act, h)
                        state = state[1]

                    flat = state.view(-1).cpu().numpy()
                    if len(flat) > 100:
                        flat = flat[:100]
                    heat_data.append(flat)

                if heat_data and HAS_PYQTGRAPH:
                    heat_arr = np.array(heat_data)
                    # Normalize
                    if heat_arr.max() > heat_arr.min():
                        heat_arr = (heat_arr - heat_arr.min()) / (
                            heat_arr.max() - heat_arr.min()
                        )
                    self.micro_heat_view.setImage(heat_arr)

            if HAS_PYQTGRAPH:
                self.micro_conv_curve.setData(deltas)

            # Stability Check
            final_delta = deltas[-1]
            if final_delta < 1e-4:
                self.stability_label.setText("STABLE (L < 1)")
                self.stability_label.setStyleSheet(
                    "background-color: #27ae60; color: white; padding: 5px; border-radius: 4px; font-weight: bold;"
                )
            elif final_delta < 1e-2:
                self.stability_label.setText("MARGINAL (Settling)")
                self.stability_label.setStyleSheet(
                    "background-color: #f39c12; color: white; padding: 5px; border-radius: 4px; font-weight: bold;"
                )
            else:
                self.stability_label.setText("UNSTABLE (Chaotic)")
                self.stability_label.setStyleSheet(
                    "background-color: #c0392b; color: white; padding: 5px; border-radius: 4px; font-weight: bold;"
                )

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", str(e))
            import traceback

            traceback.print_exc()
