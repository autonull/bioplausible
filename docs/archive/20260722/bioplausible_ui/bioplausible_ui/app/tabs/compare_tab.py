from bioplausible_ui.app.schemas.compare import COMPARE_TAB_SCHEMA
from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.core.bridge import SessionBridge
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox


class ComparisonWorker(QThread):
    progress = pyqtSignal(int, dict, dict)  # epoch, metrics1, metrics2
    completed = pyqtSignal()

    def __init__(self, config1, config2, parent=None):
        super().__init__(parent)
        self.config1 = config1
        self.config2 = config2
        self.bridge1 = SessionBridge(config1)
        self.bridge2 = SessionBridge(config2)
        self.running = True

    def run(self):
        # We need to interleave generators manually
        gen1 = self.bridge1.session.start()
        gen2 = self.bridge2.session.start()

        from bioplausible.pipeline.events import CompletedEvent, ProgressEvent

        active1 = True
        active2 = True
        epoch = 0

        metrics1 = {}
        metrics2 = {}

        while (active1 or active2) and self.running:
            # Step 1
            if active1:
                try:
                    event = next(gen1)
                    if isinstance(event, ProgressEvent):
                        metrics1 = event.metrics
                    elif isinstance(event, CompletedEvent):
                        metrics1 = event.final_metrics
                        active1 = False
                except StopIteration:
                    active1 = False
                except Exception as e:
                    print(f"Error Model 1: {e}")
                    active1 = False

            # Step 2
            if active2:
                try:
                    event = next(gen2)
                    if isinstance(event, ProgressEvent):
                        metrics2 = event.metrics
                    elif isinstance(event, CompletedEvent):
                        metrics2 = event.final_metrics
                        active2 = False
                except StopIteration:
                    active2 = False
                except Exception as e:
                    print(f"Error Model 2: {e}")
                    active2 = False

            if metrics1 or metrics2:
                self.progress.emit(epoch, metrics1, metrics2)

            epoch += 1

        self.completed.emit()

    def stop(self):
        self.running = False
        self.bridge1.stop()
        self.bridge2.stop()


class CompareTab(BaseTab):
    """Comparison tab - UI auto-generated from schema."""

    SCHEMA = COMPARE_TAB_SCHEMA

    def _post_init(self):
        self.results_manager = None  # Initialized on demand usually, or here
        from bioplausible.pipeline.results import ResultsManager

        self.results_manager = ResultsManager()

    def _compare_saved_runs(self):
        run_id_1 = self.run_selector_1.get_run_id()
        run_id_2 = self.run_selector_2.get_run_id()

        if not run_id_1 or not run_id_2:
            QMessageBox.warning(self, "Warning", "Please select two runs to compare.")
            return

        run1 = self.results_manager.load_run(run_id_1)
        run2 = self.results_manager.load_run(run_id_2)

        if not run1 or not run2:
            QMessageBox.critical(self, "Error", "Could not load run data.")
            return

        # Extract history
        # New format saves history in 'metrics' as 'history' list?
        # Or did we change _save_results to save dict with history?
        # See session.py update: `self._save_results(final_results)` where final_results = {final, history}
        # But `ResultsManager.save_run` takes `metrics`. So the `metrics` field in JSON will now contain `history`.

        metrics1 = run1.get("metrics", {})
        metrics2 = run2.get("metrics", {})

        hist1 = metrics1.get("history", [])
        hist2 = metrics2.get("history", [])

        if not hist1 and not hist2:
            QMessageBox.warning(
                self,
                "Warning",
                "No history data found in selected runs (maybe old format?).",
            )
            return

        # Plot
        self.plot_comparison_plot.clear()
        self.plot_loss_plot.clear()

        # We need to add legend manually or use our widget wrapper if it supports it.
        # `BasePlotWidget` has `add_legend(labels)`.

        import pyqtgraph as pg

        # Helper to extract
        def extract(hist, key):
            return [h.get("epoch", i) for i, h in enumerate(hist)], [
                h.get(key, 0) for h in hist
            ]

        x1, y1_acc = extract(hist1, "accuracy")
        x2, y2_acc = extract(hist2, "accuracy")

        x1_l, y1_loss = extract(hist1, "loss")
        x2_l, y2_loss = extract(hist2, "loss")

        # Plot Acc
        # BasePlotWidget wraps pg.PlotWidget in .plot_widget
        p1 = self.plot_comparison_plot.plot_widget
        p1.clear()
        if p1.plotItem.legend:
            p1.plotItem.legend.scene().removeItem(p1.plotItem.legend)
        p1.addLegend()
        p1.plot(x1, y1_acc, pen=pg.mkPen("r", width=2), name="Run 1")
        p1.plot(x2, y2_acc, pen=pg.mkPen("b", width=2), name="Run 2")

        # Plot Loss
        p2 = self.plot_loss_plot.plot_widget
        p2.clear()
        if p2.plotItem.legend:
            p2.plotItem.legend.scene().removeItem(p2.plotItem.legend)
        p2.addLegend()
        p2.plot(x1_l, y1_loss, pen=pg.mkPen("r", width=2), name="Run 1")
        p2.plot(x2_l, y2_loss, pen=pg.mkPen("b", width=2), name="Run 2")
