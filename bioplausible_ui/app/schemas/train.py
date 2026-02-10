from bioplausible_ui.core.schema import (ActionDef, PlotDef, TabSchema,
                                         WidgetDef)
from bioplausible_ui.core.widgets.dataset_picker import DatasetPicker
from bioplausible_ui.core.widgets.hyperparam_editor import HyperparamEditor
from bioplausible_ui.core.widgets.model_selector import ModelSelector
from bioplausible_ui.core.widgets.task_selector import TaskSelector
from bioplausible_ui.core.widgets.training_config import TrainingConfigWidget

TRAIN_TAB_SCHEMA = TabSchema(
    name="Train",
    widgets=[
        WidgetDef("task_selector", TaskSelector),
        WidgetDef(
            "dataset_picker", DatasetPicker, bindings={"task": "@task_selector.value"}
        ),
        WidgetDef(
            "model_selector", ModelSelector, bindings={"task": "@task_selector.value"}
        ),
        WidgetDef(
            "training_config",
            TrainingConfigWidget,
            bindings={"task": "@task_selector.value"},
        ),
        WidgetDef(
            "hyperparam_editor",
            HyperparamEditor,
            bindings={"model": "@model_selector.value"},
        ),
    ],
    actions=[
        ActionDef("start", "▶", "_start_training", style="success", shortcut="Ctrl+R"),
        ActionDef("stop", "⏹", "_stop_training", style="danger", shortcut="Ctrl+Q"),
        ActionDef("test", "👁️", "_test_model", style="primary"),
        ActionDef("analyze", "🔬", "_analyze_model", style="secondary"),
    ],
    plots=[
        PlotDef("loss", xlabel="Epoch", ylabel="Loss"),
        PlotDef("accuracy", xlabel="Epoch", ylabel="Accuracy"),
    ],
)
