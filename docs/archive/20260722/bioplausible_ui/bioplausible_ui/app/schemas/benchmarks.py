from PyQt6.QtWidgets import QCheckBox, QListWidget

from bioplausible_ui.core.schema import ActionDef, TabSchema, WidgetDef
from bioplausible_ui.core.widgets.track_selector import TrackSelector

BENCHMARKS_TAB_SCHEMA = TabSchema(
    name="Benchmarks",
    widgets=[
        WidgetDef("track_selector", TrackSelector),
        WidgetDef("parallel_check", QCheckBox, params={"text": "Parallel Execution"}),
        WidgetDef("results_list", QListWidget),
    ],
    actions=[
        ActionDef("run", "▶", "_run_benchmarks", style="primary"),
        ActionDef("clear", "🧹", "_clear_logs"),
    ],
    plots=[],
)
