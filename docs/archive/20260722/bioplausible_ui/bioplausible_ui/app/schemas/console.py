from bioplausible_ui.core.schema import ActionDef, TabSchema, WidgetDef
from bioplausible_ui.core.widgets.log_output import LogOutput

CONSOLE_TAB_SCHEMA = TabSchema(
    name="Console",
    widgets=[
        WidgetDef("log_output", LogOutput),
    ],
    actions=[
        ActionDef("run_diagnostics", "🩺", "_run_diagnostics"),
        ActionDef("save", "💾", "_save_logs"),
        ActionDef("clear", "🧹", "_clear_logs"),
    ],
    plots=[],
)
