from bioplausible_ui.core.schema import ActionDef, TabSchema, WidgetDef
from bioplausible_ui.core.widgets.export_format_selector import \
    ExportFormatSelector
from bioplausible_ui.core.widgets.run_selector import RunSelector

DEPLOY_TAB_SCHEMA = TabSchema(
    name="Deploy",
    widgets=[
        WidgetDef("run_selector", RunSelector),
        WidgetDef("format_selector", ExportFormatSelector),
        # RunSelector is the primary mechanism for deployment ("Exploit" use case).
        # Architecture export without training is deferred to future updates.
    ],
    actions=[
        ActionDef("export", "📦", "_export_model", style="primary"),
        ActionDef("serve", "🚀", "_serve_model", style="success"),
        ActionDef("refresh", "🔄", "_refresh_runs", style="secondary"),
    ],
    plots=[],
)
