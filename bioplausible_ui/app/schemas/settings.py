from bioplausible_ui.core.schema import ActionDef, TabSchema, WidgetDef
from bioplausible_ui.core.widgets.hyperparam_editor import HyperparamEditor

SETTINGS_TAB_SCHEMA = TabSchema(
    name="Settings",
    widgets=[
        WidgetDef(
            "preferences",
            HyperparamEditor,
            params={
                "defaults": {
                    "theme": "dark",
                    "auto_save": True,
                    "backend": "pytorch",  # pytorch, numpy
                    "device": "auto",  # auto, cpu, cuda
                    "logging_level": "INFO",
                    "default_results_dir": "./results",
                }
            },
        ),
    ],
    actions=[
        ActionDef("save", "💾", "_save_settings", style="primary"),
        ActionDef("reset", "🔄", "_reset_settings"),
    ],
    plots=[],
)
