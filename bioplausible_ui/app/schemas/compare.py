from bioplausible_ui.core.schema import ActionDef, PlotDef, TabSchema, WidgetDef
from bioplausible_ui.core.widgets.run_selector import RunSelector

COMPARE_TAB_SCHEMA = TabSchema(
    name="Compare",
    widgets=[
        WidgetDef("run_selector_1", RunSelector),
        WidgetDef("run_selector_2", RunSelector),
    ],
    actions=[
        ActionDef("compare_runs", "📉 Compare Runs", "_compare_saved_runs"),
    ],
    plots=[
        PlotDef("comparison_plot", xlabel="Epoch", ylabel="Accuracy"),
        PlotDef("loss_plot", xlabel="Epoch", ylabel="Loss"),
    ],
)
