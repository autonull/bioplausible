from dataclasses import dataclass, field
from typing import Any


@dataclass
class WidgetDef:
    name: str
    widget_class: type
    params: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, str] = field(default_factory=dict)  # "@other_widget.value"
    visible_when: str | None = None  # Conditional visibility
    layout: str = "vertical"


@dataclass
class ActionDef:
    name: str
    icon: str
    callback: str
    enabled_when: str | None = None
    shortcut: str | None = None
    style: str | None = None  # "primary", "danger", "success"


@dataclass
class PlotDef:
    name: str
    xlabel: str
    ylabel: str
    type: str = "line"  # "line", "scatter", "violin", "radar"


@dataclass
class LayoutDef:
    type: str  # "vertical", "horizontal", "grid", "tabs", "splitter"
    items: list[WidgetDef | ActionDef | LayoutDef]
    stretch: list[int] | None = None


@dataclass
class TabSchema:
    name: str
    widgets: list[WidgetDef]
    actions: list[ActionDef]
    plots: list[PlotDef]
    layout: LayoutDef | None = None
