from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union


@dataclass
class WidgetDef:
    name: str
    widget_class: Type
    params: Dict[str, Any] = field(default_factory=dict)
    bindings: Dict[str, str] = field(default_factory=dict)  # "@other_widget.value"
    visible_when: Optional[str] = None  # Conditional visibility
    layout: str = "vertical"


@dataclass
class ActionDef:
    name: str
    icon: str
    callback: str
    enabled_when: Optional[str] = None
    shortcut: Optional[str] = None
    style: Optional[str] = None  # "primary", "danger", "success"


@dataclass
class PlotDef:
    name: str
    xlabel: str
    ylabel: str
    type: str = "line"  # "line", "scatter", "violin", "radar"


@dataclass
class LayoutDef:
    type: str  # "vertical", "horizontal", "grid", "tabs", "splitter"
    items: List[Union[WidgetDef, ActionDef, "LayoutDef"]]
    stretch: Optional[List[int]] = None


@dataclass
class TabSchema:
    name: str
    widgets: List[WidgetDef]
    actions: List[ActionDef]
    plots: List[PlotDef]
    layout: Optional[LayoutDef] = None
