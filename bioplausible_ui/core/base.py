from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from bioplausible_ui.core.schema import TabSchema
from bioplausible_ui.core.widgets.plot_widget import BasePlotWidget

if TYPE_CHECKING:
    pass


class TabMeta(type(QWidget)):
    """Auto-generates __init__ and wires signals from schema."""

    def __new__(mcs, name, bases, dct):
        if "SCHEMA" in dct and dct["SCHEMA"] is not None:
            schema = dct["SCHEMA"]
            dct["__init__"] = mcs._generate_init(schema)

            # Auto-generate property accessors
            for widget_def in schema.widgets:
                # We use a closure to capture the name
                def make_prop(w_name):
                    return property(lambda self: self._widgets[w_name])

                dct[widget_def.name] = make_prop(widget_def.name)

            for plot_def in schema.plots:

                def make_plot_prop(p_name):
                    return property(lambda self: self._plots[p_name])

                dct[f"plot_{plot_def.name}"] = make_plot_prop(plot_def.name)

        return super().__new__(mcs, name, bases, dct)

    @staticmethod
    def _generate_init(schema):
        def __init__(self, parent=None):
            QWidget.__init__(self, parent)
            self._widgets = {}
            self._plots = {}
            self._actions = {}
            self._build_from_schema(schema)
            if hasattr(self, "_post_init"):
                self._post_init()

        return __init__


class BaseTab(QWidget, metaclass=TabMeta):
    """Base class for all tabs - UI built from schema."""

    SCHEMA = None  # Override in subclasses

    def _resolve_params(self, params):
        # Resolve any string references or return params as is
        resolved = {}
        for k, v in params.items():
            resolved[k] = v
        return resolved

    def _build_from_schema(self, schema: "TabSchema"):
        """Build UI from declarative schema."""
        if schema.layout:
            # Complex layout
            self._build_layout(schema.layout, self)
        else:
            # Default vertical layout
            layout = QVBoxLayout(self)

            # Create widgets
            for widget_def in schema.widgets:
                widget = widget_def.widget_class(
                    **self._resolve_params(widget_def.params)
                )
                self._widgets[widget_def.name] = widget

                # Auto-wire signal if binding specified
                if widget_def.bindings:
                    for param, binding in widget_def.bindings.items():
                        self._setup_binding(widget, param, binding)

                layout.addWidget(widget)

            # Actions row
            if schema.actions:
                action_layout = QHBoxLayout()
                for action_def in schema.actions:
                    btn = QPushButton(action_def.name)
                    if action_def.callback and hasattr(self, action_def.callback):
                        btn.clicked.connect(getattr(self, action_def.callback))

                    if action_def.style == "danger":
                        btn.setStyleSheet("background-color: #ff6b6b; color: white;")
                    elif action_def.style == "success":
                        btn.setStyleSheet("background-color: #2ecc71; color: white;")

                    action_layout.addWidget(btn)
                    self._actions[action_def.name] = btn
                layout.addLayout(action_layout)

            # Plots
            for plot_def in schema.plots:
                plot = BasePlotWidget(
                    title=plot_def.name, xlabel=plot_def.xlabel, ylabel=plot_def.ylabel
                )
                layout.addWidget(plot)
                self._plots[plot_def.name] = plot

    def _setup_binding(self, widget, param, binding):
        """
        binding format: "@other_widget.value" -> triggers update
        """
        if binding.startswith("@"):
            parts = binding[1:].split(".")
            target_name = parts[0]

            if target_name in self._widgets:
                target = self._widgets[target_name]
                # Try to connect generic signals
                if hasattr(target, "valueChanged"):
                    target.valueChanged.connect(
                        lambda val: self._update_param(widget, param, val)
                    )
                elif hasattr(target, "currentTextChanged"):
                    target.currentTextChanged.connect(
                        lambda val: self._update_param(widget, param, val)
                    )
                elif hasattr(target, "currentIndexChanged"):
                    target.currentIndexChanged.connect(
                        lambda val: self._update_param(widget, param, val)
                    )

    def _update_param(self, widget, param, value):
        setter = f"set_{param}"
        # try simple attribute set if method not found
        if hasattr(widget, setter):
            getattr(widget, setter)(value)
        elif hasattr(widget, param):
            setattr(widget, param, value)

    def _build_layout(self, layout_def, parent_widget):
        # Placeholder for complex layout builder
        pass
