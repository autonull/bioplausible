from bioplausible_ui.core.base import BaseTab
from bioplausible_ui.core.schema import TabSchema, WidgetDef
from bioplausible_ui.core.widgets.task_selector import TaskSelector
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget


class SourceWidget(QWidget):
    valueChanged = pyqtSignal(str)

    def emit_val(self, v):
        self.valueChanged.emit(v)


class TargetWidget(QWidget):
    def __init__(self, val="", parent=None):
        super().__init__(parent)
        self.val = val

    def set_val(self, v):
        self.val = v


def test_schema_based_tab_creation(qtbot):
    schema = TabSchema(
        name="Test", widgets=[WidgetDef("selector", TaskSelector)], actions=[], plots=[]
    )

    class TestTab(BaseTab):
        SCHEMA = schema

    tab = TestTab()
    qtbot.addWidget(tab)

    assert hasattr(tab, "selector")
    assert isinstance(tab.selector, TaskSelector)


def test_binding(qtbot):
    schema = TabSchema(
        name="Test Binding",
        widgets=[
            WidgetDef("source", SourceWidget),
            WidgetDef("target", TargetWidget, bindings={"val": "@source.value"}),
        ],
        actions=[],
        plots=[],
    )

    class BindingTab(BaseTab):
        SCHEMA = schema

    tab = BindingTab()
    qtbot.addWidget(tab)

    # Trigger signal
    tab.source.emit_val("new_value")

    assert tab.target.val == "new_value"
