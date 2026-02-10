from bioplausible.models.registry import ModelSpec
from bioplausible_ui.lab.registry import ToolRegistry


def test_tool_registry():
    @ToolRegistry.register("test_tool", requires=["magic"])
    class TestTool:
        pass

    spec = ModelSpec(
        name="TestModel",
        description="",
        model_type="test",
        family="test",
        task_compat=[],
    )
    # Monkeypatch supports_magic
    spec.supports_magic = True

    tools = ToolRegistry.get_compatible_tools(spec)
    assert "test_tool" in tools

    spec.supports_magic = False
    tools = ToolRegistry.get_compatible_tools(spec)
    assert "test_tool" not in tools


def test_lab_window(qtbot):
    from bioplausible_ui.lab.window import LabMainWindow

    window = LabMainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "Bioplausible Lab (biopl-lab)"
