class ToolRegistry:
    """Auto-discovery for lab analysis tools."""

    _tools = {}

    @classmethod
    def register(cls, name, requires=None):
        """Decorator to register tools."""

        def decorator(tool_class):
            cls._tools[name] = {"class": tool_class, "requires": requires or []}
            return tool_class

        return decorator

    @classmethod
    def get_compatible_tools(cls, model_spec):
        """Get tools compatible with model capabilities."""
        compatible = []
        for name, info in cls._tools.items():
            # Check if model has required capabilities
            # model_spec is ModelSpec object
            # capabilities are fields starting with supports_

            is_compatible = True
            for req in info["requires"]:
                # Mapping requirement to field: "dynamics" -> "supports_dynamics"
                field_name = f"supports_{req}"
                if not getattr(model_spec, field_name, False):
                    is_compatible = False
                    break

            if is_compatible:
                compatible.append(name)
        return compatible

    @classmethod
    def get_tool_class(cls, name):
        return cls._tools[name]["class"]
