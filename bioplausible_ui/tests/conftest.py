import pytest

try:
    pass
except ImportError:
    # If pytest-qt is missing, skip tests that require it
    @pytest.fixture
    def qtbot():
        pytest.skip("pytest-qt not installed. Please install it to run UI tests.")

    @pytest.fixture
    def qapp():
        pytest.skip("pytest-qt not installed. Please install it to run UI tests.")
