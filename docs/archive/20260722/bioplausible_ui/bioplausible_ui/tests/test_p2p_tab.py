from unittest.mock import MagicMock, patch

from PyQt6.QtWidgets import QLabel, QPushButton

from bioplausible_ui.app.tabs.p2p_tab import P2PTab
from bioplausible_ui.app.window import AppMainWindow


def test_p2p_tab_ui_structure(qtbot):
    """Verifies that P2PTab constructs its UI elements correctly."""
    tab = P2PTab()
    qtbot.addWidget(tab)

    # Check if key widgets exist
    connect_btn = tab.findChild(QPushButton)
    # The connect button text changes, but initially it should contain "Join Network"
    assert "Join Network" in connect_btn.text()

    # Check status label
    labels = tab.findChildren(QLabel)
    status_label_found = False
    for lbl in labels:
        if "DISCONNECTED" in lbl.text():
            status_label_found = True
            break
    assert status_label_found, "Status label not found initialized to DISCONNECTED"

    # Check that worker is None initially
    assert tab.worker is None


def test_p2p_tab_cleanup(qtbot):
    """Verifies that cleanup doesn't crash even if no worker runs."""
    tab = P2PTab()
    qtbot.addWidget(tab)
    tab.close()


def test_main_window_integration(qtbot):
    """Verifies that P2PTab is integrated into AppMainWindow."""
    window = AppMainWindow()
    qtbot.addWidget(window)
    window.show()

    # Check tab existence
    tabs = window.tabs
    found = False
    for i in range(tabs.count()):
        if tabs.tabText(i) == "Community":
            found = True
            assert isinstance(tabs.widget(i), P2PTab)
            break

    assert found, "Community tab not found in AppMainWindow"


def test_shutdown_called_on_close(qtbot):
    """Verifies AppMainWindow calls shutdown on tabs."""
    window = AppMainWindow()
    qtbot.addWidget(window)
    window.show()

    # Mock shutdown on P2PTab
    p2p_tab = window.p2p_tab
    p2p_tab.shutdown = MagicMock()

    window.close()

    p2p_tab.shutdown.assert_called_once()


def test_p2p_local_preset_parsing(qtbot):
    """Test parsing of '127.0.0.1:8468 (Local Test)' preset."""
    tab = P2PTab()
    qtbot.addWidget(tab)

    # Switch to DHT mode
    tab.radio_dht.setChecked(True)

    # Select "127.0.0.1:8468 (Local Test)"
    # Combo box items: [bootstrap1, bootstrap2, local, empty]
    # Local is index 2
    tab.bootstrap_combo.setCurrentIndex(2)
    selected_text = tab.bootstrap_combo.currentText()
    assert "Local Test" in selected_text

    with (
        patch("bioplausible_ui.app.tabs.p2p_tab.P2PEvolution") as MockEvolution,
        patch("bioplausible_ui.app.tabs.p2p_tab.P2PWorkerBridge"),
    ):
        tab.connect_btn.click()

        # Check args passed to P2PEvolution
        args, kwargs = MockEvolution.call_args
        assert kwargs["bootstrap_ip"] == "127.0.0.1"
        assert kwargs["bootstrap_port"] == 8468
