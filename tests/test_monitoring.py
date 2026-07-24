import time
from unittest.mock import MagicMock, patch

from bioplausible.execution.monitoring import InterferenceMonitor


def test_interference_monitor_init():
    monitor = InterferenceMonitor(threshold_cpu=20.0, sustain_duration=5.0)
    assert monitor.threshold_cpu == 20.0
    assert monitor.sustain_duration == 5.0
    assert not monitor.running
    assert not monitor.check_interference()


@patch("bioplausible.execution.monitoring.psutil")
@patch("bioplausible.execution.monitoring.os")
def test_monitor_detection(mock_os, mock_psutil):
    # Setup mocks
    mock_os.getpid.return_value = 1234

    mock_process = MagicMock()
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_count.return_value = 4

    # We set interval to small value
    monitor = InterferenceMonitor(
        threshold_cpu=20.0, sustain_duration=0.1, interval=0.05
    )

    # Simulation:
    # Call 1 (Prime): cpu_percent -> ignored
    # Call 2 (Loop 1): sys=50, proc=10 (share 2.5) -> back=47.5 > 20. Start timer.
    # Call 3 (Loop 2): sys=50, proc=10 -> back=47.5 > 20. Time diff > 0.1? Yes. Detect.

    # side_effect needs enough values for:
    # 1. psutil.cpu_percent() (prime)
    # 2. psutil.cpu_percent(interval=None) (loop 1)
    # 3. psutil.cpu_percent(interval=None) (loop 2)
    # ...
    mock_psutil.cpu_percent.side_effect = [0.0, 50.0, 50.0, 50.0, 50.0, 50.0]

    # 1. p.cpu_percent() (prime)
    # 2. p.cpu_percent(interval=None) (loop 1)
    # ...
    mock_process.cpu_percent.side_effect = [0.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    monitor.start()
    time.sleep(0.3)  # Wait enough for detection
    monitor.stop()

    assert monitor.check_interference() is True


@patch("bioplausible.execution.monitoring.psutil")
@patch("bioplausible.execution.monitoring.os")
def test_monitor_no_interference(mock_os, mock_psutil):
    mock_os.getpid.return_value = 1234
    mock_process = MagicMock()
    mock_psutil.Process.return_value = mock_process
    mock_psutil.cpu_count.return_value = 4

    monitor = InterferenceMonitor(
        threshold_cpu=20.0, sustain_duration=0.1, interval=0.05
    )

    # Always low load
    mock_psutil.cpu_percent.return_value = 10.0
    mock_process.cpu_percent.return_value = 10.0  # share 2.5 -> back 7.5

    monitor.start()
    time.sleep(0.2)
    monitor.stop()

    assert monitor.check_interference() is False
