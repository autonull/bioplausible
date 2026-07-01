"""
System Monitoring Module.

Provides utilities for detecting interference from other processes
and monitoring system resource usage to ensure experiment integrity.
"""

import logging
import os
import threading
import time
from typing import Optional

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("InterferenceMonitor")


class InterferenceMonitor:
    """
    Monitors system resource usage to detect interference from other processes.

    Attributes:
        threshold_cpu (float): Max allowed background CPU usage
            (percentage of total system).
        sustain_duration (float): Time in seconds that violation
            must persist to trigger detection.
        interval (float): Sampling interval in seconds.
    """

    def __init__(
        self,
        threshold_cpu: float = 20.0,
        sustain_duration: float = 5.0,
        interval: float = 1.0,
    ) -> None:
        """
        Initialize the InterferenceMonitor.

        Args:
            threshold_cpu: Background CPU usage threshold to trigger detection.
            sustain_duration: Duration to sustain violation before alerting.
            interval: Sampling interval in seconds.
        """
        self.threshold_cpu = threshold_cpu
        self.sustain_duration = sustain_duration
        self.interval = interval
        self.running = False
        self.interference_detected = False
        self.thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the monitoring thread."""
        if not psutil:
            logger.warning("psutil not installed. Monitoring disabled.")
            return

        self.running = True
        self.interference_detected = False
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the monitoring thread."""
        if not self.running:
            return
        self.running = False
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def check_interference(self) -> bool:
        """
        Check if interference has been detected.

        Returns:
            bool: True if interference detected, False otherwise.
        """
        return self.interference_detected

    def _monitor_loop(self) -> None:
        """
        Internal loop to monitor CPU usage.
        """
        if not psutil:
            return

        violation_start_time: Optional[float] = None
        try:
            p = psutil.Process(os.getpid())
            # Prime the counters (first call returns 0.0)
            p.cpu_percent()
            psutil.cpu_percent()
        except Exception as e:
            logger.error(f"Failed to initialize monitor process: {e}")
            return

        while not self._stop_event.is_set():
            # Wait for interval or stop event
            if self._stop_event.wait(self.interval):
                break

            try:
                # System-wide CPU (avg across cores)
                sys_cpu = psutil.cpu_percent(interval=None)

                # Process CPU (sum across threads, can be > 100% per core)
                proc_cpu = p.cpu_percent(interval=None)

                num_cores = psutil.cpu_count() or 1
                # Normalize process CPU to system scale (0-100%)
                proc_cpu_share = proc_cpu / num_cores

                # Background load is what's left
                background = sys_cpu - proc_cpu_share

                # Handle negative jitter or precision issues
                background = max(0.0, background)

                if background > self.threshold_cpu:
                    if violation_start_time is None:
                        violation_start_time = time.time()
                    elif time.time() - violation_start_time > self.sustain_duration:
                        self.interference_detected = True
                        logger.warning(
                            "Interference detected! Background: %.1f%%"
                            " (Sys: %.1f%%, Proc: %.1f%%)",
                            background, sys_cpu, proc_cpu_share,
                        )
                else:
                    violation_start_time = None

            except Exception as e:
                logger.error(f"Monitor error: {e}")
