import threading
import time
import logging
import os

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("InterferenceMonitor")


class InterferenceMonitor:
    """
    Monitors system resource usage to detect interference from other processes.
    """

    def __init__(self, threshold_cpu=20.0, sustain_duration=5.0, interval=1.0):
        """
        Args:
            threshold_cpu: Max allowed background CPU usage (percentage of total system).
            sustain_duration: Time in seconds that violation must persist to trigger detection.
            interval: Sampling interval in seconds.
        """
        self.threshold_cpu = threshold_cpu
        self.sustain_duration = sustain_duration
        self.interval = interval
        self.running = False
        self.interference_detected = False
        self.thread = None
        self._stop_event = threading.Event()

    def start(self):
        if not psutil:
            logger.warning("psutil not installed. Monitoring disabled.")
            return

        self.running = True
        self.interference_detected = False
        self._stop_event.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def check_interference(self) -> bool:
        return self.interference_detected

    def _monitor_loop(self):
        # Note: GPU monitoring is handled in `resources.py` for pause decisions.
        # This monitor focuses on CPU interference detection for now.

        violation_start_time = None
        try:
            p = psutil.Process(os.getpid())
            # Prime the counters
            p.cpu_percent()
            psutil.cpu_percent()
        except Exception as e:
            logger.error(f"Failed to initialize monitor process: {e}")
            return

        while not self._stop_event.is_set():
            # Wait for interval
            if self._stop_event.wait(self.interval):
                break

            try:
                # System-wide CPU (avg across cores)
                sys_cpu = psutil.cpu_percent(interval=None)

                # Process CPU (sum across threads, can be > 100%)
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
                            f"Interference detected! Background Load: {background:.1f}% "
                            f"(Sys: {sys_cpu:.1f}%, Proc: {proc_cpu_share:.1f}%)"
                        )
                else:
                    violation_start_time = None

            except Exception as e:
                logger.error(f"Monitor error: {e}")
