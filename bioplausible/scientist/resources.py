import logging
import sys

# psutil needed for resource monitoring
try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("AutoScientist")

class ResourceMonitor:
    """Monitors system resources to prevent overload."""

    def __init__(self, cpu_limit=90.0, mem_limit=90.0):
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit

    def should_pause(self) -> bool:
        if not psutil:
            return False

        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent

        if cpu > self.cpu_limit or mem > self.mem_limit:
            logger.warning(f"System Load High: CPU={cpu}%, Mem={mem}%. Pausing...")
            return True
        return False
