import logging
import sys
import shutil

# psutil needed for resource monitoring
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("AutoScientist")


class ResourceMonitor:
    """Monitors system resources to prevent overload."""

    def __init__(self, cpu_limit=90.0, mem_limit=90.0, gpu_limit=90.0, disk_limit=95.0):
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit
        self.gpu_limit = gpu_limit
        self.disk_limit = disk_limit

    def should_pause(self) -> bool:
        if not psutil:
            return False

        # CPU & RAM Check
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent

        if cpu > self.cpu_limit:
            logger.warning(f"System Load High: CPU={cpu}%. Pausing...")
            return True

        if mem > self.mem_limit:
            logger.warning(f"System Load High: Mem={mem}%. Pausing...")
            return True

        # GPU Check
        if torch and torch.cuda.is_available():
            try:
                # Get global free memory (free, total) for device 0
                free, total = torch.cuda.mem_get_info(0)
                used_ratio = (total - free) / total * 100.0
                if used_ratio > self.gpu_limit:
                    logger.warning(f"GPU Load High: Mem={used_ratio:.1f}%. Pausing...")
                    return True
            except Exception:
                pass  # Ignore GPU check errors

        # Disk Check (cwd)
        total, used, free = shutil.disk_usage(".")
        disk_percent = (used / total) * 100.0
        if disk_percent > self.disk_limit:
            logger.warning(f"Disk Space Low: Used={disk_percent:.1f}%. Pausing...")
            return True

        return False
