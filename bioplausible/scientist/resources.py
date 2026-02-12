"""
Resource Monitoring for AutoScientist.

This module provides system resource monitoring to prevent the autonomous
agent from overloading the host machine (CPU, RAM, Disk, GPU).
"""

import logging
import shutil
from typing import Optional, Tuple

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
    """
    Monitors system resources to prevent overload.

    Checks CPU, Memory, Disk, and GPU usage against defined thresholds.
    """

    def __init__(
        self,
        cpu_limit: float = 98.0,
        mem_limit: float = 98.0,
        gpu_limit: float = 98.0,
        disk_limit: float = 99.0,
    ) -> None:
        """
        Initialize the resource monitor.

        Args:
            cpu_limit (float): Max CPU usage percentage allowed.
            mem_limit (float): Max RAM usage percentage allowed.
            gpu_limit (float): Max GPU memory usage percentage allowed.
            disk_limit (float): Max Disk usage percentage allowed.
        """
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit
        self.gpu_limit = gpu_limit
        self.disk_limit = disk_limit

    def should_pause(self) -> bool:
        """
        Check if any resource usage exceeds the defined limits.

        Returns:
            bool: True if execution should pause, False otherwise.
        """
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
        if self._check_gpu_overload():
            return True

        # Disk Check (cwd)
        if self._check_disk_overload():
            return True

        return False

    def _check_gpu_overload(self) -> bool:
        """Check if GPU memory usage is too high."""
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
        return False

    def _check_disk_overload(self) -> bool:
        """Check if disk usage is too high."""
        total, used, free = shutil.disk_usage(".")
        disk_percent = (used / total) * 100.0
        if disk_percent > self.disk_limit:
            logger.warning(f"Disk Space Low: Used={disk_percent:.1f}%. Pausing...")
            return True
        return False
