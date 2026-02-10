"""
Distributed Hash Table (DHT) Node for True P2P Mode.
Uses Kademlia protocol via the 'kademlia' python package.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from kademlia.network import Server
except ImportError:
    Server = None

logger = logging.getLogger("DHTNode")


class DHTNode:
    """
    A Kademlia DHT Node running in a separate thread.
    """

    def __init__(self, port: int = 8468, bootstrap_nodes: List[tuple] = None):
        self.port = port
        self.bootstrap_nodes = bootstrap_nodes or []
        self.loop = None
        self.server = None
        self.thread = None
        self.running = False
        self._ready_event = threading.Event()

        if Server is None:
            logger.error("Kademlia not installed. DHT disabled.")

    def start(self):
        """Start the DHT node in a background thread."""
        if self.running or Server is None:
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        # Wait for loop to be ready
        if not self._ready_event.wait(timeout=5):
            logger.error("DHT start timed out")

    def stop(self):
        """Stop the DHT node."""
        if not self.running:
            return
        self.running = False
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("DHT stopped")

    def _run_loop(self):
        """Internal asyncio loop runner."""
        logger.info(f"DHT Node starting on port {self.port}...")

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.server = Server()
        self.loop.run_until_complete(self.server.listen(self.port))

        if self.bootstrap_nodes:
            logger.info(f"Bootstrapping from {self.bootstrap_nodes}...")
            try:
                self.loop.run_until_complete(
                    self.server.bootstrap(self.bootstrap_nodes)
                )
            except Exception as e:
                logger.warning(f"Bootstrap failed: {e}")

        self._ready_event.set()
        logger.info("DHT Node Ready")

        try:
            self.loop.run_forever()
        except Exception as e:
            logger.error(f"DHT Loop Error: {e}")
        finally:
            self.server.stop()
            self.loop.close()

    def get(self, key: str) -> Optional[Any]:
        """Synchronous get wrapper."""
        if not self.running:
            return None
        try:
            future = asyncio.run_coroutine_threadsafe(self.server.get(key), self.loop)
            result = future.result(timeout=10)
            if result:
                return json.loads(result)
            return None
        except Exception as e:
            logger.debug(f"DHT Get Error ({key}): {e}")
            return None

    def set(self, key: str, value: Any):
        """Synchronous set wrapper."""
        if not self.running:
            return
        try:
            val_str = json.dumps(value)
            future = asyncio.run_coroutine_threadsafe(
                self.server.set(key, val_str), self.loop
            )
            future.result(timeout=5)
        except Exception as e:
            logger.error(f"DHT Set Error ({key}): {e}")

    def get_best_model(self, task: str) -> Optional[Dict]:
        """Retrieve the best model for a task."""
        key = f"best_model_{task}"
        return self.get(key)

    def publish_best_model(self, task: str, config: Dict, score: float):
        """Publish a new best model."""
        key = f"best_model_{task}"

        # Optimistic concurrency check (simple)
        current = self.get(key)
        if current and current.get("score", -999) >= score:
            return  # Someone else is better already

        data = {
            "config": config,
            "score": score,
            "timestamp": time.time(),
            "author": "anonymous",  # or self.id
        }
        self.set(key, data)
        logger.info(f"Published new best model for {task} (Score: {score:.4f})")

    def get_known_peers(self) -> List[Dict]:
        """
        Get list of known peers from the routing table.
        Returns [{'id': str, 'ip': str, 'port': int}, ...]
        """
        if not self.running or not self.server:
            return []

        peers = []
        try:
            # Safely access routing table. Accessing cross-thread is risky but reading buckets is generally okay-ish
            # or we should schedule it on the loop.

            async def _get_peers():
                # kademlia.protocol.RoutingTable
                # Accessing buckets
                p_list = []
                for bucket in self.server.protocol.router.buckets:
                    for node in bucket.get_nodes():
                        p_list.append(
                            {"id": node.id.hex(), "ip": node.ip, "port": node.port}
                        )
                return p_list

            future = asyncio.run_coroutine_threadsafe(_get_peers(), self.loop)
            peers = future.result(timeout=2)
        except Exception as e:
            logger.debug(f"Error getting peers: {e}")

        return peers
