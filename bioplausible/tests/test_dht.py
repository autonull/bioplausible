import logging
import time
import unittest

from bioplausible.p2p.dht import DHTNode

# Configure logging to see output during tests
logging.basicConfig(level=logging.DEBUG)


class TestDHT(unittest.TestCase):
    def setUp(self):
        # Only run if kademlia is installed
        try:
            import kademlia  # noqa: F401
        except ImportError:
            self.skipTest("kademlia not installed")

    def test_dht_connectivity(self):
        # Create two nodes
        node1 = DHTNode(port=8470)
        node2 = DHTNode(port=8471, bootstrap_nodes=[("127.0.0.1", 8470)])

        try:
            node1.start()
            time.sleep(2)  # Wait for node1 to be ready

            node2.start()
            time.sleep(2)  # Wait for bootstrap

            # Set on Node 1
            node1.set("test_key", {"data": "hello"})
            time.sleep(1)

            # Get on Node 2
            val = node2.get("test_key")

            self.assertIsNotNone(val)
            self.assertEqual(val.get("data"), "hello")

        finally:
            node2.stop()
            node1.stop()

    def test_best_model_propagation(self):
        node1 = DHTNode(port=8472)
        node2 = DHTNode(port=8473, bootstrap_nodes=[("127.0.0.1", 8472)])

        try:
            node1.start()
            time.sleep(2)
            node2.start()
            time.sleep(2)

            # Publish best model on Node 1
            config = {"model": "TestModel", "lr": 0.01}
            node1.publish_best_model("test_task", config, 0.95)
            time.sleep(1)

            # Retrieve on Node 2
            best = node2.get_best_model("test_task")
            self.assertIsNotNone(best)
            self.assertEqual(best["score"], 0.95)
            self.assertEqual(best["config"]["model"], "TestModel")

            # Try to publish worse model on Node 2 (should be ignored by
            # Node 1 logic if we implemented robust checks,
            # but currently DHT is simple KV, so it overwrites.
            # The implementation of publish_best_model does an optimistic check
            # *locally* before setting. Verify Node 2 checks locally before
            # overwriting.

            # Node 2 sees 0.95. Try to publish 0.90.
            node2.publish_best_model("test_task", config, 0.90)
            time.sleep(1)

            # Verify it is still 0.95
            # Note: The 'publish_best_model' logic:
            # current = self.get(key)
            # if current and current >= score: return

            best_after = node2.get_best_model("test_task")
            self.assertEqual(best_after["score"], 0.95)

        finally:
            node2.stop()
            node1.stop()


if __name__ == "__main__":
    unittest.main()
