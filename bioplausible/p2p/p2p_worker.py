"""
P2P Worker CLI

Entry point for running a headless P2P NAS worker.
"""

import argparse
import logging
import signal
import sys
import time

from bioplausible.p2p.evolution import P2PEvolution

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("P2PWorker")


def signal_handler(sig, frame):
    logger.info("Shutdown signal received.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="BioPlausible P2P Worker")
    parser.add_argument(
        "--bootstrap-ip", type=str, default=None, help="IP of a known bootstrap node"
    )
    parser.add_argument(
        "--bootstrap-port", type=int, default=8468, help="Port of bootstrap node"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="shakespeare",
        help="Task to run (shakespeare, tiny_shakespeare, mnist, cifar10, cartpole)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "deep"],
        help="Discovery mode",
    )
    parser.add_argument(
        "--max-hidden", type=int, default=None, help="Constraint: Max hidden dim"
    )
    parser.add_argument(
        "--max-layers", type=int, default=None, help="Constraint: Max layers"
    )

    args = parser.parse_args()

    constraints = {}
    if args.max_hidden:
        constraints["max_hidden"] = args.max_hidden
    if args.max_layers:
        constraints["max_layers"] = args.max_layers

    logger.info(f"Starting P2P Worker (Mode: {args.mode}, Task: {args.task})")

    worker = P2PEvolution(
        bootstrap_ip=args.bootstrap_ip,
        bootstrap_port=args.bootstrap_port,
        discovery_mode=args.mode,
        constraints=constraints,
        task=args.task,
    )

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        worker.start(auto_nice=True)

        # Keep main thread alive
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping worker...")
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
