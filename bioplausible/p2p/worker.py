"""
Headless P2P Worker Entry Point.
Usage: python -m bioplausible.p2p.worker --join <COORDINATOR_URL> [--client-id <ID>]
"""

import argparse
import logging
import time

from bioplausible.p2p.node import Worker


def main():
    parser = argparse.ArgumentParser(description="Bio-Plausible P2P Worker")
    parser.add_argument(
        "--join", type=str, required=True, help="URL of the Coordinator node"
    )
    parser.add_argument(
        "--client-id", type=str, default=None, help="Optional Client ID"
    )

    args = parser.parse_args()

    # Setup logging to console
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print(f"Starting Worker connecting to {args.join}...")

    worker = Worker(args.join, client_id=args.client_id)

    # We need to block the main thread since worker uses a daemon thread loop
    # worker.start_loop() starts a thread.
    worker.start_loop()

    try:
        while True:
            time.sleep(1)
            # Maybe print status updates periodically
    except KeyboardInterrupt:
        print("\nStopping worker...")
        worker.stop()


if __name__ == "__main__":
    main()
