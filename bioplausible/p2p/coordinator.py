"""
P2P Coordinator CLI.

Usage:
    python -m bioplausible.p2p.coordinator --port 8000
"""

import argparse
import logging
import time

from bioplausible.p2p.node import Coordinator


def main():
    parser = argparse.ArgumentParser(description="Bio-Plausible P2P Coordinator")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host interface to bind to"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("CoordinatorCLI")

    print(f"Starting Coordinator on {args.host}:{args.port}...")
    print("Press Ctrl+C to stop.")

    coord = Coordinator(host=args.host, port=args.port)
    coord.start()

    try:
        while True:
            time.sleep(5)
            # Log periodic status
            with coord.lock:
                node_count = len(coord.nodes)
                job_count = coord.jobs_completed
                queue_len = len(coord.job_queue)

            logger.info(
                f"Status: {node_count} nodes connected | {job_count} jobs completed | {queue_len} jobs in queue"
            )

    except KeyboardInterrupt:
        print("\nStopping Coordinator...")
        coord.stop()


if __name__ == "__main__":
    main()
