import sys
import argparse
from PyQt6.QtWidgets import QApplication
from bioplausible_ui.apps.equitile_ui.window import EquiTileWindow

def parse_args():
    parser = argparse.ArgumentParser(description="EquiTile Live UI Demo")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--tiles-per-layer", type=int, default=16, help="Number of tiles per layer")
    parser.add_argument("--neurons-per-tile", type=int, default=32, help="Neurons per tile")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--dataset", type=str, default="Tiny Shakespeare", help="Dataset name")
    # Use parse_known_args to allow standard Qt arguments (e.g. -platform, -style)
    # to pass through to QApplication without argparse raising an error.
    args, unknown = parser.parse_known_args()
    return args

def main():
    """Main entry point for EquiTile UI."""
    args = parse_args()

    # Qt Arguments need to be handled carefully if mixing with argparse
    # We pass sys.argv to QApplication but argparse has already handled its part
    app = QApplication(sys.argv)

    # Set Application Metadata
    app.setApplicationName("EquiTile Demo")
    app.setApplicationVersion("1.0.0")

    # Build config from args
    config = {
        "num_layers": args.num_layers,
        "tiles_per_layer": args.tiles_per_layer,
        "neurons_per_tile": args.neurons_per_tile,
        "dataset_name": args.dataset,
        "batch_size": args.batch_size,
        "max_seq_len": args.seq_len
    }

    # Create and Show Window
    window = EquiTileWindow(initial_config=config)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
