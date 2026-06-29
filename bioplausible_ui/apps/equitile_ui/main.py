import sys
import argparse
from PyQt6.QtWidgets import QApplication
from bioplausible_ui.apps.equitile_ui.window import EquiTileWindow

def parse_args():
    parser = argparse.ArgumentParser(description="Bio-Plausible Studio Live UI")
    parser.add_argument("--model", type=str, default="EquiTile", help="Model name to load")
    parser.add_argument("--task", type=str, default="lm", help="Task type (lm, vision)")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--tiles-per-layer", type=int, default=16, help="Hidden units/tiles per layer")
    parser.add_argument("--neurons-per-tile", type=int, default=32, help="Neurons per tile (or unit)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--dataset", type=str, default="Tiny Shakespeare", help="Dataset name")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate")

    args, unknown = parser.parse_known_args()
    return args

def main():
    """Main entry point for Bio-Plausible Studio."""
    args = parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName("Bio-Plausible Studio")
    app.setApplicationVersion("2.0.0")

    # Build config from args
    config = {
        "name": args.model,
        "task_type": args.task,
        "num_layers": args.num_layers,
        "tiles_per_layer": args.tiles_per_layer, # Maps to hidden_dim for standard models
        "neurons_per_tile": args.neurons_per_tile,
        "dataset_name": args.dataset,
        "batch_size": args.batch_size,
        "max_seq_len": args.seq_len,
        "learning_rate": args.lr
    }

    # Create and Show Window
    window = EquiTileWindow(initial_config=config)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
