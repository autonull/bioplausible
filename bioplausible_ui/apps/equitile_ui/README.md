# EquiTile Live UI Demo

Interactive visualization and training interface for the EquiTile language model.

**Note: This UI is now integrated into the main Bioplausible Studio.**

## Features

- **Visualizer**: Real-time view of tile activity and importance. Tiles glow based on activity and are colored by importance.
- **Inspector**: Detailed view of individual tile states, including neuron activation heatmaps.
- **Controls**: Live tuning of hyperparameters (Learning Rate, Sparsity, etc.) and architecture reconfiguration.
- **Diagnostics**: Real-time plots for Loss, Accuracy, Throughput, and Sparsity.
- **Live Generation**: Watch the model generate text as it trains.

## Running the UI

Use the provided shell script in the repository root to launch the unified Studio:

```bash
./run_ui.sh [options]
```

This will launch the Bioplausible Studio. Select **"🧠 EquiTile Demo"** from the sidebar to access this interface.

### Options (passed to Studio)

- `--num-layers INT`: Number of transformer layers (default: 4).
- `--tiles-per-layer INT`: Tiles per layer (default: 16).
- `--neurons-per-tile INT`: Neurons per tile (default: 32).
- `--dataset STR`: Dataset name (default: "Tiny Shakespeare").

### Example

```bash
./run_ui.sh --num-layers 6 --tiles-per-layer 64
```

## Dependencies

Requires `PyQt6` and `pyqtgraph`. See `requirements.txt` in the root directory.
