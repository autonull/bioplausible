"""
Central Registry for Validation Tracks.

Aggregates all track definitions from various modules into a single lookup dictionary.
This allows the Verifier to easily access all available experiments.
"""

from collections.abc import Callable

# Import all KEPT track modules (Phase 4 deleted: advanced_tracks, analysis_tracks,
# engine_validation_tracks, enhanced_validation_tracks, framework_validation,
# honest_tradeoff, new_tracks, rapid_validation, special_tracks)
from . import (
    application_tracks,
    architecture_comparison,
    core_tracks,
    hardware_tracks,
    nebc_tracks,
    negative_results,
    research_tracks,
    scaling_tracks,
    signal_tracks,
    tradeoff_tracks,
)

# Initialize registry
ALL_TRACKS: dict[int, Callable] = {}


def register_tracks_from_module(module):
    """Register all functions starting with 'track_' or in a TRACKS dict."""
    try:
        # Check for explicit registry dict
        found_dict = False
        if hasattr(module, "TRACKS"):
            ALL_TRACKS.update(module.TRACKS)
            found_dict = True

        if hasattr(module, "NEW_TRACKS"):
            ALL_TRACKS.update(module.NEW_TRACKS)
            found_dict = True

        # If no explicit dictionary is found, scan for functions starting with 'track_'
        # Format expected: track_ID_name(verifier)
        if not found_dict:
            import inspect

            for name, func in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("track_"):
                    try:
                        # Parse ID from name: track_42_something -> 42
                        parts = name.split("_")
                        if len(parts) >= 2 and parts[1].isdigit():
                            track_id = int(parts[1])
                            ALL_TRACKS[track_id] = func
                    except Exception:
                        pass
    except Exception as e:
        print(f"Warning: Failed to register tracks from module {module.__name__}: {e}")


# Register all tracks (only kept modules)
# 1. Core & Standard
register_tracks_from_module(core_tracks)

# 2. Scaling
register_tracks_from_module(scaling_tracks)

# 3. Hardware
register_tracks_from_module(hardware_tracks)

# 4. Applications
register_tracks_from_module(application_tracks)

# 5. NEBC / Research / Negative Results
register_tracks_from_module(nebc_tracks)
register_tracks_from_module(negative_results)
register_tracks_from_module(architecture_comparison)
register_tracks_from_module(tradeoff_tracks)
register_tracks_from_module(research_tracks)

# 6. Signal Propagation
register_tracks_from_module(signal_tracks)


def get_track(track_id: int) -> Callable:
    """Get a track function by ID."""
    if track_id not in ALL_TRACKS:
        raise ValueError(f"Track {track_id} not found in registry.")
    return ALL_TRACKS[track_id]


def get_track_metadata(track_id: int) -> dict[str, str]:
    """Get metadata for a track (name, description)."""
    func = get_track(track_id)
    name = func.__name__
    description = getattr(func, "description", "No description available.")
    category = getattr(func, "category", "General")

    # Clean up name
    if name.startswith("track_"):
        parts = name.split("_")
        if len(parts) >= 3:
            name = " ".join(parts[2:]).title()

    return {
        "id": track_id,
        "name": name,
        "description": description,
        "category": category,
        "func_name": func.__name__,
    }


def list_tracks() -> dict[int, str]:
    """Return dictionary of track ID -> function name."""
    return {tid: func.__name__ for tid, func in sorted(ALL_TRACKS.items())}
