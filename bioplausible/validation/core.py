import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .notebook import TrackResult, VerificationNotebook
from .tracks import track_registry


class Verifier:
    """Complete verification suite for all research tracks."""

    def __init__(
        self,
        quick_mode: bool = False,
        intermediate_mode: bool = False,
        seed: int = 42,
        n_seeds_override: Optional[int] = None,
        export_data: bool = False,
        output_dir: Optional[str] = None,
    ):
        self.quick_mode = quick_mode
        self.intermediate_mode = intermediate_mode
        self.seed = seed
        self.export_data = export_data
        self.notebook = VerificationNotebook()

        # Set output directory (default to ./results if not specified)
        if output_dir is None:
            self.output_dir = Path("results")
        else:
            self.output_dir = Path(output_dir)

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Validation Mode Configuration
        # Quick:        ~2 min - mechanics only (smoke test)
        # Intermediate: ~1 hour - directional validation
        # Full:         ~4+ hr  - statistically significant claims
        if quick_mode:
            self.epochs = 5
            self.n_samples = 200
            self.n_seeds = 1
            self.evidence_level = "smoke"
        elif intermediate_mode:
            self.epochs = 50
            self.n_samples = 5000
            self.n_seeds = 3
            self.evidence_level = "intermediate"
        else:
            self.epochs = 100
            self.n_samples = 10000
            self.n_seeds = 5
            self.evidence_level = "full"

        # Allow override of seeds
        if n_seeds_override is not None:
            self.n_seeds = n_seeds_override

        self.data_records = []  # For CSV export
        self.current_seed = seed  # Track current seed for logging

        # Track definitions loaded from central registry
        self.tracks = {}
        self._load_tracks()

    def _load_tracks(self):
        """Load tracks from the registry into local format."""
        # Convert raw functions to (name, function) tuples expected by Verifier
        for tid, func in track_registry.ALL_TRACKS.items():
            # Attempt to extract nice name from docstring or function name
            name = func.__doc__.split("\n")[0] if func.__doc__ else func.__name__
            # Clean up name (remove "Track X: " prefix if present)
            if ":" in name and "Track" in name.split(":")[0]:
                name = name.split(":", 1)[1].strip()

            self.tracks[tid] = (name, func)

    def print_header(self):
        evidence_labels = {
            "smoke": "🧪 Smoke Test (mechanics only)",
            "intermediate": "📊 Intermediate (directional)",
            "full": "✅ Full Validation (statistically significant)",
        }
        mode_name = (
            "Quick"
            if self.quick_mode
            else ("Intermediate" if self.intermediate_mode else "Full")
        )
        mode_icon = (
            "⚡" if self.quick_mode else ("📊" if self.intermediate_mode else "🔬")
        )

        print("=" * 70)
        print("       TOREQPROP COMPREHENSIVE VERIFICATION SUITE")
        print("       Undeniable Evidence for All Research Claims")
        print("=" * 70)
        print(f"\\n📋 Configuration:")
        print(f"   Seed: {self.seed}")
        print(f"   Mode: {mode_icon} {mode_name}")
        print(f"   Evidence: {evidence_labels[self.evidence_level]}")
        print(f"   Epochs: {self.epochs}")
        print(f"   Samples: {self.n_samples}")
        print(f"   Seeds: {self.n_seeds}")
        print(f"   Tracks: {len(self.tracks)}")
        if self.export_data:
            print(f"   Export: Enabled (results/data.csv)")
        print("=" * 70)

    def record_metric(
        self, track_id: int, seed: int, step: int, metric_name: str, value: float
    ):
        """Record a data point for export."""
        if self.export_data:
            self.data_records.append(
                {
                    "track_id": track_id,
                    "seed": seed,
                    "step": step,
                    "metric": metric_name,
                    "value": value,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def evaluate_robustness(self, track_fn, n_seeds: int = 3) -> Dict:
        """Run a track logic multiple times with different seeds."""
        scores = []
        metrics_list = []

        # Determine number of seeds to run
        # override rules:
        # 1. if quick_mode -> 1
        # 2. if --seeds X provided -> X
        # 3. if default (3) -> use track-specific n_seeds (arg)

        run_count = self.n_seeds
        if self.n_seeds == 3 and not self.quick_mode:
            run_count = n_seeds

        print(f"      Running robustness check ({run_count} seeds)...")

        for i in range(run_count):
            seed = self.seed + i * 100
            self.current_seed = seed  # Update state for loggers

            # Temporarily set seed
            torch.manual_seed(seed)
            np.random.seed(seed)

            try:
                score, metrics = track_fn()
                scores.append(score)
                metrics_list.append(metrics)

                # Record aggregations for export
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        self.record_metric(
                            0, seed, 0, k, v
                        )  # Track ID 0 is generic/unknown here

            except Exception as e:
                print(f"        Seed {seed}: Failed ({e})")
                import traceback

                traceback.print_exc()
                scores.append(0)
                metrics_list.append({})

        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0

        # Calculate 95% Confidence Interval
        n = len(scores)
        se = std_score / np.sqrt(n) if n > 1 else 0.0
        ci_95 = 1.96 * se

        # Aggregate metrics with confidence intervals
        agg_metrics = {}
        if metrics_list:
            keys = metrics_list[0].keys()
            for k in keys:
                vals = [
                    m[k]
                    for m in metrics_list
                    if k in m and isinstance(m[k], (int, float))
                ]
                if vals:
                    m_mean = np.mean(vals)
                    m_std = np.std(vals) if len(vals) > 1 else 0.0
                    m_se = m_std / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
                    m_ci = 1.96 * m_se
                    agg_metrics[f"{k}_mean"] = m_mean
                    agg_metrics[f"{k}_std"] = m_std
                    agg_metrics[f"{k}_ci95"] = m_ci  # New: CI for each metric

        return {
            "mean_score": mean_score,
            "std_score": std_score,
            "ci_95": ci_95,  # New: 95% CI half-width
            "metrics": agg_metrics,
            "all_scores": scores,
        }

    def run_tracks(
        self, track_ids: Optional[List[int]] = None, parallel: bool = False
    ) -> Dict:
        """Run specified tracks (or all if None)."""
        self.print_header()
        self.notebook.add_header(self.seed)

        if track_ids is None:
            track_ids = list(self.tracks.keys())

        # Auto-run Track 0 (Framework Validation) in intermediate/full modes
        if (self.intermediate_mode or (not self.quick_mode)) and 0 not in track_ids:
            print("\n⚙️  Running Track 0 (Framework Validation) automatically...")
            track_ids = [0] + track_ids

        results = {}
        start_time = time.time()

        # Helper to run a single track
        def _execute_track(tid):
            if tid not in self.tracks:
                return tid, None, f"Unknown track: {tid}"
            name, method = self.tracks[tid]
            try:
                # Pass self (Verifier) to the track method
                result = method(self)
                return tid, result, None
            except Exception as e:
                import traceback

                return tid, None, f"Failed: {e}\n{traceback.format_exc()}"

        if parallel and len(track_ids) > 1:
            import concurrent.futures

            print(f"🚀 Running {len(track_ids)} tracks in parallel...")

            # Use ThreadPoolExecutor because tracks are largely I/O bound (PyTorch/CUDA releases GIL often)
            # or CPU bound but numpy releases GIL.
            # ProcessPoolExecutor would require pickling everything which is hard with Modules.
            max_workers = min(
                len(track_ids), 4
            )  # Cap at 4 to avoid resource contention
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_track = {
                    executor.submit(_execute_track, tid): tid for tid in track_ids
                }

                completed = 0
                for future in concurrent.futures.as_completed(future_to_track):
                    tid, result, error = future.result()

                    if error:
                        print(f"\n❌ Track {tid} error: {error}")
                    elif result:
                        results[tid] = result
                        # Note: notebook.add_track_result is not thread-safe by default,
                        # but we are collecting results sequentially here as they complete.
                        self.notebook.add_track_result(result)

                        icon = {
                            "pass": "✅",
                            "fail": "❌",
                            "partial": "⚠️",
                            "stub": "🔧",
                        }.get(result.status, "?")
                        name, _ = self.tracks[tid]
                        print(
                            f"\n{icon} Track {tid}: {name} - {result.status.upper()} ({result.score:.0f}/100)"
                        )

                    completed += 1
                    elapsed = time.time() - start_time
                    print(
                        f"   Progress: {completed}/{len(track_ids)} | Elapsed: {elapsed:.0f}s"
                    )

        else:
            # Sequential Execution
            for i, track_id in enumerate(track_ids):
                tid, result, error = _execute_track(track_id)

                if error:
                    print(f"\n❌ Track {track_id} failed: {error}")
                elif result:
                    results[track_id] = result
                    self.notebook.add_track_result(result)
                    icon = {
                        "pass": "✅",
                        "fail": "❌",
                        "partial": "⚠️",
                        "stub": "🔧",
                    }.get(result.status, "?")
                    name, _ = self.tracks[track_id]
                    print(
                        f"\n{icon} Track {track_id}: {name} - {result.status.upper()} ({result.score:.0f}/100)"
                    )

                # Progress
                elapsed = time.time() - start_time
                completed = i + 1
                remaining = len(track_ids) - completed
                if remaining > 0:
                    eta = (elapsed / completed) * remaining
                    print(
                        f"   Progress: {completed}/{len(track_ids)} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
                    )

        # Save
        total_time = time.time() - start_time

        output_path = self.output_dir / "verification_notebook.md"
        self.notebook.save(output_path)

        # Summary
        print("\n" + "=" * 70)
        print("🎉 VERIFICATION COMPLETE")
        print("=" * 70)
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"📓 Output: {output_path}")

        passed = sum(1 for r in results.values() if r.status == "pass")
        total = len(results)
        print(f"\n📊 Results: {passed}/{total} tracks passed")

        if self.export_data and self.data_records:
            import csv

            csv_path = self.output_dir / "data.csv"
            keys = self.data_records[0].keys()
            with open(csv_path, "w", newline="") as f:
                dict_writer = csv.DictWriter(f, keys)
                dict_writer.writeheader()
                dict_writer.writerows(self.data_records)
            print(f"💾 Data exported to: {csv_path}")

        return results

    def list_tracks(self):
        """Print all available tracks."""
        print("\nAvailable Verification Tracks:")
        print("-" * 60)
        for tid, (name, _) in self.tracks.items():
            print(f"  {tid:2d}. {name}")
        print("-" * 60)
