import threading
import time
import unittest

from bioplausible.validation.core import TrackResult, Verifier


class TestParallelValidation(unittest.TestCase):
    def test_parallel_execution(self):
        """Test that parallel execution is faster than sequential for blocking tasks."""

        # Define dummy tracks that just sleep
        def track_slow_1(verifier):
            time.sleep(0.5)
            return TrackResult(
                track_id=1,
                name="Slow Track 1",
                status="pass",
                score=100,
                metrics={},
                evidence="N/A",
                time_seconds=0.5,
            )

        def track_slow_2(verifier):
            time.sleep(0.5)
            return TrackResult(
                track_id=2,
                name="Slow Track 2",
                status="pass",
                score=100,
                metrics={},
                evidence="N/A",
                time_seconds=0.5,
            )

        def track_slow_3(verifier):
            time.sleep(0.5)
            return TrackResult(
                track_id=3,
                name="Slow Track 3",
                status="pass",
                score=100,
                metrics={},
                evidence="N/A",
                time_seconds=0.5,
            )

        # Mock tracks in Verifier
        verifier = Verifier(quick_mode=True)
        verifier.tracks = {
            1: ("Slow Track 1", track_slow_1),
            2: ("Slow Track 2", track_slow_2),
            3: ("Slow Track 3", track_slow_3),
        }

        start_time = time.time()
        # Run in parallel
        # Note: We explicitly pass track_ids to avoid auto-running generic Track 0
        results = verifier.run_tracks(track_ids=[1, 2, 3], parallel=True)
        duration = time.time() - start_time

        print(f"Parallel duration: {duration:.2f}s")

        # Verify correctness
        self.assertEqual(len(results), 3)
        self.assertEqual(results[1].status, "pass")

        # Verify speedup (should be significantly less than 1.5s)
        # Using 1.0s as a safe upper bound for 0.5s parallel sleep
        self.assertLess(
            duration, 1.0, "Parallel execution should be faster than sequential sum"
        )

    def test_thread_safety(self):
        """Test that results are aggregated correctly without race conditions."""

        def track_fast(verifier):
            return TrackResult(
                track_id=1,
                name="Fast Track",
                status="pass",
                score=100,
                metrics={},
                evidence="N/A",
                time_seconds=0.0,
            )

        verifier = Verifier(quick_mode=True)
        # 20 identical tracks
        verifier.tracks = {i: (f"Track {i}", track_fast) for i in range(20)}

        results = verifier.run_tracks(track_ids=list(range(20)), parallel=True)

        self.assertEqual(len(results), 20)
        # Check notebook aggregation
        self.assertEqual(len(verifier.notebook.track_results), 20)


if __name__ == "__main__":
    unittest.main()
