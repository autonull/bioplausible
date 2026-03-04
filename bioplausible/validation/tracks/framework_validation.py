"""
Track 0: Framework Validation (Infrastructure Self-Test)

Validates that the statistical rigor infrastructure works correctly.
This track tests the validation framework itself rather than EqProp models.

Runs automatically in intermediate and full modes to ensure framework integrity.
"""

import sys
import time
from pathlib import Path

import numpy as np

root_path = Path(__file__).parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from bioplausible.validation.notebook import TrackResult
from bioplausible.validation.utils import (classify_evidence_level,
                                           compute_cohens_d,
                                           compute_reproducibility_hash,
                                           format_statistical_comparison,
                                           interpret_effect_size,
                                           interpret_pvalue, paired_ttest)


def track_0_framework_validation(verifier) -> TrackResult:
    """
    Track 0: Framework Validation

    Self-test of statistical rigor infrastructure.
    Ensures all statistical functions work correctly.
    """
    print("\n" + "=" * 60)
    print("TRACK 0: Framework Validation (Infrastructure Self-Test)")
    print("=" * 60)

    start = time.time()
    tests_passed = []
    failures = []

    # Test 1: Cohen's d
    print("\n[0.1] Testing Cohen's d calculation...")
    try:
        # Negligible effect
        d1 = compute_cohens_d([1.0, 1.1, 0.9], [1.0, 1.0, 1.0])
        assert abs(d1) < 0.2, f"Negligible test failed: d={d1}"

        # Large effect
        d2 = compute_cohens_d([1.0, 1.1, 1.2], [0.0, 0.1, 0.2])
        assert abs(d2) > 0.8, f"Large effect test failed: d={d2}"

        tests_passed.append("Cohen's d")
        print("      ✅ Cohen's d works correctly")
    except AssertionError as e:
        failures.append(f"Cohen's d: {e}")
        print(f"      ❌ Cohen's d failed: {e}")

    # Test 2: T-tests
    print("\n[0.2] Testing statistical significance...")
    try:
        # Identical groups -> p=1.0
        t, p = paired_ttest([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        assert p == 1.0, f"Identical groups test failed: p={p}"

        # Different groups -> p<0.001
        np.random.seed(42)
        group1 = np.random.normal(10.0, 1.0, 20).tolist()
        group2 = np.random.normal(5.0, 1.0, 20).tolist()
        t, p = paired_ttest(group1, group2)
        assert p < 0.001, f"Significant difference test failed: p={p}"

        tests_passed.append("T-tests")
        print("      ✅ T-tests work correctly")
    except AssertionError as e:
        failures.append(f"T-tests: {e}")
        print(f"      ❌ T-tests failed: {e}")

    # Test 3: Evidence classification
    print("\n[0.3] Testing evidence classification...")
    try:
        level_smoke = classify_evidence_level(100, 1, 5)
        assert level_smoke == "smoke", f"Smoke level failed: {level_smoke}"

        level_dir = classify_evidence_level(500, 3, 15)
        assert level_dir == "directional", f"Directional level failed: {level_dir}"

        level_conc = classify_evidence_level(5000, 5, 50)
        assert level_conc == "conclusive", f"Conclusive level failed: {level_conc}"

        tests_passed.append("Evidence classification")
        print("      ✅ Evidence classification works correctly")
    except AssertionError as e:
        failures.append(f"Evidence classification: {e}")
        print(f"      ❌ Evidence classification failed: {e}")

    # Test 4: Interpretations
    print("\n[0.4] Testing human-readable interpretations...")
    try:
        interp_neg = interpret_effect_size(0.1)
        assert "negligible" in interp_neg, "Negligible interpretation failed"

        interp_large = interpret_effect_size(1.5)
        assert "large" in interp_large.lower(), "Large interpretation failed"

        interp_sig = interpret_pvalue(0.0001)
        assert "highly significant" in interp_sig, "Significance interpretation failed"

        tests_passed.append("Interpretations")
        print("      ✅ Interpretations work correctly")
    except AssertionError as e:
        failures.append(f"Interpretations: {e}")
        print(f"      ❌ Interpretations failed: {e}")

    # Test 5: Statistical comparison formatting
    print("\n[0.5] Testing statistical comparison formatting...")
    try:
        result = format_statistical_comparison(
            "Method A", [0.95, 0.96, 0.94], "Method B", [0.50, 0.52, 0.51], "accuracy"
        )
        assert "Method A" in result and "Method B" in result, "Missing method names"
        assert "Effect Size" in result, "Missing effect size"
        assert "Significance" in result, "Missing significance"

        tests_passed.append("Statistical formatting")
        print("      ✅ Statistical formatting works correctly")
    except AssertionError as e:
        failures.append(f"Statistical formatting: {e}")
        print(f"      ❌ Statistical formatting failed: {e}")

    # Test 6: Reproducibility hash
    print("\n[0.6] Testing reproducibility hash...")
    try:
        hash1 = compute_reproducibility_hash(42, 1000, 50, "LoopedMLP")
        hash2 = compute_reproducibility_hash(42, 1000, 50, "LoopedMLP")
        hash3 = compute_reproducibility_hash(43, 1000, 50, "LoopedMLP")

        assert hash1 == hash2, "Same params should produce same hash"
        assert hash1 != hash3, "Different params should produce different hash"
        assert len(hash1) == 8, f"Hash should be 8 chars, got {len(hash1)}"

        tests_passed.append("Reproducibility hash")
        print("      ✅ Reproducibility hash works correctly")
    except AssertionError as e:
        failures.append(f"Reproducibility hash: {e}")
        print(f"      ❌ Reproducibility hash failed: {e}")

    # Scoring
    total_tests = 6
    passed_count = len(tests_passed)
    score = (passed_count / total_tests) * 100

    if passed_count == total_tests:
        status = "pass"
    elif passed_count >= total_tests * 0.5:
        status = "partial"
    else:
        status = "fail"

    # Evidence
    evidence = f"""
**Framework Self-Test Results**

| Test | Status |
|------|--------|
| Cohen's d calculation | {'✅' if 'Cohen\'s d' in tests_passed else '❌'} |
| Statistical significance (t-tests) | {'✅' if 'T-tests' in tests_passed else '❌'} |
| Evidence classification | {'✅' if 'Evidence classification' in tests_passed else '❌'} |
| Human-readable interpretations | {'✅' if 'Interpretations' in tests_passed else '❌'} |
| Statistical comparison formatting | {'✅' if 'Statistical formatting' in tests_passed else '❌'} |
| Reproducibility hashing | {'✅' if 'Reproducibility hash' in tests_passed else '❌'} |

**Tests Passed**: {passed_count}/{total_tests}

**Purpose**: This track validates the validation framework itself, ensuring all statistical
functions work correctly before running model validation tracks.
"""

    if failures:
        evidence += "\n**Failures**:\n"
        for failure in failures:
            evidence += f"- {failure}\n"

    return TrackResult(
        track_id=0,
        name="Framework Validation",
        status=status,
        score=score,
        metrics={"tests_passed": passed_count, "total_tests": total_tests},
        evidence=evidence,
        time_seconds=time.time() - start,
        improvements=(
            [] if status == "pass" else ["Fix failing statistical function tests"]
        ),
        evidence_level="smoke",  # Always smoke - this is infrastructure testing
        limitations=["Framework-level test only, does not validate EqProp models"],
    )


# Export for registration
def register_track():
    return {0: track_0_framework_validation}
