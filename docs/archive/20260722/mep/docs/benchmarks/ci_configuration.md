# CI Benchmark Configuration

This directory contains CI benchmark configurations for automated performance regression testing.

## Quick Start

```bash
# Run performance regression tests (fast, ~30 seconds)
pytest tests/regression/test_performance_baseline.py -v

# Run full benchmark suite (slower, ~5 minutes)
python -m mep.benchmarks.tuned_compare --epochs 3 --model mlp_small --output ci_results.json

# Check results against baselines
python .github/scripts/check_benchmarks.py ci_results.json
```

## Benchmark Tiers

### Tier 1: Quick Sanity (CI on every PR)
- **Time:** ~30 seconds
- **Tests:** `test_performance_baseline.py`
- **Thresholds:**
  - XOR 100 steps: ≥75% accuracy
  - MNIST 1 epoch: ≥80% accuracy
  - Configuration defaults: Verified

### Tier 2: Full Regression (Nightly)
- **Time:** ~5 minutes
- **Tests:** Full benchmark suite
- **Thresholds:**
  - MNIST 3 epochs: ≥88% accuracy
  - No performance degradation >2%

### Tier 3: Extended Validation (Weekly)
- **Time:** ~30 minutes
- **Tests:** Extended benchmarks, multiple seeds
- **Thresholds:**
  - MNIST 10 epochs: ≥93% accuracy
  - Statistical significance testing

## GitHub Actions Workflow

```yaml
# .github/workflows/benchmarks.yml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  performance-regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest torchvision
      
      - name: Run Tier 1 benchmarks
        run: |
          pytest tests/regression/test_performance_baseline.py -v
      
      - name: Run Tier 2 benchmarks (nightly)
        if: github.event_name == 'schedule'
        run: |
          python -m mep.benchmarks.tuned_compare --epochs 3 --output results.json
          python .github/scripts/check_benchmarks.py results.json
```

## Baseline Results File

Store baseline results in `.github/benchmarks/baseline_results.json`:

```json
{
  "mnist_3epoch_mlp_small": {
    "smep": {"accuracy": 0.914, "tolerance": 0.03},
    "sgd": {"accuracy": 0.910, "tolerance": 0.03},
    "adam": {"accuracy": 0.902, "tolerance": 0.03}
  },
  "mnist_10epoch_10k": {
    "smep": {"accuracy": 0.9537, "tolerance": 0.02},
    "adam": {"accuracy": 0.9575, "tolerance": 0.02}
  },
  "xor_100step": {
    "smep": {"accuracy": 1.0, "tolerance": 0.25}
  }
}
```

## Benchmark Check Script

`.github/scripts/check_benchmarks.py`:

```python
#!/usr/bin/env python3
"""Check benchmark results against baselines."""

import json
import sys

def check_benchmarks(results_file, baseline_file):
    with open(results_file) as f:
        results = json.load(f)
    
    with open(baseline_file) as f:
        baselines = json.load(f)
    
    failures = []
    for benchmark, optimizers in results.items():
        if benchmark not in baselines:
            continue
        
        for optimizer, data in optimizers.items():
            if optimizer not in baselines[benchmark]:
                continue
            
            baseline = baselines[benchmark][optimizer]
            accuracy = data.get('val_acc', 0)
            threshold = baseline['accuracy'] - baseline['tolerance']
            
            if accuracy < threshold:
                failures.append(
                    f"{benchmark}/{optimizer}: {accuracy:.2%} < {threshold:.2%}"
                )
    
    if failures:
        print("❌ Performance regression detected:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("✅ All benchmarks passed")
        sys.exit(0)

if __name__ == '__main__':
    check_benchmarks(sys.argv[1], sys.argv[2])
```

## Performance Dashboard

Generate a performance dashboard with:

```bash
python .github/scripts/generate_dashboard.py
```

This creates an HTML report showing:
- Current vs baseline accuracy
- Historical trends
- Speed comparisons
- Memory usage

## Adding New Benchmarks

1. Add benchmark to `mep/benchmarks/`
2. Add baseline to `.github/benchmarks/baseline_results.json`
3. Update `check_benchmarks.py` if needed
4. Run full test suite to verify

## Troubleshooting

### Benchmark fails intermittently
- Increase tolerance in baseline configuration
- Use multiple seeds and average results
- Check for non-deterministic operations

### Benchmark too slow
- Reduce dataset size for CI
- Use `--subset-train` and `--subset-test` flags
- Consider moving to Tier 3 (weekly)

### False positives
- Verify baseline is still achievable
- Check for environmental differences
- Consider hardware-specific variations
