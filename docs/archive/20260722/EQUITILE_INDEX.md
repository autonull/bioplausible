# EquiTile Documentation Index

## Core Documentation

| Document | Purpose | Audience |
|----------|---------|----------|
| [EQUITILE.md](EQUITILE.md) | **Complete architecture specification** | Researchers, developers |
| [EQUITILE_LM_DEMO.md](EQUITILE_LM_DEMO.md) | User guide and quickstart | Users |
| [EQUITILE_LM_ARCHITECTURE.md](EQUITILE_LM_ARCHITECTURE.md) | Technical deep-dive | Developers |
| [OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md) | Performance tuning | Advanced users |
| [BREAKTHROUGH_EVIDENCE.md](BREAKTHROUGH_EVIDENCE.md) | Evidence for performance claims | Researchers |
| [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) | NanoGPT comparison results | Researchers |
| [KERNEL_OPTIMIZATION_REPORT.md](KERNEL_OPTIMIZATION_REPORT.md) | Kernel optimization analysis | Developers |
| [PERPLEXITY_INVESTIGATION.md](PERPLEXITY_INVESTIGATION.md) | Ablation study results | Researchers |

---

## Quick Reference

### For Researchers

Start here: **[EQUITILE.md](EQUITILE.md)**

This document provides:
- Complete architectural specification
- Novelty assessment vs. prior art
- Research questions enabled
- Implementation details
- Comparison tables

### For Users

Start here: **[EQUITILE_LM_DEMO.md](EQUITILE_LM_DEMO.md)**

This document provides:
- Quick start guide
- Usage examples
- Configuration options
- Troubleshooting

### For Developers

Start here: **[EQUITILE_LM_ARCHITECTURE.md](EQUITILE_LM_ARCHITECTURE.md)**

This document provides:
- Component-level details
- Mathematical formulations
- Implementation notes
- Extension points

### For Performance Tuning

Start here: **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)**

This document provides:
- All optimization options
- Configuration recommendations
- Performance benchmarks
- Hardware-specific guidance

---

## API Reference

### Core Classes

```python
from bioplausible.models.equitile.lm_demo import (
    FastLMEquiTile,  # Main model
    FastLMConfig,  # Configuration
    LMTrainer,  # Training loop
    TrainingConfig,  # Training configuration
)
```

### Tokenizers

```python
from bioplausible.models.equitile.lm_demo import (
    CharacterTokenizer,  # Character-level
    BPETokenizer,  # BPE (GPT-2 style)
    WordPieceTokenizer,  # WordPiece (BERT style)
)
```

### Benchmarks

```python
from bioplausible.models.equitile.benchmarks import (
    compare_nanoGPT,  # NanoGPT comparison
    run_rigorous_benchmark,  # Statistical benchmarking
    analyze_parameter_efficiency,
    analyze_flop_efficiency,
)
```

### Utilities

```python
from bioplausible.models.equitile.utils import (
    ReproducibilityTracker,  # Experiment tracking
    set_reproducible_mode,  # Seed control
)
```

---

## Command Line Interface

### Training

```bash
# Basic training
python -m bioplausible.models.equitile.lm_demo.demo \
    --task shakespeare \
    --epochs 10

# With all optimizations
python -m bioplausible.models.equitile.lm_demo.demo \
    --task shakespeare \
    --attention-type flash \
    --sliding-window 256 \
    --num-kv-heads 2 \
    --use-compile \
    --compile-mode max-autotune \
    --epochs 10
```

### Benchmarking

```bash
# Quick comparison
python -c "from bioplausible.models.equitile.benchmarks import compare_nanoGPT; compare_nanoGPT()"

# Rigorous benchmark (5 runs, 95% CI)
python -m bioplausible.models.equitile.benchmarks.rigorous
```

### Validation

```bash
# Full validation suite
python -m bioplausible.models.equitile.validate

# Quick smoke test
python -m bioplausible.models.equitile.validate --quick
```

---

## File Structure

```
bioplausible/models/equitile/
├── lm_demo/
│   ├── __init__.py
│   ├── fast_lm.py           # Core architecture
│   ├── data.py              # Character tokenizer
│   ├── data_advanced.py     # BPE/WordPiece tokenizers
│   ├── training.py          # Training loop
│   ├── demo.py              # CLI interface
│   ├── profiling.py         # Memory/bandwidth profiling
│   ├── ablation_study.py    # Ablation framework
│   └── train_tinystories.py # Large-scale training
├── benchmarks/
│   ├── __init__.py
│   ├── rigorous.py          # Statistical benchmarking
│   ├── compare_nanoGPT.py   # NanoGPT comparison
│   └── efficiency_analysis.py
├── utils/
│   ├── __init__.py
│   └── reproducibility.py   # Reproducibility framework
├── kernels.py               # Kernel utilities
└── validate.py              # Automated validation
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial release with full documentation |

---

## Getting Help

- **Architecture questions:** [docs/EQUITILE.md](docs/EQUITILE.md)
- **Usage questions:** [docs/EQUITILE_LM_DEMO.md](docs/EQUITILE_LM_DEMO.md)
- **Performance issues:** [docs/OPTIMIZATION_GUIDE.md](docs/OPTIMIZATION_GUIDE.md)
- **Bug reports:** GitHub Issues
- **Research collaboration:** GitHub Discussions

---

## Citation

```bibtex
@software{equitile2024,
  title = {EquiTile: Tile-Based Local Learning for Language Modeling},
  author = {BioPlausible Team},
  year = {2024},
  url = {https://github.com/bioplausible/equitile},
}
```
