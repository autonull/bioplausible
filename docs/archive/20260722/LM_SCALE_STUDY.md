# Language Modeling Scale Study

## Purpose

Comprehensive multi-hour experiment to understand when and why EqProp outperforms Backprop on language modeling.

## Running the Experiment

### Full Study (2-4 hours)
```bash
python experiments/lm_scale_study.py
```

### Quick Test (~15 mins)
```bash
python experiments/lm_scale_study.py --quick
```

## What It Tests

### Experiment 1: Dataset Size Scaling
Tests dataset sizes: 1K, 2K, 5K, 10K, 20K, 50K characters
- Compares Backprop vs 3 EqProp variants (looped_mlp, recurrent_core, full)
- Finds crossover point where EqProp begins to win
- Full config: 128 hidden, 3 layers, 64 seq_len, 30 epochs

### Experiment 2: Sequence Length Scaling
Tests sequence lengths: 16, 32, 64, 128 tokens
- Uses 10K char dataset
- Tests best EqProp variant vs Backprop
- Validates if advantage holds for longer sequences

## Expected Outcomes

Based on Track 37 and our ablation:
- **Small data (<5K):** Backprop wins
- **Medium data (10-20K):** EqProp begins to win
- **Large data (>20K):** EqProp wins decisively (2× better PPL)

## Output Files

All saved to `results/lm_scale_study/`:
- `results_final.json` - Complete results with all metrics
- `results_final.csv` - Spreadsheet-friendly format
- `summary.md` - Auto-generated summary with key findings
- `results_intermediate.json` - Saves progress (in case of crash)

## What You'll Learn

1. **Crossover point** - Exact dataset size where EqProp becomes better
2. **Best variant** - Which EqProp architecture performs best
3. **Scale dependence** - How advantage grows with data size
4. **Sequence effects** - If longer sequences favor EqProp

## Resource Requirements

- **GPU:** Highly recommended (20× faster)
- **Memory:** ~2GB GPU RAM for full study
- **Time:** 2-4 hours (full), 15 mins (quick)
- **Disk:** ~10MB for results

## Monitoring Progress

The script prints:
- Current experiment and dataset size
- Model being trained
- Intermediate perplexity results
- Comparison to Backprop baseline
- Saves intermediate results after each run

You can stop anytime (Ctrl+C) and results up to that point are saved in `results_intermediate.json`.
