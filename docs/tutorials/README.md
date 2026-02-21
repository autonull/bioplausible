# EquiTile Tutorial Notebooks

This directory contains Jupyter notebook tutorials for common EquiTile use cases.

## Available Tutorials

### 1. Getting Started (`01_getting_started.ipynb`)
Introduction to EquiTile basics:
- Creating your first model
- Training on a simple dataset
- Evaluating performance

### 2. Builder Pattern (`02_builder_pattern.ipynb`)
Using the fluent builder API:
- Production configurations
- Research configurations
- Custom architectures

### 3. Multi-GPU Training (`03_multigpu_training.ipynb`)
Scaling to multiple GPUs:
- Single-process multi-GPU
- Multi-process distributed training
- Performance tuning

### 4. Async Execution (`04_async_execution.ipynb`)
Asynchronous tile processing:
- Configuring async execution
- Performance benefits
- When to use async

### 5. Profiling and Benchmarking (`05_profiling.ipynb`)
Performance analysis:
- Timing analysis
- Memory profiling
- Benchmark suites

### 6. Research Experiments (`06_research_experiments.ipynb`)
Research utilities:
- Experiment tracking
- Metric collection
- Ablation studies
- Visualization

### 7. Dynamic Architectures (`07_dynamic_architectures.ipynb`)
Tile growth and pruning:
- Configuring dynamics
- Monitoring tile changes
- Use cases

### 8. Enhanced EP (`08_enhanced_ep.ipynb`)
Enhanced Equilibrium Propagation:
- Layer normalization
- Curriculum learning
- Weight initialization

## Running the Tutorials

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter server
cd docs/tutorials
jupyter notebook

# Or run a specific notebook
jupyter execute 01_getting_started.ipynb
```

## Prerequisites

All tutorials require:
- Python 3.9+
- PyTorch 2.0+
- EquiTile package
- matplotlib (for visualization tutorials)
- jupyter (for notebook execution)

```bash
pip install torch torchvision matplotlib jupyter
```

## Tutorial Structure

Each tutorial follows this structure:

1. **Overview**: What you'll learn
2. **Setup**: Imports and configuration
3. **Examples**: Step-by-step code examples
4. **Exercises**: Try-it-yourself sections
5. **Summary**: Key takeaways

## Contributing

To contribute new tutorials:

1. Create notebook in this directory
2. Follow the tutorial structure
3. Include clear explanations
4. Add exercises for readers
5. Submit PR with notebook

## Support

For questions about tutorials:
- Check the API documentation
- Review the migration guide
- Open a GitHub issue
