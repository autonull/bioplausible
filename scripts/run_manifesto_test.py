import os
from datetime import datetime
from bioplausible.scientist.failure_tracker import FailureTracker, FailureRecord, FailureCategory
from bioplausible.analysis.failure_manifesto import FailureManifestoGenerator

def main():
    print("Testing Phase 1.5: Failure Manifesto Generation...")
    
    db_path = "results/test_failures.db"
    
    # Clean up old test db
    if os.path.exists(db_path):
        os.remove(db_path)
    
    tracker = FailureTracker(db_path)
    
    # Mock some failures
    failures = [
        # EqProp divergence
        FailureRecord(
            timestamp=datetime.now().isoformat(), model_name="eqprop_mlp", task_name="mnist",
            tier="baseline", trial_id=1, failure_type=FailureCategory.SETTLING_DIVERGENCE.value,
            failure_epoch=5, failure_batch=100, config={"lr": 0.05}, last_metrics={}, stack_trace="State failed to settle"
        ),
        FailureRecord(
            timestamp=datetime.now().isoformat(), model_name="eqprop_mlp", task_name="mnist",
            tier="baseline", trial_id=2, failure_type=FailureCategory.SETTLING_DIVERGENCE.value,
            failure_epoch=2, failure_batch=50, config={"lr": 0.1}, last_metrics={}, stack_trace="State failed to settle"
        ),
        # Forward-Forward collapse
        FailureRecord(
            timestamp=datetime.now().isoformat(), model_name="forward_forward", task_name="cifar10",
            tier="baseline", trial_id=3, failure_type=FailureCategory.GOODNESS_COLLAPSE.value,
            failure_epoch=1, failure_batch=10, config={"lr": 0.001}, last_metrics={}, stack_trace="Goodness = 0"
        ),
        # Memory inefficient scaling
        FailureRecord(
            timestamp=datetime.now().isoformat(), model_name="backprop_mlp", task_name="cifar10",
            tier="scaling", trial_id=4, failure_type=FailureCategory.MEMORY_OOM.value,
            failure_epoch=1, failure_batch=0, config={"num_layers": 128}, last_metrics={}, stack_trace="CUDA OOM"
        ),
        # STDP Silencing
        FailureRecord(
            timestamp=datetime.now().isoformat(), model_name="spiking_stdp", task_name="mnist",
            tier="baseline", trial_id=5, failure_type=FailureCategory.SPIKE_SILENCING.value,
            failure_epoch=3, failure_batch=500, config={"lr": 1e-4}, last_metrics={}, stack_trace="0 spikes detected"
        ),
        # Early Failure (to trigger early divergence diagnostic)
        FailureRecord(
            timestamp=datetime.now().isoformat(), model_name="eqprop_mlp", task_name="mnist",
            tier="baseline", trial_id=6, failure_type=FailureCategory.GRADIENT_EXPLOSION.value,
            failure_epoch=0, failure_batch=5, config={"lr": 1.0}, last_metrics={}, stack_trace="NaN"
        ),
        FailureRecord(
            timestamp=datetime.now().isoformat(), model_name="eqprop_mlp", task_name="mnist",
            tier="baseline", trial_id=7, failure_type=FailureCategory.GRADIENT_EXPLOSION.value,
            failure_epoch=0, failure_batch=15, config={"lr": 1.0}, last_metrics={}, stack_trace="NaN"
        ),
        FailureRecord(
            timestamp=datetime.now().isoformat(), model_name="pepita", task_name="mnist",
            tier="baseline", trial_id=8, failure_type=FailureCategory.GRADIENT_EXPLOSION.value,
            failure_epoch=1, failure_batch=1, config={"lr": 0.5}, last_metrics={}, stack_trace="NaN"
        )
    ]
    
    for f in failures:
        tracker.log_failure(f)
        
    print(f"Logged {len(failures)} mock failures.")
    
    generator = FailureManifestoGenerator(db_path)
    out_path = "reports/failure_manifesto.md"
    
    generator.generate(out_path)
    
    if os.path.exists(out_path):
        print(f"Successfully generated manifesto at {out_path}")
        with open(out_path, "r") as f:
            print("\nPreview:")
            lines = f.readlines()
            for line in lines[:25]: # Print first 25 lines
                print(line.strip())
    else:
        print(f"Failed to generate {out_path}")

if __name__ == "__main__":
    main()
