
import unittest
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from bioplausible.scientist.synthesizer import ResearchSynthesizer
from bioplausible.scientist.training_dynamics import TrainingCheckpoint, TrainingTrajectory

# Helper to create mock trajectories
def create_mock_traj(
    model_name: str, 
    task_name: str, 
    final_acc: float, 
    convergence_epoch: int,
    config: Dict = None
) -> TrainingTrajectory:
    
    if config is None:
        config = {"activation": "relu"}
        
    checkpoints = []
    # Add a final checkpoint
    ckpt = TrainingCheckpoint(
        epoch=convergence_epoch + 5,
        train_acc=final_acc + 0.05,
        val_acc=final_acc,
        train_loss=0.5,
        val_loss=0.6,
        grad_norm_mean=0.1,
        grad_norm_std=0.01,
        weight_norm=1.0,
        learning_rate=0.01,
        train_val_gap=0.05,
        wall_time_seconds=10.0
    )
    checkpoints.append(ckpt)
    
    traj = TrainingTrajectory(
        trial_id=1,
        model_name=model_name,
        task_name=task_name,
        config=config,
        checkpoints=checkpoints
    )
    # Mock computed metrics
    traj.compute_convergence_speed = lambda: convergence_epoch
    traj.compute_sample_efficiency = lambda: final_acc * 100 # Dummy metric
    
    return traj

class TestResearchSynthesizer(unittest.TestCase):

    def setUp(self):
        # Create dataset
        # 1. Baseline Backprop: High Acc, Slow
        self.backprop_trajs = [
            create_mock_traj("Baseline Backprop", "mnist", 0.95, 20),
            create_mock_traj("Baseline Backprop", "cifar10", 0.85, 30),
        ]
        
        # 2. EqProp: Lower Acc, Fast
        self.eqprop_trajs = [
            create_mock_traj("EqProp MLP", "mnist", 0.92, 5),
            create_mock_traj("EqProp Conv", "cifar10", 0.80, 8),
        ]
        
        # 3. GELU vs ReLU config test
        self.gelu_traj = create_mock_traj("GELU Model", "mnist", 0.98, 20, config={"activation": "gelu"})
        self.relu_traj = create_mock_traj("ReLU Model", "mnist", 0.94, 20, config={"activation": "relu"})
        
        self.all_trajs = self.backprop_trajs + self.eqprop_trajs + [self.gelu_traj, self.relu_traj]
        
        self.synth = ResearchSynthesizer(self.all_trajs)

    def test_cross_algorithm_insights(self):
        """Test that insights are generated correctly."""
        insights = self.synth.generate_cross_algorithm_insights()
        
        # Check that we have insights for tasks
        mnist_perf = next((i for i in insights if i.task == "mnist" and i.metric == "final_accuracy"), None)
        self.assertIsNotNone(mnist_perf)
        
        # In setup, Backprop (0.95) > EqProp (0.92) for MNIST
        # So Baseline should be ranked higher than EqProp 
        # (Assuming model names map to families: "Baseline..." -> "baseline", "EqProp..." -> "eqprop")
        if mnist_perf:
             self.assertTrue("higher than" in mnist_perf.narrative or "outperforming" in mnist_perf.narrative)
        
        # Check speed comparison
        # EqProp (5) < Baseline (20) -> EqProp is faster (better)
        mnist_speed = next((i for i in insights if i.task == "mnist" and i.metric == "convergence_speed"), None)
        self.assertIsNotNone(mnist_speed)
        self.assertEqual(mnist_speed.ranking[0], "eqprop") # Eqprop should be first (best)

    def test_architecture_recommendations(self):
        """Test hybrid recommendation generation."""
        recs = self.synth.generate_architecture_recommendations()
        
        # We set up EqProp to be faster but less accurate, so we expect the hybrid recommendation
        hybrid_rec = next((r for r in recs if "Hybrid" in r.name), None)
        self.assertIsNotNone(hybrid_rec)
        self.assertIn("EqProp converges", hybrid_rec.motivation)

    def test_quick_wins(self):
        """Test detection of activation function wins."""
        # Create isolated synthesizer to avoid pollution from backprop/eqprop trajs which might default to Relu
        iso_synth = ResearchSynthesizer([self.gelu_traj, self.relu_traj])
        wins = iso_synth.find_quick_wins()
        
        # GELU (0.98) > ReLU (0.94) + 0.02
        gelu_win = next((w for w in wins if "GELU" in w["title"]), None)
        self.assertIsNotNone(gelu_win)
        self.assertIn("4.0%", gelu_win["impact"]) # 0.98 - 0.94 = 0.04

    def test_research_gaps(self):
        """Test gap detection."""
        gaps = self.synth.identify_research_gaps()
        
        # We didn't include graph task
        self.assertTrue(any("graph" in g.lower() for g in gaps))
        
        # We didn't include PReLU
        self.assertTrue(any("PReLU" in g for g in gaps))

if __name__ == "__main__":
    unittest.main()
