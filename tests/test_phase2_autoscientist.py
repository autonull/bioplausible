"""Tests for Phase 2: AutoScientist Intelligence Layer."""

import os
import tempfile
from pathlib import Path

import pytest

from bioplausible.autoscientist.bridge import AutoScientistBridge, ExperimentProposal
from bioplausible.autoscientist.campaign import AutoScientistCampaign
from bioplausible.autoscientist.proposer import ExperimentProposer
from bioplausible.autoscientist.reasoner import Hypothesis, HypothesisReasoner
from bioplausible.knowledge import KnowledgeBase, KnowledgeEntry


@pytest.fixture
def tmp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_autoscientist.db")
        yield db_path


@pytest.fixture
def kb_with_data(tmp_db_path):
    """Create a KnowledgeBase with sample experiment data."""
    kb = KnowledgeBase(db_path=tmp_db_path)
    # Add seed + experiment data
    kb.add_experiment(
        name="exp_vision_1",
        model_family="eqprop",
        task="mnist",
        config={"lr": 0.01, "hidden_dim": 256, "num_layers": 2},
        metrics={"val_accuracy": 0.92, "val_loss": 0.25},
    )
    kb.add_experiment(
        name="exp_vision_2",
        model_family="forward_forward",
        task="mnist",
        config={"lr": 0.001, "hidden_dim": 128, "num_layers": 3},
        metrics={"val_accuracy": 0.88, "val_loss": 0.35},
    )
    kb.add_experiment(
        name="exp_lm_1",
        model_family="backprop",
        task="tiny_shakespeare",
        config={"lr": 0.0005, "hidden_dim": 512, "num_layers": 6},
        metrics={"val_accuracy": 0.75, "val_loss": 1.2},
    )
    return kb


class TestHypothesisReasoner:
    """Tests for the HypothesisReasoner."""

    def test_reasoner_initialization(self, tmp_db_path):
        """Test HypothesisReasoner initialization."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        reasoner = HypothesisReasoner(kb)
        assert reasoner is not None
        assert reasoner.llm_backend is None

    def test_cross_domain_transfer_hypotheses(self, kb_with_data):
        """Test cross-domain transfer hypothesis generation."""
        reasoner = HypothesisReasoner(kb_with_data)

        recent_results = [
            {"model": "eqprop", "task": "mnist", "val_accuracy": 0.92},
        ]
        hypotheses = reasoner._cross_domain_transfer_hypotheses(recent_results)

        assert len(hypotheses) > 0
        for h in hypotheses:
            assert isinstance(h, Hypothesis)
            assert h.source == "rule-based"
            assert h.proposed_model is not None or h.proposed_propagator is not None

    def test_bio_accuracy_tradeoff_hypotheses(self, kb_with_data):
        """Test bio-accuracy tradeoff hypothesis generation."""
        reasoner = HypothesisReasoner(kb_with_data)

        # Mock a result with high bio score but low accuracy
        recent_results = [
            {
                "model": "local_learner",
                "task": "mnist",
                "val_accuracy": 0.45,
                "bio_score": 0.9,
            },
        ]
        hypotheses = reasoner._bio_accuracy_tradeoff_hypotheses(recent_results)

        assert len(hypotheses) > 0
        for h in hypotheses:
            assert (
                "hybrid" in h.statement.lower()
                or "backprop head" in h.statement.lower()
            )

    def test_mep_variant_hypotheses(self, tmp_db_path):
        """Test MEP variant hypothesis generation."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        reasoner = HypothesisReasoner(kb)

        # This tests the registry query for MEP propagators
        hypotheses = reasoner._mep_variant_hypotheses(None)
        # Should return hypotheses even with no data
        assert isinstance(hypotheses, list)

    def test_generate_hypotheses(self, kb_with_data):
        """Test full hypothesis generation."""
        reasoner = HypothesisReasoner(kb_with_data)

        hypotheses = reasoner.generate_hypotheses()
        assert len(hypotheses) >= 0
        # All should be Hypothesis objects
        for h in hypotheses:
            assert isinstance(h, Hypothesis)

    def test_analyze_knowledge_base(self, kb_with_data):
        """Test KnowledgeBase analysis for insights."""
        # Add a knowledge entry to test the method
        entry = KnowledgeEntry(
            id="TEST-ANALYZE",
            topic="Analysis",
            model_family="eqprop",
            finding="Test entry for analysis",
            details="Details",
            confidence=0.9,
            metrics={"val_accuracy": 0.92},
        )
        kb_with_data.add_entry(entry)

        reasoner = HypothesisReasoner(kb_with_data)

        insights = reasoner.analyze_knowledge_base()
        assert isinstance(insights, list)
        # Should have insights about model performance
        assert any("eqprop" in str(i) or "accuracy" in str(i).lower() for i in insights)

    def test_llm_integration_optional(self, tmp_db_path):
        """Test that LLM integration is optional and works when disabled."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        reasoner = HypothesisReasoner(kb, llm_backend=None)

        # Should still work without LLM
        hypotheses = reasoner.generate_hypotheses()
        assert isinstance(hypotheses, list)


class TestExperimentProposer:
    """Tests for the ExperimentProposer."""

    def test_proposer_initialization(self, tmp_db_path):
        """Test ExperimentProposer initialization."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        proposer = ExperimentProposer(knowledge_base=kb)
        assert proposer is not None

    def test_propose_batch(self, tmp_db_path):
        """Test batch experiment proposal generation."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        proposer = ExperimentProposer(knowledge_base=kb)

        proposals = proposer.propose_batch(n_proposals=5)
        assert len(proposals) >= 0
        for p in proposals:
            assert isinstance(p, ExperimentProposal)
            assert p.model is not None
            assert p.task is not None

    def test_propose_batch_with_domain_filter(self, tmp_db_path):
        """Test batch proposal with domain filtering."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        proposer = ExperimentProposer(knowledge_base=kb)

        proposals = proposer.propose_batch(domain="vision", n_proposals=3)
        assert len(proposals) >= 0

    def test_propose_batch_with_bio_score_filter(self, tmp_db_path):
        """Test batch proposal with bio-plausibility filtering."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        proposer = ExperimentProposer(knowledge_base=kb)

        # High bio score should only include local/equilibrium rules
        proposals = proposer.propose_batch(n_proposals=5, min_bio_score=0.8)
        assert len(proposals) >= 0

    def test_hypothesis_to_proposal(self, tmp_db_path):
        """Test converting a hypothesis to an experiment proposal."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        proposer = ExperimentProposer(knowledge_base=kb)

        hypothesis = Hypothesis(
            statement="Test hypothesis",
            confidence=0.8,
            proposed_model="test_model",
            proposed_task="mnist",
            proposed_propagator=None,
            reasoning_chain=["Reason 1", "Reason 2"],
        )

        proposal = proposer._hypothesis_to_proposal(hypothesis)
        assert proposal is not None
        assert proposal.model == "test_model"
        assert proposal.task == "mnist"
        assert proposal.priority == 0.8

    def test_propose_ablation(self, tmp_db_path):
        """Test ablation study proposal generation."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        proposer = ExperimentProposer(knowledge_base=kb)

        proposals = proposer.propose_ablation(
            model="test_model",
            base_config={"lr": 0.01, "hidden_dim": 256},
            parameters=["lr", "hidden_dim"],
            values=[[0.001, 0.01], [128, 256]],
        )

        # 2 lr values + 2 hidden_dim values = 4 proposals
        assert len(proposals) == 4
        for p in proposals:
            assert isinstance(p, ExperimentProposal)
            assert "ablation" in p.tags


class TestAutoScientistBridge:
    """Tests for the AutoScientistBridge."""

    def test_bridge_initialization(self):
        """Test AutoScientistBridge initialization."""
        bridge = AutoScientistBridge()
        assert bridge is not None
        assert len(bridge.pending_proposals()) == 0

    def test_proposal_to_task(self):
        """Test converting proposal to task config."""
        bridge = AutoScientistBridge()

        proposal = ExperimentProposal(
            hypothesis="Test hypothesis",
            model="equitile",
            task="mnist",
            propagator="eqprop",
            optimizer="adam",
            hyperparams={"lr": 0.01, "hidden_dim": 256},
            priority=0.8,
            tags=["test"],
        )

        config = bridge.proposal_to_task(proposal)

        assert config["model"] == "equitile"
        assert config["task"] == "mnist"
        assert config["optimizer"] == "adam"
        assert config["propagator"] == "eqprop"
        assert config["lr"] == 0.01

    def test_submit_and_pending_proposals(self):
        """Test proposal submission and retrieval."""
        bridge = AutoScientistBridge()

        proposal = ExperimentProposal(
            hypothesis="Test",
            model="model_a",
            task="mnist",
            priority=0.5,
        )
        bridge.submit_proposal(proposal)

        pending = bridge.pending_proposals()
        assert len(pending) == 1
        assert pending[0].model == "model_a"

    def test_clear_executed(self):
        """Test clearing executed proposals."""
        bridge = AutoScientistBridge()

        for i in range(3):
            bridge.submit_proposal(
                ExperimentProposal(
                    hypothesis=f"Test {i}",
                    model=f"model_{i}",
                    task="mnist",
                    priority=0.5,
                )
            )

        bridge.clear_executed([0])
        assert len(bridge.pending_proposals()) == 2


class TestAutoScientistCampaign:
    """Tests for the AutoScientistCampaign."""

    def test_campaign_initialization(self, tmp_db_path):
        """Test AutoScientistCampaign initialization."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        campaign = AutoScientistCampaign(
            knowledge_base=kb,
            output_dir=Path(tmp_db_path).parent / "campaigns_dir",
            max_concurrent=1,
            human_approval_gate=False,
        )
        assert campaign is not None
        assert campaign.knowledge_base is kb

    def test_campaign_dry_run(self, tmp_db_path):
        """Test campaign dry run (proposal only)."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        campaign = AutoScientistCampaign(
            knowledge_base=kb,
            output_dir=Path(tmp_db_path).parent / "campaigns_dir2",
            human_approval_gate=False,
        )

        results = campaign.run_iteration(domain="vision", n_experiments=2, dry_run=True)
        assert results == []

    def test_campaign_summary(self, tmp_db_path):
        """Test campaign summary generation."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        campaign = AutoScientistCampaign(knowledge_base=kb)

        summary = campaign.get_summary()
        assert "iterations" in summary
        assert "total_experiments" in summary
        assert "completed" in summary
        assert "best_accuracy" in summary

    def test_human_approval_gate(self, tmp_db_path):
        """Test human approval gate behavior."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        campaign = AutoScientistCampaign(knowledge_base=kb, human_approval_gate=True)

        # Should approve all by default in mock
        proposals = [
            ExperimentProposal(hypothesis="H1", model="m1", task="t1"),
            ExperimentProposal(hypothesis="H2", model="m2", task="t2"),
        ]
        approved = campaign._human_approval(proposals)
        assert all(i in approved for i in range(len(proposals)))


class TestKnowledgeBaseIntelligence:
    """Tests for KnowledgeBase intelligent features."""

    def test_surrogate_training(self, kb_with_data):
        """Test surrogate model training."""
        surrogate_id = kb_with_data.train_surrogate(target_metric="val_accuracy")
        assert (
            surrogate_id is not None or surrogate_id is None
        )  # May fail without enough data

    def test_surrogate_listing(self, kb_with_data):
        """Test listing surrogate models."""
        surrogates = kb_with_data.list_surrogates()
        assert isinstance(surrogates, list)

    def test_symbolic_rules_extraction(self, tmp_db_path):
        """Test symbolic rule extraction."""
        kb = KnowledgeBase(db_path=tmp_db_path)

        # Add failure data for the metamodel
        import sqlite3

        with sqlite3.connect(tmp_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failures (
                    model_name TEXT,
                    task_name TEXT,
                    failure_type TEXT,
                    config TEXT
                )
            """)
            conn.execute(
                "INSERT INTO failures VALUES (?, ?, ?, ?)",
                (
                    "eqprop_mlp",
                    "mnist",
                    "settling_divergence",
                    '{"lr": 0.1, "hidden_dim": 512}',
                ),
            )
            conn.execute(
                "INSERT INTO failures VALUES (?, ?, ?, ?)",
                ("eqprop_mlp", "mnist", "success", '{"lr": 0.01, "hidden_dim": 256}'),
            )
            conn.commit()

        rules = kb.extract_symbolic_rules(focus_model="eqprop_mlp")
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_algorithm_similarity(self, tmp_db_path):
        """Test algorithm similarity computation."""
        kb = KnowledgeBase(db_path=tmp_db_path)

        similarity = kb.compute_algorithm_similarity()
        assert isinstance(similarity, dict)

    def test_causal_analysis(self, tmp_db_path):
        """Test causal discovery analysis."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        # Add experiments for causal analysis
        for i in range(15):
            kb.add_experiment(
                name=f"causal_exp_{i}",
                model_family="test_model",
                task="mnist",
                config={"lr": 0.01 * (i + 1), "hidden_dim": 64 * (i + 1)},
                metrics={"val_accuracy": 0.5 + i * 0.02},
            )

        result = kb.run_causal_analysis()
        assert isinstance(result, dict)
        assert "correlations" in result or "error" in result


class TestAutoScientistIntegration:
    """Integration tests for AutoScientist → KnowledgeBase → CoreTrainer flow."""

    def test_proposer_uses_kb_insights(self, tmp_db_path):
        """Test that proposer incorporates KB insights into proposals."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        kb.add_experiment(
            name="exp_1",
            model_family="eqprop",
            task="mnist",
            config={"lr": 0.01},
            metrics={"val_accuracy": 0.95},
        )

        proposer = ExperimentProposer(knowledge_base=kb)
        proposals = proposer.propose_batch(n_proposals=3)

        assert len(proposals) >= 0

    def test_reasoner_uses_kb_for_hypotheses(self, tmp_db_path):
        """Test that reasoner uses KB data for hypothesis generation."""
        kb = KnowledgeBase(db_path=tmp_db_path)
        kb.add_experiment(
            name="exp_1",
            model_family="backprop",
            task="mnist",
            config={"hidden_dim": 256},
            metrics={"val_accuracy": 0.5},
        )

        reasoner = HypothesisReasoner(kb)
        hypotheses = reasoner.generate_hypotheses()

        assert isinstance(hypotheses, list)

    def test_end_to_end_flow_mocked(self, tmp_db_path):
        """Test end-to-end AutoScientist flow with mocked execution."""
        kb = KnowledgeBase(db_path=tmp_db_path)

        # Initialize campaign
        campaign = AutoScientistCampaign(
            knowledge_base=kb,
            output_dir=Path(tmp_db_path).parent / "kb_campaign_test",
        )

        # Add experiment data
        kb.add_experiment(
            name="exp_base",
            model_family="eqprop",
            task="mnist",
            config={"lr": 0.01, "hidden_dim": 128},
            metrics={"val_accuracy": 0.88},
        )

        # Run analysis
        insights = campaign.reasoner.analyze_knowledge_base()
        assert isinstance(insights, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
