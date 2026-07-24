"""Tests for Cross-Domain Benchmark Suite (Phase 3)."""

import tempfile

import pytest

from bioplausible.evaluation.cross_domain import BenchmarkSuiteConfig
from bioplausible.evaluation.cross_domain import BenchmarkSuiteResult
from bioplausible.evaluation.cross_domain import CrossDomainBenchmarkSuite


def test_suite_config():
    config = BenchmarkSuiteConfig()
    assert config.epochs == 5
    assert config.batch_size == 64
    assert config.track_energy is True


def test_suite_initialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        suite = CrossDomainBenchmarkSuite(output_dir=tmpdir)
        assert suite is not None
        assert suite.output_dir.exists()


def test_get_benchmark_tasks():
    suite = CrossDomainBenchmarkSuite()
    tasks = suite.get_benchmark_tasks()
    assert "vision" in tasks
    assert "lm" in tasks
    assert "tabular" in tasks


def test_get_models_for_domain():
    suite = CrossDomainBenchmarkSuite()
    models = suite.get_models_for_domain("vision")
    assert isinstance(models, list)
    assert len(models) > 0


def test_suite_result_to_dict():
    config = BenchmarkSuiteConfig()
    result = BenchmarkSuiteResult(config=config)
    d = result.to_dict()
    assert "n_results" in d
    assert "total_time_s" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
