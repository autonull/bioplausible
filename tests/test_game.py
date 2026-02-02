"""
Tests for ResearchGame.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import MagicMock, patch
from bioplausible.scientist.game import ResearchGame
from bioplausible.scientist.core import ExperimentTask, PatientLevel

@pytest.fixture
def temp_files():
    """Create temp DB and stats file."""
    fd_db, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd_db)

    fd_stats, stats_path = tempfile.mkstemp(suffix=".json")
    os.close(fd_stats)

    yield db_path, stats_path

    if os.path.exists(db_path): os.remove(db_path)
    if os.path.exists(stats_path): os.remove(stats_path)

def test_game_initialization(temp_files):
    db, stats = temp_files
    game = ResearchGame(db_path=db, stats_path=stats)

    assert game.level == 1
    assert game.xp == 0
    assert game.science_points == 0.0

def test_level_gating(temp_files):
    db, stats = temp_files

    with patch('bioplausible.scientist.game.ExperimentState'), \
         patch('bioplausible.scientist.game.ScientistStrategy') as MockStrategy:

        game = ResearchGame(db_path=db, stats_path=stats)

        # Mock strategy to return tasks of different levels
        t1 = ExperimentTask("M1", "T1", PatientLevel.SMOKE, "s1", 10.0)
        t2 = ExperimentTask("M2", "T2", PatientLevel.STANDARD, "s2", 20.0) # Req Level 2
        t3 = ExperimentTask("M3", "T3", PatientLevel.DEEP, "s3", 30.0) # Req Level 8

        game.strategy.generate_candidates.return_value = [t1, t2, t3]

        # Level 1: Only t1
        game.stats["level"] = 1
        avail = game.get_available_experiments()
        assert len(avail) == 1
        assert avail[0] == t1

        # Level 2: t1 and t2
        game.stats["level"] = 2
        avail = game.get_available_experiments()
        assert len(avail) == 2
        assert t2 in avail

        # Level 8: All
        game.stats["level"] = 8
        avail = game.get_available_experiments()
        assert len(avail) == 3

def test_execute_task_rewards(temp_files):
    db, stats = temp_files

    with patch('bioplausible.scientist.game.run_single_trial_task') as mock_run, \
         patch('bioplausible.scientist.game.create_optuna_space'), \
         patch('bioplausible.scientist.game.get_evaluation_config') as mock_conf:

        mock_conf.return_value = MagicMock(epochs=1, batch_size=1)
        mock_run.return_value = {"accuracy": 0.5, "loss": 0.1}

        game = ResearchGame(db_path=db, stats_path=stats)

        # Mock study
        mock_study = MagicMock()
        game.state.get_optuna_study = MagicMock(return_value=mock_study)

        task = ExperimentTask("M1", "T1", PatientLevel.SMOKE, "s1", 10.0)

        acc = game.execute_task(task)

        assert acc == 0.5
        assert game.xp == 10 # Smoke gives 10
        assert game.science_points == 5.0 # 0.5 * 10
        assert game.stats["experiments_run"] == 1

        # Verify persistence
        with open(stats, "r") as f:
            saved = json.load(f)
            assert saved["xp"] == 10

def test_level_up(temp_files):
    db, stats = temp_files
    game = ResearchGame(db_path=db, stats_path=stats)

    # 100 XP needed for Level 2
    game.add_xp(100)
    assert game.level == 2

    # Check persistence
    with open(stats, "r") as f:
        saved = json.load(f)
        assert saved["level"] == 2
