"""
Tests for Research RPG (Console).
"""

import pytest
import os
import tempfile
import json
import sys
from unittest.mock import MagicMock, patch

# Ensure game_rpg is in path
sys.path.append(os.getcwd())

from game_rpg.core import ResearchGame
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

    assert game.stats["experiments_run_session"] == 0
    assert game.stats["session_start"] > 0

def test_get_top_discoveries(temp_files):
    db, stats = temp_files
    game = ResearchGame(db_path=db, stats_path=stats)

    # Mock storage return
    mock_trial_1 = MagicMock(status="completed", accuracy=0.8, model_name="M1")
    mock_trial_2 = MagicMock(status="completed", accuracy=0.9, model_name="M2")
    mock_trial_fail = MagicMock(status="failed", accuracy=0.0)

    with patch.object(game.state.storage, 'get_all_trials', return_value=[mock_trial_1, mock_trial_2, mock_trial_fail]):
        top = game.get_top_discoveries()
        assert len(top) == 2
        assert top[0].model_name == "M2" # Sorted by acc desc

def test_execute_task_simple(temp_files):
    db, stats = temp_files

    with patch('game_rpg.core.run_single_trial_task') as mock_run, \
         patch('game_rpg.core.create_optuna_space'), \
         patch('game_rpg.core.get_evaluation_config') as mock_conf:

        mock_conf.return_value = MagicMock(epochs=1, batch_size=1)
        mock_run.return_value = {"accuracy": 0.5, "loss": 0.1}

        game = ResearchGame(db_path=db, stats_path=stats)

        # Mock study
        mock_study = MagicMock()
        game.state.get_optuna_study = MagicMock(return_value=mock_study)

        task = ExperimentTask("M1", "T1", PatientLevel.SMOKE, "s1", 10.0)

        acc = game.execute_task(task)

        assert acc == 0.5
        assert game.stats["experiments_run_session"] == 1
