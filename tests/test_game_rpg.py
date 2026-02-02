"""
Tests for Research RPG.
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
from game_rpg.upgrades import UpgradeManager
from game_rpg.quests import QuestManager
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
    # Quests should be auto-generated
    assert len(game.stats["quests"]) > 0

def test_upgrades(temp_files):
    db, stats = temp_files
    game = ResearchGame(db_path=db, stats_path=stats)

    # Give lots of SP
    game.add_science_points(1000.0)

    um = game.upgrade_manager
    up_id = "grant_writing" # 50 cost

    # Check cost
    assert um.get_cost(up_id) == 50.0

    # Purchase
    assert um.purchase(up_id)
    assert um.get_level(up_id) == 1
    assert game.science_points == 950.0

    # Check effect
    # Base multiplier is 1.0 + (level * 0.1) = 1.1
    assert um.get_multiplier("xp_gain") == 1.1

def test_quest_completion(temp_files):
    db, stats = temp_files
    game = ResearchGame(db_path=db, stats_path=stats)

    # Clear quests and add a specific one
    game.stats["quests"] = [{
        "id": "test_q",
        "description": "Test Quest",
        "target_type": "run_count",
        "target_count": 1,
        "progress": 0,
        "reward_xp": 100,
        "reward_sp": 50,
        "completed": False
    }]

    qm = game.quest_manager

    # Simulate experiment finish
    task = MagicMock()
    metrics = {"accuracy": 0.5}

    qm.check_quests(task, metrics)

    # Verify completion
    q = game.stats["quests"][0]
    assert q["completed"] is True
    assert game.xp == 100
    assert game.science_points == 50.0

def test_xp_multiplier(temp_files):
    db, stats = temp_files
    game = ResearchGame(db_path=db, stats_path=stats)

    # Buy XP upgrade
    game.add_science_points(100.0)
    game.upgrade_manager.purchase("grant_writing") # Lvl 1: +10%

    # Add XP
    game.add_xp(100)

    # Should be 110
    assert game.xp == 110
