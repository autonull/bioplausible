"""
Quest System for Research RPG.
"""

import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class QuestSpec:
    id: str
    description: str
    target_type: str # "run_count", "accuracy", "model_type"
    target_value: Any
    target_count: int
    reward_xp: int
    reward_sp: float

class QuestManager:
    def __init__(self, game):
        self.game = game

    def refresh_daily_quests(self):
        """Generates 3 quests if none exist or if manually refreshed (game day logic?)."""
        current_quests = self.game.stats.get("quests", [])

        # Simple Logic: If no active quests, generate 3.
        # In a real game, we'd check timestamps.
        if not current_quests:
            self.generate_quests()

    def generate_quests(self):
        level = self.game.level
        new_quests = []

        # 1. Run Experiments Quest
        count = random.randint(3, 5 + level)
        new_quests.append({
            "id": f"daily_run_{time.time()}",
            "description": f"Complete {count} experiments.",
            "target_type": "run_count",
            "target_count": count,
            "progress": 0,
            "reward_xp": 50 + (level * 10),
            "reward_sp": 10.0 + (level * 2.0),
            "completed": False
        })

        # 2. Accuracy Quest
        target_acc = 0.5 + min(0.4, (level * 0.05))
        new_quests.append({
            "id": f"daily_acc_{time.time()}",
            "description": f"Achieve > {target_acc:.0%} accuracy in a single experiment.",
            "target_type": "accuracy",
            "target_value": target_acc,
            "progress": 0,
            "reward_xp": 100 + (level * 20),
            "reward_sp": 50.0 + (level * 5.0),
            "completed": False
        })

        self.game.stats["quests"] = new_quests
        self.game.save_stats()

    def check_quests(self, task, metrics):
        """
        Called after an experiment finishes. Updates progress.
        """
        quests = self.game.stats.get("quests", [])
        updated = False

        for q in quests:
            if q.get("completed"): continue

            if q["target_type"] == "run_count":
                q["progress"] += 1
                if q["progress"] >= q["target_count"]:
                    self._complete_quest(q)
                updated = True

            elif q["target_type"] == "accuracy":
                acc = metrics.get("accuracy", 0.0)
                if acc > q["target_value"]:
                    # Instant complete
                    self._complete_quest(q)
                updated = True

        if updated:
            self.game.stats["quests"] = quests
            self.game.save_stats()

    def _complete_quest(self, quest):
        quest["completed"] = True
        print(f"\n✨ QUEST COMPLETED: {quest['description']}")
        print(f"   Rewards: {quest['reward_xp']} XP, {quest['reward_sp']:.1f} SP")
        self.game.add_xp(quest['reward_xp'])
        self.game.add_science_points(quest['reward_sp'])
