"""
Upgrade System for Research RPG.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class UpgradeSpec:
    id: str
    name: str
    description: str
    base_cost: float
    cost_multiplier: float
    effect_type: str # "xp_gain", "sp_gain", "auto_speed"
    effect_per_level: float
    max_level: int = 10

UPGRADES = [
    UpgradeSpec(
        id="grant_writing",
        name="Grant Writing Workshop",
        description="Increases XP gained from experiments.",
        base_cost=50.0,
        cost_multiplier=1.5,
        effect_type="xp_gain",
        effect_per_level=0.10 # +10% per level
    ),
    UpgradeSpec(
        id="lab_equipment",
        name="Advanced Lab Equipment",
        description="Increases Science Points gained from results.",
        base_cost=100.0,
        cost_multiplier=1.6,
        effect_type="sp_gain",
        effect_per_level=0.10
    ),
    UpgradeSpec(
        id="cluster_access",
        name="Cluster Access Priority",
        description="Reduces downtime in Auto-Scientist mode.",
        base_cost=200.0,
        cost_multiplier=2.0,
        effect_type="auto_speed",
        effect_per_level=0.15 # 15% faster
    )
]

class UpgradeManager:
    def __init__(self, game):
        self.game = game
        self.specs = {u.id: u for u in UPGRADES}

    def get_level(self, upgrade_id: str) -> int:
        return self.game.stats.get("upgrades", {}).get(upgrade_id, 0)

    def get_cost(self, upgrade_id: str) -> float:
        spec = self.specs.get(upgrade_id)
        if not spec: return 999999.0
        lvl = self.get_level(upgrade_id)
        if lvl >= spec.max_level: return float('inf')
        return spec.base_cost * (spec.cost_multiplier ** lvl)

    def get_multiplier(self, effect_type: str) -> float:
        mult = 1.0
        for uid, spec in self.specs.items():
            if spec.effect_type == effect_type:
                lvl = self.get_level(uid)
                # print(f"DEBUG: {uid} lvl={lvl} effect={spec.effect_per_level}")
                mult += (lvl * spec.effect_per_level)
        # print(f"DEBUG: Type {effect_type} Mult={mult}")
        return mult

    def can_afford(self, upgrade_id: str) -> bool:
        return self.game.science_points >= self.get_cost(upgrade_id)

    def purchase(self, upgrade_id: str) -> bool:
        if not self.can_afford(upgrade_id):
            return False

        cost = self.get_cost(upgrade_id)
        self.game.stats["science_points"] -= cost

        upgrades = self.game.stats.get("upgrades", {})
        upgrades[upgrade_id] = upgrades.get(upgrade_id, 0) + 1
        self.game.stats["upgrades"] = upgrades

        self.game.save_stats()
        return True

    def get_all_upgrades(self) -> List[UpgradeSpec]:
        return UPGRADES
