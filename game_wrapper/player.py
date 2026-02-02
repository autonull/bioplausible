class PlayerState:
    def __init__(self):
        self.xp = 0
        self.level = 1
        self.max_energy = 100.0
        self.energy = 100.0
        self.energy_regen = 0.05
        self.scan_cost = 20.0
        self.unlocks = {"compass": False, "turbo_scan": False, "auto_probe": False}

    def update(self):
        self.energy = min(self.max_energy, self.energy + self.energy_regen)

    def gain_xp(self, amount):
        self.xp += amount
        # Simple level curve
        req = self.level * 500
        if self.xp >= req:
            self.xp -= req
            self.level_up()
            return True
        return False

    def level_up(self):
        self.level += 1
        self.max_energy += 20
        self.energy = self.max_energy
        # Unlocks based on level
        if self.level == 2:
            self.unlocks["compass"] = True
        if self.level == 3:
            self.unlocks["turbo_scan"] = True
        if self.level == 4:
            self.unlocks["auto_probe"] = True
