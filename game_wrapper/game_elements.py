class Probe:
    def __init__(self, pos, vel, launcher_ref):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.launcher_ref = launcher_ref
        self.life = 600 # 10 seconds default
        self.scan_timer = 0
        
    def update(self, game):
        self.pos += self.vel
        self.life -= 1
        self.scan_timer += 1
        
        # Scan every 2 seconds
        if self.scan_timer > 120:
            self.scan_timer = 0
            # Trigger a scan at probe location
            # We need to access Game to trigger scan? 
            # Or just use the launcher ref if we pass params.
            # Ideally we call back into game or duplicate logic.
            game.trigger_remote_scan(self.pos)
            
        return self.life > 0

# Visual Modes
# 1. Standard (Color by Model)
# 2. Performance (Color by Accuracy: Red->Green)
# 3. Chronology (Color by ID/Time: Dark->Bright)

