import pygame
import numpy as np
import sys
import math
import random
import time
import json

from bioplausible.models.registry import MODEL_REGISTRY

from .engine import Engine3D, Camera, Colors
from .data import DataManager
from .launcher import JobLauncher
from .audio import audio, music

# Constants
WIDTH, HEIGHT = 1200, 800
FPS = 60

# Colors
COL_HUD = (0, 255, 128)
COL_HUD_DIM = (0, 100, 50)
COL_WARN = (255, 50, 0)
COL_BG = (5, 5, 15)
COL_GRID = (20, 40, 60)

class Particle:
    def __init__(self, pos, vel, color, life):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.color = color
        self.life = life
        self.max_life = life

    def update(self):
        self.pos += self.vel
        self.life -= 1
        return self.life > 0

class Objective:
    def __init__(self, text, condition_func):
        self.text = text
        self.condition_func = condition_func
        self.completed = False
        self.completed_time = 0

class Probe:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.life = 1000 
        self.scan_timer = 0
        self.history = []

    def update(self, game):
        self.pos += self.vel
        self.life -= 1
        self.scan_timer += 1
        
        if self.scan_timer % 10 == 0:
            self.history.append(self.pos.copy())
            if len(self.history) > 20: self.history.pop(0)

        # Scan every 1.5 seconds
        if self.scan_timer > 90:
            self.scan_timer = 0
            game.trigger_remote_scan(self.pos)
            # Ping
            if game.view_dist(self.pos) < 100:
                 audio.play_tone(1200, 0.1, 0.1, 'sine')
            
        return self.life > 0

class Nebula:
    def __init__(self, pos):
        self.pos = np.array(pos, dtype=float)
        self.size = np.random.uniform(20, 50)
        self.color = (np.random.randint(20, 50), np.random.randint(10, 30), np.random.randint(30, 60))
        
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Auto-Scientist Navigator: DEEP SEARCH")
        self.clock = pygame.time.Clock()
        
        self.font_small = pygame.font.SysFont("monospace", 14)
        self.font_large = pygame.font.SysFont("monospace", 17)
        self.font_title = pygame.font.SysFont("impact", 36)
        self.font_sector = pygame.font.SysFont("impact", 24)

        self.engine = Engine3D(WIDTH, HEIGHT, fov=500)
        self.camera = Camera(pos=[0, 0, -30])
        self.trail = []
        
        self.data_mgr = DataManager()
        self.data_mgr.start()
        
        self.launcher = JobLauncher()
        
        self.particles = []
        self.probes = []
        self.nebulas = [Nebula(np.random.uniform(-50, 50, 3)) for _ in range(30)]
        
        self.scan_pulse = 0.0
        self.scan_active = False
        
        self.mouse_locked = False
        pygame.mouse.set_visible(True)

        self.available_models = MODEL_REGISTRY
        self.model_idx = 0
        self.tasks = ["vision", "lm", "rl"]
        self.task_idx = 0
        
        self.scan_cooldown = 0
        self.last_stars_count = 0
        self.inspected_star = None
        
        self.view_modes = ["STANDARD", "PERFORMANCE", "CHRONO"]
        self.view_mode_idx = 0
        self.photo_mode = False
        
        # Gameplay
        self.objectives = [
            Objective("System Initialization: Perform a SCAN (SPACE)", lambda g: g.last_stars_count > 0),
            Objective("Discovery: Find a model > 50% Accuracy", lambda g: any(s.raw_data.get('accuracy',0)>0.5 for s in g.data_mgr.stars)),
            Objective("High Performance: Find a model > 90% Accuracy", lambda g: any(s.raw_data.get('accuracy',0)>0.9 for s in g.data_mgr.stars)),
            Objective("Diversity: Run 3 different Model Types", lambda g: len(set(s.raw_data.get('model_name') for s in g.data_mgr.stars)) >= 3),
            Objective("Automation: Launch a Probe (F)", lambda g: len(g.probes) > 0)
        ]
        
        # Audio
        audio.play_tone(220, 1.0, 0.3, 'sine')
        music.start()

    def get_sector_name(self, pos):
        # Generate flavor text based on coordinates
        prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Echo", "Void", "Deep", "High", "Low", "Hyper"]
        suffixes = ["Prime", "Cluster", "Expanse", "Zone", "Field", "Nebula", "Limit", "Horizon"]
        
        # Seed slightly with pos to be stable-ish?
        # Use abs coordinates for index
        s_idx = int(abs(pos[0]) + abs(pos[1])) % len(prefixes)
        p_idx = int(abs(pos[2]) * 5) % len(suffixes)
        
        name = f"{prefixes[s_idx]} {suffixes[p_idx]}"
        
        # Add descriptor based on params
        if pos[1] > 5: name += " (High Capacity)"
        elif pos[1] < -5: name += " (Minimalist)"
        
        return name

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS)
            running = self.handle_events()
            self.update()
            self.draw()
        self.cleanup()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.mouse_locked = not self.mouse_locked
                    pygame.mouse.set_visible(not self.mouse_locked)
                    pygame.event.set_grab(self.mouse_locked)
                if event.key == pygame.K_TAB:
                    direction = -1 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1
                    self.model_idx = (self.model_idx + direction) % len(self.available_models)
                    audio.play_tone(600, 0.05, 0.2, 'square')
                if event.key == pygame.K_t:
                    self.task_idx = (self.task_idx + 1) % len(self.tasks)
                    audio.play_tone(500, 0.1, 0.3, 'sine')
                if event.key == pygame.K_SPACE:
                    if self.scan_cooldown <= 0:
                        self.trigger_scan()
                if event.key == pygame.K_f:
                    self.launch_probe()
                if event.key == pygame.K_v:
                    self.view_mode_idx = (self.view_mode_idx + 1) % len(self.view_modes)
                    audio.play_tone(700, 0.1, 0.2, 'sine')
                if event.key == pygame.K_p:
                    self.photo_mode = not self.photo_mode
                if event.key == pygame.K_r:
                    self.camera.pos = np.array([0, 0, -30.0])
                    self.camera.vel *= 0
                    self.camera.rot *= 0
                    self.trail = []
        return True

    def update(self):
        self.camera.update()
        
        # Trail update
        if len(self.trail) == 0 or np.linalg.norm(self.trail[-1] - self.camera.pos) > 1.0:
            self.trail.append(self.camera.pos.copy())
            if len(self.trail) > 100: self.trail.pop(0)

        # Gravity Assist
        # Find high accuracy stars nearby
        for s in self.data_mgr.stars:
            if s.raw_data.get('accuracy', 0) > 0.8:
                dist_vec = s.pos - self.camera.pos
                dist = np.linalg.norm(dist_vec)
                if dist < 10 and dist > 1: # Don't pull if too close or too far
                    # Pull vector
                    pull = (dist_vec / dist) * 0.02 * (1.0 / dist) # Falloff
                    self.camera.vel += pull
                    # Subtle audio cue?
                    
        if self.mouse_locked:
            mdx, mdy = pygame.mouse.get_rel()
            self.camera.rotate_impulse(mdx * 0.002, mdy * 0.002)
            
            keys = pygame.key.get_pressed()
            dx, dy, dz = 0, 0, 0
            if keys[pygame.K_w]: dz += 1
            if keys[pygame.K_s]: dz -= 1
            if keys[pygame.K_a]: dx -= 1
            if keys[pygame.K_d]: dx += 1
            if keys[pygame.K_q]: dy -= 1
            if keys[pygame.K_e]: dy += 1
            if keys[pygame.K_LSHIFT]:
                 dx*=2; dy*=2; dz*=2
            
            if dx or dy or dz:
                self.camera.thrust(dx, dy, dz)
                
        self.particles = [p for p in self.particles if p.update()]
        self.probes = [p for p in self.probes if p.update(self)]
        
        if self.scan_cooldown > 0:
            self.scan_cooldown -= 1
        if self.scan_pulse > 0:
            self.scan_pulse -= 0.02
        else:
            self.scan_active = False

        current_stars_count = len(self.data_mgr.stars)
        if current_stars_count > self.last_stars_count:
            audio.play_tone(1000, 0.2, 0.5, 'sine')
            self.last_stars_count = current_stars_count
            
        for obj in self.objectives:
            if not obj.completed and obj.condition_func(self):
                obj.completed = True
                obj.completed_time = time.time()
                audio.play_tone(880, 0.5, 0.5, 'square') 

    def draw(self):
        self.screen.fill(COL_BG)
        
        # Nebulas
        clouds = np.array([n.pos for n in self.nebulas])
        cloud_proj = self.engine.project(clouds, self.camera)
        
        for i, p in enumerate(cloud_proj):
            if p[3] > 0:
                neb = self.nebulas[i]
                r = int(neb.size * p[2] * 2)
                s = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*neb.color, 10), (r, r), r)
                self.screen.blit(s, (p[0]-r, p[1]-r))

        self.draw_grid()
        self.draw_trail()
        
        stars = self.data_mgr.get_stars()
        points = np.array([s.pos for s in stars]) if stars else np.array([])
        draw_list = []
        
        min_dist_angle = 0.995 
        closest_star = None
        yaw, pitch = self.camera.rot[0], self.camera.rot[1]
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        fwd = np.array([sy * cp, -sp, cy * cp])
        cam_pos = self.camera.pos
        
        if len(points) > 0:
            proj = self.engine.project(points, self.camera)
            for i, p in enumerate(proj):
                star = stars[i]
                if p[3] > 0: 
                    draw_list.append((p[2], p[0], p[1], star))
                    vec_to_star = star.pos - cam_pos
                    dist = np.linalg.norm(vec_to_star)
                    dir_to_star = vec_to_star / (dist + 1e-5)
                    dot = np.dot(fwd, dir_to_star)
                    if dot > min_dist_angle and dist < 50:
                        min_dist_angle = dot
                        closest_star = star

        self.inspected_star = closest_star

        if self.particles:
            ppoints = np.array([p.pos for p in self.particles])
            pproj = self.engine.project(ppoints, self.camera)
            for i, p in enumerate(pproj):
                if p[3] > 0:
                    part = self.particles[i]
                    draw_list.append((p[2], p[0], p[1], part))
                    
        if self.probes:
            pr_points = np.array([p.pos for p in self.probes])
            pr_proj = self.engine.project(pr_points, self.camera)
            for i, p in enumerate(pr_proj):
                if p[3] > 0:
                     probe = self.probes[i]
                     draw_list.append((p[2], p[0], p[1], probe))

        draw_list.sort(key=lambda x: x[0])
        view_mode = self.view_modes[self.view_mode_idx]
        
        for item in draw_list:
            depth, sx, sy, obj = item
            if isinstance(obj, Probe):
                 r = int(5 * depth)
                 pygame.draw.circle(self.screen, (255, 100, 255), (int(sx), int(sy)), r)
                 pygame.draw.rect(self.screen, (255, 255, 255), (sx-r, sy-r, r*2, r*2), 1)
            elif isinstance(obj, Particle):
                dim_col = tuple(max(0, min(255, int(c * (obj.life/obj.max_life)))) for c in obj.color)
                pygame.draw.circle(self.screen, dim_col, (int(sx), int(sy)), 2)
            else: 
                r = max(2, obj.size * depth)
                color = obj.color
                if view_mode == "PERFORMANCE":
                    acc = obj.raw_data.get('accuracy', 0)
                    color = (int(255*(1-acc)), int(255*acc), 0)
                elif view_mode == "CHRONO":
                    age = max(0, min(1, (obj.id / (len(stars)+1))))
                    v = int(age * 255)
                    color = (v, v, v)
                if obj == self.inspected_star:
                     pygame.draw.circle(self.screen, (255, 255, 255), (int(sx), int(sy)), int(r)+6, 1)
                if obj.raw_data['status'] == 'running':
                     pygame.draw.circle(self.screen, (200, 255, 200), (int(sx), int(sy)), int(r)+4, 1)
                elif obj.raw_data['status'] == 'failed':
                     pygame.draw.circle(self.screen, (255, 0, 0), (int(sx), int(sy)), int(r)+4, 1)
                pygame.draw.circle(self.screen, color, (int(sx), int(sy)), int(r))

        if not self.photo_mode:
            self.draw_hud()
            
            # Draw Sector Name Center (Fade in/out?)
            # Just static for now
            sname = self.get_sector_name(self.camera.pos)
            stxt = self.font_sector.render(sname, True, (100, 255, 255))
            self.screen.blit(stxt, (WIDTH//2 - stxt.get_width()//2, HEIGHT - 50))
            
        pygame.display.flip()

    def draw_trail(self):
        if len(self.trail) < 2: return
        t_arr = np.array(self.trail)
        t_proj = self.engine.project(t_arr, self.camera)
        
        # Line strip
        pts = []
        for i in range(len(t_proj)):
             if t_proj[i][3] > 0:
                 pts.append((t_proj[i][0], t_proj[i][1]))
             else:
                 # Break strip if behind camera?
                 if len(pts) > 1:
                     pygame.draw.lines(self.screen, (50, 100, 200), False, pts, 1)
                 pts = []
        if len(pts) > 1:
            pygame.draw.lines(self.screen, (50, 100, 200), False, pts, 1)
        
    def draw_grid(self):
        # Draw a grid at y = -10 (floor of the parameter space generally)
        # Or centered at y=0?
        # Let's draw a grid at y = -10 (small hidden dim) and y = 10 (large hidden dim)
        pass # Too expensive in python loop properly without optimization?
        # Let's do a few axis lines
        # X Axis (Steps)
        starts = np.array([[-20, 0, 0], [0, -20, 0], [0, 0, -20]])
        ends = np.array([[20, 0, 0], [0, 20, 0], [0, 0, 20]])
        colors = [(255, 50, 50), (50, 255, 50), (50, 50, 255)] # R, G, B axes
        
        # Project starts/ends
        p_starts = self.engine.project(starts, self.camera)
        p_ends = self.engine.project(ends, self.camera)
        
        for i in range(3):
            if p_starts[i][3]>0 and p_ends[i][3]>0:
                pygame.draw.line(self.screen, colors[i], (p_starts[i][0], p_starts[i][1]), (p_ends[i][0], p_ends[i][1]), 1)

    def draw_hud(self):
        cx, cy = WIDTH//2, HEIGHT//2
        col = COL_WARN if self.scan_cooldown > 0 else COL_HUD
        
        offset_x = -self.camera.rot_vel[0] * 500
        offset_y = self.camera.rot_vel[1] * 500
        
        pygame.draw.line(self.screen, col, (cx-15+offset_x, cy+offset_y), (cx+15+offset_x, cy+offset_y), 2)
        pygame.draw.line(self.screen, col, (cx+offset_x, cy-15+offset_y), (cx+offset_x, cy+15+offset_y), 2)
        pygame.draw.circle(self.screen, COL_HUD_DIM, (cx, cy), 200, 1)
        
        if self.scan_active:
            r = 200 * (1.0 - self.scan_pulse)
            pygame.draw.circle(self.screen, (0, 255, 255), (cx, cy), int(r), 2)

        self.draw_panel_left()
        self.draw_panel_right()
        self.draw_inspector()
        self.draw_objectives()
        
        task_txt = self.font_title.render(f"OPERATION: {self.tasks[self.task_idx].upper()}", True, (100, 200, 255))
        self.screen.blit(task_txt, (WIDTH//2 - task_txt.get_width()//2, 10))

    def draw_objectives(self):
        y = 100
        x = 20
        self.screen.blit(self.font_large.render("MISSION LOG:", True, COL_HUD), (x, y))
        y += 25
        for obj in self.objectives:
            col = (100, 255, 100) if obj.completed else (150, 150, 150)
            mark = "[X]" if obj.completed else "[ ]"
            txt = self.font_small.render(f"{mark} {obj.text}", True, col)
            self.screen.blit(txt, (x, y))
            y += 20

    def draw_panel_left(self):
        params = self.pos_to_params(self.camera.pos)
        lines = [
            "NAVIGATOR STATUS",
            f"XYZ: {self.camera.pos[0]:.1f} {self.camera.pos[1]:.1f} {self.camera.pos[2]:.1f}",
            "",
            "TARGET HYPERPARAMS:",
            f" [STEPS]  {params['steps']}",
            f" [HIDDEN] {params['hidden_dim']}",
            f" [L_RATE] {params['learning_rate']:.5f}",
        ]
        x, y = 20, HEIGHT - 180
        for l in lines:
            col = COL_HUD if l.strip() else COL_HUD
            txt = self.font_small.render(l, True, col)
            self.screen.blit(txt, (x, y))
            y += 18

    def draw_panel_right(self):
        model_spec = self.available_models[self.model_idx]
        task = self.tasks[self.task_idx]
        compat = not (model_spec.task_compat and task not in model_spec.task_compat)
        col_model = COL_HUD if compat else COL_WARN
        
        lines = [
            "VEHICLE SYSTEM:",
            f" > {model_spec.name.upper()}",
            f"   Type: {model_spec.model_type}",
            f"   Compat: {'OK' if compat else 'INCOMPATIBLE'}",
            "",
            f"THREADS: {len(self.launcher.active_jobs)}",
            f"DATA: {len(self.data_mgr.stars)}"
        ]
        x = WIDTH - 300
        y = 50
        for l in lines:
            txt = self.font_small.render(l, True, col_model)
            self.screen.blit(txt, (x, y))
            y += 18
            
    def draw_inspector(self):
        if not self.inspected_star:
            return
        s = self.inspected_star
        d = s.raw_data
        rect = pygame.Rect(WIDTH//2 - 200, HEIGHT - 220, 400, 200)
        pygame.draw.rect(self.screen, (0, 0, 0), rect) 
        pygame.draw.rect(self.screen, COL_HUD, rect, 1)
        
        lines = [
            f"TRIAL ID: {s.id}",
            f"MODEL: {d['model_name']}",
            f"STATUS: {d['status'].upper()}",
            f"ACCURACY: {d.get('accuracy', 0):.2%}",
            f"LOSS: {d.get('final_loss', 0):.4f}",
            f"TIME: {d.get('iteration_time', 0):.2f}s",
            "",
            "CONFIG:",
            str(d['config_json'])[:45] + "..."
        ]
        x = rect.x + 10
        y = rect.y + 10
        for l in lines:
            self.screen.blit(self.font_large.render(l, True, (255, 255, 255)), (x, y))
            y += 24

    def pos_to_params(self, pos):
        steps = int(np.clip(55 + pos[0] * 2, 5, 200))
        power = 7 + (pos[1] / 5.0) 
        hidden = int(2 ** power)
        hidden = max(16, min(2048, hidden))
        power = -3 + (pos[2] / 10.0)
        lr = 10 ** power
        return {
            'steps': steps,
            'hidden_dim': hidden,
            'learning_rate': lr
        }

    def cleanup(self):
        music.stop()
        self.data_mgr.stop()
        self.launcher.cleanup()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    Game().run()
