import math

import numpy as np


class Camera:
    def __init__(self, pos=[0, 0, -10], rot=[0, 0]):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array([0.0, 0.0, 0.0], dtype=float)
        self.rot = np.array(rot, dtype=float)  # yaw, pitch
        self.rot_vel = np.array([0.0, 0.0], dtype=float)

        # Physics constants
        self.friction = 0.92
        self.accel_speed = 0.05
        self.max_speed = 2.0
        self.rot_friction = 0.8
        self.mouse_sensitivity = 0.002

    def update(self, dt_scale=1.0):
        # Apply velocity
        self.pos += self.vel * dt_scale
        self.rot += self.rot_vel * dt_scale

        # Apply friction
        # Friction is per frame? 0.92 per frame.
        # correct way: vel = vel * (friction ** dt_scale)
        self.vel *= self.friction**dt_scale
        self.rot_vel *= self.rot_friction**dt_scale

    def thrust(self, dx, dy, dz):
        # Move relative to looking direction
        yaw = self.rot[0]
        pitch = self.rot[1]

        # Forward vector (3D)
        # pitch affects y component
        # yaw affects x/z

        # Simplified "flight" mechanics:
        # We want meaningful 3D movement.
        c_yaw, s_yaw = math.cos(yaw), math.sin(yaw)
        c_pitch, s_pitch = math.cos(pitch), math.sin(pitch)

        # Forward vector
        fwd = np.array([s_yaw * c_pitch, -s_pitch, c_yaw * c_pitch])

        # Right vector (flat on ground mostly)
        right = np.array([c_yaw, 0, -s_yaw])

        # Up vector (relative to camera)
        # Use relative Up for space flight feel.
        up = np.cross(fwd, right)  # Might be inverted

        # Accumulate forces
        accel = (fwd * dz) + (right * dx) + (up * dy)

        self.vel += accel * self.accel_speed

        # Cap speed? (Optional, adds "drag" naturally via friction)

    def rotate_impulse(self, dyaw, dpitch):
        self.rot_vel[0] += dyaw * 0.1
        self.rot_vel[1] += dpitch * 0.1


class Engine3D:
    def __init__(self, width, height, fov=400):
        self.width = width
        self.height = height
        self.cx = width // 2
        self.cy = height // 2
        self.fov = fov

    def project(self, points, camera):
        """
        Project 3D points to 2D.
        points: (N, 3) numpy array
        camera: Camera object
        Returns: (N, 3) array where col 0,1 are x,y screen coords and col 2 is scale/depth
        """
        if len(points) == 0:
            return []

        # Relative to camera
        rel = points - camera.pos

        # Rotate (Yaw)
        yaw = -camera.rot[0]
        c, s = math.cos(yaw), math.sin(yaw)
        # Rotation matrix around Y
        x = rel[:, 0] * c - rel[:, 2] * s
        z = rel[:, 0] * s + rel[:, 2] * c
        y = rel[:, 1]

        # Rotate (Pitch)
        pitch = -camera.rot[1]
        c, s = math.cos(pitch), math.sin(pitch)
        # Rotation around X
        y_final = y * c - z * s
        z_final = y * s + z * c

        # Perspective divide
        # Avoid division by zero, discard points behind camera
        valid = z_final > 0.1

        # We process only valid points, but to keep mapping we might return masks
        # For simplicity in this retro engine, we might just filter

        # z_final is depth
        f = self.fov / (z_final + 1e-5)
        sx = x * f + self.cx
        sy = -y_final * f + self.cy  # Flip Y for screen coords

        return np.column_stack((sx, sy, f, valid))


class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (50, 255, 50)
    RED = (255, 50, 50)
    BLUE = (50, 50, 255)
    YELLOW = (255, 255, 50)
