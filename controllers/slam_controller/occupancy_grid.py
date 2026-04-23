"""
occupancy_grid.py - Log-odds occupancy grid (ENU: X=East, Y=North).

WeBots E-puck LiDAR angles increase CLOCKWISE, but standard math is CCW.
So: world_angle = robot_heading - local_angle
(NOT + local_angle)

Verified from debug data:
  heading=0 (East), raw+90deg = South wall (0.15m) → raw+90 = -Y direction
  world_angle = 0 - 90deg = -90deg = South ✓
"""
import math
import numpy as np


class OccupancyGrid:
    def __init__(self, width, height, resolution=0.05):
        self.width      = width
        self.height     = height
        self.resolution = resolution
        self.grid_w     = int(math.ceil(width  / resolution))
        self.grid_h     = int(math.ceil(height / resolution))
        self.grid       = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self.l_occ      =  2
        self.l_free     = -0.1
        self.l_max      =  4.0
        self.l_min      = -4.0

    def world_to_grid(self, wx, wy):
        col = max(0, min(self.grid_w-1, int(wx / self.resolution)))
        row = max(0, min(self.grid_h-1, int(wy / self.resolution)))
        return col, row

    def update(self, robot_x, robot_y, robot_theta, scan_ranges, max_range=1.5):
        """
        world_angle = robot_theta - local_angle
        (LiDAR is clockwise, math is counter-clockwise)
        """
        r_col, r_row = self.world_to_grid(robot_x, robot_y)

        for local_angle, r in scan_ranges:
            hit         = r < max_range * 0.99
            # KEY FIX: subtract local_angle (clockwise → CCW conversion)
            world_angle = robot_theta - local_angle
            end_r       = r if hit else max_range

            hit_x = robot_x + end_r * math.cos(world_angle)
            hit_y = robot_y + end_r * math.sin(world_angle)
            h_col, h_row = self.world_to_grid(hit_x, hit_y)

            cells = self._bresenham(r_col, r_row, h_col, h_row)
            if not cells:
                continue
            for col, row in cells[:-1]:
                self.grid[row, col] = max(self.l_min,
                                          self.grid[row, col] + self.l_free)
            if hit:
                col, row = cells[-1]
                self.grid[row, col] = min(self.l_max,
                                          self.grid[row, col] + self.l_occ)

    def _bresenham(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1-x0); dy = abs(y1-y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            if 0 <= x0 < self.grid_w and 0 <= y0 < self.grid_h:
                cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 <  dx: err += dx; y0 += sy
        return cells

    def get_rgb_array(self):
        """(H,W,3) uint8: 128=unknown, 255=free, 0=occupied. Row-0=Y=0(south)."""
        g    = np.clip(self.grid, -8, 8)
        prob = 1.0 - 1.0 / (1.0 + np.exp(g))
        v    = (255 * (1.0 - prob)).astype(np.uint8)
        v[np.abs(self.grid) < 0.05] = 128
        return np.stack([v, v, v], axis=2)
