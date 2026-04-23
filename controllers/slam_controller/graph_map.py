"""
graph_map.py — Sparse graph-based occupancy map (Milestone 3).

Unlike a fixed-size numpy array, this map uses a Python dictionary
keyed by (col, row) grid coordinates.  The graph grows dynamically
as the robot explores — no hardcoded world size, no GPS offset
assumptions, and no starting-position restrictions.

Each occupancy node stores a log-odds value:
  positive → occupied (wall)
  negative → free
  absent/0 → unknown

8-connected neighbors are implicit (computed on demand).
Bresenham ray-tracing adds/updates nodes along each LiDAR ray.

No external libraries beyond math.
"""
import math


# ── Tuneable constants ───────────────────────────────────────────────────────

FREE_DELTA = -0.1
OCC_DELTA  =  2.0
CLAMP_MIN  = -4.0
CLAMP_MAX  =  4.0


class GraphMap:
    """
    Sparse graph-based occupancy map.

    The occupancy grid is a sparse dict keyed by (col, row),
    giving infinite-grid behaviour with no fixed array bounds.
    Works regardless of starting position or map shape.
    """

    _NEIGHBORS_8 = [
        (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
        (-1, -1, 1.4142), (-1,  1, 1.4142),
        ( 1, -1, 1.4142), ( 1,  1, 1.4142),
    ]

    def __init__(self, resolution=0.05,
                 l_occ=OCC_DELTA, l_free=FREE_DELTA,
                 l_max=CLAMP_MAX, l_min=CLAMP_MIN):
        self.resolution = resolution
        self.nodes = {}        # (col, row) → log-odds float
        self.l_occ  = l_occ
        self.l_free = l_free
        self.l_max  = l_max
        self.l_min  = l_min

    # ── Coordinate conversion ────────────────────────────────────────────────

    def world_to_grid(self, wx, wy):
        """World (metres) → grid (col, row).  No clamping — infinite grid."""
        return int(math.floor(wx / self.resolution)), \
               int(math.floor(wy / self.resolution))

    def grid_to_world(self, col, row):
        """Grid (col, row) → world (metres) at cell centre."""
        return (col + 0.5) * self.resolution, (row + 0.5) * self.resolution

    # ── Node access ──────────────────────────────────────────────────────────

    def get(self, col, row):
        """Log-odds of cell.  Returns 0.0 (unknown) if never observed."""
        return self.nodes.get((col, row), 0.0)

    def set(self, col, row, value):
        self.nodes[(col, row)] = max(self.l_min, min(self.l_max, value))

    def is_wall(self, col, row, threshold=0.3):
        return self.get(col, row) > threshold

    def is_free(self, col, row, threshold=-0.5):
        return self.get(col, row) < threshold

    def is_observed(self, col, row):
        return abs(self.get(col, row)) > 0.05

    # ── Graph topology ───────────────────────────────────────────────────────

    def neighbors(self, col, row):
        """Yield (nc, nr, cost) for all 8 adjacent cells."""
        for dc, dr, cost in self._NEIGHBORS_8:
            yield col + dc, row + dr, cost

    # ── Bounds ───────────────────────────────────────────────────────────────

    def get_bounds(self):
        """(min_col, min_row, max_col, max_row) of all observed cells."""
        if not self.nodes:
            return 0, 0, 1, 1
        min_c = min(c for c, _ in self.nodes)
        max_c = max(c for c, _ in self.nodes)
        min_r = min(r for _, r in self.nodes)
        max_r = max(r for _, r in self.nodes)
        return min_c, min_r, max_c, max_r

    def mapped_count(self):
        """Number of cells with significant log-odds (observed)."""
        return sum(1 for v in self.nodes.values() if abs(v) > 0.05)

    # ── Wall-cell query (used by planner for inflation) ──────────────────────

    def get_wall_set(self, threshold=0.3):
        """Return set of (col, row) classified as walls."""
        return {k for k, v in self.nodes.items() if v > threshold}

    # ── LiDAR update ─────────────────────────────────────────────────────────

    def update(self, robot_x, robot_y, robot_theta, scan_ranges, max_range,
               excluded_bearings=None, **_kwargs):
        """
        Integrate one LiDAR scan into the graph via Bresenham ray-casting.

        Parameters
        ----------
        robot_x, robot_y, robot_theta : current robot pose
        scan_ranges : list of (local_angle, range)
        max_range   : sensor maximum range
        excluded_bearings : list of (world_angle_centre, half_width) or None
            Angular zones to skip (detected moving objects).
        """
        r_col, r_row = self.world_to_grid(robot_x, robot_y)

        for local_angle, r in scan_ranges:
            world_angle = robot_theta - local_angle   # CW LiDAR → CCW math

            # Skip rays aimed at detected moving objects
            if excluded_bearings and self._in_exclusion(world_angle,
                                                        excluded_bearings):
                continue

            hit   = r < max_range * 0.99
            end_r = r if hit else max_range

            hit_x = robot_x + end_r * math.cos(world_angle)
            hit_y = robot_y + end_r * math.sin(world_angle)
            h_col, h_row = self.world_to_grid(hit_x, hit_y)

            cells = self._bresenham(r_col, r_row, h_col, h_row)
            if not cells:
                continue

            # Free-space rays
            for c, rr in cells[:-1]:
                old = self.get(c, rr)
                self.set(c, rr, old + self.l_free)

            # Occupied endpoint
            if hit:
                c, rr = cells[-1]
                old = self.get(c, rr)
                self.set(c, rr, old + self.l_occ)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _in_exclusion(angle, zones):
        for centre, half_w in zones:
            diff = angle - centre
            while diff >  math.pi: diff -= 2 * math.pi
            while diff < -math.pi: diff += 2 * math.pi
            if abs(diff) < half_w:
                return True
        return False

    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        """Bresenham line — no bounds check (infinite graph)."""
        cells = []
        dx = abs(x1 - x0); dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy: err -= dy; x0 += sx
            if e2 <  dx: err += dx; y0 += sy
        return cells

    # ── Frontier detection (for autonomous exploration) ──────────────────────

    def nearest_frontier(self, robot_x, robot_y):
        """
        Find the nearest frontier cell to the robot.

        A frontier is a FREE cell (log-odds < -0.1) that has at least one
        8-connected neighbor that is UNKNOWN (not in self.nodes).

        Returns (wx, wy) world coordinates of the nearest frontier,
        or None if no frontier exists.
        """
        rc, rr = self.world_to_grid(robot_x, robot_y)
        best = None
        best_dist2 = float('inf')

        for (c, r), val in self.nodes.items():
            if val >= -0.1:
                continue   # not free — skip walls and barely-seen cells

            # Check if any neighbor is unknown
            is_frontier = False
            for dc, dr, _ in self._NEIGHBORS_8:
                if (c + dc, r + dr) not in self.nodes:
                    is_frontier = True
                    break

            if is_frontier:
                dist2 = (c - rc) ** 2 + (r - rr) ** 2
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best = (c, r)

        if best is None:
            return None
        return self.grid_to_world(*best)
