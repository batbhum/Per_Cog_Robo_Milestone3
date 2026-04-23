"""
path_planning.py - A* path planner for E-puck SLAM (Milestone 2+).

Operates directly on the OccupancyGrid log-odds array.

WALL threshold: cells with log-odds > WALL_THRESH are treated as occupied.
Cells at or below WALL_THRESH (including ~0 = unknown) are WALKABLE —
this lets the robot explore unmapped territory without getting stuck.

C-Space inflation
-----------------
Before A* runs, every confirmed wall cell is dilated outward by INFLATION_R
grid cells in all directions.  This converts the discrete grid into
Configuration Space (C-Space) — the robot is treated as a point but the
obstacles grow by the robot's body radius, so any path A* finds through the
inflated-free space is guaranteed to clear the physical walls.

Goal snapping
-------------
If the requested goal lands inside an inflated (or real) wall cell, a BFS
spiral outward finds the nearest free cell and snaps the goal there
automatically, so A* never fails just because a waypoint was placed in a wall.

Returns a smoothed list of world-coordinate waypoints (wx, wy).

No external libraries — only math, numpy, and heapq (stdlib).
"""
import math
import heapq
import numpy as np


# ── Tuneable constants ────────────────────────────────────────────────────────

WALL_THRESH    = 0.3    # log-odds above this → cell classified as wall
                        # (lowered from 0.5 so weakly-seen walls also inflate)
INFLATION_R    = 3      # cells of C-Space padding  (3 × 0.05 m = 0.15 m ≈ 4×body radius)
WAYPOINT_DIST  = 0.20   # m — minimum spacing between kept waypoints

# ─────────────────────────────────────────────────────────────────────────────


def _inflate_grid(binary_wall, radius):
    """
    Morphological dilation of binary_wall by a square structuring element
    of side (2*radius + 1) — pure numpy, no scipy.

    Algorithm: loop over each of the (2r+1)² kernel offsets (dr, dc) and
    OR-shift the source array into the output.  For r=3 this is 48 vectorised
    numpy operations on the whole grid, far faster than iterating over
    individual wall pixels.

    Returns
    -------
    inflated : bool ndarray same shape as binary_wall
        True = original wall  OR  within `radius` cells of a wall.
    """
    if radius <= 0:
        return binary_wall.copy()

    rows, cols = binary_wall.shape
    inflated   = binary_wall.copy()

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr == 0 and dc == 0:
                continue   # centre — already copied

            # Compute non-overlapping source / destination slices so we never
            # read out-of-bounds.  A positive dr means "shift rows downward":
            #   source rows  : [max(0,-dr) … rows-dr)
            #   dest   rows  : [max(0, dr) … rows+dr)  (clamped to rows)
            r_src = slice(max(0, -dr), min(rows, rows - dr))
            c_src = slice(max(0, -dc), min(cols, cols - dc))
            r_dst = slice(max(0,  dr), min(rows, rows + dr))
            c_dst = slice(max(0,  dc), min(cols, cols + dc))

            inflated[r_dst, c_dst] |= binary_wall[r_src, c_src]

    return inflated


def _heuristic(r0, c0, r1, c1):
    """Euclidean distance heuristic (admissible for 8-connected grid)."""
    dr = r1 - r0
    dc = c1 - c0
    return math.sqrt(dr * dr + dc * dc)


# 8-connected neighbours with move costs
_NEIGHBOURS = [
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, 1.4142),
    (-1,  1, 1.4142),
    ( 1, -1, 1.4142),
    ( 1,  1, 1.4142),
]


def _astar_grid(walkable, start, goal):
    """
    Core A* search on a boolean walkable grid.
    walkable: (H, W) bool — True = can traverse
    start, goal: (row, col) ints
    Returns list of (row, col) from start→goal, or [] if no path.
    """
    rows, cols = walkable.shape
    sr, sc = start
    gr, gc = goal

    if not walkable[sr, sc]:
        # If start is inside a wall (shouldn't happen), relax it
        pass
    if not walkable[gr, gc]:
        return []   # goal is a hard wall

    # g_cost array — infinity everywhere
    g = np.full((rows, cols), math.inf, dtype=np.float64)
    g[sr, sc] = 0.0

    parent = {}         # (r,c) → (pr, pc)
    parent[(sr, sc)] = None

    # Min-heap: (f, r, c)
    open_heap = []
    heapq.heappush(open_heap, (0.0 + _heuristic(sr, sc, gr, gc), sr, sc))

    closed = np.zeros((rows, cols), dtype=bool)

    while open_heap:
        f, r, c = heapq.heappop(open_heap)

        if closed[r, c]:
            continue
        closed[r, c] = True

        if r == gr and c == gc:
            # Reconstruct path
            path = []
            node = (gr, gc)
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        for dr, dc, cost in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if not walkable[nr, nc]:
                continue
            if closed[nr, nc]:
                continue
            new_g = g[r, c] + cost
            if new_g < g[nr, nc]:
                g[nr, nc] = new_g
                parent[(nr, nc)] = (r, c)
                f_new = new_g + _heuristic(nr, nc, gr, gc)
                heapq.heappush(open_heap, (f_new, nr, nc))

    return []   # no path found


def _smooth_path(grid_path, walkable):
    """
    Greedy line-of-sight smoother (string-pulling).
    Removes intermediate waypoints that have a clear straight-line path
    between their neighbours, reducing the path to essential turns only.
    """
    if len(grid_path) <= 2:
        return grid_path

    rows, cols = walkable.shape

    def los(r0, c0, r1, c1):
        """Bresenham line-of-sight check between two grid cells."""
        dx = abs(c1 - c0); dy = abs(r1 - r0)
        sx = 1 if c0 < c1 else -1
        sy = 1 if r0 < r1 else -1
        err = dx - dy
        x, y = c0, r0
        while True:
            if not (0 <= y < rows and 0 <= x < cols):
                return False
            if not walkable[y, x]:
                return False
            if x == c1 and y == r1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x += sx
            if e2 < dx:
                err += dx; y += sy

    smoothed = [grid_path[0]]
    i = 0
    while i < len(grid_path) - 1:
        # Find the farthest point reachable in a straight line
        j = len(grid_path) - 1
        while j > i + 1:
            r0, c0 = grid_path[i]
            r1, c1 = grid_path[j]
            if los(r0, c0, r1, c1):
                break
            j -= 1
        smoothed.append(grid_path[j])
        i = j

    return smoothed


def _thin_waypoints(world_path, min_dist):
    """
    Remove waypoints that are closer than `min_dist` to the previous kept one.
    Always keeps the final waypoint.
    """
    if not world_path:
        return []
    out = [world_path[0]]
    for pt in world_path[1:-1]:
        dx = pt[0] - out[-1][0]
        dy = pt[1] - out[-1][1]
        if math.sqrt(dx * dx + dy * dy) >= min_dist:
            out.append(pt)
    if len(world_path) > 1:
        out.append(world_path[-1])
    return out


# ─────────────────────────────────────────────────────────────────────────────

class AStarPlanner:
    """
    Stateless A* planner. Call plan() each time a new path is needed.

    Usage:
        planner = AStarPlanner(grid_resolution=0.05)
        waypoints = planner.plan(occ_grid.grid, start_gxy, goal_gxy, occ_grid)

    grid_resolution: metres per cell (must match OccupancyGrid.resolution)
    """

    def __init__(self, grid_resolution=0.05,
                 wall_thresh=WALL_THRESH,
                 inflation_r=INFLATION_R,
                 waypoint_dist=WAYPOINT_DIST):
        self.res          = grid_resolution
        self.wall_thresh  = wall_thresh
        self.inflation_r  = inflation_r
        self.waypoint_dist = waypoint_dist

    # ── public ────────────────────────────────────────────────────────────────

    def world_to_grid(self, wx, wy, grid_w, grid_h):
        """Convert world (m) → (col, row) grid indices."""
        col = int(wx / self.res)
        row = int(wy / self.res)
        col = max(0, min(grid_w - 1, col))
        row = max(0, min(grid_h - 1, row))
        return col, row

    def grid_to_world(self, col, row):
        """Convert grid (col, row) → world (wx, wy) at cell centre."""
        wx = (col + 0.5) * self.res
        wy = (row + 0.5) * self.res
        return wx, wy

    def plan(self, log_odds_grid, start_world, goal_world):
        """
        Run A* from start to goal.

        Parameters
        ----------
        log_odds_grid : np.ndarray shape (H, W), dtype float32
            The raw OccupancyGrid.grid array.
        start_world   : (wx, wy) in metres
        goal_world    : (wx, wy) in metres

        Returns
        -------
        list of (wx, wy) world-coordinate waypoints, or [] on failure.
        The list always starts just ahead of start and ends at goal.
        """
        grid_h, grid_w = log_odds_grid.shape

        # 1. Build binary wall map
        wall_binary = log_odds_grid > self.wall_thresh   # True = wall

        # 2. Inflate walls for robot body clearance
        inflated = _inflate_grid(wall_binary, self.inflation_r)

        # 3. walkable = NOT inflated wall
        walkable = ~inflated

        # 4. Convert start / goal to grid coords
        sc, sr = self.world_to_grid(start_world[0], start_world[1], grid_w, grid_h)
        gc, gr = self.world_to_grid(goal_world[0],  goal_world[1],  grid_w, grid_h)

        # ── Goal snapping ─────────────────────────────────────────────────────
        # If the goal lands in a real wall OR in the inflated safety margin,
        # BFS outward to find the nearest genuinely free cell.
        # This handles goals placed against walls, inside pillars, or in
        # corners where inflation has consumed the originally-requested cell.
        if not walkable[gr, gc]:
            orig_wx, orig_wy = self.grid_to_world(gc, gr)
            snapped_c, snapped_r = self._nearest_free(walkable, gc, gr)
            if snapped_c is None:
                print("[A*] Goal is completely enclosed — no path possible.")
                return []
            gc, gr = snapped_c, snapped_r
            snap_wx, snap_wy = self.grid_to_world(gc, gr)
            print(f"[A*] Goal snapped: ({orig_wx:.2f},{orig_wy:.2f}) → "
                  f"({snap_wx:.2f},{snap_wy:.2f})  "
                  f"[requested cell was inside wall/inflation zone]")

        # Ensure the start cell is walkable — the robot is physically there,
        # so even if inflation has marked it occupied, A* must begin there.
        walkable[sr, sc] = True

        # 5. Run A*
        # A* works in (row, col) convention internally
        grid_path = _astar_grid(walkable, (sr, sc), (gr, gc))

        if not grid_path:
            print(f"[A*] No path from ({sc},{sr}) to ({gc},{gr})")
            return []

        # 6. Smooth path (string-pulling)
        smoothed = _smooth_path(grid_path, walkable)

        # 7. Convert to world coordinates
        world_path = [self.grid_to_world(c, r) for r, c in smoothed]

        # 8. Thin waypoints
        thinned = _thin_waypoints(world_path, self.waypoint_dist)

        print(f"[A*] Path found: {len(grid_path)} cells → "
              f"{len(smoothed)} smoothed → {len(thinned)} waypoints")
        return thinned

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _nearest_free(walkable, c, r):
        """BFS outward from (c, r) to find nearest walkable cell."""
        rows, cols = walkable.shape
        visited = np.zeros((rows, cols), dtype=bool)
        queue   = [(r, c)]
        visited[r, c] = True
        while queue:
            cr, cc = queue.pop(0)
            if walkable[cr, cc]:
                return cc, cr   # return (col, row)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                    visited[nr, nc] = True
                    queue.append((nr, nc))
        return None, None   # completely enclosed
