"""
graph_planner.py — A* path planner operating on GraphMap (Milestone 3).

Unlike the Milestone 2 planner that needed a fixed-size numpy array,
this planner works directly on the sparse graph dictionary.

C-Space inflation
-----------------
All wall cells are collected from the graph and dilated by INFLATION_R
cells (Chebyshev distance).  The inflated set is built once per plan()
call — no numpy required.

Goal snapping
-------------
If the goal lands inside an inflated/wall cell, BFS outward finds the
nearest free cell.

Returns a smoothed list of world-coordinate waypoints (wx, wy).

No external libraries — only math and heapq.
"""
import math
import heapq
from collections import deque

# ── Tuneable constants ────────────────────────────────────────────────────────

WALL_THRESH   = 0.3     # log-odds above this → wall
INFLATION_R   = 3       # cells of C-Space padding
WAYPOINT_DIST = 0.20    # m — minimum spacing between kept waypoints
MAX_EXPANSIONS = 8000   # A* gives up after this many node expansions


# ── Helpers ──────────────────────────────────────────────────────────────────

def _heuristic(c0, r0, c1, r1):
    """Euclidean grid-distance heuristic."""
    dc = c1 - c0; dr = r1 - r0
    return math.sqrt(dc * dc + dr * dr)


_MOVES = [
    (-1,  0, 1.0), ( 1,  0, 1.0), ( 0, -1, 1.0), ( 0,  1, 1.0),
    (-1, -1, 1.4142), (-1,  1, 1.4142),
    ( 1, -1, 1.4142), ( 1,  1, 1.4142),
]


def _inflate(wall_set, radius):
    """
    Dilate wall_set by `radius` cells (Chebyshev distance).
    Returns a new set containing original walls + inflated cells.
    """
    if radius <= 0:
        return set(wall_set)
    inflated = set(wall_set)
    for (wc, wr) in wall_set:
        for dc in range(-radius, radius + 1):
            for dr in range(-radius, radius + 1):
                inflated.add((wc + dc, wr + dr))
    return inflated


def _smooth_path(path, blocked_set):
    """
    Greedy line-of-sight smoother (string-pulling) on grid cells.
    Removes intermediate waypoints with clear Bresenham lines between them.
    """
    if len(path) <= 2:
        return path

    def los(c0, r0, c1, r1):
        dx = abs(c1 - c0); dy = abs(r1 - r0)
        sx = 1 if c0 < c1 else -1
        sy = 1 if r0 < r1 else -1
        err = dx - dy; x, y = c0, r0
        while True:
            if (x, y) in blocked_set:
                return False
            if x == c1 and y == r1:
                return True
            e2 = 2 * err
            if e2 > -dy: err -= dy; x += sx
            if e2 <  dx: err += dx; y += sy

    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if los(path[i][0], path[i][1], path[j][0], path[j][1]):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed


def _thin_waypoints(world_path, min_dist):
    """Remove waypoints closer than min_dist to the previous kept one."""
    if not world_path:
        return []
    out = [world_path[0]]
    for pt in world_path[1:-1]:
        dx = pt[0] - out[-1][0]; dy = pt[1] - out[-1][1]
        if math.sqrt(dx*dx + dy*dy) >= min_dist:
            out.append(pt)
    if len(world_path) > 1:
        out.append(world_path[-1])
    return out


# ── Planner class ────────────────────────────────────────────────────────────

class GraphPlanner:
    """
    A* planner operating directly on a GraphMap (sparse dict).

    Usage
    -----
        planner  = GraphPlanner(resolution=0.05)
        waypoints = planner.plan(graph_map, start_world, goal_world)
    """

    def __init__(self, resolution=0.05,
                 wall_thresh=WALL_THRESH,
                 inflation_r=INFLATION_R,
                 waypoint_dist=WAYPOINT_DIST):
        self.res          = resolution
        self.wall_thresh  = wall_thresh
        self.inflation_r  = inflation_r
        self.wp_dist      = waypoint_dist

    # ── Public ───────────────────────────────────────────────────────────────

    def plan(self, graph_map, start_world, goal_world):
        """
        Run A* from start_world to goal_world on the graph map.

        Parameters
        ----------
        graph_map          : GraphMap
        start_world        : (x, y)  robot position
        goal_world         : (x, y)  target position

        Returns list of (wx, wy) world waypoints, or [] on failure.
        """
        # 1. Build inflated blocked set from static walls
        walls    = graph_map.get_wall_set(self.wall_thresh)
        blocked  = _inflate(walls, self.inflation_r)

        # 2. Convert start / goal
        sc, sr = graph_map.world_to_grid(*start_world)
        gc, gr = graph_map.world_to_grid(*goal_world)

        # 3. Goal snapping: if goal lands inside a wall/inflated cell,
        #    BFS outward to nearest free cell.
        if (gc, gr) in blocked:
            snapped = self._nearest_free(blocked, gc, gr)
            if snapped is None:
                print("[A*-G] Goal completely enclosed — no path.")
                return []
            oc, or_ = gc, gr
            gc, gr = snapped
            ow = graph_map.grid_to_world(oc, or_)
            nw = graph_map.grid_to_world(gc, gr)
            print(f"[A*-G] Goal snapped ({ow[0]:.2f},{ow[1]:.2f}) → "
                  f"({nw[0]:.2f},{nw[1]:.2f})")

        # Ensure start is walkable (robot is physically there)
        blocked.discard((sc, sr))

        # 4. Run A*
        path = self._astar(blocked, sc, sr, gc, gr)
        if not path:
            print(f"[A*-G] No path ({sc},{sr})→({gc},{gr})")
            return []

        # 5. Smooth
        smoothed = _smooth_path(path, blocked)

        # 6. Convert to world coords
        world_path = [graph_map.grid_to_world(c, r) for c, r in smoothed]

        # 7. Thin
        thinned = _thin_waypoints(world_path, self.wp_dist)

        print(f"[A*-G] Path: {len(path)} cells → {len(smoothed)} smooth → "
              f"{len(thinned)} waypoints")
        return thinned

    # ── A* core ──────────────────────────────────────────────────────────────

    @staticmethod
    def _astar(blocked, sc, sr, gc, gr):
        """
        A* search on infinite 8-connected grid.
        blocked : set of (col, row) that cannot be traversed.
        Returns list of (col, row) from start to goal, or [].

        Has a MAX_EXPANSIONS safety limit to prevent freezing
        when the goal is unreachable or very far away.
        """
        start = (sc, sr); goal = (gc, gr)
        g_cost = {start: 0.0}
        parent = {start: None}
        open_heap = [(_heuristic(sc, sr, gc, gr), sc, sr)]
        closed = set()
        expansions = 0

        while open_heap:
            _, c, r = heapq.heappop(open_heap)
            node = (c, r)
            if node in closed:
                continue
            closed.add(node)
            expansions += 1

            # Safety: abort if search takes too long
            if expansions > MAX_EXPANSIONS:
                print(f"[A*-G] Exceeded {MAX_EXPANSIONS} expansions — aborting")
                return []

            if node == goal:
                path = []
                n = goal
                while n is not None:
                    path.append(n)
                    n = parent[n]
                path.reverse()
                return path

            for dc, dr, cost in _MOVES:
                nc, nr = c + dc, r + dr
                nb = (nc, nr)
                if nb in blocked or nb in closed:
                    continue
                new_g = g_cost[node] + cost
                if new_g < g_cost.get(nb, math.inf):
                    g_cost[nb] = new_g
                    parent[nb] = node
                    f = new_g + _heuristic(nc, nr, gc, gr)
                    heapq.heappush(open_heap, (f, nc, nr))

        return []

    # ── Goal snapping ────────────────────────────────────────────────────────

    @staticmethod
    def _nearest_free(blocked, c, r, max_radius=200):
        """BFS outward to find nearest cell NOT in blocked."""
        visited = {(c, r)}
        queue = deque([(c, r)])
        while queue:
            cc, cr = queue.popleft()
            if (cc, cr) not in blocked:
                return (cc, cr)
            for dc, dr, _ in _MOVES:
                nb = (cc + dc, cr + dr)
                if nb not in visited:
                    visited.add(nb)
                    # Safety: don't search infinitely
                    if abs(nb[0] - c) > max_radius:
                        continue
                    queue.append(nb)
        return None
