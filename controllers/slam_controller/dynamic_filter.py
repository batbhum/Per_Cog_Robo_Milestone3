"""
dynamic_filter.py — Camera + LiDAR fusion for moving-object detection
                    (Milestone 3).

Three independent motion signals combined with OR-fusion:

1. CAMERA — Vivid-colour detection + frame-to-frame differencing.
   Walls and floors are grey/brown (low saturation).  Moving balls are
   vivid magenta, cyan, red, or orange (high saturation).  Any camera
   column that contains a highly-saturated region is flagged.

2. LIDAR — Scan-to-scan range consistency.  For every beam we compare
   the current range to the previous scan.  A significant range jump
   indicates something appeared or disappeared.

3. MAP CONSISTENCY — If a LiDAR hit lands in a cell that the map has
   previously confirmed as FREE (negative log-odds below threshold),
   this indicates a transient object has appeared in known free space.

FUSION RULE (OR — aggressive):
   A ray is excluded from map creation when ANY of:
     (a) Camera detects vivid + changing object at that bearing, OR
     (b) LiDAR shows a large range jump at that beam index, OR
     (c) The hit lands in confirmed free space (map-consistency).
   UNLESS the hit cell already has very high log-odds (permanent wall).

Active cell erasure decays wrongly-mapped cells toward free-space.

Works WITHOUT a camera — signals (b) and (c) still function.
"""
import math
import numpy as np


class DynamicFilter:
    """
    Combines camera colour/motion detection with LiDAR scan-to-scan
    consistency and map-consistency to identify and exclude moving
    objects from mapping.
    """

    def __init__(self, camera_fov=0.84, camera_width=160,
                 saturation_threshold=0.40,
                 diff_threshold=12,
                 min_motion_cols=3,
                 lidar_jump_threshold=0.12,
                 free_space_threshold=-1.5,
                 wall_confirm_thresh=3.0,
                 angular_margin=0.10,
                 max_excluded=80):
        """
        Parameters
        ----------
        camera_fov           : float — camera horizontal FOV in radians
        camera_width         : int   — camera image width in pixels
        saturation_threshold : float — HSV saturation above which a pixel
                               is "vivid" (non-wall)
        diff_threshold       : float — per-pixel brightness difference
                               to confirm motion
        min_motion_cols      : int   — minimum flagged columns to form a zone
        lidar_jump_threshold : float — minimum range change (m) between
                               consecutive scans to flag a beam
        free_space_threshold : float — log-odds below which a cell is
                               considered confirmed free space
        wall_confirm_thresh  : float — log-odds above which a cell is
                               permanently confirmed wall (never excluded)
        angular_margin       : float — extra radians padding around zones
        max_excluded         : int   — cap on excluded rays per scan
        """
        self.cam_fov        = camera_fov
        self.cam_w          = camera_width
        self.sat_thr        = saturation_threshold
        self.diff_thr       = diff_threshold
        self.min_cols       = min_motion_cols
        self.lidar_jump_thr = lidar_jump_threshold
        self.free_thr       = free_space_threshold
        self.wall_thr       = wall_confirm_thresh
        self.margin         = angular_margin
        self._max_excluded  = max_excluded

        # State
        self._prev_gray     = None
        self._prev_scan     = None
        self._cam_zones     = []       # [(world_angle_centre, half_width)]
        self._lidar_flags   = set()    # beam indices with range jumps
        self._dyn_count     = 0
        self._has_camera    = False    # set True after first process_camera

    # ══════════════════════════════════════════════════════════════════════════
    #  CAMERA: vivid-colour + frame difference
    # ══════════════════════════════════════════════════════════════════════════

    def process_camera(self, rgb_array, robot_theta):
        """
        Detect moving objects via high-saturation colour detection
        combined with frame-to-frame brightness change.

        Parameters
        ----------
        rgb_array   : (H, W, 3) uint8
        robot_theta : heading in radians

        Returns
        -------
        list of (world_angle_centre, half_width) exclusion zones
        """
        self._has_camera = True
        H, W = rgb_array.shape[:2]
        r = rgb_array[:, :, 0].astype(np.float32)
        g = rgb_array[:, :, 1].astype(np.float32)
        b = rgb_array[:, :, 2].astype(np.float32)

        # ── Per-pixel saturation (simplified HSV-S) ──
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        sat = np.where(mx > 1.0, 1.0 - mn / mx, 0.0)

        # Per-column: fraction of pixels with high saturation
        vivid_mask = sat > self.sat_thr
        vivid_frac = np.mean(vivid_mask, axis=0)    # shape (W,)
        # Column is "vivid" if >12% of its pixels are saturated
        col_vivid = vivid_frac > 0.12

        # ── Frame-to-frame difference ──
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        if self._prev_gray is None:
            self._prev_gray = gray
            self._cam_zones = []
            return []

        diff = np.abs(gray - self._prev_gray)
        self._prev_gray = gray.copy()

        # Per-column mean difference
        col_diff = np.mean(diff, axis=0)
        # Ego-motion compensation
        median_diff = np.median(col_diff)
        residual = col_diff - median_diff

        # Column has motion if brightness changed
        col_motion = residual > self.diff_thr

        # ── Combined: vivid AND (motion OR very vivid) ──
        # Very vivid columns (>25% saturation) always flagged
        col_very_vivid = vivid_frac > 0.25
        flagged = col_vivid & (col_motion | col_very_vivid)

        # ── Cluster contiguous columns into zones ──
        zones = []
        start = None
        for x in range(W):
            if flagged[x]:
                if start is None:
                    start = x
            else:
                if start is not None:
                    if x - start >= self.min_cols:
                        zones.append((start, x))
                    start = None
        if start is not None and W - start >= self.min_cols:
            zones.append((start, W))

        # ── Convert pixel columns → world-frame angular bearings ──
        exclusions = []
        for x1, x2 in zones:
            cx = (x1 + x2) / 2.0
            offset = (self.cam_w / 2.0 - cx) / self.cam_w * self.cam_fov
            world_angle = robot_theta + offset
            half_w = ((x2 - x1) / self.cam_w * self.cam_fov / 2.0
                      + self.margin)
            exclusions.append((world_angle, half_w))

        self._cam_zones = exclusions
        return exclusions

    # ══════════════════════════════════════════════════════════════════════════
    #  LIDAR: scan-to-scan range jump detection
    # ══════════════════════════════════════════════════════════════════════════

    def _process_lidar(self, scan_ranges, max_range):
        """
        Compare current LiDAR ranges to the previous scan.
        Flag beam indices where a significant range jump occurs.
        """
        current = [min(r, max_range) for _, r in scan_ranges]
        self._lidar_flags = set()

        if self._prev_scan is not None and len(current) == len(self._prev_scan):
            for i, (cur, prv) in enumerate(zip(current, self._prev_scan)):
                # Ignore miss → miss transitions
                if cur >= max_range * 0.95 and prv >= max_range * 0.95:
                    continue
                if abs(cur - prv) > self.lidar_jump_thr:
                    self._lidar_flags.add(i)
                    # Flag neighbors (±3) to cover the full object width
                    for di in range(-3, 4):
                        if 0 <= i + di < len(current):
                            self._lidar_flags.add(i + di)

        self._prev_scan = current

    # ══════════════════════════════════════════════════════════════════════════
    #  FUSION: OR-based — any signal excludes the ray
    # ══════════════════════════════════════════════════════════════════════════

    def filter_scan(self, scan_ranges, robot_x, robot_y, robot_theta,
                    graph_map, max_range):
        """
        Fuse camera, LiDAR-jump, and map-consistency to produce exclusions.

        A ray is excluded when ANY of:
          • Camera detects vivid+motion at that bearing, OR
          • LiDAR detects a range jump at that beam index, OR
          • The hit cell is confirmed free space (map says free, LiDAR says hit)
        UNLESS the hit cell is a confirmed permanent wall.

        Parameters
        ----------
        scan_ranges  : [(local_angle, range), …]
        graph_map    : GraphMap instance
        max_range    : float

        Returns
        -------
        excluded : list of (world_angle, half_width)
        """
        self._process_lidar(scan_ranges, max_range)

        excluded = []
        self._dyn_count = 0

        for i, (local_angle, r) in enumerate(scan_ranges):
            if self._dyn_count >= self._max_excluded:
                break
            if r >= max_range * 0.97:
                continue

            world_angle = robot_theta - local_angle

            # ── Signal 1: camera vivid+motion zone ──
            cam_flag = self._in_any_zone(world_angle, self._cam_zones)

            # ── Signal 2: LiDAR range jump ──
            lidar_flag = i in self._lidar_flags

            # ── Signal 3: map consistency ──
            # If the map says this cell is free, the hit is suspicious
            hit_x = robot_x + r * math.cos(world_angle)
            hit_y = robot_y + r * math.sin(world_angle)
            hc, hr = graph_map.world_to_grid(hit_x, hit_y)
            cell_val = graph_map.get(hc, hr)

            # Cell is confirmed free if log-odds is well below zero
            map_flag = cell_val < self.free_thr

            # ── Don't exclude confirmed permanent walls ──
            if cell_val >= self.wall_thr:
                continue   # well-established wall — trust it

            # ── OR fusion: any signal triggers exclusion ──
            if not (cam_flag or lidar_flag or map_flag):
                continue   # no signal → allow

            # ── Exclude this ray ──
            self._dyn_count += 1
            excluded.append((world_angle, 0.10))

            # Active erasure: aggressively decay toward free-space
            if cell_val > -5.0:
                graph_map.set(hc, hr, cell_val - 2.5)

        return excluded

    # ══════════════════════════════════════════════════════════════════════════
    #  Queries
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def dynamic_count(self):
        """Number of rays suppressed in the last filter_scan() call."""
        return self._dyn_count

    @property
    def camera_zones(self):
        return list(self._cam_zones)

    @property
    def lidar_jump_count(self):
        return len(self._lidar_flags)

    @property
    def has_camera(self):
        return self._has_camera

    # ══════════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _in_any_zone(angle, zones):
        for centre, half_w in zones:
            diff = angle - centre
            while diff >  math.pi: diff -= 2 * math.pi
            while diff < -math.pi: diff += 2 * math.pi
            if abs(diff) < half_w:
                return True
        return False
