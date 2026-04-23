"""
map_display.py — SLAM map + camera feed in one window (Milestone 3).

Left:  scrolling viewport of the graph-based occupancy map
Right: live camera feed with dynamic-object detection overlays

The viewport follows the robot — the displayed region scrolls as the
robot moves through an arbitrarily-large environment.
"""
import math
import numpy as np

try:
    import pygame
    import pygame.surfarray
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False


# ── Colour detection (vectorised, no CV libraries) ───────────────────────────

def _rgb_to_hsv(rgb):
    """Vectorised RGB (H,W,3 float32 0-1) → H(0-360), S, V."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d  = mx - mn
    v  = mx
    s  = np.where(mx > 0, d / mx, 0.0)
    h  = np.zeros_like(r)
    mr = (mx == r) & (d > 0)
    mg = (mx == g) & (d > 0)
    mb = (mx == b) & (d > 0)
    h[mr] = (60 * ((g[mr] - b[mr]) / d[mr])) % 360
    h[mg] = (60 * ((b[mg] - r[mg]) / d[mg]) + 120)
    h[mb] = (60 * ((r[mb] - g[mb]) / d[mb]) + 240)
    return h, s, v


def detect_color(rgb_u8, h_min, h_max, s_min=0.45, v_min=0.25):
    """
    Find pixels matching hue range.  rgb_u8: (H,W,3) uint8.
    Returns (x1,y1,x2,y2,cx,cy,count) or None.
    """
    f = rgb_u8.astype(np.float32) / 255.0
    h, s, v = _rgb_to_hsv(f)
    sv = (s >= s_min) & (v >= v_min)
    if h_max >= h_min:
        hm = (h >= h_min) & (h <= h_max)
    else:
        hm = (h >= h_min) | (h <= h_max)
    mask = sv & hm
    ys, xs = np.where(mask)
    if len(xs) < 12:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return (x1, y1, x2, y2, (x1+x2)//2, (y1+y2)//2, len(xs))


# ── Display ──────────────────────────────────────────────────────────────────

CAM_SCALE = 3
VP_CELLS  = 140    # viewport size in grid cells (140 × 0.05 = 7 m shown)
CELL_PX   = 4      # pixels per cell in viewport


class MapDisplay:
    """
    Scrolling-viewport map + embedded camera feed.

    The viewport is VP_CELLS × VP_CELLS grid cells, centred on the robot,
    rendered at CELL_PX pixels per cell.
    """

    def __init__(self, cell_px=CELL_PX, cam_w=160, cam_h=120):
        self.cell_px = cell_px
        self.cam_w   = cam_w
        self.cam_h   = cam_h
        self.enabled = PYGAME_OK
        self._init   = False
        self._last   = 0
        self._gap    = 120          # ~8 fps
        self.traj    = []
        self._cam_rgb  = None
        self._cam_dets = []
        self._dyn_info = ""         # dynamic filter info string
        self._tracks   = []         # TrackedObject list from DynamicTracker
        self._arena_bounds = None   # for prediction trajectory rendering

        if self.enabled:
            pygame.init()

    def _setup(self):
        self.vp_cells = VP_CELLS
        self.map_px   = self.vp_cells * self.cell_px
        self.cam_dw   = self.cam_w * CAM_SCALE
        self.cam_dh   = self.cam_h * CAM_SCALE
        self.panel_h  = max(self.map_px, self.cam_dh)
        self.total_w  = self.map_px + 8 + self.cam_dw
        self.total_h  = self.panel_h + 36

        self.screen  = pygame.display.set_mode((self.total_w, self.total_h))
        pygame.display.set_caption("SLAM — Graph Map + Camera  (Milestone 3)")
        self.clock   = pygame.time.Clock()
        self.font    = pygame.font.SysFont("monospace", 11)
        self.font_b  = pygame.font.SysFont("monospace", 12, bold=True)
        self.map_surf = pygame.Surface((self.vp_cells, self.vp_cells))
        self.cam_surf = pygame.Surface((self.cam_w, self.cam_h))
        self._init   = True

    # ── World → viewport pixel conversion ────────────────────────────────────

    def _w2vp(self, wx, wy, vp_origin_c, vp_origin_r, resolution):
        """World coords → viewport pixel coords."""
        gc = int(math.floor(wx / resolution))
        gr = int(math.floor(wy / resolution))
        # Viewport pixel
        px = (gc - vp_origin_c) * self.cell_px
        py = (self.vp_cells - 1 - (gr - vp_origin_r)) * self.cell_px  # Y flip
        return (max(0, min(self.map_px - 1, px)),
                max(0, min(self.map_px - 1, py)))

    # ── Camera ───────────────────────────────────────────────────────────────

    def update_camera(self, camera_device):
        """Read WeBots camera, run colour detection. Call every timestep."""
        if not self.enabled or camera_device is None:
            return
        img = camera_device.getImage()
        if not img:
            return
        w = camera_device.getWidth()
        h = camera_device.getHeight()

        arr = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 4))
        rgb = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])  # BGRA→RGB
        self._cam_rgb = rgb

        dets = []

        # Ball 1: emissive MAGENTA
        bm = detect_color(rgb, h_min=285, h_max=315, s_min=0.5, v_min=0.4)
        if bm: dets.append(("BALL-M", (255, 0, 255), bm))

        # Ball 2: emissive CYAN
        bc = detect_color(rgb, h_min=175, h_max=195, s_min=0.5, v_min=0.4)
        if bc: dets.append(("BALL-C", (0, 255, 255), bc))

        self._cam_dets = dets

    def set_dynamic_info(self, info_str):
        """Set a status string about dynamic filtering for the info bar."""
        self._dyn_info = info_str

    def set_tracks(self, tracks, arena_bounds=None):
        """Update tracked dynamic objects for rendering."""
        self._tracks = list(tracks) if tracks else []
        if arena_bounds:
            self._arena_bounds = arena_bounds

    # ── Main update ──────────────────────────────────────────────────────────

    def update(self, graph_map, est_pose, true_pose, landmarks, cov,
               sim_time):
        """
        Render the scrolling viewport + camera panel.

        graph_map : GraphMap instance (sparse dict)
        est_pose  : (rx, ry, rt)
        """
        if not self.enabled:
            return
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.enabled = False; return

        now = pygame.time.get_ticks()
        if now - self._last < self._gap:
            return
        self._last = now

        if not self._init:
            self._setup()

        rx, ry, rt = est_pose
        res = graph_map.resolution

        # ── Compute viewport origin (bottom-left corner in grid coords) ──────
        robot_c, robot_r = graph_map.world_to_grid(rx, ry)
        half = self.vp_cells // 2
        vp_c0 = robot_c - half
        vp_r0 = robot_r - half

        self.screen.fill((18, 18, 18))

        # ── Render graph map into viewport ───────────────────────────────────
        # Build a small numpy array for the visible cells
        vp = np.full((self.vp_cells, self.vp_cells, 3), 128, dtype=np.uint8)

        for (c, r), val in graph_map.nodes.items():
            vc = c - vp_c0
            vr = r - vp_r0
            if 0 <= vc < self.vp_cells and 0 <= vr < self.vp_cells:
                # Flip Y: grid row 0 = bottom → pixel row = vp_cells-1
                py = self.vp_cells - 1 - vr
                prob = 1.0 - 1.0 / (1.0 + math.exp(max(-8, min(8, val))))
                intensity = int(255 * (1.0 - prob))
                if abs(val) < 0.05:
                    intensity = 128     # unknown
                vp[py, vc] = (intensity, intensity, intensity)

        # Transpose for surfarray (expects (W, H, 3))
        vp_t = np.ascontiguousarray(np.transpose(vp, (1, 0, 2)))
        pygame.surfarray.blit_array(self.map_surf, vp_t)
        self.screen.blit(
            pygame.transform.scale(self.map_surf,
                                   (self.map_px, self.map_px)), (0, 0))
        pygame.draw.rect(self.screen, (60, 60, 60),
                         (0, 0, self.map_px, self.map_px), 1)
        self.screen.blit(
            self.font_b.render("GRAPH MAP (scrolling)", True, (180, 180, 180)),
            (4, 4))

        # ── Robot trajectory ─────────────────────────────────────────────────
        self.traj.append((rx, ry))
        if len(self.traj) > 800:
            self.traj = self.traj[-800:]
        if len(self.traj) > 1:
            pts = [self._w2vp(p[0], p[1], vp_c0, vp_r0, res)
                   for p in self.traj]
            pygame.draw.lines(self.screen, (0, 180, 80), False, pts, 2)

        # Robot marker
        sx, sy = self._w2vp(rx, ry, vp_c0, vp_r0, res)
        pygame.draw.circle(self.screen, (255, 60, 60), (sx, sy), 5)
        hx, hy = self._w2vp(rx + 0.15 * math.cos(rt),
                             ry + 0.15 * math.sin(rt),
                             vp_c0, vp_r0, res)
        pygame.draw.line(self.screen, (255, 220, 0), (sx, sy), (hx, hy), 2)

        # ── Tracked dynamic objects ──────────────────────────────────────────
        _TRACK_COLORS = {
            "BALL-M": (255, 80, 255),   # magenta ball
            "BALL-C": (80, 255, 255),   # cyan ball
        }
        for track in self._tracks:
            tc = _TRACK_COLORS.get(track.label, (255, 160, 0))
            # Current position
            tx, ty = self._w2vp(track.x, track.y, vp_c0, vp_r0, res)
            pygame.draw.circle(self.screen, tc, (tx, ty), 7, 2)
            # Label
            self.screen.blit(
                self.font.render(f"#{track.id}", True, tc),
                (tx + 9, ty - 6))

            # Predicted trajectory (dotted line)
            if track.has_velocity:
                pred_pts = track.predict_trajectory(
                    dt_total=3.0, n_points=8,
                    arena_bounds=self._arena_bounds)
                prev = (tx, ty)
                for i, (ppx, ppy) in enumerate(pred_pts):
                    px_, py_ = self._w2vp(ppx, ppy, vp_c0, vp_r0, res)
                    # Fade colour as prediction goes further
                    alpha = max(80, 255 - i * 25)
                    pc = (min(255, tc[0]), min(alpha, tc[1]), min(alpha, tc[2]))
                    if i % 2 == 0:  # dotted effect
                        pygame.draw.line(self.screen, pc, prev, (px_, py_), 1)
                    # Draw small dot at each prediction point
                    pygame.draw.circle(self.screen, pc, (px_, py_), 2)
                    prev = (px_, py_)

                # Velocity arrow from current position
                arrow_scale = 40.0  # pixels per m/s
                ax = tx + int(track.vx * arrow_scale)
                ay = ty - int(track.vy * arrow_scale)  # Y flip
                ax = max(0, min(self.map_px - 1, ax))
                ay = max(0, min(self.map_px - 1, ay))
                pygame.draw.line(self.screen, tc, (tx, ty), (ax, ay), 2)

        # ── Divider ──────────────────────────────────────────────────────────
        div_x = self.map_px + 4
        pygame.draw.line(self.screen, (50, 50, 50),
                         (div_x, 0), (div_x, self.panel_h), 2)
        cam_x = div_x + 4

        # ── Camera panel ─────────────────────────────────────────────────────
        self.screen.blit(
            self.font_b.render("CAMERA FEED", True, (180, 180, 180)),
            (cam_x + 4, 4))

        if self._cam_rgb is not None:
            cam_t = np.ascontiguousarray(
                np.transpose(self._cam_rgb, (1, 0, 2)))
            pygame.surfarray.blit_array(self.cam_surf, cam_t)
            self.screen.blit(
                pygame.transform.scale(self.cam_surf,
                                       (self.cam_dw, self.cam_dh)),
                (cam_x, 0))

            for label, color, det in self._cam_dets:
                x1, y1, x2, y2, cx, cy, cnt = det
                pygame.draw.rect(self.screen, color,
                                 (cam_x + x1 * CAM_SCALE,
                                  y1 * CAM_SCALE,
                                  (x2 - x1) * CAM_SCALE,
                                  (y2 - y1) * CAM_SCALE), 2)
                self.screen.blit(
                    self.font_b.render(label, True, color),
                    (cam_x + x1 * CAM_SCALE, max(0, y1 * CAM_SCALE - 14)))

            if self._cam_dets:
                txt = "DETECTED: " + ", ".join(d[0] for d in self._cam_dets)
                col = (0, 255, 100)
            else:
                txt = "No objects detected"
                col = (120, 120, 120)
            self.screen.blit(self.font.render(txt, True, col),
                             (cam_x, self.cam_dh + 4))
        else:
            pygame.draw.rect(self.screen, (30, 30, 30),
                             (cam_x, 0, self.cam_dw, self.cam_dh))
            self.screen.blit(
                self.font.render("Camera initialising...", True,
                                 (100, 100, 100)),
                (cam_x + 10, self.cam_dh // 2))

        # ── Info bar ─────────────────────────────────────────────────────────
        dyn = f"  {self._dyn_info}" if self._dyn_info else ""
        trk = f"  tracks={len(self._tracks)}" if self._tracks else ""
        info = (f"t={sim_time:.0f}s  pos({rx:.2f},{ry:.2f})  "
                f"hdg={math.degrees(rt):.0f}°  "
                f"nodes={len(graph_map.nodes)}{dyn}{trk}")
        self.screen.blit(self.font.render(info, True, (180, 180, 180)),
                         (4, self.panel_h + 4))
        pygame.display.flip()

    def close(self):
        if self.enabled:
            pygame.quit()
