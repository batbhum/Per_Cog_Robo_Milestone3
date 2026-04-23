"""
camera_display.py - Live camera feed display for E-puck (Milestone 2).

Shows the camera image in a pygame window with:
  - Raw RGB feed from WeBots camera
  - Green blob detection (goal identification)
  - Red/orange blob detection (moving balls)
  - Bounding boxes drawn over detected objects

Color detection uses simple HSV-range thresholding — no CV libraries.
All image processing is done in pure numpy/math.
"""
import math
import numpy as np

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False


def rgb_to_hsv(r, g, b):
    """Convert single RGB pixel (0-255) to HSV (H:0-360, S:0-1, V:0-1)."""
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b); mn = min(r, g, b)
    v  = mx
    s  = 0 if mx == 0 else (mx - mn) / mx
    if mx == mn:
        h = 0
    elif mx == r:
        h = 60 * ((g - b) / (mx - mn) % 6)
    elif mx == g:
        h = 60 * ((b - r) / (mx - mn) + 2)
    else:
        h = 60 * ((r - g) / (mx - mn) + 4)
    return h, s, v


def detect_blobs(rgb_array, h_min, h_max, s_min=0.4, v_min=0.3):
    """
    Find pixels matching a hue range. Returns (cx, cy, pixel_count) or None.
    rgb_array: (H, W, 3) uint8
    Returns bounding box (x1,y1,x2,y2) and center, or None if not found.
    """
    H, W = rgb_array.shape[:2]
    matches_x = []
    matches_y = []

    for y in range(0, H, 2):      # sample every 2 rows for speed
        for x in range(0, W, 2):
            r, g, b = int(rgb_array[y, x, 0]), int(rgb_array[y, x, 1]), int(rgb_array[y, x, 2])
            h, s, v = rgb_to_hsv(r, g, b)
            in_range = s > s_min and v > v_min
            if h_max >= h_min:
                in_range = in_range and h_min <= h <= h_max
            else:  # wraps around 360 (e.g. red: 340-360 + 0-10)
                in_range = in_range and (h >= h_min or h <= h_max)
            if in_range:
                matches_x.append(x)
                matches_y.append(y)

    if len(matches_x) < 8:   # too few pixels → not a real detection
        return None

    x1, x2 = min(matches_x), max(matches_x)
    y1, y2 = min(matches_y), max(matches_y)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (x1, y1, x2, y2, cx, cy, len(matches_x))


class CameraDisplay:
    """Displays live camera feed with object detection overlays."""

    # Scale factor for display (camera is 160x120, display at 2x)
    SCALE = 3

    def __init__(self, cam_w=160, cam_h=120):
        self.cam_w   = cam_w
        self.cam_h   = cam_h
        self.disp_w  = cam_w * self.SCALE
        self.disp_h  = cam_h * self.SCALE
        self.enabled = PYGAME_OK
        self.screen  = None
        self.font    = None
        self.surf    = None   # unscaled surface

        if self.enabled:
            # Don't call pygame.init() here — slam_controller already did it
            # Just create a new window
            self.screen = pygame.display.set_mode(
                (self.disp_w, self.disp_h + 30),
                pygame.NOFRAME,
                display=0
            )
            # Actually we need a separate window — use a named surface trick
            # Better: use pygame to open a second display isn't easy.
            # Solution: draw camera in a sub-region of a combined window,
            # or just call pygame.display.set_mode again with a flag.
            # Simplest: create a dedicated pygame window using a subprocess
            # OR just blit camera feed onto a separate Surface and show it
            # in the SAME window as the map but side by side.
            # For simplicity: camera feed goes in its OWN window using
            # a second pygame display via os.environ trick.
            pass

    def close(self):
        pass


class CameraWindow:
    """
    Standalone camera window using a separate pygame display instance.
    Opens as a second window alongside the map display.
    """
    SCALE = 3

    def __init__(self, cam_w=160, cam_h=120):
        self.cam_w    = cam_w
        self.cam_h    = cam_h
        self.disp_w   = cam_w  * self.SCALE
        self.disp_h   = cam_h  * self.SCALE
        self.enabled  = PYGAME_OK
        self._last_ms = 0
        self._interval = 80    # ~12 fps for camera

        if not self.enabled:
            return

        # pygame doesn't support two windows natively in one process.
        # Best approach: embed camera in the map window as a side panel.
        # We return a Surface that map_display can blit into its window.
        self.surf     = None   # set by map_display when window is ready
        self.cam_surf = None
        print("  Camera display: will embed in map window")

    def get_camera_surface(self, camera_device):
        """
        Read camera image from WeBots and return a pygame Surface.
        camera_device: WeBots Camera object (already enabled)
        Returns pygame.Surface or None if no image yet.
        """
        if not self.enabled:
            return None

        img = camera_device.getImage()
        if not img:
            return None

        w = camera_device.getWidth()
        h = camera_device.getHeight()

        # WeBots getImage() returns BGRA bytes
        arr = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 4))
        rgb = arr[:, :, :3][:, :, ::-1].copy()   # BGRA→RGB, contiguous

        return rgb, w, h

    def detect_objects(self, rgb):
        """
        Detect colored objects in the camera image.
        Returns list of (label, color, bbox) for drawing.
        """
        detections = []

        # Goal: bright green (H=100-140, high S and V)
        goal = detect_blobs(rgb, h_min=100, h_max=140, s_min=0.5, v_min=0.4)
        if goal:
            detections.append(("GOAL", (0, 255, 80), goal))

        # Red ball (H=340-360 or 0-15)
        ball_r = detect_blobs(rgb, h_min=340, h_max=15, s_min=0.5, v_min=0.3)
        if ball_r:
            detections.append(("BALL", (255, 80, 80), ball_r))

        # Orange ball (H=15-35)
        ball_o = detect_blobs(rgb, h_min=15, h_max=35, s_min=0.5, v_min=0.3)
        if ball_o:
            detections.append(("BALL", (255, 160, 0), ball_o))

        return detections

    def close(self):
        pass
