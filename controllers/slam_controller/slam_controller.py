"""
slam_controller.py — WeBots E-puck SLAM + Keyboard Navigation (Milestone 3)

ENU: X=East, Y=North, Z=Up
Compass raw (0,1,0) → heading=0° → robot faces East (+X).  Confirmed.
LiDAR raw: ray[0]=0° points North.  Offset=-π/2 corrects to East.

Milestone 3 features
---------------------
  1. Graph-based map (GraphMap — sparse dict, not fixed array).
     No hardcoded world size / GPS offset.  Works from any start position.
  2. Camera + LiDAR fusion (DynamicFilter).  Moving objects are detected
     via vivid-colour detection + LiDAR scan-to-scan consistency + map
     consistency and excluded from map creation.
  3. Scrolling viewport display follows the robot.

Controls
--------
  W / ↑   — drive forward
  S / ↓   — drive backward
  A / ←   — turn left
  D / →   — turn right

  The robot maps its environment as you drive it around.
  Moving objects (coloured balls) are automatically excluded from the map.
"""
from controller import Robot, Keyboard
import math, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils            import normalize_angle
from graph_map        import GraphMap
from dynamic_filter   import DynamicFilter

from map_display      import MapDisplay
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
GRID_RES     = 0.05        # metres per cell (same as M2)
MAX_LIDAR    = 3.0
UPDATE_EVERY = 3           # update occupancy graph every N timesteps

WHEEL_RADIUS = 0.0205
WHEEL_BASE   = 0.052
MAX_SPEED    = 6.28

# Keyboard drive speeds
DRIVE_SPEED  = 0.15        # m/s forward/backward
TURN_SPEED   = 2.0         # rad/s left/right


def try_motor(robot, *names):
    for n in names:
        d = robot.getDevice(n)
        if d:
            print(f"  Motor: '{n}'")
            return d
    return None


def compass_heading(cv):
    """ENU: atan2(East, North) gives heading relative to East (+X)."""
    return math.atan2(cv[0], cv[1])


def main():
    robot    = Robot()
    timestep = int(robot.getBasicTimeStep())
    dt       = timestep / 1000.0

    print("=" * 60)
    print("  E-puck SLAM + Keyboard Control — Milestone 3")
    print(f"  Grid res  : {GRID_RES} m   (sparse graph, no fixed size)")
    print("  Controls  : W/↑=fwd  S/↓=back  A/←=left  D/→=right")
    print("=" * 60)

    # ── Motors ────────────────────────────────────────────────────────────────
    lm = try_motor(robot, "left wheel motor",  "left_wheel_motor")
    rm = try_motor(robot, "right wheel motor", "right_wheel_motor")
    if not lm or not rm:
        print("ERROR: motors not found"); return
    lm.setPosition(float('inf')); lm.setVelocity(0)
    rm.setPosition(float('inf')); rm.setVelocity(0)

    # ── Sensors ───────────────────────────────────────────────────────────────
    lidar = robot.getDevice("lidar")
    gps   = robot.getDevice("gps")
    comp  = robot.getDevice("compass")
    if not lidar or not gps or not comp:
        print("ERROR: sensor missing"); return
    lidar.enable(timestep); lidar.enablePointCloud()
    gps.enable(timestep)
    comp.enable(timestep)

    camera = robot.getDevice("slam_camera")
    cam_fov   = 0.84
    cam_width = 160
    if camera:
        camera.enable(timestep)
        cam_width = camera.getWidth()
        cam_fov   = camera.getFov()
        print(f"  Camera: {cam_width}×{camera.getHeight()} px  "
              f"FOV={math.degrees(cam_fov):.1f}°")
    else:
        print("  WARNING: camera device not found")

    # ── Keyboard ──────────────────────────────────────────────────────────────
    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)
    print("  Keyboard enabled — click the 3D window to focus, then use WASD/arrows")

    # Warm-up step
    robot.step(timestep)

    gv = gps.getValues(); cv = comp.getValues()
    start_x = gv[0]
    start_y = gv[1]
    heading = compass_heading(cv)
    print(f"  GPS start  : ({start_x:.3f}, {start_y:.3f})")
    print(f"  Heading    : {math.degrees(heading):.1f}°")

    # ── Core objects ──────────────────────────────────────────────────────────
    graph   = GraphMap(resolution=GRID_RES)
    dfilter = DynamicFilter(camera_fov=cam_fov, camera_width=cam_width)
    display = MapDisplay()

    sim_time = 0.0
    step_cnt = 0

    # ── Main loop ─────────────────────────────────────────────────────────────
    while robot.step(timestep) != -1:
        sim_time += dt
        step_cnt += 1

        # ── GPS pose (ground truth) ──────────────────────────────────────────
        gv = gps.getValues()
        cv = comp.getValues()
        rx = gv[0]
        ry = gv[1]
        rt = compass_heading(cv)

        # ── LiDAR scan ───────────────────────────────────────────────────────
        raw   = lidar.getRangeImage()
        fov   = lidar.getFov()
        nrays = lidar.getHorizontalResolution()
        scan  = []
        if raw:
            inc   = fov / nrays
            start = -fov / 2.0
            for i in range(nrays):
                r = raw[i]
                if math.isinf(r) or math.isnan(r) or r > MAX_LIDAR:
                    r = MAX_LIDAR
                scan.append((start + i * inc, r))

        # ── Camera feed + dynamic filter ─────────────────────────────────────
        excluded = None

        # Process camera if available (vivid-colour detection)
        if camera:
            display.update_camera(camera)
            img = camera.getImage()
            if img:
                w = camera.getWidth(); h = camera.getHeight()
                arr = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 4))
                rgb = np.ascontiguousarray(arr[:, :, :3][:, :, ::-1])
                dfilter.process_camera(rgb, rt)

        # Always run scan filtering (works with or without camera)
        if scan:
            excluded = dfilter.filter_scan(
                scan, rx, ry, rt, graph, MAX_LIDAR)

            if dfilter.dynamic_count > 0:
                display.set_dynamic_info(
                    f"DYN:{dfilter.dynamic_count} rays excluded")
            else:
                display.set_dynamic_info("")

        # ── Graph map update (with dynamic exclusions) ────────────────────────
        if step_cnt % UPDATE_EVERY == 0 and scan:
            graph.update(rx, ry, rt, scan, MAX_LIDAR,
                         excluded_bearings=excluded)

        # ── Keyboard input → (v, omega) ──────────────────────────────────────
        v_cmd = 0.0
        w_cmd = 0.0

        key = keyboard.getKey()
        while key != -1:
            if key == ord('W') or key == ord('w') or key == Keyboard.UP:
                v_cmd = DRIVE_SPEED
            elif key == ord('S') or key == ord('s') or key == Keyboard.DOWN:
                v_cmd = -DRIVE_SPEED
            elif key == ord('A') or key == ord('a') or key == Keyboard.LEFT:
                w_cmd = TURN_SPEED
            elif key == ord('D') or key == ord('d') or key == Keyboard.RIGHT:
                w_cmd = -TURN_SPEED
            key = keyboard.getKey()

        # ── Convert (v, omega) → individual wheel speeds ──────────────────────
        vl_raw = (v_cmd - w_cmd * WHEEL_BASE / 2) / WHEEL_RADIUS
        vr_raw = (v_cmd + w_cmd * WHEEL_BASE / 2) / WHEEL_RADIUS
        vl = max(-MAX_SPEED, min(MAX_SPEED, vl_raw))
        vr = max(-MAX_SPEED, min(MAX_SPEED, vr_raw))
        lm.setVelocity(vl)
        rm.setVelocity(vr)

        # ── Display update ────────────────────────────────────────────────────
        if step_cnt % UPDATE_EVERY == 0 and display.enabled:
            display.update(
                graph,
                (rx, ry, rt),
                (rx, ry, rt),
                [],
                [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]],
                sim_time,
            )

        # ── Periodic console log ──────────────────────────────────────────────
        if step_cnt % 150 == 0:
            dyn_str = (f"  dyn_excluded={dfilter.dynamic_count}"
                       if dfilter.dynamic_count else "")

            print(f"t={sim_time:.1f}s | pos({rx:.2f},{ry:.2f}) "
                  f"hdg={math.degrees(rt):.0f}°  "
                  f"nodes={len(graph.nodes)}{dyn_str}")

    display.close()
    print("Done.")


if __name__ == "__main__":
    main()
