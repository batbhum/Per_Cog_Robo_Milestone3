"""
exploration.py - A* goal-directed navigation for E-puck in WeBots (ENU).

AStarExplorer follows waypoints from AStarPlanner using a proportional
heading controller and escapes wall traps via a 4-phase recovery sequence.

Proportional control
---------------------
  heading_error = normalize(waypoint_bearing - robot_heading)
  omega = Kp * heading_error          (clamped to ±turn_speed)
  v     = max_v * (1 - |err|/π)²     (zero when mis-aligned > 60°)

Obstacle detection — directional, not blind
--------------------------------------------
  The old code blocked on *any* LiDAR hit in the forward cone.  That caused
  constant false triggers when the robot was legally driving along a wall.

  New rule: a hit at local angle `a` only counts if `a` is within
  WAYPOINT_ALIGN_DEG of the bearing to the *current waypoint*.  A wall to
  the side the robot is passing does not trigger a replan.

Grace period — the core fix for infinite spinning
--------------------------------------------------
  After recovery completes AND after set_waypoints() loads a fresh path,
  obstacle detection is suppressed for GRACE_DURATION seconds.  This gives
  the robot time to physically drive away from the wall before the detector
  is live again.  Without this, the robot replans, faces the same wall on
  the new path, and blocks immediately.

Stuck detection
---------------
  If the robot's position does not change by more than STUCK_DIST meters
  over STUCK_TIMEOUT seconds, a replan is forced.  This catches oscillation
  cases where the obstacle check doesn't fire.

Recovery state machine  (triggered after RECOVERY_THRESHOLD replans in
                          RECOVERY_WINDOW s, OR on A* path-not-found)
----------------------------------------------------------------------
  REVERSE  (1.2 s) — drive straight back at 70 % speed
      ↓
  ESCAPE   (0.8 s) — arc away from the wall using the clearest LiDAR side
      ↓
  ROTATE   (0.8 s) — spin toward the most open direction
      ↓
  GRACE    (1.5 s) — drive forward freely; obstacle check suppressed
      ↓
  NONE     — set need_replan=True, normal control resumes

No external libraries — only math.
"""
import math
import random

# ── Recovery tunables ─────────────────────────────────────────────────────────
_RECOVERY_THRESHOLD  = 2      # replans within window before recovery kicks in
_RECOVERY_WINDOW     = 3.0    # seconds sliding window for replan counter
_REVERSE_DURATION    = 1.0    # s  — back away from the wall
_ESCAPE_DURATION     = 1.0    # s  — rotate away from wall (was 0.8)
_ROTATE_DURATION     = 1.0    # s  — turn toward open space (was 0.8)
_GRACE_DURATION      = 1.2    # s  — drive freely after recovery

# Absolute recovery speeds
_REVERSE_SPEED       = 0.18   # faster backup (1.0s × 0.18 = 0.18m)
_ESCAPE_LINEAR       = 0.04   # slight forward during escape to move away
_GRACE_SPEED         = 0.14   # forward during grace drive-out
_REAR_SAFETY_DIST    = 0.07   # stop reversing if rear LiDAR < this
_FRONT_SAFETY_DIST   = 0.08   # stop grace drive if front LiDAR < this

# ── Obstacle-check tunables ───────────────────────────────────────────────────
_WAYPOINT_ALIGN_DEG  = 40     # only block if obstacle within this angle of WP

# ── Stuck detection ──────────────────────────────────────────────────────────
_STUCK_TIMEOUT       = 3.0    # seconds without progress → force replan
_STUCK_DIST          = 0.08   # must move at least this far


def _norm(a):
    """Wrap angle to [-pi, pi]."""
    while a >  math.pi: a -= 2.0 * math.pi
    while a < -math.pi: a += 2.0 * math.pi
    return a


class AStarExplorer:
    """
    Waypoint-following controller with LiDAR-based replanning and wall recovery.

    Parameters
    ----------
    forward_speed      : max linear speed   [m/s]
    turn_speed         : max angular speed  [rad/s]
    obstacle_threshold : LiDAR range below which a hit counts as a blockage [m]
    waypoint_radius    : distance to waypoint counted as "reached"           [m]
    Kp_heading         : proportional gain, heading error → omega
    front_arc_deg      : half-angle of raw forward cone (pre-directional filter)
    """

    def __init__(self,
                 forward_speed=6.28,
                 turn_speed=1.2,
                 obstacle_threshold=0.12,
                 waypoint_radius=0.18,
                 Kp_heading=2.2,
                 front_arc_deg=35):

        self.forward_speed      = forward_speed
        self.turn_speed         = turn_speed
        self.obstacle_threshold = obstacle_threshold
        self.waypoint_radius    = waypoint_radius
        self.Kp_heading         = Kp_heading
        self.front_arc          = math.radians(front_arc_deg)

        self._waypoints  = []
        self._wp_idx     = 0
        self.need_replan = True   # True on init → plan immediately

        # ── Recovery / grace state ────────────────────────────────────────────
        # states: 'NONE' | 'REVERSE' | 'ESCAPE' | 'ROTATE' | 'GRACE'
        self._state         = 'NONE'
        self._state_start   = 0.0
        self._recovery_omega = 0.0   # spin direction chosen at recovery entry
        self._escape_v      = 0.0    # linear component during ESCAPE arc
        self._escape_omega  = 0.0    # angular component during ESCAPE arc
        self._grace_until   = 0.0    # sim_time after which obs-check is live

        self._replan_times  = []     # sliding window of recent replan timestamps
        self._last_scan     = []     # cached for use inside recovery phases

        # ── Stuck detection ───────────────────────────────────────────────────
        self._stuck_ref_x   = 0.0
        self._stuck_ref_y   = 0.0
        self._stuck_ref_t   = 0.0    # sim_time of last reference position

    # ── Waypoint management ───────────────────────────────────────────────────

    def set_waypoints(self, waypoints, sim_time=0.0):
        """
        Load a fresh list of world-coordinate waypoints.
        Call from slam_controller whenever A* produces a new path.

        Starts a grace period so the robot can begin moving before
        obstacle detection re-activates.
        """
        self._wp_idx     = 0
        self._state      = 'NONE'

        if waypoints:
            self._waypoints  = list(waypoints)
            self.need_replan = False
            self._replan_times.clear()
            # Grace: suppress obstacle check while robot drives off the wall
            self._grace_until = sim_time + _GRACE_DURATION
            print(f"[Explorer] Path loaded: {len(waypoints)} waypoints  "
                  f"first=({waypoints[0][0]:.2f},{waypoints[0][1]:.2f})  "
                  f"grace until t={self._grace_until:.1f}s")
        else:
            self._waypoints  = []
            print("[Explorer] A* returned no path — forcing recovery.")
            self._start_recovery(sim_time, forced=True)

    def has_waypoints(self):
        return bool(self._waypoints) and self._wp_idx < len(self._waypoints)

    def current_waypoint(self):
        return self._waypoints[self._wp_idx] if self.has_waypoints() else None

    # ── Main control ──────────────────────────────────────────────────────────

    def compute_control(self, robot_x, robot_y, robot_theta, scan_ranges,
                        sim_time=0.0):
        """
        Compute (v, omega) toward the current waypoint.

        Parameters
        ----------
        robot_x, robot_y : world position  [m]
        robot_theta       : heading         [rad]
        scan_ranges       : [(local_angle, range), …] from LiDAR
        sim_time          : simulation time [s]

        Returns
        -------
        (v, omega)
        """
        self._last_scan = scan_ranges   # cache for recovery phases

        # ── Recovery / grace state machine (highest priority) ─────────────────
        if self._state != 'NONE':
            return self._do_recovery(sim_time, scan_ranges)

        # ── No plan yet: random walk to build the map ─────────────────────────
        if not self.has_waypoints():
            # Alternate between spinning and driving forward to explore
            phase = int(sim_time * 2) % 4
            if phase < 2:
                return 0.0, self.turn_speed * 0.6  # spin
            else:
                # Drive forward if clear, else spin
                if self._front_is_clear(scan_ranges, 0.15):
                    return 0.06, 0.0
                else:
                    return 0.0, self.turn_speed * 0.8

        # ── Advance past waypoints already reached ────────────────────────────
        self._advance_waypoint(robot_x, robot_y)
        if not self.has_waypoints():
            print("[Explorer] Goal reached — requesting replan.")
            self.need_replan = True
            return 0.0, 0.0

        wp_x, wp_y = self._waypoints[self._wp_idx]

        # ── Stuck detection ───────────────────────────────────────────────
        if self._check_stuck(robot_x, robot_y, sim_time):
            print(f"[Explorer] *** STUCK *** no progress for {_STUCK_TIMEOUT:.0f}s — triggering recovery")
            self._reset_stuck_ref(robot_x, robot_y, sim_time)
            # Go straight to recovery instead of just replanning
            self._start_recovery(sim_time, forced=True)
            return self._do_recovery(sim_time, scan_ranges)

        # ── Directional obstacle check (skipped during grace period) ──────
        in_grace = sim_time < self._grace_until

        if not in_grace:
            if self._waypoint_blocked(scan_ranges, robot_x, robot_y,
                                      robot_theta, wp_x, wp_y):
                if self._record_replan(sim_time):
                    self._start_recovery(sim_time)
                    return self._do_recovery(sim_time, scan_ranges)
                else:
                    print("[Explorer] Obstacle on path — requesting replan.")
                    self.need_replan = True
                    # Short grace so the new plan has time to load & move
                    self._grace_until = sim_time + 0.5
                    return 0.0, 0.0

        # ── Proportional heading control ──────────────────────────────────────
        dx = wp_x - robot_x
        dy = wp_y - robot_y
        target_bearing = math.atan2(dy, dx)
        heading_error  = _norm(target_bearing - robot_theta)

        omega = self.Kp_heading * heading_error
        omega = max(-self.turn_speed, min(self.turn_speed, omega))

        # cos² alignment: gentler than the old (1-|e|/π)² — keeps speed
        # during moderate turns.  Zero speed only for large mis-alignment.
        abs_err = abs(heading_error)
        if abs_err > math.radians(45):
            v = 0.0     # too far off — turn in place
        else:
            align = math.cos(heading_error) ** 2
            v = self.forward_speed * align

        return v, omega

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _advance_waypoint(self, rx, ry):
        while self.has_waypoints():
            wx, wy = self._waypoints[self._wp_idx]
            is_last = (self._wp_idx == len(self._waypoints) - 1)
            # Tighter radius for the final waypoint so the robot physically
            # reaches the goal rather than skipping it from 15 cm away.
            accept_r = 0.05 if is_last else self.waypoint_radius
            if math.sqrt((wx-rx)**2 + (wy-ry)**2) <= accept_r:
                self._wp_idx += 1
                if self.has_waypoints():
                    nw = self._waypoints[self._wp_idx]
                    print(f"[Explorer] WP {self._wp_idx} → ({nw[0]:.2f},{nw[1]:.2f})")
            else:
                break

    def _waypoint_blocked(self, scan_ranges, rx, ry, rtheta, wp_x, wp_y):
        """
        Return True only if a close LiDAR hit lies in the direction of the
        current waypoint — not for walls the robot is merely passing alongside.

        A ray at local angle `a` blocks only when BOTH:
          (a) its range < obstacle_threshold
          (b) `a` is within WAYPOINT_ALIGN_DEG of the bearing to the waypoint
              (converted to the robot's local frame)
        """
        if not scan_ranges:
            return False

        wp_bearing_world = math.atan2(wp_y - ry, wp_x - rx)
        wp_bearing_local = _norm(wp_bearing_world - rtheta)
        align_rad = math.radians(_WAYPOINT_ALIGN_DEG)

        return any(
            r < self.obstacle_threshold
            and abs(_norm(a - wp_bearing_local)) < align_rad
            for a, r in scan_ranges
            if abs(a) < self.front_arc
        )

    # ── Stuck detection ──────────────────────────────────────────────────────

    def _reset_stuck_ref(self, rx, ry, sim_time):
        """Set the reference position for stuck detection."""
        self._stuck_ref_x = rx
        self._stuck_ref_y = ry
        self._stuck_ref_t = sim_time

    def _check_stuck(self, rx, ry, sim_time):
        """
        Return True if the robot has not moved STUCK_DIST in STUCK_TIMEOUT s.
        Automatically resets the reference when the robot makes progress.
        """
        dist = math.sqrt((rx - self._stuck_ref_x)**2 +
                         (ry - self._stuck_ref_y)**2)

        if dist >= _STUCK_DIST:
            # Good progress — reset reference
            self._reset_stuck_ref(rx, ry, sim_time)
            return False

        elapsed = sim_time - self._stuck_ref_t
        if elapsed >= _STUCK_TIMEOUT:
            return True

        return False

    # ── Recovery helpers ──────────────────────────────────────────────────────

    def _record_replan(self, sim_time):
        cutoff = sim_time - _RECOVERY_WINDOW
        self._replan_times = [t for t in self._replan_times if t > cutoff]
        self._replan_times.append(sim_time)
        return len(self._replan_times) >= _RECOVERY_THRESHOLD

    def _start_recovery(self, sim_time, forced=False):
        reason = ("forced (A* failed)" if forced else
                  f"{len(self._replan_times)} replans / {_RECOVERY_WINDOW:.1f}s")
        print(f"[Explorer] *** RECOVERY *** ({reason})")
        self._state        = 'REVERSE'
        self._state_start  = sim_time
        self.need_replan   = False
        self._replan_times.clear()
        self._recovery_count = getattr(self, '_recovery_count', 0) + 1

    def _do_recovery(self, sim_time, scan_ranges=None):
        """
        4-phase recovery:  REVERSE → ESCAPE → ROTATE → GRACE → NONE

        REVERSE: straight backward — physically separates from wall.
                 Stops early if rear LiDAR detects a wall behind.
        ESCAPE:  pure rotation toward the more open side — no forward
                 motion, so the robot cannot push into walls.
        ROTATE:  continue spinning toward the most open direction.
        GRACE:   drive straight forward WITH front-safety check;
                 obstacle detection suppressed but collision is not.

        All speeds are absolute constants (_REVERSE_SPEED, _GRACE_SPEED)
        so recovery works regardless of the cruise speed.
        """
        elapsed = sim_time - self._state_start
        sc = scan_ranges if scan_ranges is not None else self._last_scan

        # ── REVERSE ───────────────────────────────────────────────────────────
        if self._state == 'REVERSE':
            # Detect corner: front AND both sides have close walls
            front_min = self._sector_min(sc, lo=-math.radians(30),
                                             hi= math.radians(30))
            left_min  = self._sector_min(sc, lo= math.radians(30),
                                             hi= math.radians(90))
            right_min = self._sector_min(sc, lo=-math.radians(90),
                                             hi=-math.radians(30))
            in_corner = (front_min < 0.15 and
                         (left_min < 0.15 or right_min < 0.15))

            # Extended reverse time when in a corner or repeated recoveries
            rev_duration = _REVERSE_DURATION
            if in_corner or self._recovery_count > 2:
                rev_duration = _REVERSE_DURATION * 1.5

            # Safety: stop reversing if there's a wall behind
            rear_clear = self._rear_is_clear(sc, _REAR_SAFETY_DIST)
            if elapsed < rev_duration and rear_clear:
                return -_REVERSE_SPEED, 0.0
            if not rear_clear:
                print("[Explorer] REVERSE — rear obstacle, stopping early")

            # Choose escape rotation — TOWARD the more open side
            left_mean  = self._sector_mean(sc, lo= math.radians(15),
                                               hi= math.radians(150))
            right_mean = self._sector_mean(sc, lo=-math.radians(150),
                                               hi=-math.radians(15))

            # In a corner: also check rear-left vs rear-right
            if in_corner:
                rl_mean = self._sector_mean(sc, lo=math.radians(120),
                                                hi=math.radians(170))
                rr_mean = self._sector_mean(sc, lo=-math.radians(170),
                                                hi=-math.radians(120))
                left_mean  += rl_mean
                right_mean += rr_mean

            # Add randomness on repeated recoveries to break symmetry
            if self._recovery_count > 3:
                left_mean  += random.uniform(0, 0.3)
                right_mean += random.uniform(0, 0.3)

            if left_mean >= right_mean:
                self._escape_omega =  self.turn_speed   # rotate left (CCW)
            else:
                self._escape_omega = -self.turn_speed   # rotate right (CW)

            # In a corner: more forward push during escape to physically exit
            self._escape_v = _ESCAPE_LINEAR if not in_corner else 0.06
            # Longer escape rotation in corners
            self._corner_escape = in_corner
            self._state       = 'ESCAPE'
            self._state_start = sim_time
            print(f"[Explorer] Recovery → ESCAPE "
                  f"({'right/CW' if self._escape_omega < 0 else 'left/CCW'})  "
                  f"corner={in_corner}  "
                  f"left_mean={left_mean:.2f} right_mean={right_mean:.2f}")
            return self._escape_v, self._escape_omega

        # ── ESCAPE ────────────────────────────────────────────────────────────
        if self._state == 'ESCAPE':
            esc_dur = _ESCAPE_DURATION
            if getattr(self, '_corner_escape', False):
                esc_dur = _ESCAPE_DURATION * 1.8   # longer turn in corners
            if elapsed < esc_dur:
                return self._escape_v, self._escape_omega
            # Continue rotating toward best open direction (re-evaluate live)
            self._recovery_omega = self._best_spin_direction(sc)
            self._state       = 'ROTATE'
            self._state_start = sim_time
            print(f"[Explorer] Recovery → ROTATE "
                  f"({'CW' if self._recovery_omega < 0 else 'CCW'})")
            return 0.0, self._recovery_omega

        # ── ROTATE ────────────────────────────────────────────────────────────
        if self._state == 'ROTATE':
            if elapsed < _ROTATE_DURATION:
                return 0.0, self._recovery_omega
            self._state       = 'GRACE'
            self._state_start = sim_time
            self._grace_until = sim_time + _GRACE_DURATION
            print("[Explorer] Recovery → GRACE (driving free)")
            # Check front before driving
            if self._front_is_clear(sc, _FRONT_SAFETY_DIST):
                return _GRACE_SPEED, 0.0
            else:
                return 0.0, 0.0

        # ── GRACE ─────────────────────────────────────────────────────────────
        if self._state == 'GRACE':
            if elapsed < _GRACE_DURATION:
                # Keep checking — don't drive into a wall during grace
                if self._front_is_clear(sc, _FRONT_SAFETY_DIST):
                    return _GRACE_SPEED, 0.0
                else:
                    # Wall ahead — abort grace early, replan
                    print("[Explorer] GRACE — front obstacle, aborting")
                    self._state      = 'NONE'
                    self.need_replan = True
                    return 0.0, 0.0
            print("[Explorer] Recovery complete — requesting plan.")
            self._state      = 'NONE'
            self.need_replan = True
            return 0.0, 0.0

        # Fallback
        self._state      = 'NONE'
        self.need_replan = True
        return 0.0, 0.0

    @staticmethod
    def _sector_min(scan_ranges, lo, hi):
        """Minimum range of rays whose local angle is in [lo, hi]."""
        vals = [r for a, r in scan_ranges if lo <= a <= hi]
        return min(vals) if vals else 9.0

    @staticmethod
    def _sector_mean(scan_ranges, lo, hi):
        """Mean range of rays whose local angle is in [lo, hi]."""
        vals = [r for a, r in scan_ranges if lo <= a <= hi]
        return sum(vals) / len(vals) if vals else 0.0

    @staticmethod
    def _rear_is_clear(scan_ranges, min_dist):
        """
        Return True if no LiDAR ray in the rear arc (|angle| > 120°)
        is closer than min_dist.  Prevents reversing into a wall.
        """
        rear_thresh = math.radians(120)
        for a, r in scan_ranges:
            if abs(a) > rear_thresh and r < min_dist:
                return False
        return True

    @staticmethod
    def _front_is_clear(scan_ranges, min_dist):
        """
        Return True if no LiDAR ray in the front arc (|angle| < 30°)
        is closer than min_dist.  Prevents driving into a wall during GRACE.
        """
        front_thresh = math.radians(30)
        for a, r in scan_ranges:
            if abs(a) < front_thresh and r < min_dist:
                return False
        return True

    def _best_spin_direction(self, scan_ranges):
        """
        Return +turn_speed or -turn_speed toward whichever hemisphere
        (left vs right) has more open space (higher mean range).
        """
        left  = [r for a, r in scan_ranges if  math.radians(10) < a < math.pi]
        right = [r for a, r in scan_ranges if -math.pi < a < -math.radians(10)]
        mean_l = sum(left)  / len(left)  if left  else 0.0
        mean_r = sum(right) / len(right) if right else 0.0
        # spin toward the more open side
        return self.turn_speed if mean_l >= mean_r else -self.turn_speed

    # ── API shim ──────────────────────────────────────────────────────────────

    def update(self, omega, dt):
        """Kept for slam_controller API compatibility — no state needed."""
        pass

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self):
        if self._state != 'NONE':
            return f"RECOVERY/{self._state}"
        if not self.has_waypoints():
            return "IDLE/waiting"
        wp = self._waypoints[self._wp_idx]
        return (f"WP {self._wp_idx+1}/{len(self._waypoints)} "
                f"→ ({wp[0]:.2f},{wp[1]:.2f})  "
                f"replan={self.need_replan}  "
                f"recent_replans={len(self._replan_times)}")
