"""
ball_mover.py - Bouncing ball with constant speed enforcement.

WeBots physics damps velocity over time → ball eventually stops.
Fix: every step, read current velocity and rescale it back to the
original speed if it has dropped. Direction is preserved, only
magnitude is corrected. This keeps wall bouncing via physics while
preventing energy loss.
"""
from controller import Supervisor
import math, sys

def main():
    robot    = Supervisor()
    timestep = int(robot.getBasicTimeStep())

    try:
        vx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
        vy = float(sys.argv[2]) if len(sys.argv) > 2 else 0.22
    except Exception:
        vx, vy = 0.30, 0.22

    target_speed = math.sqrt(vx*vx + vy*vy)
    self_node    = robot.getSelf()

    # Give initial kick
    self_node.setVelocity([vx, vy, 0, 0, 0, 0])
    print(f"[ball_mover] speed={target_speed:.3f} m/s, physics handles bouncing")

    while robot.step(timestep) != -1:
        vel   = self_node.getVelocity()   # [vx, vy, vz, wx, wy, wz]
        cvx, cvy = vel[0], vel[1]
        speed = math.sqrt(cvx*cvx + cvy*cvy)

        if speed < 0.01:
            # Completely stopped — give it a fresh kick in original direction
            self_node.setVelocity([vx, vy, 0, 0, 0, 0])
        elif abs(speed - target_speed) > 0.02:
            # Drifted from target speed — rescale, preserve direction
            scale = target_speed / speed
            self_node.setVelocity([cvx * scale, cvy * scale, 0, 0, 0, 0])

if __name__ == "__main__":
    main()
