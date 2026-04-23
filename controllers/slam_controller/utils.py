"""
utils.py - Mathematical utility functions.
"""
import math


def normalize_angle(angle):
    """Wrap angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def point_distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def line_segment_intersection(p1, p2, p3, p4):
    x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
    denom = (x2-x1)*(y4-y3) - (y2-y1)*(x4-x3)
    if abs(denom) < 1e-10:
        return None
    t = ((x3-x1)*(y4-y3) - (y3-y1)*(x4-x3)) / denom
    u = ((x3-x1)*(y2-y1) - (y3-y1)*(x2-x1)) / denom
    if 0 <= t <= 1 and 0 <= u <= 1:
        return t, (x1 + t*(x2-x1), y1 + t*(y2-y1))
    return None
