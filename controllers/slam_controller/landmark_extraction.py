"""
landmark_extraction.py - Extract point landmarks from LiDAR scans.
Split-and-Merge on clustered scan points → line segment endpoints.
"""
import math
from utils import point_distance


def scan_to_cartesian(robot_x, robot_y, robot_theta, scan_ranges,
                      max_range=1.5, lidar_offset=0.0):
    """
    Convert (local_angle, range) scan to world-frame (x,y) points.
    lidar_offset corrects for sensor mounting angle.
    """
    points = []
    for local_angle, r in scan_ranges:
        if r < max_range * 0.99:
            world_angle = robot_theta + local_angle + lidar_offset
            points.append((robot_x + r * math.cos(world_angle),
                           robot_y + r * math.sin(world_angle)))
    return points


def _p2l(point, p1, p2):
    x0,y0 = point; x1,y1 = p1; x2,y2 = p2
    dx = x2-x1; dy = y2-y1
    L = math.sqrt(dx*dx+dy*dy)
    if L < 1e-10: return point_distance(point, p1)
    return abs(dx*(y1-y0)-(x1-x0)*dy)/L


def _split(pts, thr, minp):
    if len(pts) < minp: return []
    ps, pe = pts[0], pts[-1]
    md, mi = 0.0, 0
    for i in range(1, len(pts)-1):
        d = _p2l(pts[i], ps, pe)
        if d > md: md, mi = d, i
    if md > thr:
        return _split(pts[:mi+1], thr, minp) + _split(pts[mi:], thr, minp)
    return [(ps, pe)]


def _merge(segs, thr):
    if len(segs) <= 1: return segs
    merged = [segs[0]]
    for curr in segs[1:]:
        prev = merged[-1]
        mid  = ((curr[0][0]+curr[1][0])/2, (curr[0][1]+curr[1][1])/2)
        if _p2l(mid, prev[0], prev[1]) < thr:
            merged[-1] = (prev[0], curr[1])
        else:
            merged.append(curr)
    return merged


def _merge_lm(lms, d=0.5):
    if not lms: return []
    used = [False]*len(lms); out = []
    for i in range(len(lms)):
        if used[i]: continue
        cx,cy = [lms[i][0]], [lms[i][1]]; used[i]=True
        for j in range(i+1, len(lms)):
            if not used[j] and point_distance(lms[i], lms[j]) < d:
                cx.append(lms[j][0]); cy.append(lms[j][1]); used[j]=True
        out.append((sum(cx)/len(cx), sum(cy)/len(cy)))
    return out


def extract_landmarks(points, dist_threshold=0.08, min_points=5, min_seg_len=0.15):
    if len(points) < min_points: return []
    clusters = [[points[0]]]
    for i in range(1, len(points)):
        if point_distance(points[i], points[i-1]) < 0.5:
            clusters[-1].append(points[i])
        else:
            if len(clusters[-1]) >= min_points:
                clusters.append([points[i]])
            else:
                clusters[-1] = [points[i]]
    segs = []
    for c in clusters:
        if len(c) >= min_points:
            s = _split(c, dist_threshold, min_points)
            segs.extend(_merge(s, dist_threshold))
    raw = []
    for s in segs:
        if point_distance(s[0], s[1]) >= min_seg_len:
            raw += [s[0], s[1]]
    return _merge_lm(raw, 0.5)
