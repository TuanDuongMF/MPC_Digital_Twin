"""Geometry utility functions - replicates GeometryCalculations from C++."""

import math


def distance_3d(p1, p2):
    """Euclidean distance between two 3D points [x,y,z]."""
    return math.sqrt(
        (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
    )


def distance_2d(x1, y1, x2, y2):
    """2D distance."""
    return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def curvature_3points(p1, p2, p3):
    """Curvature through 3 points using triangle area formula: 4*Area/(a*b*c)."""
    a = distance_3d(p1, p2)
    b = distance_3d(p2, p3)
    c = distance_3d(p1, p3)
    if a < 1e-12 or b < 1e-12 or c < 1e-12:
        return 0.0
    s = (a + b + c) / 2.0
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq < 0:
        area_sq = 0.0
    area = math.sqrt(area_sq)
    return (4.0 * area) / (a * b * c)


def calc_banking(p1, p2):
    """Banking calculation matching C++ GeometryCalculations::calc_banking().

    C++: banking = dz/dx; if (banking > 100 || dx < 1) banking = dz/dy;
    """
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    dz = abs(p2[2] - p1[2])
    if dx > 1e-12:
        banking = dz / dx
    else:
        banking = 101  # force fallback
    if banking > 100 or dx < 1:
        banking = dz / dy if dy > 1e-12 else 0.0
    return banking


def cross_product(v1, v2):
    """Cross product of two 3D vectors."""
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


def vector_norm(v):
    """Euclidean norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def compute_angle_between_three_points(a, b, c):
    """Angle at point b formed by segments ba and bc (in degrees)."""
    ba = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if mag_ba < 1e-12 or mag_bc < 1e-12:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cos_angle))


def point_in_polygon(point, polygon):
    """Ray casting algorithm to test if point is inside polygon."""
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def line_intersection_2d(p1, p2, p3, p4):
    """Find intersection point of two line segments (p1-p2) and (p3-p4).
    Returns (x, y) or None if no intersection."""
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    return None
