"""Stage 3: TSM - compute curvature, lane width, banking for each lane."""

import json
import math
import os

import numpy as np

from .geometry import calc_banking, distance_3d


def _compute_curvature_eigen(points):
    """Compute average curvature using the same Eigen-based method as C++ TSM.

    C++ logic (TSM.cpp):
      - Build vectors = points[1..] - points[0..n-2]
      - For each i: vec1 = vectors[i], vec2 = vectors[i+1]
      - curvature[i] = 2 * |vec1 x vec2| / (|vec1| * |vec2| * (|vec1| + |vec2|))
      - avg_curvature = sum(curvature) / curvature.size()
    """
    if len(points) < 3:
        return 0.0

    mat = np.array(points)
    curvatures = []
    for i in range(1, len(mat) - 1):
        v1 = mat[i] - mat[i - 1]
        v2 = mat[i + 1] - mat[i]
        cross = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-12 or norm2 < 1e-12:
            continue
        denom = norm1 * norm2 * (norm1 + norm2)
        if denom < 1e-12:
            continue
        curvatures.append(2.0 * cross_norm / denom)

    return sum(curvatures) / len(curvatures) if curvatures else 0.0


def _compute_lane_width(left_points, right_points):
    """Average width = mean of start and end distances between left/right edges."""
    if not left_points or not right_points:
        return 0.0

    d_start = distance_3d(left_points[0], right_points[0])
    d_end = distance_3d(left_points[-1], right_points[-1])
    return (d_start + d_end) / 2.0


def _compute_banking(left_points, right_points):
    """Banking from left/right edge start point only (C++ uses only start)."""
    if not left_points or not right_points:
        return 0.0
    return calc_banking(left_points[0], right_points[0])


def compute_tsm(output_path):
    """Add CURVATURE, LANE_WIDTH, BANKING, TRACTION to each lane in Lanes.json."""
    print("[Stage 3] Computing TSM parameters...")

    lanes_path = os.path.join(output_path, "Lanes.json")
    with open(lanes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for lane in data.get("lanes", []):
        points = lane.get("points", [])

        # Match C++: compute curvature for all lanes first
        avg_curvature = _compute_curvature_eigen(points) if points else 0.0

        # C++ only writes TSM fields if Left_points, Right_points, and points exist and are non-empty
        left_pts = lane.get("Left_points")
        right_pts = lane.get("Right_points")

        if (
            left_pts
            and right_pts
            and points
            and len(left_pts) > 0
            and len(right_pts) > 0
            and len(points) > 0
        ):
            width = _compute_lane_width(left_pts, right_pts)
            banking = _compute_banking(left_pts, right_pts)

            lane["BANKING"] = str(banking)
            lane["LANE_WIDTH"] = str(width)
            lane["CURVATURE"] = str(avg_curvature)
            lane["TRACTION"] = "1"

    with open(lanes_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"  Updated: {lanes_path}")
    print("[Stage 3] TSM complete.")
