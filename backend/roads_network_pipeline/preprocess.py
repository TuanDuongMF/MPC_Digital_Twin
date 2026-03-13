"""Stage 2: PreProcessor - converts extracted CSVs to Lanes.json and Plans.json."""

import csv
import json
import os
from collections import defaultdict


def _read_points_csv(filepath):
    """Read a lane/plan points CSV into {oid: [[x,y,z], ...]}."""
    table_map = defaultdict(list)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) < 4:
                    continue
                oid = row[0].strip()
                try:
                    x, y, z = float(row[1]), float(row[2]), float(row[3])
                except (ValueError, IndexError):
                    continue
                table_map[oid].append([x, y, z])
    except FileNotFoundError:
        print(f"  Warning: {filepath} not found")
    return table_map


def _read_metadata_csv(filepath):
    """Read zone/lane metadata CSV into list of dicts."""
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        print(f"  Warning: {filepath} not found")
    return rows


def preprocess(output_path):
    """Convert CSVs to Lanes.json and Plans.json."""
    print("[Stage 2] Preprocessing CSVs to JSON...")

    # Read point data
    preferred_points = _read_points_csv(os.path.join(output_path, "LanePreferred_points.csv"))
    left_points = _read_points_csv(os.path.join(output_path, "LaneLeft_points.csv"))
    right_points = _read_points_csv(os.path.join(output_path, "LaneRight_points.csv"))
    zone_polygons = _read_points_csv(os.path.join(output_path, "ZonePolygons.csv"))

    # Read metadata
    lane_metadata = _read_metadata_csv(os.path.join(output_path, "LaneMetaData.csv"))
    plan_metadata = _read_metadata_csv(os.path.join(output_path, "ZoneMetaData.csv"))

    # Build Lanes.json
    lanes = []
    for meta in lane_metadata:
        oid = meta.get("LANE_OID", "").strip()
        if not oid:
            continue

        points = preferred_points.get(oid, [])
        l_points = left_points.get(oid, [])
        r_points = right_points.get(oid, [])

        # Elevation check: skip lanes with all points below -400
        if points and all(p[2] < -400 for p in points):
            continue

        # Convert speed limit from m/s to km/h
        speed_str = meta.get("SPEED_LIMIT", "0")
        try:
            speed_ms = float(speed_str)
            speed_kmh = speed_ms * 3.6
        except (ValueError, TypeError):
            speed_kmh = 0.0

        lane = {
            "LANE_OID": oid,
            "SPEED_LIMIT": str(speed_kmh),
            "IS_ACTIVE": meta.get("IS_ACTIVE", "0").strip(),
            "AUTONOMOUS": meta.get("AUTONOMOUS", "0").strip(),
            "MANNED": meta.get("MANNED", "0").strip(),
            "DYNAMIC_GEN": meta.get("DYNAMIC_GEN", "0").strip(),
            "TYPE": meta.get("TYPE", "").strip(),
            "points": points,
            "Left_points": l_points,
            "Right_points": r_points,
        }
        lanes.append(lane)

    lanes_json = {"lanes": lanes}  # C++ uses lowercase "lanes"
    lanes_path = os.path.join(output_path, "Lanes.json")
    with open(lanes_path, "w", encoding="utf-8") as f:
        json.dump(lanes_json, f, indent=4)
    print(f"  Written: {lanes_path} ({len(lanes)} lanes)")

    # Build Plans.json
    plans = []
    for meta in plan_metadata:
        oid = meta.get("PLAN_OID", "").strip()
        if not oid:
            continue

        polygon = zone_polygons.get(oid, [])
        plan = dict(meta)
        plan["polygon"] = polygon
        plans.append(plan)

    plans_json = {"Plans": plans}
    plans_path = os.path.join(output_path, "Plans.json")
    with open(plans_path, "w", encoding="utf-8") as f:
        json.dump(plans_json, f, indent=4)
    print(f"  Written: {plans_path} ({len(plans)} plans)")

    print("[Stage 2] Preprocessing complete.")
