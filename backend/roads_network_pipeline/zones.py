"""Stage 5: Zones Configuration - replicates Zones_Configurator from C++.

Reads Plans.json, output.json, and default template to build model.json
with load_zones, dump_zones, chargers, and service_stations.

Zone format matches C++ exactly: id, is_generated, keys, name, settings{},
destinationOID, machineOID, planOID.
"""

import csv
import json
import os

from .geometry import distance_2d, point_in_polygon


def _read_queue_exit_csv(filepath):
    """Read Zone_Queue or Zone_Exits CSV into {plan_oid: [lane_oid_int, ...]}.

    C++ reads these as long long values (strtoll).
    """
    mapping = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = str(list(row.values())[0]).strip()
                val = str(list(row.values())[1]).strip()
                if val and val != "null":
                    try:
                        mapping.setdefault(key, []).append(int(val))
                    except ValueError:
                        mapping.setdefault(key, []).append(val)
    except FileNotFoundError:
        pass
    return mapping


def _compare_coordinates(plan, nodes, roads):
    """Find inroad: road whose LAST node is closest to plan's QUEUE point.

    Replicates compareCoordinates() from C++.
    Distance threshold: 0.1 units (strict, no fallback).
    """
    str_x = str(plan.get("QUEUE_X", "")).strip()
    str_y = str(plan.get("QUEUE_Y", "")).strip()
    str_z = str(plan.get("QUEUE_Z", "")).strip()

    if not str_x or not str_y or not str_z:
        return 0, 0

    try:
        qx = float(str_x)
        qy = float(str_y)
    except (ValueError, TypeError):
        return 0, 0

    # C++ logic: check each node, if distance < 0.1, find road ending at that node
    for node in nodes:
        coords = node.get("coords", [0, 0, 0])
        # coords stored as [y, x, z]
        nx, ny = coords[1], coords[0]
        dist = distance_2d(qx, qy, nx, ny)
        if dist < 0.1:
            for road in roads:
                road_nodes = road.get("nodes", [])
                if road_nodes and road_nodes[-1] == node["id"]:
                    return road["id"], node["id"]

    return 0, 0


def _compare_coordinates_entry(plan, nodes, roads):
    """Find outroad: road whose FIRST node is closest to plan's EXIT point.

    Replicates compareCoordinates2() from C++.
    Uses EXIT_X/EXIT_Y only (no ENTRY fallback). Strict 0.1 threshold.
    """
    str_x = str(plan.get("EXIT_X", "")).strip()
    str_y = str(plan.get("EXIT_Y", "")).strip()
    str_z = str(plan.get("EXIT_Z", "")).strip()

    if not str_x or not str_y or not str_z:
        return 0, 0

    try:
        ex = float(str_x)
        ey = float(str_y)
    except (ValueError, TypeError):
        return 0, 0

    # C++ logic: check each node, if distance < 0.1, find road starting at that node
    for node in nodes:
        coords = node.get("coords", [0, 0, 0])
        nx, ny = coords[1], coords[0]
        dist = distance_2d(ex, ey, nx, ny)
        if dist < 0.1:
            for road in roads:
                road_nodes = road.get("nodes", [])
                if road_nodes and road_nodes[0] == node["id"]:
                    return road["id"], node["id"]

    return 0, 0


def _make_zone_settings(inroad, innode, outroad, outnode, zonetype="standard",
                        extra=None):
    """Build the settings object matching C++ zone format."""
    settings = {
        "n_spots": 1,
        "n_entrances": 1,
        "dtheta": 0,
        "flip": False,
        "queing": False,
        "roadlength": 100,
        "rolling_resistance": 1,
        "speed_limit": 10,
        "zonetype": zonetype,
        "innode_ids": [innode] if innode else [],
        "inroad_ids": [inroad] if inroad else [],
        "outnode_ids": [outnode] if outnode else [],
        "outroad_ids": [outroad] if outroad else [],
    }
    if extra:
        settings.update(extra)
    return settings


def configure_zones(output_path, template_path):
    """Build model.json with zones from Plans.json + output.json + template.

    Replicates Zones_Configurator::dozones() from C++.
    """
    print("[Stage 5] Configuring zones...")

    # Read input files
    plans_path = os.path.join(output_path, "Plans.json")
    output_json_path = os.path.join(output_path, "output.json")
    default_json_path = os.path.join(template_path, "default.json")

    with open(plans_path, "r", encoding="utf-8") as f:
        plans_data = json.load(f)
    with open(output_json_path, "r", encoding="utf-8") as f:
        road_data = json.load(f)

    # Load default template
    if os.path.exists(default_json_path):
        with open(default_json_path, "r", encoding="utf-8") as f:
            model = json.load(f)
    else:
        print(f"  Warning: template {default_json_path} not found, using empty model")
        model = {}

    nodes = road_data.get("nodes", [])
    roads = road_data.get("roads", [])

    # Merge road network into model (C++: defaultJson["roads"] = modelJson["roads"])
    model["nodes"] = nodes
    model["roads"] = roads

    # Read queue/exit lane mappings (C++: planQueue_json, planExit_json)
    queue_map = _read_queue_exit_csv(os.path.join(output_path, "Zone_Queue.csv"))
    exit_map = _read_queue_exit_csv(os.path.join(output_path, "Zone_Exits.csv"))

    plans = plans_data.get("Plans", [])

    load_zones = []
    dump_zones = []
    chargers = []
    service_stations = []

    zone_id = 0  # C++ starts at i = 0
    for plan in plans:
        disc = str(plan.get("DISCRIMINATOR_ID", "")).strip()
        plan_oid = str(plan.get("PLAN_OID", "")).strip()
        name = str(plan.get("NAME", "")).strip()

        # Convert planOID to int for JSON (C++ uses strtoll)
        try:
            plan_oid_int = int(plan_oid) if plan_oid else 0
        except ValueError:
            plan_oid_int = 0

        if disc == "LoadPlanImpl":
            # Find inroad/outroad via coordinate matching
            inroad, innode = _compare_coordinates(plan, nodes, roads)
            outroad, outnode = _compare_coordinates_entry(plan, nodes, roads)

            # C++: only create zone when both inroad and outroad found
            if inroad != 0 and outroad != 0:
                load_zone = {
                    "id": zone_id,
                    "is_generated": True,
                    "keys": "load_zones",
                    "name": name,
                    "settings": _make_zone_settings(inroad, innode, outroad, outnode),
                    "destinationOID": "",
                    "machineOID": "",
                    "planOID": plan_oid_int,
                }
                load_zones.append(load_zone)
                zone_id += 1

        elif disc == "DumpPlanImpl":
            processor_val = str(plan.get("PROCESSOR", "")).strip()
            if not processor_val:
                continue

            inroad, innode = _compare_coordinates(plan, nodes, roads)
            outroad, outnode = _compare_coordinates_entry(plan, nodes, roads)

            # C++: only create zone when both inroad and outroad found
            if inroad != 0 and outroad != 0:
                # machineOID = processor OID (C++ uses strtoll)
                try:
                    machine_oid = int(processor_val) if processor_val else ""
                except ValueError:
                    machine_oid = processor_val

                dump_zone = {
                    "id": zone_id,
                    "is_generated": True,
                    "keys": "dump_zones",
                    "name": name,
                    "settings": _make_zone_settings(inroad, innode, outroad, outnode),
                    "destinationOID": "",
                    "machineOID": machine_oid,
                    "planOID": plan_oid_int,
                }
                dump_zones.append(dump_zone)
                zone_id += 1

        elif disc == "FuelBayPlanImpl":
            fuel_bay_val = str(plan.get("FUEL_BAY", "")).strip()
            if not fuel_bay_val:
                continue

            # For chargers: use queue/exit CSV lane OIDs as road IDs
            queue_lanes = queue_map.get(plan_oid, [])
            exit_lanes = exit_map.get(plan_oid, [])
            inroad = queue_lanes[0] if queue_lanes else 0
            outroad = exit_lanes[0] if exit_lanes else 0

            # Get innode/outnode from road endpoints
            innode = 0
            outnode = 0
            road_ids = {r["id"] for r in roads}
            found_q = False
            found_e = False
            for road in roads:
                if road["id"] == inroad and inroad in road_ids:
                    road_nodes = road.get("nodes", [])
                    if road_nodes:
                        innode = road_nodes[-1]
                    found_q = True
                if road["id"] == outroad and outroad in road_ids:
                    road_nodes = road.get("nodes", [])
                    if road_nodes:
                        outnode = road_nodes[0]
                    found_e = True

            # C++: only append zone when both roads found and valid
            if found_e and found_q and inroad != 0 and outroad != 0:
                try:
                    machine_oid = int(fuel_bay_val) if fuel_bay_val else ""
                except ValueError:
                    machine_oid = fuel_bay_val

                charger = {
                    "id": zone_id,
                    "is_generated": True,
                    "type": "diesel",
                    "name": name,
                    "settings": _make_zone_settings(
                        inroad, innode, outroad, outnode,
                        zonetype="drivethrough",
                        extra={"reverse_speed_limit": 5},
                    ),
                    "destinationOID": "",
                    "machineOID": machine_oid,
                    "planOID": plan_oid_int,
                }
                chargers.append(charger)
                zone_id += 1

        elif disc == "WorkshopPlanImpl":
            queue_lanes = queue_map.get(plan_oid, [])
            exit_lanes = exit_map.get(plan_oid, [])
            inroad = queue_lanes[0] if queue_lanes else 0
            outroad = exit_lanes[0] if exit_lanes else 0

            innode = 0
            outnode = 0
            road_ids = {r["id"] for r in roads}
            found_q = False
            found_e = False
            for road in roads:
                if road["id"] == inroad and inroad in road_ids:
                    rn = road.get("nodes", [])
                    if rn:
                        innode = rn[-1]
                    found_q = True
                if road["id"] == outroad and outroad in road_ids:
                    rn = road.get("nodes", [])
                    if rn:
                        outnode = rn[0]
                    found_e = True

            # C++: only append zone when both roads found and valid
            if found_e and found_q and inroad != 0 and outroad != 0:
                workshop_val = str(plan.get("WORKSHOP", "")).strip()
                try:
                    dest_oid = int(workshop_val) if workshop_val else ""
                except ValueError:
                    dest_oid = workshop_val

                station = {
                    "id": zone_id,
                    "is_generated": True,
                    "is_show_service": False,
                    "is_deactivate": False,
                    "name": name,
                    "settings": _make_zone_settings(
                        inroad, innode, outroad, outnode,
                        zonetype="serviceparallel",
                    ),
                    "destinationOID": dest_oid,
                    "machineOID": "",
                    "planOID": plan_oid_int,
                }
                service_stations.append(station)
                zone_id += 1

        elif disc == "LayoverStationPlanImpl":
            layover_val = str(plan.get("LAYOVER_STATION", "")).strip()
            if not layover_val:
                continue

            queue_lanes = queue_map.get(plan_oid, [])
            exit_lanes = exit_map.get(plan_oid, [])
            inroad = queue_lanes[0] if queue_lanes else 0
            outroad = exit_lanes[0] if exit_lanes else 0

            innode = 0
            outnode = 0
            road_ids = {r["id"] for r in roads}
            found_q = False
            found_e = False
            for road in roads:
                if road["id"] == inroad and inroad in road_ids:
                    rn = road.get("nodes", [])
                    if rn:
                        innode = rn[-1]
                    found_q = True
                if road["id"] == outroad and outroad in road_ids:
                    rn = road.get("nodes", [])
                    if rn:
                        outnode = rn[0]
                    found_e = True

            # C++: only append zone when both roads found and valid
            if found_e and found_q and inroad != 0 and outroad != 0:
                try:
                    machine_oid = int(layover_val) if layover_val else ""
                except ValueError:
                    machine_oid = layover_val

                station = {
                    "id": zone_id,
                    "is_generated": True,
                    "is_show_service": False,
                    "is_deactivate": False,
                    "name": name,
                    "settings": _make_zone_settings(
                        inroad, innode, outroad, outnode,
                        zonetype="serviceparallel",
                    ),
                    "destinationOID": "",
                    "machineOID": machine_oid,
                    "planOID": plan_oid_int,
                }
                service_stations.append(station)
                zone_id += 1

    model["load_zones"] = load_zones
    model["dump_zones"] = dump_zones
    model["chargers"] = chargers
    model["service_stations"] = service_stations

    # Write model.json
    model_path = os.path.join(output_path, "model.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=4)

    print(f"  Load zones: {len(load_zones)}, Dump zones: {len(dump_zones)}")
    print(f"  Chargers: {len(chargers)}, Service stations: {len(service_stations)}")
    print(f"  Written: {model_path}")

    # Apply speed zones
    _apply_speed_zones(output_path)

    print("[Stage 5] Zones complete.")


def _apply_speed_zones(output_path):
    """Apply speed zone limits to nodes in model.json based on SpeedZones.json."""
    speed_zones_path = os.path.join(output_path, "SpeedZones.json")
    model_path = os.path.join(output_path, "model.json")

    if not os.path.exists(speed_zones_path):
        return

    with open(speed_zones_path, "r", encoding="utf-8") as f:
        speed_zones = json.load(f)
    with open(model_path, "r", encoding="utf-8") as f:
        model = json.load(f)

    nodes = model.get("nodes", [])
    updated = 0

    for zone in speed_zones:
        is_active = str(zone.get("IS_ACTIVE", "0")).strip().lower()
        speed_limit = zone.get("SPEED_LIMIT", "0")
        try:
            speed_val = float(speed_limit)
        except (ValueError, TypeError):
            continue

        if is_active not in ("1", "true", "yes") or speed_val == 0:
            continue

        polygon = zone.get("BOUNDARY_POLYGON", [])
        if not polygon or len(polygon) < 3:
            continue

        for node in nodes:
            coords = node.get("coords", [0, 0, 0])
            # coords stored as [y, x, z]
            point = [coords[1], coords[0]]
            if point_in_polygon(point, polygon):
                node["speed_limit"] = speed_val
                updated += 1

    model["nodes"] = nodes
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=4)

    if updated:
        print(f"  Applied speed zones: {updated} nodes updated")
