"""Stage 1: SQL Data Extraction - replicates TestSql from C++.

Adapted for DB schemas that store coordinates in separate point tables
(LANE_PATH_POINT, LANE_LEFT_POINT, LANE_RIGHT_POINT, PLAN_POLYGON,
ZONE_POLYGON_POINT) instead of spatial geometry columns.
"""

import csv
import json
import os

import pyodbc


def get_connection(server, username, password, database):
    """Create pyodbc connection to SQL Server."""
    for driver in [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
    ]:
        try:
            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={server};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password};"
                "TrustServerCertificate=yes;"
            )
            return pyodbc.connect(conn_str)
        except pyodbc.Error:
            continue
    raise ConnectionError(f"Cannot connect to {server}/{database}")


def write_csv(filepath, headers, rows):
    """Write rows to CSV file with given headers."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)
    print(f"  Written: {os.path.basename(filepath)} ({len(rows)} rows)")


def query_fetchall(conn, sql):
    """Execute query and return all rows as list of tuples."""
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    cursor.close()
    return rows


def _table_exists(conn, table_name):
    """Check if a table exists in the current database."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = ?",
        table_name,
    )
    result = cursor.fetchone()
    cursor.close()
    return result is not None


def _column_exists(conn, table_name, column_name):
    """Check if a column exists in a table."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ? AND COLUMN_NAME = ?",
        table_name, column_name,
    )
    result = cursor.fetchone()
    cursor.close()
    return result is not None


def _extract_lane_points(conn, output_path):
    """Extract lane points - handles both spatial and table-based schemas."""
    # Try separate point tables first (non-spatial schema)
    if _table_exists(conn, "LANE_PATH_POINT"):
        rows = query_fetchall(conn, "SELECT LANE_OID, X, Y, Z FROM LANE_PATH_POINT ORDER BY LANE_OID, POINT_NR")
        write_csv(os.path.join(output_path, "LanePreferred_points.csv"), ["lane_oid", "x", "y", "z"], rows)
    elif _column_exists(conn, "LANE", "PREFERRED_PATH_POINTS"):
        # Spatial schema - use AsTextZM()
        rows = query_fetchall(conn, "SELECT LANE_OID, PREFERRED_PATH_POINTS.AsTextZM() FROM LANE")
        points = []
        for lane_oid, wkt in rows:
            for pt in _parse_linestring(wkt):
                points.append([lane_oid, pt[0], pt[1], pt[2]])
        write_csv(os.path.join(output_path, "LanePreferred_points.csv"), ["lane_oid", "x", "y", "z"], points)
    else:
        write_csv(os.path.join(output_path, "LanePreferred_points.csv"), ["lane_oid", "x", "y", "z"], [])

    if _table_exists(conn, "LANE_LEFT_POINT"):
        rows = query_fetchall(conn, "SELECT LANE_OID, X, Y, Z FROM LANE_LEFT_POINT ORDER BY LANE_OID, POINT_NR")
        write_csv(os.path.join(output_path, "LaneLeft_points.csv"), ["lane_oid", "x", "y", "z"], rows)
    elif _column_exists(conn, "LANE", "LEFT_EDGE_POINTS"):
        rows = query_fetchall(conn, "SELECT LANE_OID, LEFT_EDGE_POINTS.AsTextZM() FROM LANE")
        points = []
        for lane_oid, wkt in rows:
            for pt in _parse_linestring(wkt):
                points.append([lane_oid, pt[0], pt[1], pt[2]])
        write_csv(os.path.join(output_path, "LaneLeft_points.csv"), ["lane_oid", "x", "y", "z"], points)
    else:
        write_csv(os.path.join(output_path, "LaneLeft_points.csv"), ["lane_oid", "x", "y", "z"], [])

    if _table_exists(conn, "LANE_RIGHT_POINT"):
        rows = query_fetchall(conn, "SELECT LANE_OID, X, Y, Z FROM LANE_RIGHT_POINT ORDER BY LANE_OID, POINT_NR")
        write_csv(os.path.join(output_path, "LaneRight_points.csv"), ["lane_oid", "x", "y", "z"], rows)
    elif _column_exists(conn, "LANE", "RIGHT_EDGE_POINTS"):
        rows = query_fetchall(conn, "SELECT LANE_OID, RIGHT_EDGE_POINTS.AsTextZM() FROM LANE")
        points = []
        for lane_oid, wkt in rows:
            for pt in _parse_linestring(wkt):
                points.append([lane_oid, pt[0], pt[1], pt[2]])
        write_csv(os.path.join(output_path, "LaneRight_points.csv"), ["lane_oid", "x", "y", "z"], points)
    else:
        write_csv(os.path.join(output_path, "LaneRight_points.csv"), ["lane_oid", "x", "y", "z"], [])


def _extract_plan_polygons(conn, output_path):
    """Extract plan polygons - handles both schemas."""
    if _table_exists(conn, "PLAN_POLYGON"):
        rows = query_fetchall(conn, "SELECT PLAN_OID, X, Y, Z FROM PLAN_POLYGON ORDER BY PLAN_OID, POINT_NR")
        write_csv(os.path.join(output_path, "ZonePolygons.csv"), ["PLAN_OID", "x", "y", "z"], rows)
    elif _column_exists(conn, "PLAN_MODEL", "PLAN_POLYGON"):
        rows = query_fetchall(conn, "SELECT PLAN_OID, PLAN_POLYGON.AsTextZM() FROM PLAN_MODEL")
        points = []
        for plan_oid, wkt in rows:
            for pt in _parse_polygon(wkt):
                points.append([plan_oid] + pt)
        write_csv(os.path.join(output_path, "ZonePolygons.csv"), ["PLAN_OID", "x", "y", "z"], points)
    else:
        write_csv(os.path.join(output_path, "ZonePolygons.csv"), ["PLAN_OID", "x", "y", "z"], [])


def _extract_zones(conn, output_path):
    """Extract zone data with polygon points for SpeedZones."""
    # Zone metadata (no polygon in this query)
    zone_meta_headers = [
        "ZONE_OID", "DISCRIMINATOR_ID", "ID", "NAME", "IS_ACTIVE",
        "AUTONOMOUS_INC_ZONE", "AUTONOMOUS_EXC_ZONE", "PASSABLE", "SPEED_LIMIT",
    ]

    rows = query_fetchall(conn, """
        SELECT ZONE_OID, DISCRIMINATOR_ID, ID, NAME, IS_ACTIVE,
               AUTONOMOUS_INC_ZONE, AUTONOMOUS_EXC_ZONE, PASSABLE, SPEED_LIMIT
        FROM ZONE
    """)
    # Write basic zone CSV (without polygon)
    write_csv(os.path.join(output_path, "SpeedZones.csv"), zone_meta_headers, rows)

    # Build SpeedZones.json with polygon data
    zones_json = []
    zone_polygons = {}

    if _table_exists(conn, "ZONE_POLYGON_POINT"):
        poly_rows = query_fetchall(conn, """
            SELECT ZONE_OID, POINT_X, POINT_Y, POINT_Z
            FROM ZONE_POLYGON_POINT
            ORDER BY ZONE_OID, POINT_NR
        """)
        for zone_oid, px, py, pz in poly_rows:
            zone_oid_str = str(zone_oid)
            zone_polygons.setdefault(zone_oid_str, []).append(
                [float(px or 0), float(py or 0), float(pz or 0)]
            )

    for row in rows:
        zone_oid = str(row[0])
        polygon = zone_polygons.get(zone_oid, [])
        if not polygon:
            continue
        zones_json.append({
            "ZONE_OID": zone_oid,
            "DISCRIMINATOR_ID": str(row[1] or ""),
            "ID": str(row[2] or ""),
            "NAME": str(row[3] or ""),
            "IS_ACTIVE": str(row[4] or ""),
            "AUTONOMOUS_INC_ZONE": str(row[5] or ""),
            "AUTONOMOUS_EXC_ZONE": str(row[6] or ""),
            "PASSABLE": str(row[7] or ""),
            "SPEED_LIMIT": str(row[8] or ""),
            "BOUNDARY_POLYGON": polygon,
        })

    json_path = os.path.join(output_path, "SpeedZones.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(zones_json, f, indent=4)
    print(f"  Written: SpeedZones.json ({len(zones_json)} zones with polygons)")


def _extract_waypoints(conn, output_path):
    """Extract waypoints - handles both schemas."""
    # Check if ROUTE_POINT_X columns exist (non-spatial)
    if _column_exists(conn, "VIRTUALBEACON", "ROUTE_POINT_X"):
        rows = query_fetchall(conn, """
            SELECT VIRTUALBEACON_OID, NAME, TYPE, LOADERID, ID,
                   ROUTE_POINT_X, ROUTE_POINT_Y, ROUTE_POINT_Z
            FROM VIRTUALBEACON
        """)
        write_csv(
            os.path.join(output_path, "Waypoints.csv"),
            ["VIRTUALBEACON_OID", "NAME", "TYPE", "LOADERID", "ID",
             "ROUTE_POINT_X", "ROUTE_POINT_Y", "ROUTE_POINT_Z"],
            rows,
        )
    elif _column_exists(conn, "VIRTUALBEACON", "ROUTE_POINT"):
        # Spatial schema
        rows = query_fetchall(conn, """
            SELECT VIRTUALBEACON_OID, NAME, TYPE, LOADERID, ID,
                   ROUTE_POINT.AsTextZM()
            FROM VIRTUALBEACON
        """)
        wp_rows = []
        for row in rows:
            pt = _parse_point(row[5])
            if pt:
                wp_rows.append([row[0], row[1], row[2], row[3], row[4], pt[0], pt[1], pt[2]])
            else:
                wp_rows.append([row[0], row[1], row[2], row[3], row[4], "", "", ""])
        write_csv(
            os.path.join(output_path, "Waypoints.csv"),
            ["VIRTUALBEACON_OID", "NAME", "TYPE", "LOADERID", "ID",
             "ROUTE_POINT_X", "ROUTE_POINT_Y", "ROUTE_POINT_Z"],
            wp_rows,
        )
    else:
        write_csv(
            os.path.join(output_path, "Waypoints.csv"),
            ["VIRTUALBEACON_OID", "NAME", "TYPE", "LOADERID", "ID",
             "ROUTE_POINT_X", "ROUTE_POINT_Y", "ROUTE_POINT_Z"],
            [],
        )


def _parse_linestring(wkt):
    """Parse LINESTRING ZM WKT into [[x,y,z], ...]."""
    import re
    if not wkt:
        return []
    text = re.sub(r"^LINESTRING\s*Z?M?\s*\(", "", str(wkt).strip())
    text = text.rstrip(")")
    points = []
    for part in text.split(","):
        coords = part.strip().split()
        if len(coords) >= 3:
            points.append([float(coords[0]), float(coords[1]), float(coords[2])])
        elif len(coords) == 2:
            points.append([float(coords[0]), float(coords[1]), 200.0])
    return points


def _parse_point(wkt):
    """Parse POINT ZM WKT into [x, y, z]."""
    import re
    if not wkt:
        return []
    text = re.sub(r"^POINT\s*Z?M?\s*\(", "", str(wkt).strip())
    text = text.rstrip(")")
    coords = text.strip().split()
    if len(coords) >= 3:
        return [float(coords[0]), float(coords[1]), float(coords[2])]
    return []


def _parse_polygon(wkt):
    """Parse POLYGON ZM WKT into [[x,y,z], ...]."""
    import re
    if not wkt:
        return []
    text = re.sub(r"^POLYGON\s*Z?M?\s*\(\(", "", str(wkt).strip())
    text = text.rstrip(")").rstrip(")")
    points = []
    for part in text.split(","):
        coords = part.strip().split()
        if len(coords) >= 2:
            pt = [float(c) for c in coords[:3]]
            if len(pt) == 2:
                pt.append(0.0)
            points.append(pt)
    return points


def extract_all(server, username, password, output_path):
    """Extract all data from SQL Server databases and write CSVs."""
    os.makedirs(output_path, exist_ok=True)
    print("[Stage 1] Extracting data from SQL Server...")

    # ── msmodel database ──
    conn = get_connection(server, username, password, "msmodel")
    print("  Connected to msmodel")

    # Lane Metadata
    rows = query_fetchall(conn, """
        SELECT LANE_OID, SPEED_LIMIT, IS_ACTIVE, AUTONOMOUS, MANNED, DYNAMIC_GEN, TYPE
        FROM LANE
    """)
    write_csv(
        os.path.join(output_path, "LaneMetaData.csv"),
        ["LANE_OID", "SPEED_LIMIT", "IS_ACTIVE", "AUTONOMOUS", "MANNED", "DYNAMIC_GEN", "TYPE"],
        rows,
    )

    # Lane Points (preferred, left, right)
    _extract_lane_points(conn, output_path)

    # Plan Metadata
    rows = query_fetchall(conn, """
        SELECT PLAN_OID, DISCRIMINATOR_ID, NAME, ENTRY_X, ENTRY_Y, ENTRY_Z,
               EXIT_X, EXIT_Y, EXIT_Z, QUEUE_X, QUEUE_Y, QUEUE_Z,
               LOAD_PLAN_TYPE, DUMP_PLAN_TYPE, FUEL_BAY, PROCESSOR,
               LANE_SPEED_LIMIT, SERVER_SCRIPT, ASSOCIATED_ZONE,
               WORKSHOP, WATER_REFILL_STATION, LAYOVER_STATION
        FROM PLAN_MODEL
    """)
    write_csv(
        os.path.join(output_path, "ZoneMetaData.csv"),
        ["PLAN_OID", "DISCRIMINATOR_ID", "NAME", "ENTRY_X", "ENTRY_Y", "ENTRY_Z",
         "EXIT_X", "EXIT_Y", "EXIT_Z", "QUEUE_X", "QUEUE_Y", "QUEUE_Z",
         "LOAD_PLAN_TYPE", "DUMP_PLAN_TYPE", "FUEL_BAY", "PROCESSOR",
         "LANE_SPEED_LIMIT", "SERVER_SCRIPT", "ASSOCIATED_ZONE",
         "WORKSHOP", "WATER_REFILL_STATION", "LAYOVER_STATION"],
        rows,
    )

    # Plan Polygons
    _extract_plan_polygons(conn, output_path)

    # Zones + SpeedZones.json
    _extract_zones(conn, output_path)

    # Plan Queue
    rows = query_fetchall(conn, "SELECT PLAN_OID, LANE_OID FROM PLAN_LANE_INFO_QUEUE")
    write_csv(os.path.join(output_path, "Zone_Queue.csv"), ["PLAN_OID", "QUEUE_LANE_OID"], rows)

    # Plan Exit
    rows = query_fetchall(conn, "SELECT PLAN_OID, LANE_OID FROM PLAN_LANE_INFO_EXIT")
    write_csv(os.path.join(output_path, "Zone_Exits.csv"), ["PLAN_OID", "EXIT_LANE_OID"], rows)

    # Spot Points
    if _table_exists(conn, "SPOT_POINT"):
        rows = query_fetchall(conn, "SELECT SPOT_POINT_OID, PLAN_OID, X, Y, Z FROM SPOT_POINT")
        write_csv(os.path.join(output_path, "Spot_Points.csv"), ["SPOT_POINT_OID", "PLAN_OID", "X", "Y", "Z"], rows)

    # Spot Entry/Exit
    if _table_exists(conn, "SPOT_POINT_ENTRY_LANES"):
        rows = query_fetchall(conn, "SELECT SPOT_POINT_OID, LANE_OID FROM SPOT_POINT_ENTRY_LANES")
        write_csv(os.path.join(output_path, "Spot_Entry.csv"), ["SPOT_POINT_OID", "ENTRY_LANE_OID"], rows)

    if _table_exists(conn, "SPOT_POINT_EXIT_LANES"):
        rows = query_fetchall(conn, "SELECT SPOT_POINT_OID, LANE_OID FROM SPOT_POINT_EXIT_LANES")
        write_csv(os.path.join(output_path, "Spot_Exit.csv"), ["SPOT_POINT_OID", "EXIT_LANE_OID"], rows)

    # Roads
    rows = query_fetchall(conn, """
        SELECT ROADSEGMENT_OID, IS_ACTIVE, DESCRIPTION, ROLLINGRESIST,
               SPEEDLIMIT, START1, ENDWAYPOINT, ROADTYPE
        FROM ROADSEGMENT
    """)
    write_csv(
        os.path.join(output_path, "Roads.csv"),
        ["ROADSEGMENT_OID", "IS_ACTIVE", "DESCRIPTION", "ROLLINGRESIST",
         "SPEEDLIMIT", "START1", "ENDWAYPOINT", "ROADTYPE"],
        rows,
    )

    # Waypoints
    _extract_waypoints(conn, output_path)

    # Machines
    rows = query_fetchall(conn, """
        SELECT NAME, ECF_CLASS_ID, IS_ACTIVE, MACHINE_OID, SERIALNUMBER,
               ID, CLASS, IGNOREFORASSIGNMENT, PRODUCTIVITY, FUELCAPACITY,
               TKPHRATE, MAXPAYLOAD, NOMINALPAYLOAD, DEFAULTMATERIAL,
               LOAD_PLAN_OID, MATERIAL, ENDPOINT_ID, EXTERNALREF,
               EXTERNALDESC, IS_CRUSHER
        FROM MACHINE
    """)
    write_csv(
        os.path.join(output_path, "Machine.csv"),
        ["NAME", "ECF_CLASS_ID", "IS_ACTIVE", "MACHINE_OID", "SERIALNUMBER",
         "ID", "CLASS", "IGNOREFORASSIGNMENT", "PRODUCTIVITY", "FUELCAPACITY",
         "TKPHRATE", "MAXPAYLOAD", "NOMINALPAYLOAD", "DEFAULTMATERIAL",
         "LOAD_PLAN_OID", "MATERIAL", "ENDPOINT_ID", "EXTERNALREF",
         "EXTERNALDESC", "IS_CRUSHER"],
        rows,
    )

    # Machine Comms
    if _table_exists(conn, "MACHINE_COMMS_URL"):
        try:
            rows = query_fetchall(conn, """
                SELECT MCU.OID,
                       LEFT(REPLACE(MCU.COMMS_URL, 'tmac://', ''),
                            CHARINDEX(':', REPLACE(MCU.COMMS_URL, 'tmac://', '')) - 1) AS ip,
                       SUBSTRING(REPLACE(MCU.COMMS_URL, 'tmac://', ''),
                            CHARINDEX(':', REPLACE(MCU.COMMS_URL, 'tmac://', '')) + 1,
                            LEN(REPLACE(MCU.COMMS_URL, 'tmac://', ''))
                            - CHARINDEX(':', REPLACE(MCU.COMMS_URL, 'tmac://', ''))) AS port
                FROM MACHINE_COMMS_URL AS MCU
                WHERE MCU.INTERFACE_NAME = 'Assignment'
            """)
        except Exception:
            rows = []
        write_csv(os.path.join(output_path, "MachineComms.csv"), ["OID", "ip", "port"], rows)

    # Machine Class
    rows = query_fetchall(conn, "SELECT NAME, ECF_CLASS_ID, IS_ACTIVE, MACHINECLASS_OID FROM MACHINECLASS")
    write_csv(
        os.path.join(output_path, "Machine_Class.csv"),
        ["NAME", "ECF_CLASS_ID", "IS_ACTIVE", "MACHINECLASS_OID"], rows,
    )

    # Assignment Group
    rows = query_fetchall(conn, "SELECT NAME, ASSIGNMENTGROUP_OID FROM ASSIGNMENTGROUP")
    write_csv(os.path.join(output_path, "Machine_Assignment_group.csv"), ["NAME", "ASSIGNMENTGROUP_OID"], rows)

    # Crew
    rows = query_fetchall(conn, "SELECT NAME, OID FROM CREW")
    write_csv(os.path.join(output_path, "Machine_Crew.csv"), ["NAME", "OID"], rows)

    # Fleet
    rows = query_fetchall(conn, "SELECT NAME, FLEET_OID FROM FLEET")
    write_csv(os.path.join(output_path, "Machine_Fleet.csv"), ["NAME", "FLEET_OID"], rows)

    # Fleet Element
    rows = query_fetchall(conn, "SELECT OID, MACHINE_OID FROM FLEET_ELEMENT")
    write_csv(os.path.join(output_path, "Machine_Fleet_Element.csv"), ["OID", "MACHINE_OID"], rows)

    # Materials
    rows = query_fetchall(conn, """
        SELECT MATERIAL_OID, CODECAES, COLOUR, DESCRIPTION, ID,
               BANKDENSITY, LOOSEDENSITY, MATUNIT, NAME, IS_ACTIVE,
               MATERIALGROUP, EXTERNALREF, EXTERNALDESC,
               MODEL_UPDATE_VERSION, LAYER_UPDATE_VERSION
        FROM MATERIAL
    """)
    write_csv(
        os.path.join(output_path, "Materials.csv"),
        ["MATERIAL_OID", "CODECAES", "COLOUR", "DESCRIPTION", "ID",
         "BANKDENSITY", "LOOSEDENSITY", "MATUNIT", "NAME", "IS_ACTIVE",
         "MATERIALGROUP", "EXTERNALREF", "EXTERNALDESC",
         "MODEL_UPDATE_VERSION", "LAYER_UPDATE_VERSION"],
        rows,
    )

    # Loading Tool Materials
    rows = query_fetchall(conn, "SELECT OID, MATERIAL_OID FROM LOADINGTOOL_MATERIAL")
    write_csv(os.path.join(output_path, "LoadingToolMaterials.csv"), ["OID", "MATERIAL_OID"], rows)

    # Processor Materials
    rows = query_fetchall(conn, "SELECT OID, MATERIAL_OID FROM PROCESSOR_MATERIAL")
    write_csv(os.path.join(output_path, "ProcessorMaterials.csv"), ["OID", "MATERIAL_OID"], rows)

    # Destinations
    rows = query_fetchall(conn, """
        SELECT loc.LOCATION_OID, loc.NAME, loc_way.WAYPOINT_OID,
               loc.IS_ACTIVE, loc.ISSOURCE, loc.ISSINK, loc.ISSTATION,
               loc.EXTERNALREF, vb.LOADERID, vb.DESCRIPTION
        FROM LOCATION AS loc
        JOIN LOCATION_WAYPOINT AS loc_way ON loc_way.OID = loc.LOCATION_OID
        JOIN VIRTUALBEACON AS vb ON vb.VIRTUALBEACON_OID = loc_way.WAYPOINT_OID
    """)
    write_csv(
        os.path.join(output_path, "Destinations.csv"),
        ["LOCATION_OID", "NAME", "WAYPOINT_OID", "IS_ACTIVE", "ISSOURCE",
         "ISSINK", "ISSTATION", "EXTERNALREF", "LOADERID", "DESCRIPTION"],
        rows,
    )

    # Delay Category
    rows = query_fetchall(conn, "SELECT DELAYCATEGORY_OID, NAME, COLOUR, BOLD, ITALIC FROM DELAYCATEGORY")
    write_csv(os.path.join(output_path, "DELAYCATEGORY.csv"), ["DELAYCATEGORY_OID", "NAME", "COLOUR", "BOLD", "ITALIC"], rows)

    # Delay Class
    rows = query_fetchall(conn, """
        SELECT DELAYCLASS_OID, ACKNOWLEDGEMENTREQ, INFIELD, DELAYCATEGORY,
               DESCRIPTION, DURATION, ID, NAME, IS_ACTIVE,
               PRODUCTIONREPORTINGONLY, EXTERNALDESC, EXTERNALREF,
               EMAIL, ENGINESTOPPED, ISVALIDFORROAD, ISVALIDFORLOCATION,
               DELAYACTIVITYCLASS, MODEL_UPDATE_VERSION, LAYER_UPDATE_VERSION
        FROM DELAYCLASS
    """)
    write_csv(
        os.path.join(output_path, "DELAYCLASS.csv"),
        ["DELAYCLASS_OID", "ACKNOWLEDGEMENTREQ", "INFIELD", "DELAYCATEGORY",
         "DESCRIPTION", "DURATION", "ID", "NAME", "IS_ACTIVE",
         "PRODUCTIONREPORTINGONLY", "EXTERNALDESC", "EXTERNALREF",
         "EMAIL", "ENGINESTOPPED", "ISVALIDFORROAD", "ISVALIDFORLOCATION",
         "DELAYACTIVITYCLASS", "MODEL_UPDATE_VERSION", "LAYER_UPDATE_VERSION"],
        rows,
    )

    conn.close()

    # ── mshist database (Production Plans) ──
    try:
        conn = get_connection(server, username, password, "mshist")
        print("  Connected to mshist")
        rows = query_fetchall(conn, """
            SELECT PROD_REQUEST_OID, MIN_RATE, MAX_RATE,
                   FROM_SERVER, TO_SERVER, MATERIAL
            FROM PROD_REQUEST_GOAL
        """)
        write_csv(
            os.path.join(output_path, "Production_plans.csv"),
            ["PROD_REQUEST_OID", "MIN_RATE", "MAX_RATE", "FROM_SERVER", "TO_SERVER", "MATERIAL"],
            rows,
        )
        conn.close()
    except Exception as e:
        print(f"  Warning: mshist not available ({e}), writing empty Production_plans.csv")
        write_csv(
            os.path.join(output_path, "Production_plans.csv"),
            ["PROD_REQUEST_OID", "MIN_RATE", "MAX_RATE", "FROM_SERVER", "TO_SERVER", "MATERIAL"],
            [],
        )

    # ── mspitmodel database (Machine Positions) ──
    try:
        conn = get_connection(server, username, password, "mspitmodel")
        print("  Connected to mspitmodel")
        rows = query_fetchall(conn, """
            SELECT MACHINE_OID, LOCATION_OID, WAYPOINT_OID, X, Y, Z, HEADING
            FROM MACHINE_IN_PIT
        """)
        write_csv(
            os.path.join(output_path, "Machine_POS.csv"),
            ["MACHINE_OID", "DESTINATION_OID", "WAYPOINT_OID", "X", "Y", "Z", "HEADING"],
            rows,
        )
        conn.close()
    except Exception as e:
        print(f"  Warning: mspitmodel not available ({e}), writing empty Machine_POS.csv")
        write_csv(
            os.path.join(output_path, "Machine_POS.csv"),
            ["MACHINE_OID", "DESTINATION_OID", "WAYPOINT_OID", "X", "Y", "Z", "HEADING"],
            [],
        )

    print("[Stage 1] Extraction complete.")
