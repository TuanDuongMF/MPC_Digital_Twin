"""
Generate All Simulation Data

Creates both model and simulation files from database telemetry:
1. Model file (nodes, roads, zones) - for road network visualization
2. DES Inputs file - for simulation engine configuration
3. Events Ledger file - for animation playback

Usage:
    python scripts/simulation_generator.py                           # Use config.json
    python scripts/simulation_generator.py --config custom.json      # Use custom config
    python scripts/simulation_generator.py --site "BhpEscondida"     # Override site from CLI
    python scripts/simulation_generator.py --list-sites              # List available sites

Config file (config.json) contains all configurable parameters.
CLI arguments override config file values.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import deque
import math

# Add webapp directory to path for imports
# backend/scripts/simulation_generator.py -> backend/ -> webapp/
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
webapp_root = os.path.dirname(backend_dir)
sys.path.insert(0, webapp_root)

try:
    import pymysql
    pymysql.install_as_MySQLdb()
except ImportError:
    print("Error: pymysql not found. Install with: pip install pymysql")
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(iterable, desc=None, total=None, disable=False):
        return iterable

from backend.core.db_config import DB_CONFIG, OUTPUT_PATH, EXAMPLE_JSON_PATH
from backend.simulation_analysis import GPSToEventsConverter


# =============================================================================
# Configuration
# =============================================================================

# Default config file path (relative to this script)
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

# Resolve paths
scripts_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(scripts_dir)

def resolve_path(path_var, default_relative):
    """Resolve path: if relative, make it relative to backend_dir; if absolute, use as-is."""
    if os.path.isabs(path_var):
        return path_var
    return os.path.join(backend_dir, path_var)

# Machine templates file path (resolved from EXAMPLE_JSON_PATH)
example_json_resolved = resolve_path(EXAMPLE_JSON_PATH, "../exampleJSON")
MACHINE_TEMPLATES_PATH = os.path.join(
    example_json_resolved,
    "simulation", "machine_templates.json"
)

# Default configuration values
DEFAULT_CONFIG = {
    "site": None,
    "output_dir": None,  # Will use OUTPUT_PATH from env if None
    "machine_templates_path": None,  # Use default if None
    "data_fetching": {
        "limit": 100000,
        "sample_interval": 5,
    },
    "road_detection": {
        "grid_size": 5.0,
        "min_density": 3,
        "simplify_epsilon": 5.0,
    },
    "zone_detection": {
        "grid_size": 10.0,
        "min_stop_count": 20,
    },
    "simulation": {
        "sim_time": 480,
    },
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file. If None, uses default config.json
        
    Returns:
        Configuration dictionary with all parameters
    """
    config = DEFAULT_CONFIG.copy()
    
    # Deep copy nested dicts
    config["data_fetching"] = DEFAULT_CONFIG["data_fetching"].copy()
    config["road_detection"] = DEFAULT_CONFIG["road_detection"].copy()
    config["zone_detection"] = DEFAULT_CONFIG["zone_detection"].copy()
    config["simulation"] = DEFAULT_CONFIG["simulation"].copy()
    
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = json.load(f)
            
            # Merge file config into default config
            if "site" in file_config:
                config["site"] = file_config["site"]
            if "output_dir" in file_config:
                config["output_dir"] = file_config["output_dir"]
            if "machine_templates_path" in file_config:
                config["machine_templates_path"] = file_config["machine_templates_path"]
            
            if "data_fetching" in file_config:
                config["data_fetching"].update(file_config["data_fetching"])
            if "road_detection" in file_config:
                config["road_detection"].update(file_config["road_detection"])
            if "zone_detection" in file_config:
                config["zone_detection"].update(file_config["zone_detection"])
            if "simulation" in file_config:
                config["simulation"].update(file_config["simulation"])
            
            print(f"  Loaded config from: {config_path}")
        except Exception as e:
            print(f"  Warning: Could not load config file: {e}")
            print(f"  Using default configuration")
    else:
        print(f"  Config file not found: {config_path}")
        print(f"  Using default configuration")
    
    return config


def save_default_config(config_path: str = None):
    """Save default configuration to file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    
    print(f"Default config saved to: {config_path}")


def load_machine_templates(templates_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load machine templates from JSON file.
    
    Args:
        templates_path: Path to machine templates file. If None, uses default from env.
        
    Returns:
        Dictionary containing hauler and loader templates
    """
    if templates_path is None:
        templates_path = MACHINE_TEMPLATES_PATH
    else:
        # Resolve relative path if provided and not absolute
        if not os.path.isabs(templates_path):
            templates_path = os.path.join(backend_dir, templates_path)
    
    if os.path.exists(templates_path):
        try:
            with open(templates_path, "r", encoding="utf-8") as f:
                templates = json.load(f)
            print(f"  Loaded machine templates from: {templates_path}")
            return templates
        except Exception as e:
            print(f"  Warning: Could not load machine templates: {e}")
            print(f"  Using hardcoded defaults")
    else:
        print(f"  Machine templates file not found: {templates_path}")
        print(f"  Using hardcoded defaults")
    
    # Return empty dict to signal using hardcoded defaults
    return {}


def deep_copy_dict(d: Dict) -> Dict:
    """Deep copy a dictionary (handles nested dicts and lists)."""
    import copy
    return copy.deepcopy(d)


def merge_dict(base: Dict, overrides: Dict) -> Dict:
    """
    Merge overrides into base dict recursively.
    
    Args:
        base: Base dictionary
        overrides: Dictionary with values to override
        
    Returns:
        Merged dictionary
    """
    result = deep_copy_dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = deep_copy_dict(value) if isinstance(value, (dict, list)) else value
    return result


# =============================================================================
# Database Functions
# =============================================================================

def get_connection():
    """Create database connection."""
    try:
        connection = pymysql.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database=DB_CONFIG["database"],
            charset=DB_CONFIG["charset"],
            autocommit=DB_CONFIG["autocommit"],
            connect_timeout=30,
            read_timeout=120,
            write_timeout=120,
        )
        return connection
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def fetch_sites(cursor) -> List[Dict]:
    """Fetch list of available sites."""
    query = """
        SELECT DISTINCT m.`Site Name`, s.`SiteNameShort`, s.`SiteId`
        FROM amt_cycleprodinfo cp
        INNER JOIN machines m ON cp.`Machine Unique Id` = m.`Machine Unique Id`
        INNER JOIN site s ON m.`Site Name` = s.`Site Name`
        ORDER BY m.`Site Name`
    """
    cursor.execute(query)
    sites = []
    for row in cursor.fetchall():
        sites.append({
            "site_name": row[0],
            "site_short": row[1],
            "site_id": row[2],
        })
    return sites


def fetch_machines(cursor, site_name: Optional[str] = None) -> Dict[int, Dict]:
    """Fetch machine information."""
    query = """
        SELECT DISTINCT m.`Machine Unique Id`, m.`Machine Id`, m.`Name`, 
               m.`TypeName`, m.`Autonomous`, m.`Site Name`
        FROM machines m
        INNER JOIN amt_cycleprodinfo cp ON m.`Machine Unique Id` = cp.`Machine Unique Id`
    """
    params = []
    if site_name:
        query += " WHERE m.`Site Name` = %s"
        params.append(site_name)
    
    cursor.execute(query, params)
    machines = {}
    for row in cursor.fetchall():
        machines[row[0]] = {
            "machine_unique_id": row[0],
            "machine_id": row[1],
            "name": row[2],
            "type_name": row[3],
            "autonomous": row[4],
            "site_name": row[5],
        }
    return machines


def fetch_telemetry_data(
    cursor,
    machine_ids: List[int],
    limit: int = 100000,
    sample_interval: int = 5,
) -> List[Tuple]:
    """
    Fetch telemetry data from database.
    
    Returns list of tuples:
    (machine_id, segment_id, cycle_id, interval, pathEasting, pathNorthing,
     pathElevation, expectedSpeed, actualSpeed, pathBank, pathHeading,
     leftWidth, rightWidth, payloadPercent)
    """
    print("    Fetching segment metadata...")
    
    placeholders = ",".join(["%s"] * len(machine_ids))
    meta_query = f"""
        SELECT `Machine Unique Id`, segmentId, cycleId, cycleProdInfoHandle
        FROM amt_cycleprodinfo
        WHERE `Machine Unique Id` IN ({placeholders})
        ORDER BY `Machine Unique Id`, segmentId
        LIMIT {limit // 10}
    """
    
    cursor.execute(meta_query, machine_ids)
    metadata = cursor.fetchall()
    
    if not metadata:
        return []
    
    print(f"    Found {len(metadata)} segments")
    
    handle_to_meta = {}
    handles = []
    for row in metadata:
        handle = row[3]
        handles.append(handle)
        handle_to_meta[handle] = {
            "machine_id": row[0],
            "segment_id": row[1],
            "cycle_id": row[2],
        }
    
    print("    Fetching telemetry points...")
    
    results = []
    batch_size = 100
    
    batch_iterator = range(0, len(handles), batch_size)
    if TQDM_AVAILABLE:
        batch_iterator = tqdm(
            batch_iterator,
            desc="      Batches",
            total=(len(handles) + batch_size - 1) // batch_size,
            unit="batch",
        )
    
    for i in batch_iterator:
        batch_handles = handles[i:i + batch_size]
        placeholders = ",".join(["%s"] * len(batch_handles))
        
        telem_query = f"""
            SELECT 
                cycleProdInfoHandle,
                `interval`,
                pathEasting,
                pathNorthing,
                pathElevation,
                expectedSpeed,
                actualSpeed,
                pathBank,
                pathHeading,
                leftWidth,
                rightWidth,
                payloadPercent
            FROM amt_cycleprodinfo_handle
            WHERE cycleProdInfoHandle IN ({placeholders})
                AND pathEasting IS NOT NULL 
                AND pathNorthing IS NOT NULL
                AND pathElevation IS NOT NULL
                AND MOD(`interval`, %s) = 0
            ORDER BY cycleProdInfoHandle, `interval`
        """
        
        batch_params = batch_handles + [sample_interval]
        cursor.execute(telem_query, batch_params)
        batch_results = cursor.fetchall()
        
        for row in batch_results:
            handle = row[0]
            if handle in handle_to_meta:
                meta = handle_to_meta[handle]
                combined = (
                    meta["machine_id"],
                    meta["segment_id"],
                    meta["cycle_id"],
                    row[1],   # interval
                    row[2],   # pathEasting
                    row[3],   # pathNorthing
                    row[4],   # pathElevation
                    row[5],   # expectedSpeed
                    row[6],   # actualSpeed
                    row[7],   # pathBank
                    row[8],   # pathHeading
                    row[9],   # leftWidth
                    row[10],  # rightWidth
                    row[11],  # payloadPercent
                )
                results.append(combined)
        
        if len(results) >= limit:
            break
    
    # Sort by machine_id, cycle_id, segment_id, interval to ensure correct time order
    results.sort(key=lambda x: (x[0], x[2], x[1], x[3]))
    return results[:limit]


# =============================================================================
# Model Generation Functions
# =============================================================================

def convert_coordinates(path_easting: int, path_northing: int, path_elevation: int) -> Tuple[float, float, float]:
    """Convert database coordinates (mm) to meters."""
    return (
        round(path_easting / 1000.0, 3),
        round(path_northing / 1000.0, 3),
        round(path_elevation / 1000.0, 3),
    )


def calculate_distance(coord1: Tuple, coord2: Tuple) -> float:
    """Calculate 3D distance between two points."""
    dx = coord2[0] - coord1[0]
    dy = coord2[1] - coord1[1]
    dz = coord2[2] - coord1[2] if len(coord1) > 2 and len(coord2) > 2 else 0
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def douglas_peucker(points: List[Tuple], epsilon: float) -> List[Tuple]:
    """Douglas-Peucker path simplification algorithm."""
    if len(points) <= 2:
        return points
    
    start, end = points[0], points[-1]
    max_dist = 0
    max_idx = 0
    
    for i in range(1, len(points) - 1):
        dist = perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    if max_dist > epsilon:
        left = douglas_peucker(points[:max_idx + 1], epsilon)
        right = douglas_peucker(points[max_idx:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]


def perpendicular_distance(point: Tuple, line_start: Tuple, line_end: Tuple) -> float:
    """Calculate perpendicular distance from point to line (2D)."""
    x0, y0 = point[0], point[1]
    x1, y1 = line_start[0], line_start[1]
    x2, y2 = line_end[0], line_end[1]
    
    line_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if line_len == 0:
        return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / line_len


def create_roads_from_trajectories(
    telemetry_data: List[Tuple],
    simplify_epsilon: float = 10.0,
    min_segment_distance: float = 15.0,
    coordinates_in_meters: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Create road network from actual vehicle trajectories.
    
    This ensures nodes and roads follow the actual path of vehicles,
    maintaining correct sequence order for animation playback.
    
    Args:
        telemetry_data: Sorted telemetry data (by machine, cycle, segment, interval)
        simplify_epsilon: Douglas-Peucker simplification threshold (meters)
        min_segment_distance: Minimum distance between nodes (meters)
        coordinates_in_meters: If True, coordinates are already in meters (for imported data).
                              If False, coordinates are in millimeters (for database data).
    
    Returns:
        Tuple of (nodes list, roads list)
    """
    if not telemetry_data:
        return [], []
    
    print(f"    Creating roads from trajectories (epsilon={simplify_epsilon}m)...")
    
    # Group telemetry by machine
    machine_trajectories = {}
    for row in telemetry_data:
        machine_id = row[0]
        # Convert coordinates based on source
        if coordinates_in_meters:
            # Coordinates already in meters (imported data)
            coord = (
                round(float(row[4]), 3),
                round(float(row[5]), 3),
                round(float(row[6]), 3),
            )
        else:
            # Coordinates in millimeters (database data) - convert to meters
            coord = convert_coordinates(row[4], row[5], row[6])
        
        if machine_id not in machine_trajectories:
            machine_trajectories[machine_id] = []
        machine_trajectories[machine_id].append(coord)
    
    print(f"    Found {len(machine_trajectories)} machine trajectories")
    
    # Create unified node map (avoid duplicate nodes at same location)
    all_nodes = []
    node_id = 1
    coord_to_node_id = {}  # (x, y) -> node_id (with tolerance)
    
    def get_or_create_node(coord: Tuple[float, float, float], tolerance: float = 5.0) -> int:
        nonlocal node_id
        x, y, z = coord
        
        # Check if a node already exists nearby
        for (nx, ny), nid in coord_to_node_id.items():
            if math.sqrt((x - nx) ** 2 + (y - ny) ** 2) < tolerance:
                return nid
        
        # Create new node
        new_node = {
            "id": node_id,
            "name": f"Node_{node_id}",
            "coords": [x, y, z],
            "speed_limit": 40.0,
            "rolling_resistance": 2.5,
            "banking": 0,
            "curvature": "",
            "lane_width": 14,
            "traction": 0.6,
        }
        all_nodes.append(new_node)
        coord_to_node_id[(x, y)] = node_id
        node_id += 1
        return new_node["id"]
    
    # Create roads from each machine's trajectory
    roads = []
    road_id = 1
    
    for machine_id, trajectory in machine_trajectories.items():
        if len(trajectory) < 2:
            continue
        
        # Simplify trajectory using Douglas-Peucker
        simplified = douglas_peucker(trajectory, simplify_epsilon)
        
        if len(simplified) < 2:
            continue
        
        # Further filter by minimum segment distance
        filtered_points = [simplified[0]]
        for point in simplified[1:]:
            if calculate_distance(filtered_points[-1], point) >= min_segment_distance:
                filtered_points.append(point)
        
        # Ensure last point is included
        if len(filtered_points) >= 1 and filtered_points[-1] != simplified[-1]:
            if calculate_distance(filtered_points[-1], simplified[-1]) >= min_segment_distance / 2:
                filtered_points.append(simplified[-1])
        
        if len(filtered_points) < 2:
            continue
        
        # Create nodes for this trajectory
        road_node_ids = []
        for point in filtered_points:
            nid = get_or_create_node(point)
            # Avoid consecutive duplicates
            if not road_node_ids or road_node_ids[-1] != nid:
                road_node_ids.append(nid)
        
        if len(road_node_ids) >= 2:
            road = {
                "id": road_id,
                "name": f"Road_{road_id}",
                "nodes": road_node_ids,
                "is_generated": False,
                "ways_num": 2,
                "lanes_num": 1,
                "banking": "",
                "lane_width": "",
                "speed_limit": "",
                "rolling_resistance": "",
                "traction_coefficient": "",
                "offset": 0,
            }
            roads.append(road)
            road_id += 1
    
    print(f"    Created {len(all_nodes)} nodes and {len(roads)} roads from trajectories")
    return all_nodes, roads


def detect_road_network(
    all_points: List[Tuple],
    grid_size: float = 5.0,
    min_density: int = 3,
    simplify_epsilon: float = 5.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect road network from GPS points using grid-based density analysis.
    
    NOTE: This is the legacy method. Consider using create_roads_from_trajectories()
    for better alignment between events and road network.
    
    Returns:
        Tuple of (nodes list, roads list)
    """
    if not all_points:
        return [], []
    
    print(f"    Detecting roads (grid={grid_size}m, min_density={min_density})...")
    
    # Build density grid
    grid = {}
    for point in all_points:
        i = int(point[0] / grid_size)
        j = int(point[1] / grid_size)
        key = (i, j)
        if key not in grid:
            grid[key] = []
        grid[key].append(point)
    
    # Filter by density
    road_cells = {}
    for key, points in grid.items():
        if len(points) >= min_density:
            avg_x = sum(p[0] for p in points) / len(points)
            avg_y = sum(p[1] for p in points) / len(points)
            avg_z = sum(p[2] for p in points) / len(points)
            road_cells[key] = {'center': (avg_x, avg_y, avg_z), 'density': len(points)}
    
    print(f"    Road cells: {len(road_cells):,}")
    
    if not road_cells:
        return [], []
    
    # Find connected components using BFS
    visited = set()
    components = []
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for start_cell in road_cells:
        if start_cell in visited:
            continue
        
        component = []
        queue = deque([start_cell])
        visited.add(start_cell)
        
        while queue:
            cell = queue.popleft()
            component.append(cell)
            
            i, j = cell
            for di, dj in neighbors:
                neighbor = (i + di, j + dj)
                if neighbor in road_cells and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if len(component) >= 2:
            components.append(component)
    
    print(f"    Found {len(components):,} road segments")
    
    # Extract paths from components
    roads = []
    all_nodes = []
    node_id = 1
    road_id = 1
    coord_to_node_id = {}
    
    for component in components:
        centers = [road_cells[cell]['center'] for cell in component]
        if len(centers) < 2:
            continue
        
        # Order points along path
        ordered_path = order_path_points(centers)
        
        # Simplify path
        if len(ordered_path) > 2 and simplify_epsilon > 0:
            ordered_path = douglas_peucker(ordered_path, simplify_epsilon)
        
        if len(ordered_path) < 2:
            continue
        
        # Create nodes
        road_node_ids = []
        for coords in ordered_path:
            rounded = (round(coords[0], 2), round(coords[1], 2), round(coords[2], 2))
            
            if rounded in coord_to_node_id:
                road_node_ids.append(coord_to_node_id[rounded])
            else:
                node = {
                    "id": node_id,
                    "speed_limit": "",
                    "rolling_resistance": "",
                    "banking": "",
                    "curvature": "",
                    "lane_width": "",
                    "traction": "",
                    "coords": list(coords),
                }
                all_nodes.append(node)
                coord_to_node_id[rounded] = node_id
                road_node_ids.append(node_id)
                node_id += 1
        
        # Remove consecutive duplicates
        unique_node_ids = []
        for nid in road_node_ids:
            if not unique_node_ids or unique_node_ids[-1] != nid:
                unique_node_ids.append(nid)
        
        if len(unique_node_ids) >= 2:
            road = {
                "id": road_id,
                "name": f"Road_{road_id}",
                "nodes": unique_node_ids,
                "is_generated": False,
                "ways_num": 2,
                "lanes_num": 1,
                "banking": "",
                "lane_width": "",
                "speed_limit": "",
                "rolling_resistance": "",
                "traction_coefficient": "",
                "offset": 0,
            }
            roads.append(road)
            road_id += 1
    
    print(f"    Created {len(all_nodes):,} nodes and {len(roads):,} roads")
    return all_nodes, roads


def order_path_points(points: List[Tuple]) -> List[Tuple]:
    """Order scattered points into continuous path using nearest neighbor."""
    if len(points) <= 2:
        return points
    
    # Find most distant points as endpoints
    max_dist = 0
    start_idx = 0
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = math.sqrt(
                (points[i][0] - points[j][0])**2 +
                (points[i][1] - points[j][1])**2
            )
            if dist > max_dist:
                max_dist = dist
                start_idx = i
    
    # Build path using nearest neighbor
    remaining = set(range(len(points)))
    remaining.remove(start_idx)
    ordered = [points[start_idx]]
    current_idx = start_idx
    
    while remaining:
        min_dist = float('inf')
        nearest_idx = None
        
        for idx in remaining:
            dist = math.sqrt(
                (points[current_idx][0] - points[idx][0])**2 +
                (points[current_idx][1] - points[idx][1])**2
            )
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        if nearest_idx is not None:
            ordered.append(points[nearest_idx])
            remaining.remove(nearest_idx)
            current_idx = nearest_idx
    
    return ordered


def convert_reader_zones_to_model(
    reader_zones: List,
    nodes: List[Dict],
    roads: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Convert Reader.Zone objects to model zone dicts (same format as detect_zones output).

    Args:
        reader_zones: Zone objects from Reader.py (with zoneType, centroid, points)
        nodes: Road network nodes
        roads: Road network roads

    Returns:
        Tuple of (load_zones, dump_zones)
    """
    from backend.core.constants import ZoneType

    if not reader_zones:
        return [], []

    # Build node lookup for nearest road endpoint search
    node_lookup = {n["id"]: n for n in nodes}
    road_endpoints = []
    for road in roads:
        if len(road["nodes"]) >= 2:
            start_node = node_lookup.get(road["nodes"][0])
            end_node = node_lookup.get(road["nodes"][-1])
            if start_node and end_node:
                road_endpoints.append({
                    "road_id": road["id"],
                    "start": tuple(start_node["coords"]),
                    "end": tuple(end_node["coords"]),
                    "start_node_id": road["nodes"][0],
                    "end_node_id": road["nodes"][-1],
                })

    def find_nearest_road_endpoint(x, y):
        min_dist = float('inf')
        nearest = None
        for ep in road_endpoints:
            for label in ("start", "end"):
                node_id_key = f"{label}_node_id"
                dx = x - ep[label][0]
                dy = y - ep[label][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist
                    nearest = {"road_id": ep["road_id"], "node_id": ep[node_id_key], "distance": dist}
        return nearest

    load_zones = []
    dump_zones = []
    load_id = 1
    dump_id = 1

    for zone in reader_zones:
        # Compute centroid from zone points
        if hasattr(zone, 'centroid') and zone.centroid:
            cx, cy = zone.centroid[0][0], zone.centroid[0][1]
            avg_z = zone.centroid[0][2] if len(zone.centroid[0]) > 2 else 0.0
        elif hasattr(zone, 'points') and zone.points:
            xs = [p[0] for p in zone.points]
            ys = [p[1] for p in zone.points]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)
            zs = [p[2] for p in zone.points if len(p) > 2]
            avg_z = sum(zs) / len(zs) if zs else 0.0
        else:
            continue

        nearest = find_nearest_road_endpoint(cx, cy)
        if nearest is None or nearest["distance"] > 200:
            continue

        zone_settings = {
            "n_spots": 1,
            "roadlength": 100,
            "speed_limit": "",
            "rolling_resistance": "",
            "flip": False,
            "dtheta": 0,
            "n_entrances": 1,
            "queing": False,
            "reverse_speed_limit": "",
            "width": 50,
            "angular_spread": 80,
            "create_uturn_road": False,
            "clearance_radius": 80,
            "access_distance": 40,
            "zonetype": "standard",
            "inroad_ids": [nearest["road_id"]],
            "outroad_ids": [nearest["road_id"]],
            "innode_ids": [nearest["node_id"]],
            "outnode_ids": [nearest["node_id"]],
        }

        zone_type_name = zone.zoneType.value if hasattr(zone.zoneType, 'value') else str(zone.zoneType)

        if zone_type_name == "LOAD":
            zone_dict = {
                "id": load_id,
                "name": f"Load zone {load_id}",
                "keys": "load_zones",
                "is_generated": True,
                "connector_zone_data": [],
                "settings": zone_settings,
                "detected_location": {"x": cx, "y": cy, "z": avg_z},
            }
            load_zones.append(zone_dict)
            load_id += 1
        elif zone_type_name == "DUMP":
            zone_dict = {
                "id": dump_id,
                "name": f"Dump zone {dump_id}",
                "is_generated": True,
                "connector_zone_data": [],
                "settings": zone_settings,
                "detected_location": {"x": cx, "y": cy, "z": avg_z},
            }
            dump_zones.append(zone_dict)
            dump_id += 1

    return load_zones, dump_zones


def detect_zones(
    telemetry_data: List[Tuple],
    nodes: List[Dict],
    roads: List[Dict],
    grid_size: float = 10.0,
    min_stop_count: int = 20,
    coordinates_in_meters: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect load/dump zones from stopped points in telemetry.
    
    Args:
        telemetry_data: Telemetry data tuples
        nodes: List of nodes
        roads: List of roads
        grid_size: Grid size for zone detection
        min_stop_count: Minimum stop count for zone detection
        coordinates_in_meters: If True, coordinates are already in meters (for imported data).
                              If False, coordinates are in millimeters (for database data).
    
    Returns:
        Tuple of (load_zones, dump_zones)
    """
    if not telemetry_data:
        return [], []
    
    # Grid for stopped points
    grid = {}
    
    for row in telemetry_data:
        actual_speed = row[8] if len(row) > 8 else None
        payload = row[13] if len(row) > 13 else None
        
        if actual_speed is not None and actual_speed <= 5:
            if coordinates_in_meters:
                # Coordinates already in meters
                x = float(row[4]) if row[4] else 0
                y = float(row[5]) if row[5] else 0
                z = float(row[6]) if row[6] else 0
            else:
                # Coordinates in millimeters - convert to meters
                x = row[4] / 1000.0 if row[4] else 0
                y = row[5] / 1000.0 if row[5] else 0
                z = row[6] / 1000.0 if row[6] else 0
            
            grid_x = round(x / grid_size) * grid_size
            grid_y = round(y / grid_size) * grid_size
            key = (grid_x, grid_y)
            
            if key not in grid:
                grid[key] = {'points': [], 'payloads': [], 'elevations': []}
            
            grid[key]['points'].append((x, y, z))
            grid[key]['elevations'].append(z)
            
            if payload is not None and 0 <= payload <= 100:
                grid[key]['payloads'].append(payload)
    
    # Build node lookup for road endpoint detection
    node_lookup = {n["id"]: n for n in nodes}
    road_endpoints = []
    for road in roads:
        if len(road["nodes"]) >= 2:
            start_node = node_lookup.get(road["nodes"][0])
            end_node = node_lookup.get(road["nodes"][-1])
            if start_node and end_node:
                road_endpoints.append({
                    "road_id": road["id"],
                    "start": tuple(start_node["coords"]),
                    "end": tuple(end_node["coords"]),
                    "start_node_id": road["nodes"][0],
                    "end_node_id": road["nodes"][-1],
                })
    
    def find_nearest_road_endpoint(x, y):
        min_dist = float('inf')
        nearest = None
        for ep in road_endpoints:
            dist_end = math.sqrt((x - ep["end"][0])**2 + (y - ep["end"][1])**2)
            if dist_end < min_dist:
                min_dist = dist_end
                nearest = {"road_id": ep["road_id"], "node_id": ep["end_node_id"], "distance": dist_end}
            dist_start = math.sqrt((x - ep["start"][0])**2 + (y - ep["start"][1])**2)
            if dist_start < min_dist:
                min_dist = dist_start
                nearest = {"road_id": ep["road_id"], "node_id": ep["start_node_id"], "distance": dist_start}
        return nearest
    
    # Create zones
    load_zones = []
    dump_zones = []
    load_id = 1
    dump_id = 1
    
    for (grid_x, grid_y), data in grid.items():
        if len(data['points']) < min_stop_count:
            continue
        
        avg_z = sum(data['elevations']) / len(data['elevations']) if data['elevations'] else 0
        avg_payload = sum(data['payloads']) / len(data['payloads']) if data['payloads'] else 50
        
        nearest = find_nearest_road_endpoint(grid_x, grid_y)
        if nearest is None or nearest["distance"] > 100:
            continue
        
        zone_settings = {
            "n_spots": 1,
            "roadlength": 100,
            "speed_limit": "",
            "rolling_resistance": "",
            "flip": False,
            "dtheta": 0,
            "n_entrances": 1,
            "queing": False,
            "reverse_speed_limit": "",
            "width": 50,
            "angular_spread": 80,
            "create_uturn_road": False,
            "clearance_radius": 80,
            "access_distance": 40,
            "zonetype": "standard",
            "inroad_ids": [nearest["road_id"]],
            "outroad_ids": [nearest["road_id"]],
            "innode_ids": [nearest["node_id"]],
            "outnode_ids": [nearest["node_id"]],
        }
        
        if avg_payload < 30:  # Load zone
            zone = {
                "id": load_id,
                "name": f"Load zone {load_id}",
                "keys": "load_zones",
                "is_generated": True,
                "connector_zone_data": [],
                "settings": zone_settings,
                "detected_location": {"x": grid_x, "y": grid_y, "z": avg_z},
            }
            load_zones.append(zone)
            load_id += 1
        elif avg_payload > 70:  # Dump zone
            zone = {
                "id": dump_id,
                "name": f"Dump zone {dump_id}",
                "is_generated": True,
                "connector_zone_data": [],
                "settings": zone_settings,
                "detected_location": {"x": grid_x, "y": grid_y, "z": avg_z},
            }
            dump_zones.append(zone)
            dump_id += 1
    
    return load_zones, dump_zones


def create_model(
    nodes: List[Dict],
    roads: List[Dict],
    load_zones: List[Dict] = None,
    dump_zones: List[Dict] = None,
    version: str = "2.0.51",
) -> Dict:
    """Create complete model structure with full settings from example_model.json."""
    load_zones = load_zones or []
    dump_zones = dump_zones or []
    
    model = {
        "version": version,
        "machine_list": {"haulers": [], "loaders": []},
        "map_id": -1,
        "map_translate": {"total_northing": 0, "total_easting": 0, "total_elevation": 0, "total_angle": 0},
        "parameters": [],
        "nodes": nodes,
        "settings": {
            "ambient_temperature": 34,
            "intersection_dispatching": False,
            "reassignment_threshold_min": 5,
            "passing_bay_logic": False,
            "passing_bay_waiting_time": 0.75,
            "battery_state_of_health": 0.9,
            "fuel_when_to_fill_lvl": 0.1,
            "battery_min_pct": 0.1,
            "battery_max_pct": 0.9,
            "battery_trolley_pct": 0.9,
            "max_SOC_logic_for_DET": True,
            "det_rms_power": "off",
            "power_loss_model": False,
            "rail_resist_multiplier": 1,
            "battery_charge_pct": 0.9,
            "max_SOC_logic_for_SET": True,
            "driving_side": 0,
            "gap_between_lanes": 1.4,
            "road_merging_intersection": 0,
            "automatic_intersection_creation": True,
            "road_ways_num": 2,
            "road_lanes_num": 1,
            "road_speed_limit": 65,
            "road_rolling_resistance": 2,
            "lane_width": 14.525,
            "road_traction_coefficient": 0.6,
            "banking": 0,
            "distance_between_lanes": 15.925,
            "reduce_speed_logic": True,
            "min_foll_dist": 50,
            "bump_prevention": True,
            "intersection_system": "simple",
            "des_inputs_settings": {
                "curvature_based": False,
                "curvature_speed_application": True,
                "cfh_version": 13,
                "super_elevation": "flat_curve",
                "max_distance_curve_radius": 50,
                "driver_speed_behavior": False,
                "dsb_distance": 500,
                "dsb_threshold": 20,
                "grade_limits": False,
                "interpolate_speed_limit": False,
                "loaded_flat_speed_limits": 40,
                "empty_flat_speed_limits": 60,
                "loaded_steep_downhill_grades": -15,
                "empty_steep_downhill_grades": -15,
                "loaded_steep_downhill_speed_limits": 10,
                "empty_steep_downhill_speed_limits": 15,
                "loaded_downhill_grades": -5,
                "empty_downhill_grades": -5,
                "loaded_downhill_speed_limits": 20,
                "empty_downhill_speed_limits": 30,
                "loaded_uphill_grades": 5,
                "empty_uphill_grades": 5,
                "loaded_uphill_speed_limits": 30,
                "empty_uphill_speed_limits": 45,
                "loaded_steep_uphill_grades": 15,
                "empty_steep_uphill_grades": 15,
                "loaded_steep_uphill_speed_limits": 20,
                "empty_steep_uphill_speed_limits": 30,
                "lane_width": 10,
                "driving_side": 0,
                "gap_between_lanes": 1.4,
                "road_network_logic": False,
                "require_energy_zone": True,
                "max_SOC_logic_for_SET": True,
                "max_SOC_logic_for_DET": True,
                "road_merging_intersection": 0,
                "reassignment_threshold_min": 5,
                "road_traction_coefficient": 0.6,
                "bump_prevention": True,
                "create_intersection_object": False,
                "automatic_intersection_creation": True,
                "intersection_dispatching": False,
                "det_rms_power": "off",
                "banking": 0,
                "corner_speed_limit": False,
            },
            "target_payload": 100,
            "variation_model": -1,
            "loader_time_variation": 0,
            "loader_payload_variation": 0,
            "truck_time_variation": 0,
            "truck_payload_variation": 0,
            "payload_precision": 0.05,
            "tires": {
                "calculate_TKPH": False,
                "TKPH_RollingWindow": 60,
                "TKPH_ambient_temperature": 34,
                "TKPH_speed_adjustment": False,
                "front_tire_TKPH_limit": 1394,
                "rear_tire_TKPH_limit": 1394,
                "front_tire_TKPH_deactivate_limit": 1000,
                "rear_tire_TKPH_deactivate_limit": 1000,
                "TKPH_limiting_speed": 25,
            },
            "fueling_charging_dispatching": {
                "prerun_soc_buffer": 0.05,
                "prerun_soc_buffer_trolley": 0.05,
                "soe_penalty": 1,
                "charging_strategy": "rule_based",
                "fueling_charging_dispatching_policy": "when_empty",
            },
            "create_intersection_object": False,
            "new_prerun": False,
            "new_fpc": True,
            "sim_time": 240,
            "random_seed": 1234,
            "material_density": 1700,
            "operational_delays": "off",
            "require_energy_zone": True,
            "road_network_logic": True,
            "intersections": {
                "intersect_logic": True,
                "intersect_length": 0,
                "intersect_yield": True,
                "intersect_yield_distance": 100,
                "intersect_yield_speed": 10,
            },
            "curvature_speed_application": True,
            "power_averaging_and_derate": False,
            "secondary_check_for_CZ_dispatch": True,
        },
        "default_powernode_priorities": {
            "loaders": 0,
            "crushers": 0,
            "trolleys": 2,
            "chargers": 4,
        },
        "economic_settings": {},
        "zone_defaults": {
            "trolley": {
                "type": 0,
                "stop_propel": True,
                "queue_at_entry": False,
                "speed_reduction": False,
                "adaptive_speed_reduction_soc_target": 0.7,
                "connect_speed": 20,
                "maximum_speed": 45,
                "line_rail_efficiency": 0.95,
                "rejection_rate": 0,
                "substation_output_power_limit": 12000,
                "power_module_rms_limit": 9000,
                "substation_efficiency": 0.965,
                "rms_moving_time_window": 30,
                "power_factor": 0.9,
                "power_factor_lagging": "Lagging (Inductive)",
                "max_propel_preferred": False,
                "extra_buffer_demand_strategy": 1,
                "power_prioritization": 0,
                "trolley_length": 1,
                "battery_charge_pct": 0.9,
                "max_SOC_logic_for_DET": True,
            },
            "fuel_charge": {
                "connect_time": 3,
                "disconnect_time": 2,
                "ramup_time": 0.8,
                "output_power": 4000,
                "efficiency": 0.945,
                "cable_efficiency": 0.98,
                "power_factor": 0.9,
                "power_factor_lagging": "Lagging (Inductive)",
                "fuel_rate": 500,
                "fuel_connect_time": 3,
                "fuel_disconnect_time": 2,
                "auto_generate": {
                    "speed_limit": 20,
                    "rolling_resistance": 2,
                },
                "fuel_auto_generate": {
                    "speed_limit": 20,
                    "rolling_resistance": 2,
                },
            },
            "load": {
                "auto_generate": {
                    "speed_limit": 20,
                    "rolling_resistance": 2,
                    "reverse_speed_limit": 5,
                },
            },
            "dump": {
                "auto_generate": {
                    "speed_limit": 20,
                    "rolling_resistance": 2,
                    "reverse_speed_limit": 5,
                    "power_factor": 1,
                },
            },
            "service": {
                "auto_generate": {
                    "speed_limit": 20,
                    "rolling_resistance": 2,
                },
            },
        },
        "roads": roads,
        "trolleys": [],
        "chargers": [],
        "service_stations": [],
        "dump_zones": dump_zones,
        "load_zones": load_zones,
        "routes": [],
        "haulers": [],
        "loaders": [],
        "simulates": [],
        "esses": [],
        "batteries": [],
        "crushers": [],
        "cameraPosition": {"x": 0, "y": 1000, "z": 0},
        "controlTarget": {"x": 0, "y": 0, "z": 0},
    }
    
    # Update camera position
    if nodes:
        eastings = [n["coords"][0] for n in nodes]
        northings = [n["coords"][1] for n in nodes]
        elevations = [n["coords"][2] for n in nodes]
        
        center_x = (min(eastings) + max(eastings)) / 2
        center_y = (min(northings) + max(northings)) / 2
        center_z = (min(elevations) + max(elevations)) / 2
        span = max(max(eastings) - min(eastings), max(northings) - min(northings))
        
        model["cameraPosition"] = {"x": center_x, "y": center_z + span, "z": center_y}
        model["controlTarget"] = {"x": center_x, "y": center_z, "z": center_y}
    
    return model


# =============================================================================
# DES Inputs Generation Functions
# =============================================================================

def _create_default_hauler_spec(spec_id: int, model_name: str, is_electric: bool) -> Dict:
    """Create a default hauler spec (fallback when no template file)."""
    hauler_machine = {
        "ID": spec_id,
        "Model": model_name,
        "Engine": "CAT 3516C HD EUI",
        "FlywheelPower": 1976.0,
        "TransType": "7SPD PS",
        "Capacity": 129.0,
        "Payload": 227000.0,
        "OwnOp": 0.0,
        "Availability": 0.0,
        "MachineType": 6,
        "IsLoader": 0,
        "IsHauler": 1,
        "IsExcavator": 0,
        "IsDozer": 0,
        "IsSupport": 0,
        "MachineCode": "",
        "LoaderType": 0,
        "StdBucketCapacity": 0.0,
        "StdBucketLoad": 0.0,
        "StdBucketMins": 0.0,
        "HaulExchTime": 0.0,
        "WheelBase": 0.0,
        "StdBucketSize": 0.0,
        "MaxBurnRate": 227.0,
        "IdleBurnPct": 15.0,
        "ManeuverBurnPct": 90.0,
        "RetardingBurnPct": 50.0,
        "LoadedFrontPct": 33.0,
        "LoadedRearPct": 67.0,
        "LoadedTrailPct": 0.0,
        "EmptyFrontPct": 45.0,
        "EmptyRearPct": 55.0,
        "EmptyTrailPct": 0.0,
        "TiresFront": 2,
        "TiresRear": 4,
        "TiresTrail": 0,
        "NumAxles": 2,
        "GrossPower": 1976.0,
        "Description": f"{model_name} Mining Truck",
        "Comments": "",
        "RimpullTireType": None,
        "RimpullTireSize": "46/90R57",
        "RimpullDLR": 1778.0,
        "RetardingBasis": 1,
        "RetardingPackage": "Standard",
        "TotalReduction": "35",
        "RetardingMax": None,
        "ArcType": 0,
        "ArcRpmMin": 0,
        "ArcRpmMax": 0,
        "ArcRpmDefault": 0,
        "Altitude7500": 1.0,
        "Altitude10000": 1.0,
        "Altitude12500": 1.0,
        "Altitude15000": 1.0,
        "IsCat": 1,
        "HasRimpullTail": 0,
        "DumpManeuver": 1.2,
        "ShiftLogic": 0,
        "FuelLogic": 0,
        "InletRestriction": 0.0,
        "IntakeTempRise": 0.0,
        "TrolleyVoltage": 0.0,
        "TrolleyConnectPct": 0.0,
        "TrolleyOnlinePct": 0.0,
        "TrolleyFuelPct": None,
        "TrolleyWeight": 0.0,
        "SecuredAccess": 0,
        "IsCompetitive": 0,
        "battery_type": "LFP" if is_electric else None,
        "battery_size": 2230.0 if is_electric else None,
        "GHGFreeConfig": "BEM" if is_electric else None,
        "HaulerMax": 392000.0,
        "EmptyWeight": 165000.0,
        "PayloadIndex": 227000.0,
        "Machine Fuel Tank Capacity (L)": 3785,
        "machine_type": {
            "power_source": "battery" if is_electric else "diesel",
            "driveline": "electric" if is_electric else "mechanical",
            "is_hauler": True,
            "is_loader": False,
            "det_capable": is_electric,
            "machine type": "Mining Truck",
        },
    }
    
    return {
        "machine": {
            "machine": hauler_machine,
            "tires": [{
                "ID": spec_id,
                "Order": 0,
                "IsStandard": 1,
                "Size": "46/90R57",
                "Type": "E4",
                "SpeedCorrection": 1.0,
                "HaulerEmpty": 165000.0,
                "HaulerMax": 392000.0,
                "PayloadIndex": 227000.0,
            }],
            "retarding": [],
            "rimpull": [],
            "shift_points": [],
            "fuel_consumption": [],
            "gear_ratios": [],
        },
        "has_fuel": not is_electric,
        "is_hybrid": False,
        "machine_type": "electric" if is_electric else "diesel",
    }


def _create_default_loader_spec() -> Dict:
    """Create a default loader spec (fallback when no template file)."""
    return {
        "loader": {
            "ID": 1,
            "Model": "994K HL",
            "Engine": "CAT 3516E",
            "FlywheelPower": 1296.0,
            "TransType": "3SPD PS",
            "Capacity": 0.0,
            "Payload": 0.0,
            "OwnOp": 0.0,
            "Availability": 0.0,
            "MachineType": 11,
            "IsLoader": 1,
            "IsHauler": 0,
            "IsExcavator": 0,
            "IsDozer": 0,
            "IsSupport": 0,
            "MachineCode": "",
            "LoaderType": 0,
            "StdBucketCapacity": 19.0,
            "HaulExchTime": 0.7,
            "GrossPower": 1377.0,
            "Description": "Cat 994K Wheel Loader",
            "DumpManeuver": 1.2,
            "IsCat": 1,
            "machine_type": {
                "power_source": "diesel",
                "driveline": "mechanical",
                "is_hauler": False,
                "is_loader": True,
                "det_capable": False,
                "machine type": "Wheel Loader",
            },
        },
        "buckets": [{
            "ID": 1,
            "Order": 0,
            "IsStandard": 1,
            "Type": "Rock Bucket",
            "Capacity": 19.0,
            "RatedLoad": 40823.0,
            "CycleTimeMin": 0.58,
            "FirstBucketMin": 0.1,
        }],
        "has_fuel": True,
        "is_hybrid": False,
        "machine_type": "diesel",
        "tires": [],
        "retarding": [],
        "rimpull": [],
    }


def create_des_inputs(
    model: Dict,
    machines: Dict[int, Dict],
    site_name: str,
    sim_time: int = 480,
    machines_with_events: Optional[Set[int]] = None,
    machine_templates: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Create DES Inputs structure from model and machine data.
    
    Args:
        model: Model dictionary with nodes, roads, zones
        machines: Machine info dictionary
        site_name: Site name
        sim_time: Simulation time in minutes
        machines_with_events: Set of machine IDs that have events data
        machine_templates: Machine templates loaded from JSON file
    
    Returns:
        DES Inputs dictionary
    """
    nodes = model.get("nodes", [])
    roads = model.get("roads", [])
    load_zones = model.get("load_zones", [])
    dump_zones = model.get("dump_zones", [])
    
    # Convert zones to DES format
    des_load_zones = []
    for zone in load_zones:
        des_zone = {
            "id": zone["id"],
            "name": zone["name"],
            "material": ["Ore_1700"],
            "terminal_zone": False,
            "spots": [{"id": 1, "ess_id": None, "roads": [zone.get("settings", {}).get("inroad_ids", [])]}],
        }
        des_load_zones.append(des_zone)
    
    des_dump_zones = []
    for zone in dump_zones:
        des_zone = {
            "id": zone["id"],
            "name": zone["name"],
            "terminal_zone": False,
            "spots": [{"id": 1, "roads": [zone.get("settings", {}).get("inroad_ids", [])]}],
        }
        des_dump_zones.append(des_zone)
    
    # Create hauler specs and haulers list
    # Group specs by model type (machines with same type_name share one spec)
    hauler_specs = {}
    loader_specs = {}
    haulers = []
    hauler_id = 1
    uid_base = 200
    
    # Track model types to spec IDs mapping
    model_to_spec_id = {}  # type_name -> spec_id
    spec_id = 1
    
    # Load templates from file or use defaults
    templates = machine_templates or {}
    hauler_template = templates.get("hauler_template", {})
    electric_hauler_overrides = templates.get("electric_hauler_overrides", {})
    hauler_entry_template = templates.get("hauler_entry_template", {})
    electric_hauler_entry_overrides = templates.get("electric_hauler_entry_overrides", {})
    loader_template = templates.get("loader_template", {})
    
    # Only process machines that have events data
    for machine_id, machine_info in machines.items():
        # Skip machines without events
        if machines_with_events is not None and machine_id not in machines_with_events:
            continue
        
        machine_name = machine_info.get("name", f"Hauler_{machine_id}")
        machine_type_name = machine_info.get("type_name", "Cat 793F CMD")
        
        # Determine truck type
        is_electric = "BEM" in machine_type_name or "Electric" in machine_type_name or "Battery" in machine_type_name
        
        # Create spec for this model type if not exists
        if machine_type_name not in model_to_spec_id:
            # Create new hauler spec from template
            if hauler_template:
                # Use template from file
                hauler_spec = deep_copy_dict(hauler_template)
                hauler_machine = hauler_spec.get("machine", {}).get("machine", {})
                hauler_machine["ID"] = spec_id
                hauler_machine["Model"] = machine_type_name
                
                # Update tires ID
                for tire in hauler_spec.get("machine", {}).get("tires", []):
                    tire["ID"] = spec_id
                
                # Apply electric overrides if needed
                if is_electric and electric_hauler_overrides:
                    hauler_spec = merge_dict(hauler_spec, electric_hauler_overrides)
                    # Re-apply ID after merge
                    if "machine" in hauler_spec:
                        hauler_spec["machine"]["machine"]["ID"] = spec_id
                        hauler_spec["machine"]["machine"]["Model"] = machine_type_name
                
                hauler_specs[str(spec_id)] = hauler_spec
            else:
                # Fallback to hardcoded defaults
                hauler_specs[str(spec_id)] = _create_default_hauler_spec(
                    spec_id, machine_type_name, is_electric
                )
            
            model_to_spec_id[machine_type_name] = spec_id
            spec_id += 1
        
        # Get the spec ID for this hauler's model
        hauler_model_id = model_to_spec_id[machine_type_name]
        
        # Create hauler entry from template
        if hauler_entry_template:
            hauler = deep_copy_dict(hauler_entry_template)
            hauler["id"] = hauler_id
            hauler["name"] = machine_name
            hauler["model_id"] = hauler_model_id
            hauler["machine_name"] = machine_type_name
            hauler["uid"] = uid_base + hauler_id
            hauler["initial_conditions"] = {
                "route_id": 1 if roads else None,
                "road_id": roads[0]["id"] if roads else 0,
                "node_id": nodes[0]["id"] if nodes else 0,
            }
            
            # Apply electric overrides if needed
            if is_electric and electric_hauler_entry_overrides:
                hauler = merge_dict(hauler, electric_hauler_entry_overrides)
                # Re-apply IDs after merge
                hauler["id"] = hauler_id
                hauler["name"] = machine_name
                hauler["model_id"] = hauler_model_id
                hauler["machine_name"] = machine_type_name
                hauler["uid"] = uid_base + hauler_id
        else:
            # Fallback to hardcoded defaults
            hauler = {
                "id": hauler_id,
                "name": machine_name,
                "group": "Fleet1",
                "type": "electric" if is_electric else "diesel",
                "model_id": hauler_model_id,
                "machine_name": machine_type_name,
                "hauler_group_id": 1,
                "initial_position": 1,
                "initial_conditions": {
                    "route_id": 1 if roads else None,
                    "road_id": roads[0]["id"] if roads else 0,
                    "node_id": nodes[0]["id"] if nodes else 0,
                },
                "initial_fuel_level_pct": 0.9 if not is_electric else None,
                "initial_charge_level_pct": 0.9 if is_electric else None,
                "battery_state_of_health": 0.9 if is_electric else None,
                "EndOfLifeSOH": 84.7 if is_electric else None,
                "AvgAnnualAmbientTemp": 25,
                "CoolingActivationTemperature": 25 if is_electric else None,
                "RefridgerationActivationTemperature": 25 if is_electric else None,
                "uid": uid_base + hauler_id,
            }
        
        haulers.append(hauler)
        hauler_id += 1
    
    # Add loader spec from template
    if loader_template:
        loader_specs["1"] = deep_copy_dict(loader_template)
    else:
        # Fallback to hardcoded defaults
        loader_specs["1"] = _create_default_loader_spec()
    
    # Create DES nodes
    des_nodes = []
    for node in nodes:
        des_node = {
            "id": node["id"],
            "name": f"Node_{node['id']}",
            "coords": node["coords"],
            "speed_limit": node.get("speed_limit") or 40.0,
            "rolling_resistance": node.get("rolling_resistance") or 2.5,
            "banking": node.get("banking") or 0,
            "curvature": node.get("curvature") or "",
            "lane_width": node.get("lane_width") or 14,
            "traction": node.get("traction") or 0.6,
        }
        des_nodes.append(des_node)
    
    # Create DES roads
    des_roads = []
    for road in roads:
        des_road = {
            "id": road["id"],
            "name": road["name"],
            "nodes": road["nodes"],
            "ways_num": road.get("ways_num", 2),
            "lanes_num": road.get("lanes_num", 1),
            "speed_limit": road.get("speed_limit") or 40.0,
            "rolling_resistance": road.get("rolling_resistance") or 2.5,
            "is_generated": road.get("is_generated", False),
            "lane_width": road.get("lane_width") or 14,
            "traction_coefficient": road.get("traction_coefficient") or 0.6,
        }
        des_roads.append(des_road)
    
    # Generate zone-specific roads (entry, spotting, exit) for each zone
    next_node_id = max([n["id"] for n in des_nodes], default=0) + 1
    next_road_id = max([r["id"] for r in des_roads], default=0) + 1
    zone_road_offset = 20  # Offset for zone node positions (meters)
    uid_counter = 100
    
    # Process load zones - create entry, spotting, exit roads
    for lz in des_load_zones:
        zone_loc = None
        # Get zone location from original load_zones
        for orig_lz in load_zones:
            if orig_lz["id"] == lz["id"]:
                zone_loc = orig_lz.get("detected_location")
                break
        
        if not zone_loc:
            # Use first node as fallback
            if des_nodes:
                zone_loc = {"x": des_nodes[0]["coords"][0], "y": des_nodes[0]["coords"][1], "z": des_nodes[0]["coords"][2]}
            else:
                continue
        
        x, y, z = zone_loc["x"], zone_loc["y"], zone_loc["z"]
        
        # Create 4 nodes for this zone: entry, spot_start, spot_end, exit
        # Spotting road needs 2 nodes (roads must have at least 2 nodes)
        entry_node_id = next_node_id
        spot_start_node_id = next_node_id + 1
        spot_end_node_id = next_node_id + 2
        exit_node_id = next_node_id + 3
        next_node_id += 4
        
        spot_offset = 5.0  # Small offset for spotting segment
        
        des_nodes.append({
            "id": entry_node_id,
            "name": f"LZ{lz['id']}_Entry",
            "coords": [x - zone_road_offset, y, z],
            "speed_limit": 20.0,
            "rolling_resistance": 2.5,
            "banking": 0,
            "curvature": "",
            "lane_width": 14,
            "traction": 0.6,
        })
        des_nodes.append({
            "id": spot_start_node_id,
            "name": f"LZ{lz['id']}_SpotStart",
            "coords": [x - spot_offset, y, z],
            "speed_limit": 5.0,
            "rolling_resistance": 2.5,
            "banking": 0,
            "curvature": "",
            "lane_width": 14,
            "traction": 0.6,
        })
        des_nodes.append({
            "id": spot_end_node_id,
            "name": f"LZ{lz['id']}_SpotEnd",
            "coords": [x + spot_offset, y, z],
            "speed_limit": 5.0,
            "rolling_resistance": 2.5,
            "banking": 0,
            "curvature": "",
            "lane_width": 14,
            "traction": 0.6,
        })
        des_nodes.append({
            "id": exit_node_id,
            "name": f"LZ{lz['id']}_Exit",
            "coords": [x + zone_road_offset, y, z],
            "speed_limit": 20.0,
            "rolling_resistance": 2.5,
            "banking": 0,
            "curvature": "",
            "lane_width": 14,
            "traction": 0.6,
        })
        
        # Create 3 roads: entry, spotting (reverse), exit
        entry_road_id = next_road_id
        spotting_road_id = next_road_id + 1
        exit_road_id = next_road_id + 2
        next_road_id += 3
        
        des_roads.append({
            "id": entry_road_id,
            "name": f"LZ{lz['id']}_Entry_Road",
            "nodes": [entry_node_id, spot_start_node_id],
            "ways_num": 1,
            "lanes_num": 1,
            "speed_limit": 20.0,
            "rolling_resistance": 2.5,
            "is_generated": True,
            "lane_width": 14,
            "traction_coefficient": 0.6,
        })
        des_roads.append({
            "id": spotting_road_id,
            "name": f"LZ{lz['id']}_Spotting_Road",
            "nodes": [spot_start_node_id, spot_end_node_id],
            "ways_num": 1,
            "lanes_num": 1,
            "speed_limit": 5.0,
            "rolling_resistance": 2.5,
            "is_generated": True,
            "lane_width": 14,
            "traction_coefficient": 0.6,
        })
        des_roads.append({
            "id": exit_road_id,
            "name": f"LZ{lz['id']}_Exit_Road",
            "nodes": [spot_end_node_id, exit_node_id],
            "ways_num": 1,
            "lanes_num": 1,
            "speed_limit": 20.0,
            "rolling_resistance": 2.5,
            "is_generated": True,
            "lane_width": 14,
            "traction_coefficient": 0.6,
        })
        
        # Update zone spots with the 3 road IDs
        lz["spots"] = [{
            "id": 1,
            "ess_id": None,
            "roads": [[entry_road_id, spotting_road_id, exit_road_id]],
            "uid": uid_counter,
        }]
        lz["uid"] = uid_counter + 1
        uid_counter += 2
    
    # Process dump zones - create entry, spotting, exit roads
    for dz in des_dump_zones:
        zone_loc = None
        # Get zone location from original dump_zones
        for orig_dz in dump_zones:
            if orig_dz["id"] == dz["id"]:
                zone_loc = orig_dz.get("detected_location")
                break
        
        if not zone_loc:
            # Use last node as fallback
            if des_nodes:
                zone_loc = {"x": des_nodes[-1]["coords"][0], "y": des_nodes[-1]["coords"][1], "z": des_nodes[-1]["coords"][2]}
            else:
                continue
        
        x, y, z = zone_loc["x"], zone_loc["y"], zone_loc["z"]
        
        # Create 4 nodes for this zone: entry, spot_start, spot_end, exit
        # Spotting road needs 2 nodes (roads must have at least 2 nodes)
        entry_node_id = next_node_id
        spot_start_node_id = next_node_id + 1
        spot_end_node_id = next_node_id + 2
        exit_node_id = next_node_id + 3
        next_node_id += 4
        
        spot_offset = 5.0  # Small offset for spotting segment
        
        des_nodes.append({
            "id": entry_node_id,
            "name": f"DZ{dz['id']}_Entry",
            "coords": [x - zone_road_offset, y, z],
            "speed_limit": 20.0,
            "rolling_resistance": 2.5,
            "banking": 0,
            "curvature": "",
            "lane_width": 14,
            "traction": 0.6,
        })
        des_nodes.append({
            "id": spot_start_node_id,
            "name": f"DZ{dz['id']}_SpotStart",
            "coords": [x - spot_offset, y, z],
            "speed_limit": 5.0,
            "rolling_resistance": 2.5,
            "banking": 0,
            "curvature": "",
            "lane_width": 14,
            "traction": 0.6,
        })
        des_nodes.append({
            "id": spot_end_node_id,
            "name": f"DZ{dz['id']}_SpotEnd",
            "coords": [x + spot_offset, y, z],
            "speed_limit": 5.0,
            "rolling_resistance": 2.5,
            "banking": 0,
            "curvature": "",
            "lane_width": 14,
            "traction": 0.6,
        })
        des_nodes.append({
            "id": exit_node_id,
            "name": f"DZ{dz['id']}_Exit",
            "coords": [x + zone_road_offset, y, z],
            "speed_limit": 20.0,
            "rolling_resistance": 2.5,
            "banking": 0,
            "curvature": "",
            "lane_width": 14,
            "traction": 0.6,
        })
        
        # Create 3 roads: entry, spotting (reverse), exit
        entry_road_id = next_road_id
        spotting_road_id = next_road_id + 1
        exit_road_id = next_road_id + 2
        next_road_id += 3
        
        des_roads.append({
            "id": entry_road_id,
            "name": f"DZ{dz['id']}_Entry_Road",
            "nodes": [entry_node_id, spot_start_node_id],
            "ways_num": 1,
            "lanes_num": 1,
            "speed_limit": 20.0,
            "rolling_resistance": 2.5,
            "is_generated": True,
            "lane_width": 14,
            "traction_coefficient": 0.6,
        })
        des_roads.append({
            "id": spotting_road_id,
            "name": f"DZ{dz['id']}_Spotting_Road",
            "nodes": [spot_start_node_id, spot_end_node_id],
            "ways_num": 1,
            "lanes_num": 1,
            "speed_limit": 5.0,
            "rolling_resistance": 2.5,
            "is_generated": True,
            "lane_width": 14,
            "traction_coefficient": 0.6,
        })
        des_roads.append({
            "id": exit_road_id,
            "name": f"DZ{dz['id']}_Exit_Road",
            "nodes": [spot_end_node_id, exit_node_id],
            "ways_num": 1,
            "lanes_num": 1,
            "speed_limit": 20.0,
            "rolling_resistance": 2.5,
            "is_generated": True,
            "lane_width": 14,
            "traction_coefficient": 0.6,
        })
        
        # Update zone spots with the 3 road IDs
        dz["spots"] = [{
            "id": 1,
            "roads": [[entry_road_id, spotting_road_id, exit_road_id]],
            "uid": uid_counter,
        }]
        dz["uid"] = uid_counter + 1
        uid_counter += 2

    # Create default loaders: one loader per spot in each load zone
    loaders: List[Dict] = []
    loader_id = 1

    # Determine default loader model and name from loader specs (if available)
    default_loader_model_id = 1 if loader_specs else None
    default_loader_machine_name = "Default_Loader"
    if loader_specs:
        default_spec = loader_specs.get(str(default_loader_model_id)) or next(iter(loader_specs.values()))
        loader_info = default_spec.get("loader", {})
        default_loader_machine_name = (
            loader_info.get("Model")
            or loader_info.get("LoaderName")
            or default_loader_machine_name
        )

    for lz in des_load_zones:
        zone_id = lz.get("id")
        for spot in lz.get("spots", []):
            spot_id = spot.get("id", 1)
            loader_entry = {
                "id": loader_id,
                "name": f"Loader_{zone_id}_{spot_id}",
                "model_id": default_loader_model_id or 1,
                "used_for": "Truck Loading",
                "machine_name": default_loader_machine_name,
                "initial_conditions": {
                    "load_zone_id": zone_id,
                    "spot_id": spot_id,
                },
                "fill_factor_pct": 1.0,
                "powernode_priority": 0,
            }
            loaders.append(loader_entry)
            loader_id += 1

    # Create routes from load zones to dump zones
    # Routes only contain main roads - zone roads are part of zones, not routes
    des_routes = []
    route_id = 1
    route_uid_counter = 1000  # Start UID counter for routes
    
    # Get main road IDs only (original roads, not zone-specific roads)
    main_road_ids = [road["id"] for road in roads]
    
    for lz in des_load_zones:
        lz_id = lz["id"]
        lz_name = lz["name"]
        
        for dz in des_dump_zones:
            dz_id = dz["id"]
            dz_name = dz["name"]
            
            # Routes use only main roads to connect zones
            # Haul: main roads from load zone to dump zone
            haul_roads = list(main_road_ids)
            
            # Return: main roads reversed (from dump zone back to load zone)
            return_roads = list(reversed(main_road_ids))
            
            # Create route
            route = {
                "id": route_id,
                "name": f"{lz_name} to {dz_name}",
                "haul": haul_roads,
                "return": return_roads,
                "start_zone": {
                    "id": lz_id,
                    "type": "lz",
                    "uid": route_uid_counter,
                },
                "end_zone": {
                    "id": dz_id,
                    "type": "dz",
                    "uid": route_uid_counter + 1,
                },
                "used_by_current_MMP": True,
                "production": True,
                "uid": route_uid_counter + 2,
            }
            des_routes.append(route)
            route_id += 1
            route_uid_counter += 3
    
    # If no routes created (no zones), create a default route using main roads
    if not des_routes and main_road_ids:
        des_routes.append({
            "id": 1,
            "name": "Default Route",
            "haul": main_road_ids,
            "return": list(reversed(main_road_ids)),
            "start_zone": {"id": 1, "type": "lz", "uid": 1000},
            "end_zone": {"id": 1, "type": "dz", "uid": 1001},
            "used_by_current_MMP": True,
            "production": True,
            "uid": 1002,
        })
    
    # Build DES Inputs structure
    des_inputs = {
        "version": "CAT_2.0.2",
        "machine_specs": {
            "hauler_specs": hauler_specs,
            "loader_specs": loader_specs,
        },
        "material_properties": {
            "Ore_1700": {"id": 1, "material": "Ore", "density": 1700},
            "Waste_1800": {"id": 2, "material": "Waste", "density": 1800},
        },
        "map_id": 1,
        "map_translate": {"total_northing": 0, "total_easting": 0, "total_elevation": 0, "total_angle": 0},
        "default_powernode_priorities": {"loaders": 0, "trolleys": 2, "chargers": 1, "crushers": 0},
        "settings": {
            "sim_time": sim_time,
            "random_seed": 1234,
            "intersection_system": "simple",
            "bump_prevention": "physics_based",
            "road_network_logic": True,
            "lane_width": 14.525,
            "distance_between_lanes": 11.4,
            "driving_side": 0,
            "reduce_speed_logic": True,
            "min_foll_dist": 50,
            "verbose": False,
            "braking": True,
            "objective": "simulation",
            "log_level": "record",
            "spd_lim": 65,
            "initial_fuel_level_pct": 0.9,
            "initial_charge_level_pct": 0.9,
            "calculate_BLE": True,
        },
        "zone_defaults": {
            "trolley": {
                "type": 0,
                "stop_propel": True,
                "queue_at_entry": False,
                "speed_reduction": False,
                "adaptive_speed_reduction_soc_target": 0.7,
                "connect_speed": 20,
                "maximum_speed": 45,
                "line_rail_efficiency": 0.95,
                "rejection_rate": 0,
                "substation_output_power_limit": 12000,
                "power_module_rms_limit": 9000,
                "substation_efficiency": 0.965,
                "rms_moving_time_window": 30,
                "power_factor": 0.9,
                "power_factor_lagging": "Lagging (Inductive)",
                "max_propel_preferred": False,
                "extra_buffer_demand_strategy": 1,
                "power_prioritization": 0,
                "trolley_length": 1,
                "battery_charge_pct": 0.9,
                "max_SOC_logic_for_DET": True,
            },
            "fuel_charge": {
                "connect_time": 3,
                "disconnect_time": 2,
                "ramup_time": 0.8,
                "output_power": 4000,
                "efficiency": 0.945,
                "cable_efficiency": 0.98,
                "power_factor": 0.9,
                "power_factor_lagging": "Lagging (Inductive)",
                "fuel_rate": 500,
                "fuel_connect_time": 3,
                "fuel_disconnect_time": 2,
                "auto_generate": {
                    "speed_limit": 20,
                    "rolling_resistance": 2,
                },
                "fuel_auto_generate": {
                    "speed_limit": 20,
                    "rolling_resistance": 2,
                },
            },
            "load": {
                "auto_generate": {
                    "speed_limit": 20,
                    "rolling_resistance": 2,
                    "reverse_speed_limit": 5,
                },
            },
            "dump": {
                "auto_generate": {
                    "speed_limit": 20,
                    "rolling_resistance": 2,
                    "reverse_speed_limit": 5,
                    "power_factor": 1,
                },
            },
            "service": {
                "auto_generate": {
                    "speed_limit": 20,
                    "rolling_resistance": 2,
                },
            },
        },
        "economic_settings": {},
        "nodes": des_nodes,
        "roads": des_roads,
        "trolleys": [],
        "load_zones": des_load_zones,
        "dump_zones": des_dump_zones,
        "crushers": [],
        "fuel_zones": [],
        "charge_zones": [],
        "service_zones": [],
        "routes": des_routes,
        "loaders": loaders,
        "haulers": haulers,
        "batteries": [],
        "esses": {},
        "electrical_distributions": [],
        "haulers_assignment": [],
        "operations": {
            "material_schedules": {
                "selected_material": 1,
                "all_material_schedule": [
                    {
                        "id": 1,
                        "name": "Default_MMP",
                        "hauler_assignment": {"scheduling_method": "production_target_based"},
                        "data": [],
                    }
                ],
            },
            "operational_delays": {"haulers": [], "trolleys": [], "load_zones": [], "dump_zones": []},
        },
        "override_parameters": {},
        "intersections": [],
    }
    
    return des_inputs


# =============================================================================
# Main Processing Function
# =============================================================================

def process_site(
    cursor,
    site_name: str,
    machines: Dict[int, Dict],
    output_dir: Optional[str] = None,
    limit: int = 100000,
    sample_interval: int = 5,
    grid_size: float = 5.0,
    min_density: int = 3,
    simplify_epsilon: float = 5.0,
    zone_grid_size: float = 10.0,
    zone_min_stops: int = 20,
    sim_time: int = 480,
    machine_templates: Optional[Dict[str, Any]] = None,
    telemetry_data: Optional[List[Tuple]] = None,
    coordinates_in_meters: Optional[bool] = None,
    precomputed_zones: Optional[List] = None,
) -> Dict[str, str]:
    """
    Process site data and generate all output files.

    Args:
        cursor: Database cursor (can be None if telemetry_data is provided)
        site_name: Site name
        machines: Dictionary of machine information
        output_dir: Output directory path
        limit: Limit for data fetching (ignored if telemetry_data provided)
        sample_interval: Sample interval (ignored if telemetry_data provided)
        grid_size: Grid size for road detection
        min_density: Minimum density for road detection
        simplify_epsilon: Simplify epsilon for road detection
        zone_grid_size: Grid size for zone detection
        zone_min_stops: Minimum stops for zone detection
        sim_time: Simulation time
        machine_templates: Machine templates dictionary
        telemetry_data: Optional pre-fetched telemetry data (list of tuples)
        coordinates_in_meters: If True, coordinates are in meters (import flow).
                              If False, coordinates are in millimeters (database flow).
                              If None, will be determined based on data source.
        precomputed_zones: Optional list of Reader.Zone objects from parse_cp1_data.
                          If provided, uses these instead of detect_zones().

    Returns:
        Dictionary with paths to generated files
    """
    # Use OUTPUT_PATH from env if output_dir not provided
    if output_dir is None:
        output_dir = resolve_path(OUTPUT_PATH, "../output")
        os.makedirs(output_dir, exist_ok=True)
    
    # Get machine IDs for this site
    machine_ids = [
        m["machine_unique_id"]
        for m in machines.values()
        if m.get("site_name") == site_name
    ]
    
    if not machine_ids and not telemetry_data:
        print(f"  No machines found for site: {site_name}")
        return {}
    
    print(f"\n  Processing site: {site_name} ({len(machine_ids)} machines)")
    
    # Track if telemetry data was provided (import flow) vs fetched from DB
    telemetry_data_provided = telemetry_data is not None

    # Fetch telemetry data or use provided data
    if telemetry_data is not None:
        print("  [1/5] Using provided telemetry data...")
        print(f"    Using {len(telemetry_data):,} records")
        # Extract unique machine IDs from telemetry data
        if telemetry_data:
            unique_machine_ids = set(row[0] for row in telemetry_data)
            machine_ids = list(unique_machine_ids)
            # Create minimal machine info for machines in telemetry data
            for mid in unique_machine_ids:
                if mid not in machines:
                    machines[mid] = {
                        "machine_unique_id": mid,
                        "name": f"Machine_{mid}",
                        "site_name": site_name,
                        "type_name": "Unknown"
                    }
    else:
        print("  [1/5] Fetching telemetry data...")
        telemetry_data = fetch_telemetry_data(
            cursor,
            machine_ids=machine_ids,
            limit=limit,
            sample_interval=sample_interval,
        )
        
        if not telemetry_data:
            print(f"  No telemetry data found for site: {site_name}")
            return {}
        
        print(f"    Fetched {len(telemetry_data):,} records")
    
    # Generate model (nodes and roads) from actual trajectories
    print("  [2/5] Generating road network model from trajectories...")
    # Determine coordinate format based on explicit parameter or data source
    # - If coordinates_in_meters is explicitly provided, use it
    # - If telemetry_data was provided (import flow), coordinates are in meters
    # - If fetched from DB, coordinates are in millimeters
    if coordinates_in_meters is None:
        coordinates_in_meters = telemetry_data_provided
    
    nodes, roads = create_roads_from_trajectories(
        telemetry_data,
        simplify_epsilon=simplify_epsilon,
        min_segment_distance=15.0,
        coordinates_in_meters=coordinates_in_meters,
    )
    
    if not nodes or not roads:
        print("  Error: Could not generate road network")
        return {}
    
    # Detect zones
    print("  [3/5] Detecting load/dump zones...")
    if precomputed_zones is not None:
        print("    Using precomputed zones from Reader.py (Segment classification + DBSCAN)...")
        load_zones, dump_zones = convert_reader_zones_to_model(
            precomputed_zones, nodes, roads
        )
    else:
        load_zones, dump_zones = detect_zones(
            telemetry_data, nodes, roads, zone_grid_size, zone_min_stops,
            coordinates_in_meters=coordinates_in_meters
        )
    print(f"    Found {len(load_zones)} load zones, {len(dump_zones)} dump zones")
    
    # Create model
    model = create_model(nodes, roads, load_zones, dump_zones)
    
    # Generate events
    print("  [4/5] Generating simulation events...")
    converter = GPSToEventsConverter(model_data=model)
    
    all_events = []
    machine_data = {}
    machines_with_events: Set[int] = set()
    for row in telemetry_data:
        mid = row[0]
        if mid not in machine_data:
            machine_data[mid] = []
        machine_data[mid].append(row)
    
    machine_iterator = machine_data.items()
    if TQDM_AVAILABLE:
        machine_iterator = tqdm(list(machine_iterator), desc="    Machines", unit="machine")
    
    for machine_id, data in machine_iterator:
        machine_info = machines.get(machine_id, {})
        machine_name = machine_info.get("name", f"Machine_{machine_id}")
        
        events = converter.convert_raw_telemetry(
            data,
            machine_id=machine_id,
            machine_name=machine_name,
            # Use smaller node spacing and larger search radius
            # so haulers follow more nodes along the road network.
            min_node_distance=5.0,
            max_search_distance=150.0,
            # Pass coordinates_in_meters to ensure consistent unit conversion
            # between road creation and event generation
            coordinates_in_meters=coordinates_in_meters,
        )

        # Only keep machines that actually generated events
        if events:
            machines_with_events.add(machine_id)
            all_events.extend(events)

        converter.reset()
    
    # Sort events by time and renumber
    all_events.sort(key=lambda e: (e.get("time", 0), e.get("eid", 0)))
    
    # Normalize time to start from 0 (all times in minutes from simulation start)
    if all_events:
        min_time = min(e.get("time", 0) for e in all_events)
        for event in all_events:
            event["time"] = round(event["time"] - min_time, 4)
    
    # Renumber events after sorting
    for i, event in enumerate(all_events):
        event["eid"] = i + 1
    
    print(f"    Generated {len(all_events):,} events")
    
    # Generate DES Inputs - only include machines that have events
    print("  [5/5] Generating DES inputs...")
    des_inputs = create_des_inputs(
        model, machines, site_name, sim_time, machines_with_events, machine_templates
    )
    
    # Save all files
    safe_name = site_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"model_{safe_name}.json")
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    
    # Save DES inputs
    des_inputs_path = os.path.join(output_dir, f"des_inputs_{safe_name}.json")
    with open(des_inputs_path, "w", encoding="utf-8") as f:
        json.dump(des_inputs, f, indent=2)
    
    # Save events ledger
    events_output = {
        "status": True,
        "data": {
            "version": "20250818",
            "events": all_events,
            "summary": {
                "total_events": len(all_events),
                "total_haulers": len(machine_data),
                "simulation_duration_minutes": max((e.get("time", 0) for e in all_events), default=0),
            },
        },
    }
    ledger_path = os.path.join(output_dir, f"simulation_ledger_{safe_name}.json")
    with open(ledger_path, "w", encoding="utf-8") as f:
        json.dump(events_output, f, indent=2, default=str)
    
    print(f"\n  Output files saved to: {output_dir}")
    print(f"    - Model: model_{safe_name}.json ({len(nodes)} nodes, {len(roads)} roads)")
    print(f"    - DES Inputs: des_inputs_{safe_name}.json ({len(des_inputs['haulers'])} haulers)")
    print(f"    - Events Ledger: simulation_ledger_{safe_name}.json ({len(all_events)} events)")
    
    return {
        "model": model_path,
        "des_inputs": des_inputs_path,
        "ledger": ledger_path,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate model and simulation files from AMT telemetry data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/simulation_generator.py                        # Use config.json
  python scripts/simulation_generator.py --config custom.json   # Use custom config
  python scripts/simulation_generator.py --site "BhpEscondida"  # Override site
  python scripts/simulation_generator.py --all-sites            # Process ALL sites
  python scripts/simulation_generator.py --list-sites           # List available sites
  python scripts/simulation_generator.py --init-config          # Create default config.json

Config file parameters can be overridden by CLI arguments.
        """
    )
    
    # Config file argument
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (default: scripts/config.json)",
    )
    parser.add_argument(
        "--init-config",
        action="store_true",
        help="Create default config.json and exit",
    )
    
    # Override arguments (all optional, will use config if not provided)
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="Site name to process (overrides config)",
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="List available sites and exit",
    )
    parser.add_argument(
        "--all-sites",
        action="store_true",
        help="Process ALL available sites",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum telemetry records to fetch (overrides config)",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=None,
        help="Sample every Nth record (overrides config)",
    )
    parser.add_argument(
        "--grid-size",
        type=float,
        default=None,
        help="Grid cell size for road detection in meters (overrides config)",
    )
    parser.add_argument(
        "--min-density",
        type=int,
        default=None,
        help="Minimum point density for road detection (overrides config)",
    )
    parser.add_argument(
        "--sim-time",
        type=int,
        default=None,
        help="Simulation time in minutes (overrides config)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GENERATE ALL SIMULATION DATA")
    print("=" * 70)
    
    # Handle --init-config
    if args.init_config:
        save_default_config(args.config)
        return
    
    # Load configuration
    print("\n[0/3] Loading configuration...")
    config = load_config(args.config)
    
    # Override config with CLI arguments
    site = args.site if args.site is not None else config["site"]
    # Use OUTPUT_PATH from env if not specified
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif "output_dir" in config and config["output_dir"]:
        output_dir = config["output_dir"]
    else:
        output_dir = resolve_path(OUTPUT_PATH, "../output")
    machine_templates_path = config.get("machine_templates_path")  # Path to custom templates
    limit = args.limit if args.limit is not None else config["data_fetching"]["limit"]
    sample_interval = args.sample_interval if args.sample_interval is not None else config["data_fetching"]["sample_interval"]
    grid_size = args.grid_size if args.grid_size is not None else config["road_detection"]["grid_size"]
    min_density = args.min_density if args.min_density is not None else config["road_detection"]["min_density"]
    simplify_epsilon = config["road_detection"]["simplify_epsilon"]
    zone_grid_size = config["zone_detection"]["grid_size"]
    zone_min_stops = config["zone_detection"]["min_stop_count"]
    sim_time = args.sim_time if args.sim_time is not None else config["simulation"]["sim_time"]
    
    # Print effective configuration
    print("\n  Effective configuration:")
    print(f"    site: {site}")
    print(f"    output_dir: {output_dir}")
    print(f"    data_fetching.limit: {limit}")
    print(f"    data_fetching.sample_interval: {sample_interval}")
    print(f"    road_detection.grid_size: {grid_size}")
    print(f"    road_detection.min_density: {min_density}")
    print(f"    road_detection.simplify_epsilon: {simplify_epsilon}")
    print(f"    zone_detection.grid_size: {zone_grid_size}")
    print(f"    zone_detection.min_stop_count: {zone_min_stops}")
    print(f"    simulation.sim_time: {sim_time}")
    
    print(f"\nDatabase: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    
    # Connect to database
    print("\n[1/3] Connecting to database...")
    connection = get_connection()
    if not connection:
        print("Failed to connect to database")
        return
    
    try:
        cursor = connection.cursor()
        
        # Fetch sites
        print("[2/3] Fetching site information...")
        sites = fetch_sites(cursor)
        print(f"  Found {len(sites)} sites")
        
        if args.list_sites:
            print("\nAvailable sites:")
            print("-" * 40)
            for s in sites:
                short = s["site_short"] or "N/A"
                print(f"  {s['site_name']} ({short})")
            print("-" * 40)
            return
        
        # Determine which sites to process
        site_names = [s["site_name"] for s in sites]
        
        if args.all_sites:
            # Process all sites
            sites_to_process = site_names
            print(f"\n  Processing ALL {len(sites_to_process)} sites...")
        elif site:
            # Process single site
            if site not in site_names:
                print(f"\nError: Site '{site}' not found.")
                print("Available sites:", ", ".join(site_names))
                return
            sites_to_process = [site]
        else:
            print("\nError: Site is required.")
            print("  Set 'site' in config.json, use --site argument, or use --all-sites.")
            print("  Use --list-sites to see available sites.")
            return
        
        # Process each site
        print("[3/3] Processing...")
        all_results = {}
        failed_sites = []
        
        # Load machine templates
        machine_templates = load_machine_templates(machine_templates_path)
        
        for idx, site_name in enumerate(sites_to_process, 1):
            if len(sites_to_process) > 1:
                print(f"\n{'='*70}")
                print(f"  Site {idx}/{len(sites_to_process)}: {site_name}")
                print(f"{'='*70}")
            
            machines = fetch_machines(cursor, site_name)
            
            result = process_site(
                cursor,
                site_name,
                machines,
                output_dir,
                limit=limit,
                sample_interval=sample_interval,
                grid_size=grid_size,
                min_density=min_density,
                simplify_epsilon=simplify_epsilon,
                zone_grid_size=zone_grid_size,
                zone_min_stops=zone_min_stops,
                sim_time=sim_time,
                machine_templates=machine_templates,
            )
            
            if result:
                all_results[site_name] = result
            else:
                failed_sites.append(site_name)
        
        # Print summary
        print("\n" + "=" * 70)
        print("GENERATION COMPLETE")
        print("=" * 70)
        
        if all_results:
            print(f"\nSuccessfully processed {len(all_results)} site(s):")
            for site_name, result in all_results.items():
                print(f"\n  {site_name}:")
                for key, path in result.items():
                    print(f"    {key}: {path}")
        
        if failed_sites:
            print(f"\nFailed sites ({len(failed_sites)}):")
            for site_name in failed_sites:
                print(f"  - {site_name}")
        
        print("=" * 70)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        connection.close()
        print("\nDatabase connection closed")


if __name__ == "__main__":
    main()
