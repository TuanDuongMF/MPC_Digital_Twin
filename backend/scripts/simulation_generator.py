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
import gzip
import json
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import deque, defaultdict
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

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


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

# Machines list file path (contains machine specs by model name like "793F", "797F", etc.)
MACHINES_LIST_PATH = os.path.join(
    example_json_resolved,
    "machines_list.json"
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


def extract_machine_model(type_name: str) -> Optional[str]:
    """
    Extract machine model name from TypeName.

    Examples:
        "Cat798AC CMD" -> "798AC"
        "Cat797F CMD" -> "797F"
        "CAT 793F CMD - Coal" -> "793F"
        "CA_CAT_793F-CMD" -> "793F"
        "930E CMD" -> "930E"
        "777G CMD" -> "777G"

    Args:
        type_name: TypeName from database (e.g., "Cat 793F CMD")

    Returns:
        Machine model name (e.g., "793F") or None if not found
    """
    if not type_name:
        return None

    # Pattern to match model numbers like: 793F, 798AC, 930E, 777G, 785NG, 794AC
    # Model format: 3-4 digits followed by optional letters (F, AC, NG, E, G, D)
    # Note: Don't use \b at start because TypeName may have no space (e.g., "Cat793NG CMD")
    pattern = r'(\d{3,4}[A-Z]{0,3})\b'

    match = re.search(pattern, type_name.upper())
    if match:
        return match.group(1)

    return None


def load_machines_list(machines_list_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load machines list from JSON file.

    The file contains machine specifications indexed by model name (e.g., "793F", "797F").
    Structure is similar to model.json format.

    Args:
        machines_list_path: Path to machines_list.json file. If None, uses default path.

    Returns:
        Dictionary mapping model name to machine specification data
    """
    if machines_list_path is None:
        machines_list_path = MACHINES_LIST_PATH
    else:
        # Resolve relative path if provided and not absolute
        if not os.path.isabs(machines_list_path):
            machines_list_path = os.path.join(backend_dir, machines_list_path)

    if os.path.exists(machines_list_path):
        try:
            with open(machines_list_path, "r", encoding="utf-8") as f:
                machines_list = json.load(f)
            print(f"  Loaded machines list from: {machines_list_path}")
            return machines_list
        except Exception as e:
            print(f"  Warning: Could not load machines list: {e}")
            print(f"  Machine specs will use defaults")
    else:
        print(f"  Machines list file not found: {machines_list_path}")
        print(f"  Machine specs will use defaults")

    return {}


def get_machine_spec_from_list(
    type_name: str,
    machines_list: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Get machine specification from machines_list by TypeName.

    The machines_list has format:
    {
        "version": "...",
        "machine_list": {
            "haulers": [
                {"id": 1, "name": "777G", ...},
                {"id": 2, "name": "793F", ...}
            ],
            "loaders": [...]
        }
    }

    Args:
        type_name: TypeName from database (e.g., "Cat 793F CMD")
        machines_list: Dictionary loaded from machines_list.json

    Returns:
        Machine specification dict or None if not found
    """
    if not machines_list:
        return None

    model_name = extract_machine_model(type_name)
    if not model_name:
        return None

    # Get haulers array from machine_list
    machine_list_data = machines_list.get("machine_list", {})
    haulers = machine_list_data.get("haulers", [])

    # Search for hauler by name (exact match first)
    for hauler in haulers:
        if hauler.get("name") == model_name:
            return hauler

    # Try case-insensitive match
    model_name_upper = model_name.upper()
    for hauler in haulers:
        hauler_name = hauler.get("name", "")
        if hauler_name.upper() == model_name_upper:
            return hauler

    return None


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

    machine_iter = machine_trajectories.items()
    if TQDM_AVAILABLE:
        machine_iter = tqdm(list(machine_iter), desc="    Building roads", unit="machine")

    for machine_id, trajectory in machine_iter:
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

    # Cleanup: Remove unused nodes (nodes not referenced by any road)
    used_node_ids = set()
    for road in roads:
        used_node_ids.update(road["nodes"])

    original_node_count = len(all_nodes)
    all_nodes = [node for node in all_nodes if node["id"] in used_node_ids]

    if len(all_nodes) < original_node_count:
        print(f"    Cleaned up {original_node_count - len(all_nodes)} unused nodes")

    return all_nodes, roads


def split_roads_at_intersections(
    roads: List[Dict],
) -> Tuple[List[Dict], Dict[int, List[int]]]:
    """
    Split roads at intersection and overlap points.

    Rules:
    - Roads can only share nodes at start or end points
    - If roads share nodes in the middle, split them at those points
    - Deduplicate shared segments

    Args:
        roads: List of road dictionaries with 'id' and 'nodes' keys

    Returns:
        Tuple of:
        - List of new road segments (with updated IDs and names)
        - Mapping from original road ID to list of new segment IDs
    """
    if not roads:
        return [], {}

    print(f"    Splitting roads at intersections ({len(roads)} roads)...")

    # Step 1: Build node usage map
    # node_id -> list of (road_id, position_index, is_endpoint)
    node_usage: Dict[int, List[Tuple[int, int, bool]]] = {}

    for road in roads:
        road_id = road["id"]
        nodes = road["nodes"]
        if not nodes:
            continue

        for idx, node_id in enumerate(nodes):
            is_endpoint = (idx == 0 or idx == len(nodes) - 1)

            if node_id not in node_usage:
                node_usage[node_id] = []
            node_usage[node_id].append((road_id, idx, is_endpoint))

    # Step 2: Identify critical nodes (split points)
    # A node is critical if:
    # - It's an endpoint of any road, OR
    # - It appears in more than one road
    critical_nodes: Set[int] = set()

    for node_id, usages in node_usage.items():
        # Check if endpoint of any road
        if any(is_endpoint for _, _, is_endpoint in usages):
            critical_nodes.add(node_id)
        # Check if used by multiple roads
        elif len(set(road_id for road_id, _, _ in usages)) > 1:
            critical_nodes.add(node_id)

    print(f"      Found {len(critical_nodes)} critical nodes (split points)")

    # Step 3: Split each road at critical nodes
    # raw_segments: list of (original_road_id, node_list_tuple)
    raw_segments: List[Tuple[int, Tuple[int, ...]]] = []

    for road in roads:
        road_id = road["id"]
        nodes = road["nodes"]

        if len(nodes) < 2:
            continue

        # Find split indices (positions of critical nodes in this road's middle)
        split_indices = [0]  # Always start from beginning
        for idx in range(1, len(nodes) - 1):  # Skip first and last (they're always splits)
            if nodes[idx] in critical_nodes:
                split_indices.append(idx)
        split_indices.append(len(nodes) - 1)  # Always end at last node

        # Remove duplicates and sort
        split_indices = sorted(set(split_indices))

        # Create segments between consecutive split points
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]

            segment_nodes = tuple(nodes[start_idx:end_idx + 1])
            if len(segment_nodes) >= 2:
                raw_segments.append((road_id, segment_nodes))

    print(f"      Created {len(raw_segments)} raw segments")

    # Step 4: Deduplicate segments
    # Group segments by their node sequence (to find shared segments)
    # Key: node_tuple (or reversed), Value: list of original road IDs
    segment_to_roads: Dict[Tuple[int, ...], Set[int]] = {}

    for original_road_id, segment_nodes in raw_segments:
        # Normalize segment direction (use smaller first node as canonical form)
        # This ensures [1,2,3] and [3,2,1] are treated as same segment
        if segment_nodes[0] > segment_nodes[-1]:
            canonical_nodes = tuple(reversed(segment_nodes))
        else:
            canonical_nodes = segment_nodes

        if canonical_nodes not in segment_to_roads:
            segment_to_roads[canonical_nodes] = set()
        segment_to_roads[canonical_nodes].add(original_road_id)

    # Step 5: Create final segments with proper IDs and names
    new_roads: List[Dict] = []
    new_road_id = 1

    # Map: canonical_nodes -> new_road_id (for building road composition)
    canonical_to_new_id: Dict[Tuple[int, ...], int] = {}

    for canonical_nodes, original_road_ids in segment_to_roads.items():
        is_shared = len(original_road_ids) > 1

        # Generate name
        if is_shared:
            name = f"Road_{new_road_id}_Shared"
        else:
            name = f"Road_{new_road_id}"

        new_road = {
            "id": new_road_id,
            "name": name,
            "nodes": list(canonical_nodes),
            "is_generated": False,
            "ways_num": 2,
            "lanes_num": 1,
            "banking": "",
            "lane_width": "",
            "speed_limit": "",
            "rolling_resistance": "",
            "traction_coefficient": "",
            "offset": 0,
            # Metadata for tracking
            "_original_roads": sorted(original_road_ids),
            "_is_shared": is_shared,
        }
        new_roads.append(new_road)
        canonical_to_new_id[canonical_nodes] = new_road_id
        new_road_id += 1

    # Step 6: Build road composition mapping
    # original_road_id -> list of new segment IDs in order
    road_composition: Dict[int, List[int]] = {}

    for road in roads:
        original_road_id = road["id"]
        nodes = road["nodes"]

        if len(nodes) < 2:
            road_composition[original_road_id] = []
            continue

        # Find split indices again
        split_indices = [0]
        for idx in range(1, len(nodes) - 1):
            if nodes[idx] in critical_nodes:
                split_indices.append(idx)
        split_indices.append(len(nodes) - 1)
        split_indices = sorted(set(split_indices))

        # Build ordered list of segment IDs
        segment_ids = []
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]

            segment_nodes = tuple(nodes[start_idx:end_idx + 1])
            if len(segment_nodes) < 2:
                continue

            # Find canonical form
            if segment_nodes[0] > segment_nodes[-1]:
                canonical_nodes = tuple(reversed(segment_nodes))
            else:
                canonical_nodes = segment_nodes

            if canonical_nodes in canonical_to_new_id:
                segment_ids.append(canonical_to_new_id[canonical_nodes])

        road_composition[original_road_id] = segment_ids

    # Count shared segments
    shared_count = sum(1 for r in new_roads if r.get("_is_shared", False))
    print(f"      Final: {len(new_roads)} segments ({shared_count} shared)")

    return new_roads, road_composition


def update_routes_with_split_roads(
    routes: List[Dict],
    road_composition: Dict[int, List[int]],
) -> List[Dict]:
    """
    Update route definitions to use split road segments.

    Args:
        routes: List of route dictionaries with 'haul' and 'return' keys
        road_composition: Mapping from original road ID to list of new segment IDs

    Returns:
        Updated routes with expanded road references
    """
    if not routes:
        return routes

    updated_routes = []
    for route in routes:
        new_route = route.copy()

        # Update haul path
        if "haul" in route and route["haul"]:
            new_haul = []
            for road_id in route["haul"]:
                if road_id in road_composition:
                    new_haul.extend(road_composition[road_id])
                else:
                    new_haul.append(road_id)
            new_route["haul"] = new_haul

        # Update return path
        if "return" in route and route["return"]:
            new_return = []
            for road_id in route["return"]:
                if road_id in road_composition:
                    new_return.extend(road_composition[road_id])
                else:
                    new_return.append(road_id)
            new_route["return"] = new_return

        updated_routes.append(new_route)

    return updated_routes


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

    # Cleanup: Remove unused nodes (nodes not referenced by any road)
    used_node_ids = set()
    for road in roads:
        used_node_ids.update(road["nodes"])

    original_node_count = len(all_nodes)
    all_nodes = [node for node in all_nodes if node["id"] in used_node_ids]

    if len(all_nodes) < original_node_count:
        print(f"    Cleaned up {original_node_count - len(all_nodes)} unused nodes")

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


def create_routes(
    load_zones: List[Dict],
    dump_zones: List[Dict],
    roads: List[Dict],
    nodes: List[Dict],
) -> List[Dict]:
    """
    Create routes connecting load zones to dump zones.

    A route defines the path a hauler takes:
    - Load at load_zone
    - Haul (loaded) via haul roads to dump_zone
    - Dump at dump_zone
    - Return (empty) via return roads back to load_zone

    Args:
        load_zones: List of load zone dictionaries
        dump_zones: List of dump zone dictionaries
        roads: List of road dictionaries
        nodes: List of node dictionaries

    Returns:
        List of route dictionaries with structure:
        {
            "id": int,
            "name": str,
            "haul": [road_ids],
            "return": [road_ids],
            "load_zone": int,
            "dump_zone": int
        }
    """
    if not load_zones or not dump_zones or not roads:
        return []

    # Build node lookup
    node_lookup = {n["id"]: n for n in nodes}

    # Build road lookup and connection graph
    road_lookup = {r["id"]: r for r in roads}

    # Build graph: node_id -> list of (road_id, other_endpoint_node_id)
    # Each road connects its first node to its last node
    node_to_roads = {}
    for road in roads:
        if len(road["nodes"]) < 2:
            continue
        start_node = road["nodes"][0]
        end_node = road["nodes"][-1]

        if start_node not in node_to_roads:
            node_to_roads[start_node] = []
        if end_node not in node_to_roads:
            node_to_roads[end_node] = []

        # Road can be traversed in both directions (2-way roads)
        node_to_roads[start_node].append((road["id"], end_node))
        node_to_roads[end_node].append((road["id"], start_node))

    def find_path(start_node_id: int, end_node_id: int) -> List[int]:
        """
        Find a path of roads from start_node to end_node using BFS.
        Returns list of road IDs forming the path.
        """
        if start_node_id == end_node_id:
            return []

        if start_node_id not in node_to_roads:
            return []

        # BFS to find path
        from collections import deque
        queue = deque([(start_node_id, [])])  # (current_node, road_path)
        visited = {start_node_id}

        while queue:
            current_node, path = queue.popleft()

            if current_node not in node_to_roads:
                continue

            for road_id, next_node in node_to_roads[current_node]:
                if next_node in visited:
                    continue

                new_path = path + [road_id]

                if next_node == end_node_id:
                    return new_path

                visited.add(next_node)
                queue.append((next_node, new_path))

        return []

    routes = []
    route_id = 1

    for lz in load_zones:
        lz_id = lz["id"]
        lz_name = lz.get("name", f"Load zone {lz_id}")
        lz_settings = lz.get("settings", {})
        lz_outroad_ids = lz_settings.get("outroad_ids", [])
        lz_outnode_ids = lz_settings.get("outnode_ids", [])
        lz_inroad_ids = lz_settings.get("inroad_ids", [])
        lz_innode_ids = lz_settings.get("innode_ids", [])

        for dz in dump_zones:
            dz_id = dz["id"]
            dz_name = dz.get("name", f"Dump zone {dz_id}")
            dz_settings = dz.get("settings", {})
            dz_inroad_ids = dz_settings.get("inroad_ids", [])
            dz_innode_ids = dz_settings.get("innode_ids", [])
            dz_outroad_ids = dz_settings.get("outroad_ids", [])
            dz_outnode_ids = dz_settings.get("outnode_ids", [])

            # Determine haul path: from load_zone exit to dump_zone entry
            haul_roads = []
            haul_path_found = False
            if lz_outroad_ids and dz_inroad_ids:
                # If both zones connect to the same road, that's the haul road
                if lz_outroad_ids[0] == dz_inroad_ids[0]:
                    haul_roads = [lz_outroad_ids[0]]
                    haul_path_found = True
                else:
                    # Try to find path from load_zone exit node to dump_zone entry node
                    if lz_outnode_ids and dz_innode_ids:
                        path = find_path(lz_outnode_ids[0], dz_innode_ids[0])
                        if path:
                            haul_roads = path
                            haul_path_found = True
                        else:
                            # Fallback: try to find connecting path between roads
                            lz_road = road_lookup.get(lz_outroad_ids[0])
                            dz_road = road_lookup.get(dz_inroad_ids[0])
                            if lz_road and dz_road:
                                # Try all endpoint combinations
                                lz_endpoints = [lz_road["nodes"][0], lz_road["nodes"][-1]]
                                dz_endpoints = [dz_road["nodes"][0], dz_road["nodes"][-1]]
                                for lz_ep in lz_endpoints:
                                    for dz_ep in dz_endpoints:
                                        connecting_path = find_path(lz_ep, dz_ep)
                                        if connecting_path:
                                            # Build full path: lz_outroad + connecting + dz_inroad
                                            haul_roads = lz_outroad_ids + connecting_path + [r for r in dz_inroad_ids if r not in lz_outroad_ids and r not in connecting_path]
                                            haul_path_found = True
                                            break
                                    if haul_path_found:
                                        break
                            if not haul_path_found:
                                # Last fallback: just use lz_outroad (partial path)
                                haul_roads = lz_outroad_ids
                    else:
                        haul_roads = lz_outroad_ids
            elif lz_outroad_ids:
                haul_roads = lz_outroad_ids

            # Determine return path: from dump_zone exit to load_zone entry
            return_roads = []
            return_path_found = False
            if dz_outroad_ids and lz_inroad_ids:
                # If both zones connect to the same road, that's the return road
                if dz_outroad_ids[0] == lz_inroad_ids[0]:
                    return_roads = [dz_outroad_ids[0]]
                    return_path_found = True
                else:
                    # Try to find path from dump_zone exit node to load_zone entry node
                    if dz_outnode_ids and lz_innode_ids:
                        path = find_path(dz_outnode_ids[0], lz_innode_ids[0])
                        if path:
                            return_roads = path
                            return_path_found = True
                        else:
                            # Fallback: try to find connecting path between roads
                            dz_road = road_lookup.get(dz_outroad_ids[0])
                            lz_road = road_lookup.get(lz_inroad_ids[0])
                            if dz_road and lz_road:
                                # Try all endpoint combinations
                                dz_endpoints = [dz_road["nodes"][0], dz_road["nodes"][-1]]
                                lz_endpoints = [lz_road["nodes"][0], lz_road["nodes"][-1]]
                                for dz_ep in dz_endpoints:
                                    for lz_ep in lz_endpoints:
                                        connecting_path = find_path(dz_ep, lz_ep)
                                        if connecting_path:
                                            # Build full path: dz_outroad + connecting + lz_inroad
                                            return_roads = dz_outroad_ids + connecting_path + [r for r in lz_inroad_ids if r not in dz_outroad_ids and r not in connecting_path]
                                            return_path_found = True
                                            break
                                    if return_path_found:
                                        break
                            if not return_path_found:
                                # Last fallback: just use dz_outroad (partial path)
                                return_roads = dz_outroad_ids
                    else:
                        return_roads = dz_outroad_ids
            elif dz_outroad_ids:
                return_roads = dz_outroad_ids

            # Skip routes with no valid paths
            if not haul_roads and not return_roads:
                print(f"    Warning: No path found for route {lz_name} to {dz_name}, skipping")
                continue

            # Validate route connectivity
            def validate_path_connectivity(path: List[int], path_name: str) -> bool:
                """Check if consecutive roads in path share a common endpoint."""
                if len(path) <= 1:
                    return True
                for i in range(len(path) - 1):
                    road_a = road_lookup.get(path[i])
                    road_b = road_lookup.get(path[i + 1])
                    if not road_a or not road_b:
                        continue
                    # Get endpoints of both roads
                    a_endpoints = {road_a["nodes"][0], road_a["nodes"][-1]}
                    b_endpoints = {road_b["nodes"][0], road_b["nodes"][-1]}
                    # Check if they share any endpoint
                    if not a_endpoints.intersection(b_endpoints):
                        print(f"    Warning: {path_name} roads {path[i]} and {path[i+1]} do not share a common node")
                        return False
                return True

            haul_valid = validate_path_connectivity(haul_roads, f"Route '{lz_name} to {dz_name}' haul")
            return_valid = validate_path_connectivity(return_roads, f"Route '{lz_name} to {dz_name}' return")

            # Only add route if both paths are valid (or empty)
            if not haul_valid or not return_valid:
                print(f"    Warning: Route {lz_name} to {dz_name} has invalid paths, skipping")
                continue

            route = {
                "id": route_id,
                "name": f"{lz_name} to {dz_name}",
                "haul": haul_roads,
                "return": return_roads,
                "load_zone": lz_id,
                "dump_zone": dz_id,
            }
            routes.append(route)
            route_id += 1

    return routes


def get_path_entry_exit_nodes(
    path: List[int],
    roads: List[Dict],
    start_node_hint: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Determine the entry and exit nodes of a path (sequence of roads).

    For a valid path, consecutive roads must share a common endpoint.
    This function traces through the path to find:
    - Entry node: The starting node where we enter the first road
    - Exit node: The ending node where we leave the last road

    Args:
        path: List of road IDs forming the path
        roads: List of road dictionaries
        start_node_hint: Optional hint for which node to start from (for single road paths)

    Returns:
        Tuple of (entry_node_id, exit_node_id), or (None, None) if path is invalid
    """
    if not path:
        return None, None

    road_lookup = {r["id"]: r for r in roads}

    if len(path) == 1:
        # Single road - use hint if provided, otherwise return both endpoints
        road = road_lookup.get(path[0])
        if not road or len(road["nodes"]) < 2:
            return None, None

        start_node = road["nodes"][0]
        end_node = road["nodes"][-1]

        # If hint is provided and matches one endpoint, use it to determine direction
        if start_node_hint is not None:
            if start_node_hint == start_node:
                return start_node, end_node
            elif start_node_hint == end_node:
                return end_node, start_node

        # Default: first node is entry, last node is exit
        return start_node, end_node

    # For multiple roads, trace through to find entry/exit
    # Start by determining which endpoint of first road connects to second road
    first_road = road_lookup.get(path[0])
    second_road = road_lookup.get(path[1])

    if not first_road or not second_road:
        return None, None

    first_endpoints = {first_road["nodes"][0], first_road["nodes"][-1]}
    second_endpoints = {second_road["nodes"][0], second_road["nodes"][-1]}

    shared = first_endpoints.intersection(second_endpoints)
    if not shared:
        return None, None

    # The shared node is where first road exits
    shared_node = shared.pop()
    # Entry node is the other endpoint of first road
    entry_node = first_road["nodes"][-1] if first_road["nodes"][0] == shared_node else first_road["nodes"][0]

    # Now trace through to find exit node
    current_node = shared_node
    for i in range(1, len(path)):
        road = road_lookup.get(path[i])
        if not road:
            return entry_node, None

        # Determine which direction we traverse this road
        if road["nodes"][0] == current_node:
            # Traverse forward
            current_node = road["nodes"][-1]
        elif road["nodes"][-1] == current_node:
            # Traverse backward
            current_node = road["nodes"][0]
        else:
            # Discontinuity - road doesn't connect
            return entry_node, None

    return entry_node, current_node


def update_zone_settings_for_routes(
    routes: List[Dict],
    load_zones: List[Dict],
    dump_zones: List[Dict],
    roads: List[Dict],
) -> None:
    """
    Update zone innode_ids and outnode_ids based on actual route paths.

    This ensures zone entry/exit nodes match the route definitions to avoid
    discontinuity errors during validation.

    For each route:
    - Load zone outnode_ids should include the entry node of first haul road
    - Dump zone innode_ids should include the exit node of last haul road
    - Dump zone outnode_ids should include the entry node of first return road
    - Load zone innode_ids should include the exit node of last return road

    Args:
        routes: List of route dictionaries
        load_zones: List of load zone dictionaries (will be modified in place)
        dump_zones: List of dump zone dictionaries (will be modified in place)
        roads: List of road dictionaries
    """
    if not routes or not roads:
        return

    # Build zone lookups
    lz_lookup = {z["id"]: z for z in load_zones}
    dz_lookup = {z["id"]: z for z in dump_zones}

    for route in routes:
        lz_id = route.get("load_zone")
        dz_id = route.get("dump_zone")
        haul_path = route.get("haul", [])
        return_path = route.get("return", [])

        lz = lz_lookup.get(lz_id)
        dz = dz_lookup.get(dz_id)

        if not lz or not dz:
            continue

        # Ensure settings exist
        if "settings" not in lz:
            lz["settings"] = {}
        if "settings" not in dz:
            dz["settings"] = {}

        # Get existing zone node hints for single road cases
        lz_outnode_hint = lz["settings"].get("outnode_ids", [None])[0]
        dz_outnode_hint = dz["settings"].get("outnode_ids", [None])[0]

        # Process haul path: load zone -> dump zone
        if haul_path:
            haul_entry, haul_exit = get_path_entry_exit_nodes(
                haul_path, roads, start_node_hint=lz_outnode_hint
            )

            if haul_entry is not None:
                # Update load zone outnode_ids
                outnode_ids = lz["settings"].get("outnode_ids", [])
                if haul_entry not in outnode_ids:
                    outnode_ids.append(haul_entry)
                lz["settings"]["outnode_ids"] = outnode_ids

            if haul_exit is not None:
                # Update dump zone innode_ids
                innode_ids = dz["settings"].get("innode_ids", [])
                if haul_exit not in innode_ids:
                    innode_ids.append(haul_exit)
                dz["settings"]["innode_ids"] = innode_ids

        # Process return path: dump zone -> load zone
        if return_path:
            return_entry, return_exit = get_path_entry_exit_nodes(
                return_path, roads, start_node_hint=dz_outnode_hint
            )

            if return_entry is not None:
                # Update dump zone outnode_ids
                outnode_ids = dz["settings"].get("outnode_ids", [])
                if return_entry not in outnode_ids:
                    outnode_ids.append(return_entry)
                dz["settings"]["outnode_ids"] = outnode_ids

            if return_exit is not None:
                # Update load zone innode_ids
                innode_ids = lz["settings"].get("innode_ids", [])
                if return_exit not in innode_ids:
                    innode_ids.append(return_exit)
                lz["settings"]["innode_ids"] = innode_ids


def analyze_hauler_trips_from_telemetry(
    telemetry_data: List[Tuple],
    load_zones: List[Dict],
    dump_zones: List[Dict],
    coordinates_in_meters: bool = False,
    payload_threshold: float = 50.0,
) -> Dict[int, List[Dict]]:
    """
    Analyze telemetry data to extract actual hauler trips (load zone -> dump zone).

    Detects complete cycles by tracking payload transitions:
    - LOAD: payload transitions from empty (<50%) to loaded (>50%)
    - DUMP: payload transitions from loaded (>50%) to empty (<50%)

    Args:
        telemetry_data: List of telemetry tuples
        load_zones: List of load zone dictionaries with detected_location
        dump_zones: List of dump zone dictionaries with detected_location
        coordinates_in_meters: If True, coordinates are in meters; otherwise millimeters
        payload_threshold: Payload percentage threshold (default 50%)

    Returns:
        Dictionary mapping machine_id -> list of trips
        Each trip: {"load_zone_id": int, "dump_zone_id": int, "load_zone_name": str, "dump_zone_name": str}
    """
    if not telemetry_data:
        return {}

    import math

    # Build zone lookups with locations
    def get_zone_location(zone):
        loc = zone.get("detected_location") or zone.get("settings", {}).get("detected_location")
        if loc:
            return loc.get("x", 0), loc.get("y", 0)
        return None

    lz_locations = []
    for z in load_zones:
        loc = get_zone_location(z)
        if loc:
            lz_locations.append({
                "id": z["id"],
                "name": z.get("name", f"Load zone {z['id']}"),
                "x": loc[0],
                "y": loc[1],
            })

    dz_locations = []
    for z in dump_zones:
        loc = get_zone_location(z)
        if loc:
            dz_locations.append({
                "id": z["id"],
                "name": z.get("name", f"Dump zone {z['id']}"),
                "x": loc[0],
                "y": loc[1],
            })

    def find_nearest_zone(x, y, zones):
        """Find nearest zone to (x, y) coordinates."""
        if not zones:
            return None
        best = None
        best_dist = float("inf")
        for z in zones:
            dx = x - z["x"]
            dy = y - z["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best = z
        return best if best_dist < 500 else None  # Max 500m from zone center

    # Group telemetry by machine_id and sort by time
    machine_data = {}
    for row in telemetry_data:
        machine_id = row[1] if len(row) > 1 else None
        if machine_id is None:
            continue

        if machine_id not in machine_data:
            machine_data[machine_id] = []

        # Extract data: (timestamp, x, y, payload)
        timestamp = row[2] if len(row) > 2 else 0
        if coordinates_in_meters:
            x = float(row[4]) if len(row) > 4 and row[4] else 0
            y = float(row[5]) if len(row) > 5 and row[5] else 0
        else:
            x = row[4] / 1000.0 if len(row) > 4 and row[4] else 0
            y = row[5] / 1000.0 if len(row) > 5 and row[5] else 0
        payload = row[13] if len(row) > 13 and row[13] is not None else 0

        machine_data[machine_id].append({
            "timestamp": timestamp,
            "x": x,
            "y": y,
            "payload": payload,
        })

    # Analyze trips for each machine
    trips_by_machine = {}
    for machine_id, data_points in machine_data.items():
        # Sort by timestamp
        data_points.sort(key=lambda p: p["timestamp"])

        trips = []
        prev_loaded = None
        current_load_zone = None

        for point in data_points:
            payload = point["payload"]
            is_loaded = payload >= payload_threshold

            if prev_loaded is not None:
                # Detect LOAD transition (empty -> loaded)
                if not prev_loaded and is_loaded:
                    zone = find_nearest_zone(point["x"], point["y"], lz_locations)
                    if zone:
                        current_load_zone = zone

                # Detect DUMP transition (loaded -> empty)
                elif prev_loaded and not is_loaded:
                    zone = find_nearest_zone(point["x"], point["y"], dz_locations)
                    if zone and current_load_zone:
                        # Complete trip detected
                        trips.append({
                            "load_zone_id": current_load_zone["id"],
                            "load_zone_name": current_load_zone["name"],
                            "dump_zone_id": zone["id"],
                            "dump_zone_name": zone["name"],
                        })
                    current_load_zone = None

            prev_loaded = is_loaded

        if trips:
            trips_by_machine[machine_id] = trips

    return trips_by_machine


def create_material_schedule_from_trips(
    trips_by_machine: Dict[int, List[Dict]],
    machines: Dict[int, Dict] = None,
    default_density: float = 1960.19,
    default_material: str = "Ore",
) -> List[Dict]:
    """
    Create material schedule data from analyzed hauler trips.

    Each unique combination of (machine_id, load_zone, dump_zone) creates
    one material schedule item with num_of_hauler = 1 (since each hauler group
    contains exactly 1 hauler).

    Args:
        trips_by_machine: Dictionary from analyze_hauler_trips_from_telemetry
        machines: Optional machine info dictionary
        default_density: Default material density
        default_material: Default material name

    Returns:
        List of material schedule data items
    """
    if not trips_by_machine:
        return []

    # Collect unique (machine_id, lz_name, dz_name) combinations
    unique_routes = set()
    for machine_id, trips in trips_by_machine.items():
        for trip in trips:
            key = (machine_id, trip["load_zone_name"], trip["dump_zone_name"])
            unique_routes.add(key)

    # Build material schedule items - one per unique (hauler, route) combination
    material_data = []
    hauler_group_id = 1
    for idx, (machine_id, lz_name, dz_name) in enumerate(sorted(unique_routes), start=1):
        item = {
            "id": idx,
            "load_zone": lz_name,
            "dump_zone": dz_name,
            "route": "",
            "auto_generate_route": True,
            "material": default_material,
            "density": default_density,
            "num_of_hauler": 1,
            "assigned_machine_type": "Hauler",
            "multiple_routes": False,
            "hauler_group_id": hauler_group_id,
        }
        material_data.append(item)
        hauler_group_id += 1

    return material_data


def create_material_schedule_data(
    routes: List[Dict],
    load_zones: List[Dict],
    dump_zones: List[Dict],
    haulers: List[Dict] = None,
    telemetry_data: List[Tuple] = None,
    coordinates_in_meters: bool = False,
    default_density: float = 1960.19,
    default_material: str = "Ore",
) -> List[Dict]:
    """
    Create material schedule data based on actual telemetry trips or routes.

    If telemetry_data is provided, analyzes actual hauler trips to determine
    which haulers traveled between which zones. Otherwise, falls back to
    generating items from routes.

    Args:
        routes: List of route dictionaries with load_zone and dump_zone references
        load_zones: List of load zone dictionaries
        dump_zones: List of dump zone dictionaries
        haulers: Optional list of hauler dictionaries
        telemetry_data: Optional telemetry data for actual trip analysis
        coordinates_in_meters: Whether telemetry coordinates are in meters
        default_density: Default material density in kg/m³
        default_material: Default material name

    Returns:
        List of material schedule data items
    """
    # Try to analyze actual trips from telemetry data first
    if telemetry_data and load_zones and dump_zones:
        trips_by_machine = analyze_hauler_trips_from_telemetry(
            telemetry_data, load_zones, dump_zones, coordinates_in_meters
        )
        if trips_by_machine:
            return create_material_schedule_from_trips(
                trips_by_machine,
                default_density=default_density,
                default_material=default_material,
            )

    # Fallback: generate from routes
    if not routes:
        return []

    # Build zone name lookups
    lz_lookup = {z["id"]: z.get("name", f"Load zone {z['id']}") for z in load_zones}
    dz_lookup = {z["id"]: z.get("name", f"Dump zone {z['id']}") for z in dump_zones}

    # Count haulers per route if haulers provided
    haulers_per_route = {}
    if haulers:
        for hauler in haulers:
            route_id = hauler.get("initial_conditions", {}).get("route_id")
            if route_id is not None:
                haulers_per_route[route_id] = haulers_per_route.get(route_id, 0) + hauler.get("number_of_haulers", 1)

    # Default haulers per route if no hauler info available
    total_haulers = sum(haulers_per_route.values()) if haulers_per_route else len(routes) * 4
    default_haulers = max(1, total_haulers // len(routes)) if routes else 4

    material_data = []
    for idx, route in enumerate(routes, start=1):
        lz_id = route.get("load_zone")
        dz_id = route.get("dump_zone")

        lz_name = lz_lookup.get(lz_id, f"Load zone {lz_id}")
        dz_name = dz_lookup.get(dz_id, f"Dump zone {dz_id}")
        route_name = route.get("name", "")

        # Get hauler count for this route
        num_haulers = haulers_per_route.get(route.get("id"), default_haulers)

        item = {
            "id": idx,
            "load_zone": lz_name,
            "dump_zone": dz_name,
            "route": route_name,
            "auto_generate_route": True,
            "material": default_material,
            "density": default_density,
            "num_of_hauler": num_haulers,
            "assigned_machine_type": "Hauler",
            "multiple_routes": False,
            "hauler_group_id": 1,
        }
        material_data.append(item)

    return material_data


def create_operations_structure(
    routes: List[Dict],
    load_zones: List[Dict],
    dump_zones: List[Dict],
    haulers: List[Dict] = None,
    telemetry_data: List[Tuple] = None,
    coordinates_in_meters: bool = False,
    schedule_name: str = "Material Schedule 1",
    scheduling_method: str = "grouped_assignment",
) -> Dict:
    """
    Create the complete operations structure with material schedules.

    Args:
        routes: List of route dictionaries
        load_zones: List of load zone dictionaries
        dump_zones: List of dump zone dictionaries
        haulers: Optional list of hauler dictionaries
        telemetry_data: Optional telemetry data for actual trip analysis
        coordinates_in_meters: Whether telemetry coordinates are in meters
        schedule_name: Name for the material schedule
        scheduling_method: Scheduling method (grouped_assignment, production_target_based, etc.)

    Returns:
        Operations dictionary with material_schedules structure
    """
    material_data = create_material_schedule_data(
        routes, load_zones, dump_zones, haulers,
        telemetry_data=telemetry_data,
        coordinates_in_meters=coordinates_in_meters,
    )

    return {
        "material_schedules": {
            "selected_material": 1,
            "all_material_schedule": [
                {
                    "id": 1,
                    "name": schedule_name,
                    "hauler_assignment": {"scheduling_method": scheduling_method},
                    "mixed_fleet_based_initial_assignment": False,
                    "data": material_data,
                }
            ],
        },
        "operational_delays": {
            "haulers": [],
            "trolleys": [],
            "load_zones": [],
            "dump_zones": [],
        },
    }


def export_route_excel(
    nodes: List[Dict],
    roads: List[Dict],
    load_zones: List[Dict],
    dump_zones: List[Dict],
    routes: List[Dict],
    output_path: str,
) -> Optional[str]:
    """
    Export route data to Excel file following Route_Template format.

    The Route_Data sheet contains node coordinates for each route with
    the following columns:
    - Easting (m): X coordinate
    - Northing (m): Y coordinate
    - Elevation (m): Z coordinate
    - RouteIndex: Route identifier (road_id)
    - Segment: "haul" or "return"
    - Load Zone: Load zone name
    - Dump Zone: Dump zone name
    - Rolling Resistance (%): Optional
    - Speed Limit (kph): Optional
    - Trolley: Optional
    - Banking (%): Optional
    - Curvature (1/m): Optional
    - Lane Width (m): Optional
    - Traction Coefficient: Optional

    Args:
        nodes: List of node dictionaries with coords
        roads: List of road dictionaries with node IDs
        load_zones: List of load zone dictionaries
        dump_zones: List of dump zone dictionaries
        routes: List of route dictionaries connecting zones
        output_path: Output file path for Excel file

    Returns:
        Path to generated Excel file, or None if pandas not available
    """
    if not PANDAS_AVAILABLE:
        print("    Warning: pandas not available, skipping Excel export")
        return None

    if not nodes or not roads:
        print("    Warning: No nodes or roads to export")
        return None

    # Build node lookup
    node_lookup = {n["id"]: n for n in nodes}

    # Build zone name lookups
    load_zone_lookup = {z["id"]: z.get("name", f"Load zone {z['id']}") for z in (load_zones or [])}
    dump_zone_lookup = {z["id"]: z.get("name", f"Dump zone {z['id']}") for z in (dump_zones or [])}

    # Build road lookup
    road_lookup = {r["id"]: r for r in roads}

    # Prepare data rows
    rows = []

    if routes:
        # Export based on routes (with Load Zone / Dump Zone info)
        for route in routes:
            route_id = route["id"]
            load_zone_id = route.get("load_zone")
            dump_zone_id = route.get("dump_zone")
            load_zone_name = load_zone_lookup.get(load_zone_id, "")
            dump_zone_name = dump_zone_lookup.get(dump_zone_id, "")

            # Export haul roads
            haul_road_ids = route.get("haul", [])
            for road_id in haul_road_ids:
                road = road_lookup.get(road_id)
                if not road:
                    continue
                for node_id in road.get("nodes", []):
                    node = node_lookup.get(node_id)
                    if not node:
                        continue
                    coords = node.get("coords", [0, 0, 0])
                    rows.append({
                        "Easting (m)": round(coords[0], 3),
                        "Northing (m)": round(coords[1], 3),
                        "Elevation (m)": round(coords[2], 3),
                        "RouteIndex": route_id,
                        "Segment": "haul",
                        "Load Zone": load_zone_name,
                        "Dump Zone": dump_zone_name,
                        "Rolling Resistance (%)": node.get("rolling_resistance", ""),
                        "Speed Limit (kph)": node.get("speed_limit", ""),
                        "Trolley": "",
                        "Banking (%)": node.get("banking", ""),
                        "Curvature (1/m)": node.get("curvature", ""),
                        "Lane Width (m)": node.get("lane_width", ""),
                        "Traction Coefficient": node.get("traction", ""),
                    })

            # Export return roads
            return_road_ids = route.get("return", [])
            for road_id in return_road_ids:
                road = road_lookup.get(road_id)
                if not road:
                    continue
                for node_id in road.get("nodes", []):
                    node = node_lookup.get(node_id)
                    if not node:
                        continue
                    coords = node.get("coords", [0, 0, 0])
                    rows.append({
                        "Easting (m)": round(coords[0], 3),
                        "Northing (m)": round(coords[1], 3),
                        "Elevation (m)": round(coords[2], 3),
                        "RouteIndex": route_id,
                        "Segment": "return",
                        "Load Zone": load_zone_name,
                        "Dump Zone": dump_zone_name,
                        "Rolling Resistance (%)": node.get("rolling_resistance", ""),
                        "Speed Limit (kph)": node.get("speed_limit", ""),
                        "Trolley": "",
                        "Banking (%)": node.get("banking", ""),
                        "Curvature (1/m)": node.get("curvature", ""),
                        "Lane Width (m)": node.get("lane_width", ""),
                        "Traction Coefficient": node.get("traction", ""),
                    })
    else:
        # Fallback: Export roads without route info (Segment = "road")
        for road in roads:
            road_id = road["id"]
            for node_id in road.get("nodes", []):
                node = node_lookup.get(node_id)
                if not node:
                    continue
                coords = node.get("coords", [0, 0, 0])
                rows.append({
                    "Easting (m)": round(coords[0], 3),
                    "Northing (m)": round(coords[1], 3),
                    "Elevation (m)": round(coords[2], 3),
                    "RouteIndex": road_id,
                    "Segment": "road",
                    "Load Zone": "",
                    "Dump Zone": "",
                    "Rolling Resistance (%)": node.get("rolling_resistance", ""),
                    "Speed Limit (kph)": node.get("speed_limit", ""),
                    "Trolley": "",
                    "Banking (%)": node.get("banking", ""),
                    "Curvature (1/m)": node.get("curvature", ""),
                    "Lane Width (m)": node.get("lane_width", ""),
                    "Traction Coefficient": node.get("traction", ""),
                })

    if not rows:
        print("    Warning: No route data to export")
        return None

    # Create DataFrame with column order matching template
    columns = [
        "Easting (m)",
        "Northing (m)",
        "Elevation (m)",
        "RouteIndex",
        "Segment",
        "Rolling Resistance (%)",
        "Speed Limit (kph)",
        "Load Zone",
        "Dump Zone",
        "Trolley",
        "Banking (%)",
        "Curvature (1/m)",
        "Lane Width (m)",
        "Traction Coefficient",
    ]
    df = pd.DataFrame(rows, columns=columns)

    # Replace empty strings with NaN for cleaner Excel output
    df = df.replace("", pd.NA)

    # Write to Excel with Route_Data sheet
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Route_Data", index=False)
        return output_path
    except Exception as e:
        print(f"    Error writing Excel file: {e}")
        return None


def find_connected_components(roads: List[Dict]) -> List[Set[int]]:
    """
    Find connected components in road network using BFS.

    Args:
        roads: List of road dictionaries with 'nodes' field

    Returns:
        List of sets, each set contains node_ids belonging to same component
    """
    # Build adjacency list (undirected graph)
    graph = defaultdict(set)
    for road in roads:
        road_nodes = road.get("nodes", [])
        for i in range(len(road_nodes) - 1):
            graph[road_nodes[i]].add(road_nodes[i + 1])
            graph[road_nodes[i + 1]].add(road_nodes[i])

    # BFS to find connected components
    all_node_ids = set(graph.keys())
    visited = set()
    components = []

    for node_id in all_node_ids:
        if node_id not in visited:
            component = set()
            queue = deque([node_id])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)
            components.append(component)

    return components


def find_center_nodes_for_haulers(
    nodes: List[Dict],
    roads: List[Dict],
    machine_first_positions: Dict[int, Tuple[float, float, float]],
) -> Dict[int, int]:
    """
    Find center node for each hauler based on connected components.
    Haulers in the same network component share the same center node.

    Args:
        nodes: List of node dictionaries
        roads: List of road dictionaries
        machine_first_positions: Dict mapping machine_id to (x, y, z) coordinates

    Returns:
        Dict mapping machine_id to center_node_id
    """
    if not nodes or not roads or not machine_first_positions:
        return {}

    # Find connected components
    components = find_connected_components(roads)
    if not components:
        return {}

    # Build node_id -> coords lookup
    node_coords = {n["id"]: (n["coords"][0], n["coords"][1]) for n in nodes}

    def find_nearest_node_in_set(x: float, y: float, node_set: Set[int]) -> Optional[int]:
        """Find nearest node to (x, y) within a specific node set."""
        min_dist = float('inf')
        nearest = None
        for nid in node_set:
            if nid not in node_coords:
                continue
            nx, ny = node_coords[nid]
            dist = (x - nx) ** 2 + (y - ny) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = nid
        return nearest

    # Group haulers by component
    component_haulers = defaultdict(list)  # component_idx -> [(machine_id, nearest_node_id)]

    for machine_id, (x, y, z) in machine_first_positions.items():
        nearest_node = None
        component_idx = None
        min_dist = float('inf')

        # Find which component this hauler belongs to
        for idx, component in enumerate(components):
            node_id = find_nearest_node_in_set(x, y, component)
            if node_id:
                nx, ny = node_coords[node_id]
                dist = (x - nx) ** 2 + (y - ny) ** 2
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node_id
                    component_idx = idx

        if component_idx is not None:
            component_haulers[component_idx].append((machine_id, nearest_node))

    # Find center node for each component that has haulers
    def find_component_center(component: Set[int], hauler_nodes: List[int]) -> Optional[int]:
        """Find node in component closest to centroid of hauler positions."""
        hauler_coords_list = [node_coords[nid] for nid in hauler_nodes if nid in node_coords]
        if not hauler_coords_list:
            return None

        # Calculate centroid of hauler positions
        center_x = sum(c[0] for c in hauler_coords_list) / len(hauler_coords_list)
        center_y = sum(c[1] for c in hauler_coords_list) / len(hauler_coords_list)

        # Find actual node in component closest to centroid
        min_dist = float('inf')
        center_node = None
        for nid in component:
            if nid not in node_coords:
                continue
            nx, ny = node_coords[nid]
            dist = (center_x - nx) ** 2 + (center_y - ny) ** 2
            if dist < min_dist:
                min_dist = dist
                center_node = nid

        return center_node

    # Build result: machine_id -> center_node_id
    result = {}
    for comp_idx, haulers in component_haulers.items():
        component = components[comp_idx]
        hauler_nodes = [node_id for _, node_id in haulers]
        center_node = find_component_center(component, hauler_nodes)

        for machine_id, _ in haulers:
            result[machine_id] = center_node

    return result


def create_service_stations_for_haulers(
    center_nodes: Dict[int, int],
    nodes: List[Dict],
    roads: List[Dict],
) -> Tuple[List[Dict], List[Dict], Dict[int, Tuple[int, int]]]:
    """
    Create service stations and fuel zones at center nodes for hauler initial positions.

    Args:
        center_nodes: Dict mapping machine_id to center_node_id
        nodes: List of node dictionaries
        roads: List of road dictionaries

    Returns:
        Tuple of:
        - List of service station dictionaries
        - List of charger (fuel zone) dictionaries
        - Dict mapping machine_id to (service_zone_id, service_zone_spot_id)
    """
    if not center_nodes:
        return [], [], {}

    # Get unique center nodes
    unique_centers = set(center_nodes.values())
    node_coords = {n["id"]: n["coords"] for n in nodes}

    # Build node_to_roads mapping
    node_to_roads = defaultdict(list)
    for road in roads:
        for nid in road.get("nodes", []):
            node_to_roads[nid].append(road["id"])

    service_stations = []
    chargers = []
    node_to_service = {}  # center_node_id -> service_zone_id

    for idx, center_node_id in enumerate(sorted(unique_centers), start=1):
        if center_node_id is None:
            continue

        coords = node_coords.get(center_node_id, [0, 0, 0])
        road_ids = node_to_roads.get(center_node_id, [])

        # Use first road for in/out if available
        inroad_ids = [road_ids[0]] if road_ids else []
        outroad_ids = [road_ids[0]] if road_ids else []

        # Create service station
        service_station = {
            "id": idx,
            "name": f"Service {idx}",
            "is_generated": True,
            "is_deactive": False,
            "is_show_service": True,
            "settings": {
                "n_spots": 2,  # Always 2 spots per service station
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
                "zonetype": "servicestandard",
                "inroad_ids": inroad_ids,
                "outroad_ids": outroad_ids,
                "innode_ids": [center_node_id],
                "outnode_ids": [center_node_id],
            },
        }
        service_stations.append(service_station)

        # Create charger (fuel zone) at same location
        charger = {
            "id": idx,
            "name": f"Fuel Zone {idx}",
            "type": "diesel",
            "output_power": "",
            "connect_time": "",
            "disconnect_time": "",
            "efficiency": "",
            "ramup_time": "",
            "cable_efficiency": "",
            "is_generated": True,
            "is_deactive": False,
            "power_factor": "",
            "power_factor_lagging": "",
            "fuel_rate": "",
            "settings": {
                "n_spots": 6,
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
                "zonetype": "amtinspectionbays",
                "additional_battery": 0,
                "inroad_ids": inroad_ids,
                "outroad_ids": outroad_ids,
                "innode_ids": [center_node_id],
                "outnode_ids": [center_node_id],
            },
        }
        chargers.append(charger)

        node_to_service[center_node_id] = idx

    # Build machine_id -> (service_zone_id, spot_id) mapping
    # Distribute haulers evenly between spot 1 and spot 2
    service_spot_counters = defaultdict(int)  # service_id -> current count (for alternating)
    hauler_to_service = {}

    for machine_id, center_node_id in center_nodes.items():
        if center_node_id is None:
            continue
        service_id = node_to_service.get(center_node_id)
        if service_id:
            # Alternate between spot 1 and 2
            service_spot_counters[service_id] += 1
            spot_id = 1 if service_spot_counters[service_id] % 2 == 1 else 2
            hauler_to_service[machine_id] = (service_id, spot_id)

    return service_stations, chargers, hauler_to_service


def create_model(
    nodes: List[Dict],
    roads: List[Dict],
    load_zones: List[Dict] = None,
    dump_zones: List[Dict] = None,
    version: str = "2.0.51",
    machines: Optional[Dict[int, Dict]] = None,
    machines_list: Optional[Dict[str, Any]] = None,
    machines_with_events: Optional[Set[int]] = None,
    telemetry_data: Optional[List[Tuple]] = None,
    coordinates_in_meters: bool = False,
) -> Dict:
    """
    Create complete model structure with full settings.

    Args:
        nodes: List of node dictionaries
        roads: List of road dictionaries
        load_zones: List of load zone dictionaries
        dump_zones: List of dump zone dictionaries
        version: Model version string
        machines: Machine info dictionary (from database or import)
        machines_list: Machine specifications by model name (from machines_list.json)
        machines_with_events: Set of machine IDs that have events data
        telemetry_data: Raw telemetry data for determining hauler initial positions
        coordinates_in_meters: Whether telemetry coordinates are in meters (True) or mm (False)

    Returns:
        Complete model dictionary
    """
    load_zones = load_zones or []
    dump_zones = dump_zones or []

    # Build machine_list from machines using machines_list.json data
    machine_list_haulers = []
    machine_list_loaders = []
    added_model_names = set()  # Track added models to avoid duplicates
    model_name_to_machine_list_id = {}  # Map model_name -> machine_list hauler id

    if machines and machines_list:
        for machine_id, machine_info in machines.items():
            # Skip machines without events if filter is provided
            if machines_with_events is not None and machine_id not in machines_with_events:
                continue

            type_name = machine_info.get("type_name", "Unknown")
            model_name = extract_machine_model(type_name)

            if model_name and model_name not in added_model_names:
                # Get spec from machines_list.json
                spec_data = get_machine_spec_from_list(type_name, machines_list)
                if spec_data:
                    hauler_data = deep_copy_dict(spec_data)
                    hauler_data["model_name"] = type_name
                    machine_list_haulers.append(hauler_data)
                    # Track mapping from model_name to machine_list hauler id
                    model_name_to_machine_list_id[model_name] = hauler_data.get("id", len(machine_list_haulers))
                    added_model_names.add(model_name)

    # Add default loader from machines_list.json (first loader available)
    default_loader_machine_list_id = None
    if machines_list:
        loaders_in_list = machines_list.get("machine_list", {}).get("loaders", [])
        if loaders_in_list:
            default_loader = deep_copy_dict(loaders_in_list[0])
            machine_list_loaders.append(default_loader)
            default_loader_machine_list_id = default_loader.get("id", 1)

    # Create routes and update zone settings to ensure connectivity
    routes = create_routes(load_zones, dump_zones, roads, nodes)
    update_zone_settings_for_routes(routes, load_zones, dump_zones, roads)

    model = {
        "version": version,
        "machine_list": {"haulers": machine_list_haulers, "loaders": machine_list_loaders},
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
                "type": "DET",
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
        "routes": routes,
        "haulers": [],
        "loaders": [],
        "simulates": [],
        "esses": [],
        "batteries": [],
        "crushers": [],
        "operations": create_operations_structure(
            routes, load_zones, dump_zones,
            telemetry_data=telemetry_data,
            coordinates_in_meters=coordinates_in_meters,
            schedule_name="Material Schedule 1",
            scheduling_method="grouped_assignment"
        ),
        "cameraPosition": {"x": 0, "y": 1000, "z": 0},
        "controlTarget": {"x": 0, "y": 0, "z": 0},
    }

    # Build haulers list from machines
    model_haulers = []
    if machines and model_name_to_machine_list_id:
        routes = model.get("routes", [])
        first_route_id = routes[0]["id"] if routes else None

        # Build lookup structures for determining initial positions
        node_lookup = {n["id"]: n for n in nodes}
        road_lookup = {r["id"]: r for r in roads}

        # Build mapping: node_id -> list of road_ids that contain this node
        node_to_roads = {}
        for road in roads:
            for nid in road.get("nodes", []):
                if nid not in node_to_roads:
                    node_to_roads[nid] = []
                node_to_roads[nid].append(road["id"])

        # Build mapping: road_id -> list of route_ids that use this road
        road_to_routes = {}
        for route in routes:
            for rid in route.get("haul", []) + route.get("return", []):
                if rid not in road_to_routes:
                    road_to_routes[rid] = []
                road_to_routes[rid].append(route["id"])

        # Group telemetry by machine_id to find first position
        machine_first_positions = {}
        if telemetry_data:
            for row in telemetry_data:
                mid = row[0]
                if mid not in machine_first_positions:
                    # First telemetry point for this machine
                    # row[4]=pathEasting, row[5]=pathNorthing, row[6]=pathElevation
                    if coordinates_in_meters:
                        x, y, z = row[4], row[5], row[6]
                    else:
                        x = row[4] / 1000.0
                        y = row[5] / 1000.0
                        z = row[6] / 1000.0
                    machine_first_positions[mid] = (x, y, z)

        # Find center nodes for each hauler based on connected road components
        # Filter positions to only include machines that will be processed
        filtered_positions = {}
        for machine_id, machine_info in machines.items():
            if machines_with_events is not None and machine_id not in machines_with_events:
                continue
            type_name = machine_info.get("type_name", "Unknown")
            model_name = extract_machine_model(type_name)
            if model_name_to_machine_list_id.get(model_name) is not None:
                if machine_id in machine_first_positions:
                    filtered_positions[machine_id] = machine_first_positions[machine_id]

        # Find center nodes and create service stations and fuel zones
        center_nodes = find_center_nodes_for_haulers(nodes, roads, filtered_positions)
        service_stations, chargers, hauler_to_service = create_service_stations_for_haulers(
            center_nodes, nodes, roads
        )

        # Add service stations and chargers to model
        model["service_stations"] = service_stations
        model["chargers"] = chargers

        hauler_id = 1
        for machine_id, machine_info in machines.items():
            # Skip machines without events if filter is provided
            if machines_with_events is not None and machine_id not in machines_with_events:
                continue

            type_name = machine_info.get("type_name", "Unknown")
            model_name = extract_machine_model(type_name)

            # Get machine_list hauler id for this model
            machine_list_id = model_name_to_machine_list_id.get(model_name)
            if machine_list_id is None:
                continue

            # Get service zone assignment for this hauler
            service_info = hauler_to_service.get(machine_id)

            # Get machine spec to determine type
            spec_data = get_machine_spec_from_list(type_name, machines_list) if machines_list else None
            machine_type = spec_data.get("type", "diesel") if spec_data else "diesel"
            is_electric = machine_type == "electric"

            # Use initial_position = 2 (service zone) if service zone is assigned
            if service_info:
                service_zone_id, service_zone_spot_id = service_info
                hauler = {
                    "id": hauler_id,
                    "group_id": hauler_id,
                    "key": "haulers",
                    "name": f"Hauler {hauler_id}",
                    "machine_id": machine_list_id,
                    "is_local_machine": None,
                    "geometry_name": "_default",
                    "model_scale": 1,
                    "type": machine_type,
                    "number_of_haulers": 1,
                    "lane": 2,
                    "initial_position": 2,  # 2 = service zone
                    "initial_level_pct": {
                        "type": "exact",
                        "value": 95
                    },
                    "initial_conditions": {
                        "route_id": None,
                        "road_id": None,
                        "node_id": None,
                        "service_zone_id": service_zone_id,
                        "service_zone_spot_id": service_zone_spot_id,
                        "load_zone_id": None,
                        "assigned_load_spots": []
                    },
                    "is_deactive": False
                }
            else:
                # Fallback to route-based initial position
                hauler = {
                    "id": hauler_id,
                    "group_id": hauler_id,
                    "key": "haulers",
                    "name": f"Hauler {hauler_id}",
                    "machine_id": machine_list_id,
                    "is_local_machine": None,
                    "geometry_name": "_default",
                    "model_scale": 1,
                    "type": machine_type,
                    "number_of_haulers": 1,
                    "lane": 2,
                    "initial_position": 1,  # 1 = on route
                    "initial_level_pct": {
                        "type": "exact",
                        "value": 95
                    },
                    "initial_conditions": {
                        "route_id": first_route_id,
                        "road_id": None,
                        "node_id": None,
                        "service_zone_id": None,
                        "service_zone_spot_id": None,
                        "load_zone_id": None,
                        "assigned_load_spots": []
                    },
                    "is_deactive": False
                }

            # Add type-specific fields
            if is_electric:
                hauler["battery_state_of_health"] = 90
                hauler["battery_capacity"] = spec_data.get("battery_size", 500) if spec_data else 500
            else:
                hauler["fuel_tank"] = spec_data.get("fuel_tank", 3785) if spec_data else 3785

            model_haulers.append(hauler)
            hauler_id += 1

    model["haulers"] = model_haulers

    # Build loaders list from load_zones (one loader per load_zone)
    model_loaders = []
    if load_zones and default_loader_machine_list_id is not None:
        # Get default loader spec for configured string
        default_loader_spec = machine_list_loaders[0] if machine_list_loaders else None
        loader_model_name = default_loader_spec.get("name", "Unknown") if default_loader_spec else "Unknown"
        loader_model_id = default_loader_spec.get("model_id", "") if default_loader_spec else ""

        loader_id = 1
        for lz in load_zones:
            lz_id = lz.get("id")
            lz_name = lz.get("name", f"Load zone {lz_id}")

            # Get number of spots from load zone settings
            lz_settings = lz.get("settings", {})
            n_spots = lz_settings.get("n_spots", 1)
            assigned_spots = list(range(1, n_spots + 1)) if n_spots > 0 else [1]

            loader = {
                "id": loader_id,
                "name": f"Loader {loader_id}",
                "key": "loaders",
                "machine_id": default_loader_machine_list_id,
                "configured": f"{loader_model_name} (ID: {loader_model_id})",
                "used_for": "Truck Loading",
                "fill_factor_pct": 1.0,
                "initial_charge_fuel_levels_pct": 95,
                "initial_conditions": {
                    "load_zone_id": lz_id,
                    "assigned_load_spots": assigned_spots
                },
                "is_deactive": False
            }

            model_loaders.append(loader)
            loader_id += 1

    model["loaders"] = model_loaders

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


def _create_des_operations(
    des_routes: List[Dict],
    des_load_zones: List[Dict],
    des_dump_zones: List[Dict],
    haulers: List[Dict] = None,
    model_load_zones: List[Dict] = None,
    model_dump_zones: List[Dict] = None,
    telemetry_data: List[Tuple] = None,
    coordinates_in_meters: bool = False,
) -> Dict:
    """
    Create operations structure for DES inputs.

    If telemetry_data is provided, analyzes actual hauler trips to determine
    which haulers traveled between which zones.

    Args:
        des_routes: List of DES route dictionaries (with start_zone/end_zone format)
        des_load_zones: List of DES load zone dictionaries
        des_dump_zones: List of DES dump zone dictionaries
        haulers: Optional list of hauler dictionaries
        model_load_zones: Model load zones with detected_location (for trip analysis)
        model_dump_zones: Model dump zones with detected_location (for trip analysis)
        telemetry_data: Optional telemetry data for actual trip analysis
        coordinates_in_meters: Whether telemetry coordinates are in meters

    Returns:
        Operations dictionary with material_schedules
    """
    # Try to analyze actual trips from telemetry data first
    material_data = []

    if telemetry_data and model_load_zones and model_dump_zones:
        trips_by_machine = analyze_hauler_trips_from_telemetry(
            telemetry_data, model_load_zones, model_dump_zones, coordinates_in_meters
        )
        if trips_by_machine:
            material_data = create_material_schedule_from_trips(trips_by_machine)

    # Fallback: generate from DES routes
    if not material_data:
        # Build zone name lookups
        lz_lookup = {z["id"]: z.get("name", f"Load zone {z['id']}") for z in des_load_zones}
        dz_lookup = {z["id"]: z.get("name", f"Dump zone {z['id']}") for z in des_dump_zones}

        # Count haulers per route if available
        haulers_per_route = {}
        if haulers:
            for hauler in haulers:
                route_id = hauler.get("initial_conditions", {}).get("route_id")
                if route_id is not None:
                    haulers_per_route[route_id] = haulers_per_route.get(route_id, 0) + hauler.get("number_of_haulers", 1)

        # Default haulers per route
        total_haulers = sum(haulers_per_route.values()) if haulers_per_route else len(des_routes) * 4
        default_haulers = max(1, total_haulers // len(des_routes)) if des_routes else 4

        for idx, route in enumerate(des_routes, start=1):
            # Extract zone IDs from DES route format
            lz_id = route.get("start_zone", {}).get("id")
            dz_id = route.get("end_zone", {}).get("id")

            lz_name = lz_lookup.get(lz_id, f"Load zone {lz_id}")
            dz_name = dz_lookup.get(dz_id, f"Dump zone {dz_id}")
            route_name = route.get("name", "")

            # Get hauler count for this route
            num_haulers = haulers_per_route.get(route.get("id"), default_haulers)

            item = {
                "id": idx,
                "load_zone": lz_name,
                "dump_zone": dz_name,
                "route": route_name,
                "auto_generate_route": True,
                "material": "Ore",
                "density": 1960.19,
                "num_of_hauler": num_haulers,
                "assigned_machine_type": "Hauler",
                "multiple_routes": False,
                "hauler_group_id": 1,
            }
            material_data.append(item)

    return {
        "material_schedules": {
            "selected_material": 1,
            "all_material_schedule": [
                {
                    "id": 1,
                    "name": "Material Schedule 1",
                    "hauler_assignment": {"scheduling_method": "grouped_assignment"},
                    "mixed_fleet_based_initial_assignment": False,
                    "data": material_data,
                }
            ],
        },
        "operational_delays": {
            "haulers": [],
            "trolleys": [],
            "load_zones": [],
            "dump_zones": [],
        },
    }


def create_des_inputs(
    model: Dict,
    machines: Dict[int, Dict],
    site_name: str,
    sim_time: int = 480,
    machines_with_events: Optional[Set[int]] = None,
    machine_templates: Optional[Dict[str, Any]] = None,
    telemetry_data: Optional[List[Tuple]] = None,
    coordinates_in_meters: bool = False,
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
        telemetry_data: Optional telemetry data for actual trip analysis
        coordinates_in_meters: Whether telemetry coordinates are in meters

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

    # Create routes from load zones to dump zones using pathfinding
    # Use model routes as base and add DES-specific fields
    model_routes = model.get("routes", [])
    des_routes = []
    route_uid_counter = 1000

    # Get main road IDs as fallback
    main_road_ids = [road["id"] for road in roads]

    if model_routes:
        # Use routes from model (already computed with proper pathfinding)
        for model_route in model_routes:
            route = {
                "id": model_route["id"],
                "name": model_route["name"],
                "haul": model_route["haul"],
                "return": model_route["return"],
                "start_zone": {
                    "id": model_route["load_zone"],
                    "type": "lz",
                    "uid": route_uid_counter,
                },
                "end_zone": {
                    "id": model_route["dump_zone"],
                    "type": "dz",
                    "uid": route_uid_counter + 1,
                },
                "used_by_current_MMP": True,
                "production": True,
                "uid": route_uid_counter + 2,
            }
            des_routes.append(route)
            route_uid_counter += 3
    else:
        # Fallback: create routes for all load_zone-dump_zone pairs
        route_id = 1
        for lz in des_load_zones:
            lz_id = lz["id"]
            lz_name = lz["name"]

            for dz in des_dump_zones:
                dz_id = dz["id"]
                dz_name = dz["name"]

                route = {
                    "id": route_id,
                    "name": f"{lz_name} to {dz_name}",
                    "haul": list(main_road_ids),
                    "return": list(reversed(main_road_ids)),
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
                "type": "DET",
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
        "operations": _create_des_operations(
            des_routes, des_load_zones, des_dump_zones, haulers,
            model_load_zones=load_zones,
            model_dump_zones=dump_zones,
            telemetry_data=telemetry_data,
            coordinates_in_meters=coordinates_in_meters,
        ),
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
    machines_list: Optional[Dict[str, Any]] = None,
    telemetry_data: Optional[List[Tuple]] = None,
    coordinates_in_meters: Optional[bool] = None,
    precomputed_zones: Optional[List] = None,
    output_base_name: Optional[str] = None,
    export_model: bool = True,
    export_simulation: bool = True,
    export_routes_excel: bool = False,
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
        machines_list: Machine specifications by model name (e.g., "793F" -> spec data)
        telemetry_data: Optional pre-fetched telemetry data (list of tuples)
        coordinates_in_meters: If True, coordinates are in meters (import flow).
                              If False, coordinates are in millimeters (database flow).
                              If None, will be determined based on data source.
        precomputed_zones: Optional list of Reader.Zone objects from parse_cp1_data.
                          If provided, uses these instead of detect_zones().
        output_base_name: Optional base name for output files (e.g., "ABC" -> "ABC_model.json").
                         If not provided, uses site_name.
        export_model: If True, generate model.json file.
        export_simulation: If True, generate des_inputs.json.gz and ledger.json.gz files.
        export_routes_excel: If True, generate routes.xlsx file (Route_Template format).

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

    # Split roads at intersections and overlaps
    # This ensures roads only share nodes at endpoints
    roads, road_composition = split_roads_at_intersections(roads)
    print(f"    After splitting: {len(roads)} road segments")

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

    # Create model with machine_list from machines_list.json
    model = create_model(
        nodes, roads, load_zones, dump_zones,
        machines=machines,
        machines_list=machines_list,
        telemetry_data=telemetry_data,
        coordinates_in_meters=coordinates_in_meters,
    )

    # Generate events and DES inputs (only if export_simulation is True)
    all_events = []
    des_inputs = {}
    machines_with_events: Set[int] = set()

    if export_simulation:
        print("  [4/5] Generating simulation events...")
        converter = GPSToEventsConverter(model_data=model)

        machine_data = {}
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
            model, machines, site_name, sim_time, machines_with_events, machine_templates,
            telemetry_data=telemetry_data,
            coordinates_in_meters=coordinates_in_meters,
        )
    else:
        print("  [4/5] Skipping simulation events (export_simulation=False)")
        print("  [5/5] Skipping DES inputs (export_simulation=False)")
    
    # Save files based on export options
    # Use output_base_name if provided, otherwise use site_name
    file_base = output_base_name if output_base_name else site_name
    safe_name = file_base.replace(" ", "_").replace("/", "_").replace("\\", "_")
    os.makedirs(output_dir, exist_ok=True)

    result = {}
    print(f"\n  Output files saved to: {output_dir}", flush=True)

    # Save model (if export_model is True)
    if export_model:
        model_path = os.path.join(output_dir, f"{safe_name}_model.json")
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)
        result["model"] = model_path
        print(f"    - Model: {safe_name}_model.json ({len(nodes)} nodes, {len(roads)} roads)", flush=True)

    # Export route data to Excel (Route_Template format) - requires model data
    if export_routes_excel:
        routes = model.get("routes", [])
        route_excel_path = os.path.join(output_dir, f"{safe_name}_routes.xlsx")
        excel_result = export_route_excel(
            nodes, roads, load_zones, dump_zones, routes, route_excel_path
        )
        if excel_result:
            result["routes_excel"] = excel_result
            print(f"    - Routes Excel: {safe_name}_routes.xlsx ({len(routes)} routes)", flush=True)

    # Save simulation files (if export_simulation is True)
    if export_simulation:
        # Save DES inputs (gzip compressed)
        des_inputs_path = os.path.join(output_dir, f"{safe_name}_des_inputs.json.gz")
        with gzip.open(des_inputs_path, "wb") as f:
            f.write(json.dumps(des_inputs, indent=2).encode("utf-8"))
        result["des_inputs"] = des_inputs_path
        print(f"    - DES Inputs: {safe_name}_des_inputs.json.gz ({len(des_inputs['haulers'])} haulers)", flush=True)

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
        # Save events ledger (gzip compressed)
        ledger_path = os.path.join(output_dir, f"{safe_name}_ledger.json.gz")
        with gzip.open(ledger_path, "wb") as f:
            f.write(json.dumps(events_output, indent=2, default=str).encode("utf-8"))
        result["ledger"] = ledger_path
        print(f"    - Events Ledger: {safe_name}_ledger.json.gz ({len(all_events)} events)", flush=True)

    print(f"\n  [process_site] Returning result: {list(result.keys())}", flush=True)
    return result


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
        machines_list = load_machines_list()

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
                machines_list=machines_list,
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
