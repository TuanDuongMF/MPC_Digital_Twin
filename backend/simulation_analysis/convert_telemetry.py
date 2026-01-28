"""
Convert Telemetry to Simulation Events

Command-line tool to convert AMT telemetry data from database to simulation events
for animation playback.

Usage:
    python convert_telemetry.py --model model.json --site "SiteName" --output events.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add webapp root to path for imports
# backend/simulation_analysis/convert_telemetry.py -> backend/ -> webapp/
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

from backend.core.db_config import DB_CONFIG
from backend.simulation_analysis import GPSToEventsConverter


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
    machine_id: Optional[int] = None,
    machine_ids: Optional[List[int]] = None,
    limit: int = 50000,
    sample_interval: int = 5,
) -> List[Tuple]:
    """
    Fetch telemetry data from database.
    
    Returns list of tuples:
    (machine_id, segment_id, cycle_id, interval, pathEasting, pathNorthing,
     pathElevation, expectedSpeed, actualSpeed, pathBank, pathHeading,
     leftWidth, rightWidth, payloadPercent)
    """
    print("    Step 1: Fetching segment metadata...")
    
    meta_query = """
        SELECT `Machine Unique Id`, segmentId, cycleId, cycleProdInfoHandle
        FROM amt_cycleprodinfo
    """
    params = []
    where_clauses = []
    
    if machine_id is not None:
        where_clauses.append("`Machine Unique Id` = %s")
        params.append(machine_id)
    elif machine_ids is not None and len(machine_ids) > 0:
        placeholders = ",".join(["%s"] * len(machine_ids))
        where_clauses.append(f"`Machine Unique Id` IN ({placeholders})")
        params.extend(machine_ids)
    
    if where_clauses:
        meta_query += " WHERE " + " AND ".join(where_clauses)
    
    meta_query += " ORDER BY `Machine Unique Id`, segmentId"
    meta_query += f" LIMIT {limit // 10}"
    
    cursor.execute(meta_query, params)
    metadata = cursor.fetchall()
    
    if not metadata:
        print("    No metadata found")
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
    
    print("    Step 2: Fetching telemetry points...")
    
    results = []
    batch_size = 100
    total_batches = (len(handles) + batch_size - 1) // batch_size
    
    batch_iterator = range(0, len(handles), batch_size)
    if TQDM_AVAILABLE:
        batch_iterator = tqdm(
            batch_iterator,
            desc="      Fetching batches",
            total=total_batches,
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
    
    # Sort by machine_id, segment_id, interval
    results.sort(key=lambda x: (x[0], x[1], x[3]))
    
    return results[:limit]


def process_site(
    cursor,
    site_name: str,
    machines: Dict[int, Dict],
    model_path: str,
    output_dir: str,
    limit: int = 50000,
    sample_interval: int = 5,
    min_node_distance: float = 15.0,
) -> Optional[str]:
    """
    Process telemetry data for a single site and generate events file.
    
    Returns output file path if successful.
    """
    # Get machine IDs for this site
    machine_ids = [
        m["machine_unique_id"]
        for m in machines.values()
        if m.get("site_name") == site_name
    ]
    
    if not machine_ids:
        print(f"  No machines found for site: {site_name}")
        return None
    
    print(f"  Processing site: {site_name} ({len(machine_ids)} machines)")
    
    # Load converter with model
    print("  Loading model...")
    converter = GPSToEventsConverter(model_path=model_path)
    
    if not converter.model:
        print(f"  Error: Could not load model from {model_path}")
        return None
    
    # Fetch telemetry data
    telemetry_data = fetch_telemetry_data(
        cursor,
        machine_ids=machine_ids,
        limit=limit,
        sample_interval=sample_interval,
    )
    
    if not telemetry_data:
        print(f"  No telemetry data found for site: {site_name}")
        return None
    
    print(f"  Fetched {len(telemetry_data):,} telemetry records")
    
    # Group by machine
    machine_data = {}
    for row in telemetry_data:
        mid = row[0]
        if mid not in machine_data:
            machine_data[mid] = []
        machine_data[mid].append(row)
    
    # Convert each machine's data
    all_events = []
    machine_iterator = machine_data.items()
    if TQDM_AVAILABLE:
        machine_iterator = tqdm(
            list(machine_iterator),
            desc="  Converting machines",
            unit="machine",
        )
    
    for machine_id, data in machine_iterator:
        machine_info = machines.get(machine_id, {})
        machine_name = machine_info.get("name", f"Machine_{machine_id}")
        
        events = converter.convert_raw_telemetry(
            data,
            machine_id=machine_id,
            machine_name=machine_name,
            min_node_distance=min_node_distance,
        )
        
        all_events.extend(events)
        converter.reset()
    
    # Sort events by time
    all_events.sort(key=lambda e: (e.get("time", 0), e.get("eid", 0)))
    
    # Renumber event IDs
    for i, event in enumerate(all_events):
        event["eid"] = i + 1
    
    # Save output
    safe_name = site_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    output_filename = f"events_{safe_name}.json"
    output_path = os.path.join(output_dir, output_filename)
    
    converter.save_events(all_events, output_path, include_summary=True)
    
    print(f"  Saved: {output_filename} ({len(all_events)} events)")
    return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert AMT telemetry data to simulation events"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to road network model JSON file",
    )
    parser.add_argument(
        "--site",
        type=str,
        default=None,
        help="Site name to process (required unless --list-sites)",
    )
    parser.add_argument(
        "--list-sites",
        action="store_true",
        help="List available sites and exit",
    )
    parser.add_argument(
        "--machine-id",
        type=int,
        default=None,
        help="Process specific machine ID only",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Maximum telemetry records to fetch (default: 50000)",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="Sample every Nth record (default: 5)",
    )
    parser.add_argument(
        "--min-node-distance",
        type=float,
        default=15.0,
        help="Minimum distance between node events in meters (default: 15.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: events_<site>.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: script directory)",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AMT TELEMETRY TO SIMULATION EVENTS CONVERTER")
    print("=" * 60)
    print(f"Database: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print(f"Model: {args.model}")
    
    # Validate model exists
    if not os.path.exists(args.model):
        print(f"\nError: Model file not found: {args.model}")
        return
    
    # Connect to database
    print("\n[1/4] Connecting to database...")
    connection = get_connection()
    if not connection:
        print("Failed to connect to database")
        return
    
    try:
        cursor = connection.cursor()
        
        # Fetch sites
        print("[2/4] Fetching site information...")
        sites = fetch_sites(cursor)
        print(f"  Found {len(sites)} sites")
        
        if args.list_sites:
            print("\nAvailable sites:")
            print("-" * 40)
            for site in sites:
                short = site["site_short"] or "N/A"
                print(f"  {site['site_name']} ({short})")
            print("-" * 40)
            return
        
        if not args.site:
            print("\nError: --site is required. Use --list-sites to see available sites.")
            return
        
        # Check site exists
        site_names = [s["site_name"] for s in sites]
        if args.site not in site_names:
            print(f"\nError: Site '{args.site}' not found.")
            print("Available sites:", ", ".join(site_names))
            return
        
        # Fetch machines
        print("[3/4] Fetching machine information...")
        machines = fetch_machines(cursor, args.site)
        print(f"  Found {len(machines)} machines for site")
        
        # Determine output directory
        output_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
        os.makedirs(output_dir, exist_ok=True)
        
        # Process site
        print("[4/4] Converting telemetry to events...")
        result = process_site(
            cursor,
            args.site,
            machines,
            args.model,
            output_dir,
            limit=args.limit,
            sample_interval=args.sample_interval,
            min_node_distance=args.min_node_distance,
        )
        
        if result:
            print("\n" + "=" * 60)
            print("CONVERSION COMPLETE")
            print("=" * 60)
            print(f"Output saved to: {result}")
            print("=" * 60)
        else:
            print("\nConversion failed.")
    
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        connection.close()
        print("\nDatabase connection closed")


if __name__ == "__main__":
    main()
