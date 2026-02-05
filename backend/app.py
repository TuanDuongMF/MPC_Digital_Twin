"""
Flask RESTful API for AMT Cycle Productivity Message Reader WebApp

Endpoints:
- GET /api/sites - Get list of available sites
- POST /api/export - Export model/simulation files for a site
- POST /api/import - Import and parse raw data files
"""

import os
import sys
import json
import threading
import zipfile
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from queue import Queue
from flask import Flask, jsonify, request, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Add webapp directory to path for imports
# webapp/backend/app.py -> webapp/
backend_dir = os.path.dirname(os.path.abspath(__file__))
webapp_root = os.path.dirname(backend_dir)
sys.path.insert(0, webapp_root)

from backend.core.db_config import DB_CONFIG, OUTPUT_PATH, EXECUTE_FILE_PATH, EXAMPLE_JSON_PATH, TEMP_DIR
from backend.core.gateway_parser_wrapper import parse_gateway_files
from backend.core.gateway_data_converter import process_parser_output, convert_imported_records_to_telemetry, extract_zones_from_import
from backend.scripts.simulation_generator import (
    get_connection,
    fetch_sites,
    fetch_machines,
    process_site,
    load_machine_templates,
    DEFAULT_CONFIG,
)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
# Resolve paths relative to backend directory or use absolute paths
def resolve_path(path_var, default_relative):
    """Resolve path: if relative, make it relative to backend_dir; if absolute, use as-is."""
    if os.path.isabs(path_var):
        return path_var
    return os.path.join(backend_dir, path_var)

OUTPUT_DIR = resolve_path(OUTPUT_PATH, "../output")
EXECUTE_FILE_PATH = resolve_path(EXECUTE_FILE_PATH, "../executables")
EXAMPLE_JSON_PATH = resolve_path(EXAMPLE_JSON_PATH, "../exampleJSON")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Upload directory for import files
UPLOAD_DIR = os.path.join(backend_dir, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Maximum upload size (5GB)
MAX_UPLOAD_SIZE = 5 * 1024 * 1024 * 1024

# Machine templates path
MACHINE_TEMPLATES_PATH = os.path.join(
    EXAMPLE_JSON_PATH,
    "simulation", "machine_templates.json"
)

# Store export status
export_status = {}

# Store import status (for import + export flow)
import_status = {}

# SSE event queues for real-time updates (site_name -> list of queues)
sse_subscribers: Dict[str, List[Queue]] = {}


def get_export_status(site_name: str) -> dict:
    """Get export status for a site."""
    return export_status.get(site_name, {
        "status": "idle",
        "progress": 0,
        "message": "",
        "files": {}
    })


def set_export_status(site_name: str, status: str, progress: int = 0, message: str = "", files: dict = None):
    """Set export status for a site and notify SSE subscribers."""
    data = {
        "status": status,
        "progress": progress,
        "message": message,
        "files": files or {}
    }
    export_status[site_name] = data
    # Notify SSE subscribers
    _notify_sse(site_name, "export", data)


def get_import_status(site_name: str) -> dict:
    """Get import+export status for a site."""
    return import_status.get(site_name, {
        "status": "idle",
        "progress": 0,
        "message": "",
        "files": {}
    })


def set_import_status(site_name: str, status: str, progress: int = 0, message: str = "", files: dict = None):
    """Set import+export status for a site and notify SSE subscribers."""
    data = {
        "status": status,
        "progress": progress,
        "message": message,
        "files": files or {}
    }
    import_status[site_name] = data
    # Notify SSE subscribers
    _notify_sse(site_name, "import", data)


def _notify_sse(site_name: str, event_type: str, data: dict):
    """Push event to all SSE subscribers for a site."""
    key = f"{event_type}:{site_name}"
    print(f"[SSE] Notifying {key}, status={data.get('status')}, subscribers={len(sse_subscribers.get(key, []))}", flush=True)
    if key in sse_subscribers:
        event_data = json.dumps(data)
        dead_queues = []
        for q in sse_subscribers[key]:
            try:
                q.put_nowait(event_data)
                print(f"[SSE] Event pushed to queue successfully", flush=True)
            except Exception as e:
                print(f"[SSE] Failed to push to queue: {e}", flush=True)
                dead_queues.append(q)
        # Remove dead queues
        for q in dead_queues:
            sse_subscribers[key].remove(q)
    else:
        print(f"[SSE] No subscribers for {key}", flush=True)


def process_import_and_export(
    site_name: str,
    parse_result: Dict[str, Any],
    records: List[Dict[str, Any]],
    config: Dict[str, Any],
):
    """Process import and export in background thread."""
    try:
        set_import_status(site_name, "processing", 10, "Converting imported data to telemetry format...")
        
        # Convert imported records to telemetry tuple format
        sample_interval = config.get('sample_interval', DEFAULT_CONFIG['data_fetching']['sample_interval'])
        telemetry_data = convert_imported_records_to_telemetry(
            parse_result,
            records,
            sample_interval=sample_interval
        )
        
        if not telemetry_data:
            set_import_status(site_name, "error", 0, "Failed to convert imported data")
            return
        
        set_import_status(site_name, "processing", 20, "Extracting zones using Reader.py algorithms...")

        # Extract zones using standard Reader.py algorithms (Segment classification + DBSCAN)
        cycles, zones = extract_zones_from_import(parse_result)

        set_import_status(site_name, "processing", 30, "Preparing machine information...")

        # Create machine info from telemetry data
        machines = {}
        unique_machine_ids = set(row[0] for row in telemetry_data)
        for machine_id in unique_machine_ids:
            machines[machine_id] = {
                "machine_unique_id": machine_id,
                "name": f"Machine_{machine_id}",
                "site_name": site_name,
                "type_name": "Unknown"
            }

        # Load machine templates
        machine_templates = load_machine_templates(MACHINE_TEMPLATES_PATH)

        # Process site with imported telemetry data and precomputed zones
        set_import_status(site_name, "processing", 40, "Processing site data and generating simulation files...")
        print(f"\n[Import] Starting process_site for {site_name}...", flush=True)
        result = process_site(
            cursor=None,  # No database cursor needed for imported data
            site_name=site_name,
            machines=machines,
            output_dir=OUTPUT_DIR,
            limit=config.get('limit', DEFAULT_CONFIG['data_fetching']['limit']),
            sample_interval=sample_interval,
            grid_size=config.get('grid_size', DEFAULT_CONFIG['road_detection']['grid_size']),
            min_density=config.get('min_density', DEFAULT_CONFIG['road_detection']['min_density']),
            simplify_epsilon=config.get('simplify_epsilon', DEFAULT_CONFIG['road_detection']['simplify_epsilon']),
            zone_grid_size=config.get('zone_grid_size', DEFAULT_CONFIG['zone_detection']['grid_size']),
            zone_min_stops=config.get('zone_min_stops', DEFAULT_CONFIG['zone_detection']['min_stop_count']),
            sim_time=config.get('sim_time', DEFAULT_CONFIG['simulation']['sim_time']),
            machine_templates=machine_templates,
            telemetry_data=telemetry_data,
            coordinates_in_meters=True,  # Import data has coordinates in meters
            precomputed_zones=zones if zones else None,
        )

        print(f"\n[Import] process_site returned: {result is not None}", flush=True)
        if result:
            # Convert absolute paths to relative filenames
            files = {}
            for key, path in result.items():
                if os.path.exists(path):
                    files[key] = os.path.basename(path)

            print(f"[Import] Setting status to completed with files: {files}", flush=True)
            set_import_status(
                site_name,
                "completed",
                100,
                "Import and export completed successfully",
                files
            )
            print(f"[Import] Status set to completed successfully!", flush=True)
        else:
            print(f"[Import] process_site returned None or empty result", flush=True)
            set_import_status(site_name, "error", 0, "Failed to generate files")

    except Exception as e:
        import traceback
        print(f"[Import] Exception occurred: {str(e)}", flush=True)
        traceback.print_exc()
        set_import_status(site_name, "error", 0, f"Error: {str(e)}")


@app.route('/api/sites', methods=['GET'])
def get_sites():
    """Get list of available sites."""
    try:
        connection = get_connection()
        if not connection:
            return jsonify({"error": "Failed to connect to database"}), 500
        
        try:
            cursor = connection.cursor()
            sites = fetch_sites(cursor)
            return jsonify({"sites": sites}), 200
        finally:
            connection.close()
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/export', methods=['POST'])
def export_files():
    """Export model/simulation files for a site."""
    try:
        data = request.get_json()
        site_name = data.get('site_name')
        
        if not site_name:
            return jsonify({"error": "site_name is required"}), 400
        
        # Check if export is already in progress
        status = get_export_status(site_name)
        if status["status"] == "processing":
            return jsonify({
                "error": "Export already in progress",
                "status": status
            }), 409
        
        # Get configuration from request or use defaults
        config = data.get('config', {})
        limit = config.get('limit', DEFAULT_CONFIG['data_fetching']['limit'])
        sample_interval = config.get('sample_interval', DEFAULT_CONFIG['data_fetching']['sample_interval'])
        grid_size = config.get('grid_size', DEFAULT_CONFIG['road_detection']['grid_size'])
        min_density = config.get('min_density', DEFAULT_CONFIG['road_detection']['min_density'])
        simplify_epsilon = config.get('simplify_epsilon', DEFAULT_CONFIG['road_detection']['simplify_epsilon'])
        zone_grid_size = config.get('zone_grid_size', DEFAULT_CONFIG['zone_detection']['grid_size'])
        zone_min_stops = config.get('zone_min_stops', DEFAULT_CONFIG['zone_detection']['min_stop_count'])
        sim_time = config.get('sim_time', DEFAULT_CONFIG['simulation']['sim_time'])
        
        # Start export in background thread
        thread = threading.Thread(
            target=process_export,
            args=(
                site_name,
                limit,
                sample_interval,
                grid_size,
                min_density,
                simplify_epsilon,
                zone_grid_size,
                zone_min_stops,
                sim_time,
            )
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "Export started",
            "site_name": site_name
        }), 202
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def process_export(
    site_name: str,
    limit: int,
    sample_interval: int,
    grid_size: float,
    min_density: int,
    simplify_epsilon: float,
    zone_grid_size: float,
    zone_min_stops: int,
    sim_time: int,
):
    """Process export in background thread."""
    try:
        set_export_status(site_name, "processing", 0, "Connecting to database...")
        
        connection = get_connection()
        if not connection:
            set_export_status(site_name, "error", 0, "Failed to connect to database")
            return
        
        try:
            cursor = connection.cursor()
            
            # Fetch machines
            set_export_status(site_name, "processing", 10, "Fetching machine information...")
            machines = fetch_machines(cursor, site_name)
            
            if not machines:
                set_export_status(site_name, "error", 0, f"No machines found for site: {site_name}")
                return
            
            # Load machine templates
            machine_templates = load_machine_templates(MACHINE_TEMPLATES_PATH)
            
            # Process site
            set_export_status(site_name, "processing", 20, "Processing site data...")
            result = process_site(
                cursor=cursor,
                site_name=site_name,
                machines=machines,
                output_dir=OUTPUT_DIR,
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
                # Convert absolute paths to relative filenames
                files = {}
                for key, path in result.items():
                    if os.path.exists(path):
                        files[key] = os.path.basename(path)
                
                set_export_status(
                    site_name,
                    "completed",
                    100,
                    "Export completed successfully",
                    files
                )
            else:
                set_export_status(site_name, "error", 0, "Failed to generate files")
                
        finally:
            connection.close()
            
    except Exception as e:
        set_export_status(site_name, "error", 0, f"Error: {str(e)}")


@app.route('/api/export/status/<site_name>', methods=['GET'])
def get_export_status_endpoint(site_name: str):
    """Get export status for a site."""
    status = get_export_status(site_name)
    return jsonify(status), 200


@app.route('/api/export/events/<site_name>', methods=['GET'])
def export_events_sse(site_name: str):
    """SSE endpoint for real-time export status updates."""
    def event_stream():
        q = Queue()
        key = f"export:{site_name}"
        if key not in sse_subscribers:
            sse_subscribers[key] = []
        sse_subscribers[key].append(q)

        # Send current status immediately
        current = get_export_status(site_name)
        yield f"data: {json.dumps(current)}\n\n"

        try:
            while True:
                # Timeout 600s (10 minutes) for long-running exports
                data = q.get(timeout=600)
                yield f"data: {data}\n\n"
                # Stop if completed or error
                parsed = json.loads(data)
                if parsed.get("status") in ("completed", "error"):
                    break
        except Exception:
            pass
        finally:
            if key in sse_subscribers and q in sse_subscribers[key]:
                sse_subscribers[key].remove(q)

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/api/export/download/<site_name>/<file_type>', methods=['GET'])
def download_file(site_name: str, file_type: str):
    """Download exported file."""
    try:
        # Validate file type
        valid_types = ['model', 'des_inputs', 'ledger']
        if file_type not in valid_types:
            return jsonify({"error": f"Invalid file type. Must be one of: {valid_types}"}), 400
        
        # Get status to find filename
        status = get_export_status(site_name)
        if status["status"] != "completed":
            return jsonify({"error": "Export not completed"}), 400
        
        filename = status["files"].get(file_type)
        if not filename:
            return jsonify({"error": f"File {file_type} not found"}), 404
        
        file_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found on server"}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/json'
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/import', methods=['POST'])
def import_files():
    """
    Import and parse raw data files.
    
    Accepts:
    - Single file upload
    - Folder upload (as zip file)
    - Multiple files
    
    Query parameters:
    - export: If "true", will also export simulation files after import
    
    Returns parsed JSON data, or starts export process if export=true.
    """
    try:
        # Check if files are in request
        if 'files' not in request.files:
            return jsonify({"error": "No files provided"}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({"error": "No files selected"}), 400
        
        # Get site_name from form data (optional, default to "DefaultSite")
        site_name = request.form.get('site_name', 'DefaultSite')
        
        # Check if export is requested
        export_after_import = request.form.get('export', 'false').lower() == 'true'
        
        # Get export configuration from form data (optional)
        config = {}
        if export_after_import:
            config = {
                'limit': int(request.form.get('limit', DEFAULT_CONFIG['data_fetching']['limit'])),
                'sample_interval': int(request.form.get('sample_interval', DEFAULT_CONFIG['data_fetching']['sample_interval'])),
                'grid_size': float(request.form.get('grid_size', DEFAULT_CONFIG['road_detection']['grid_size'])),
                'min_density': int(request.form.get('min_density', DEFAULT_CONFIG['road_detection']['min_density'])),
                'simplify_epsilon': float(request.form.get('simplify_epsilon', DEFAULT_CONFIG['road_detection']['simplify_epsilon'])),
                'zone_grid_size': float(request.form.get('zone_grid_size', DEFAULT_CONFIG['zone_detection']['grid_size'])),
                'zone_min_stops': int(request.form.get('zone_min_stops', DEFAULT_CONFIG['zone_detection']['min_stop_count'])),
                'sim_time': int(request.form.get('sim_time', DEFAULT_CONFIG['simulation']['sim_time'])),
            }
        
        # Validate parser executable
        if not EXECUTE_FILE_PATH or not os.path.exists(EXECUTE_FILE_PATH):
            return jsonify({
                "error": "Parser executable not found. Please configure EXECUTE_FILE_PATH in .env"
            }), 500
        
        # Create temporary directory for uploaded files
        # Use TEMP_DIR from env or system temp to avoid Windows MAX_PATH (260) limit
        temp_base = TEMP_DIR if TEMP_DIR else None
        if temp_base and not os.path.exists(temp_base):
            os.makedirs(temp_base, exist_ok=True)
        temp_upload_dir = tempfile.mkdtemp(prefix='imp_', dir=temp_base)
        file_paths = []
        
        try:
            # Process uploaded files
            for file in files:
                if file.filename == '':
                    continue
                
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_upload_dir, filename)
                
                # Save file in chunks for large files
                with open(file_path, 'wb') as f:
                    # Read in chunks of 64MB
                    chunk_size = 64 * 1024 * 1024
                    while True:
                        chunk = file.stream.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                
                # Check if it's a zip file
                if filename.lower().endswith('.zip'):
                    # Extract zip file - use short directory name to avoid Windows MAX_PATH (260) limit
                    extract_dir = os.path.join(temp_upload_dir, f"e{len(file_paths)}")
                    os.makedirs(extract_dir, exist_ok=True)

                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            # Count valid files for progress bar
                            valid_files = [
                                zi for zi in zip_ref.infolist()
                                if not zi.is_dir() and not zi.filename.lower().endswith('.zip')
                            ]
                            total_files = len(valid_files)
                            print(f"\n[ZIP] Extracting {total_files} files from {filename}")

                            # Extract files one by one with short names to avoid MAX_PATH limit
                            file_idx = 0
                            for zip_info in tqdm(valid_files, desc="Extracting", unit="file"):
                                # Get original extension
                                orig_name = os.path.basename(zip_info.filename)
                                _, ext = os.path.splitext(orig_name)

                                # Create short filename: f0.dat, f1.dat, etc.
                                short_name = f"f{file_idx}{ext}"
                                target_path = os.path.join(extract_dir, short_name)

                                # Extract file content and write to short path
                                with zip_ref.open(zip_info) as src, open(target_path, 'wb') as dst:
                                    shutil.copyfileobj(src, dst)

                                file_paths.append(target_path)
                                file_idx += 1
                    except zipfile.BadZipFile:
                        return jsonify({"error": f"Invalid zip file: {filename}"}), 400
                else:
                    file_paths.append(file_path)
            
            if not file_paths:
                return jsonify({"error": "No valid files found"}), 400
            
            # Parse files using gateway parser
            # Pass temp_base to avoid Windows MAX_PATH (260) limit
            parse_result = parse_gateway_files(
                site_name=site_name,
                file_paths=file_paths,
                parser_exe_path=EXECUTE_FILE_PATH,
                temp_base_dir=temp_base
            )
            
            # Check if parsing was successful
            if "error" in parse_result:
                return jsonify({
                    "success": False,
                    "error": parse_result.get("error"),
                    "details": parse_result.get("details", [])
                }), 400
            
            # Process parser output using same logic as parse_gateway_messages.py
            # Returns list of dicts with database column names
            records = process_parser_output(parse_result)
            
            if export_after_import:
                # Start export in background thread
                thread = threading.Thread(
                    target=process_import_and_export,
                    args=(
                        site_name,
                        parse_result,
                        records,
                        config,
                    )
                )
                thread.daemon = True
                thread.start()
                
                return jsonify({
                    "success": True,
                    "message": "Import completed, export started",
                    "site_name": site_name,
                    "files_processed": len(file_paths),
                    "records_count": len(records),
                    "export_status": "processing"
                }), 202
            
            # Prepare response matching parse_gateway_messages.py output format
            response_data = {
                "success": True,
                "site_name": site_name,
                "files_processed": len(file_paths),
                "records_count": len(records),
                "records": records
            }
            
            return jsonify(response_data), 200
            
        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(temp_upload_dir, ignore_errors=True)
            except Exception:
                pass
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/import/status/<site_name>', methods=['GET'])
def get_import_status_endpoint(site_name: str):
    """Get import+export status for a site."""
    status = get_import_status(site_name)
    return jsonify(status), 200


@app.route('/api/import/events/<site_name>', methods=['GET'])
def import_events_sse(site_name: str):
    """SSE endpoint for real-time import status updates."""
    def event_stream():
        q = Queue()
        key = f"import:{site_name}"
        if key not in sse_subscribers:
            sse_subscribers[key] = []
        sse_subscribers[key].append(q)

        # Send current status immediately
        current = get_import_status(site_name)
        yield f"data: {json.dumps(current)}\n\n"

        try:
            while True:
                # Timeout 600s (10 minutes) for long-running imports
                data = q.get(timeout=600)
                yield f"data: {data}\n\n"
                # Stop if completed or error
                parsed = json.loads(data)
                if parsed.get("status") in ("completed", "error"):
                    break
        except Exception:
            pass
        finally:
            if key in sse_subscribers and q in sse_subscribers[key]:
                sse_subscribers[key].remove(q)

    return Response(event_stream(), mimetype='text/event-stream')


@app.route('/api/import/download/<site_name>/<file_type>', methods=['GET'])
def download_imported_file(site_name: str, file_type: str):
    """Download exported file from import."""
    try:
        # Validate file type
        valid_types = ['model', 'des_inputs', 'ledger']
        if file_type not in valid_types:
            return jsonify({"error": f"Invalid file type. Must be one of: {valid_types}"}), 400
        
        # Get status to find filename
        status = get_import_status(site_name)
        if status["status"] != "completed":
            return jsonify({"error": "Import/export not completed"}), 400
        
        filename = status["files"].get(file_type)
        if not filename:
            return jsonify({"error": f"File {file_type} not found"}), 404
        
        file_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found on server"}), 404
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/json'
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
