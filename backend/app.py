"""
Flask RESTful API for AMT Cycle Productivity Message Reader WebApp

Endpoints:
- POST /api/simulation/parse - Unified parse pipeline (model + DES inputs + events ledger)
- GET  /api/simulation/download/<file_type> - Download model, des_inputs, or ledger
- GET  /api/model/download/<file_type> - Download model.json
- GET  /api/health - Health check
"""

import io
import os
import sys
import json
import time
import threading
import zipfile
import shutil
import tempfile
import uuid
from pathlib import Path
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add webapp directory to path for imports
# webapp/backend/app.py -> webapp/
backend_dir = os.path.dirname(os.path.abspath(__file__))
webapp_root = os.path.dirname(backend_dir)
sys.path.insert(0, webapp_root)

from backend.config import OUTPUT_PATH, EXECUTE_FILE_PATH, EXAMPLE_JSON_PATH, TEMP_DIR
from backend.core.gateway_parser_wrapper import parse_gateway_files
from backend.core.gateway_data_converter import process_parser_output, convert_imported_records_to_telemetry, extract_zones_from_import
from backend.scripts.simulation_generator import (
    get_connection,
    fetch_machines,
    process_site,
    load_machine_templates,
    load_machines_list,
    DEFAULT_CONFIG,
    create_des_inputs_from_model_file,
)
from backend.roads_network_pipeline.run import run_pipeline

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

# Machines list path (contains machine specs by model name)
MACHINES_LIST_PATH = os.path.join(
    EXAMPLE_JSON_PATH,
    "machines_list.json"
)

# In-memory job store for /api/simulation/parse progress tracking
# Note: resets on server restart.
_parse_jobs = {}
_parse_jobs_lock = threading.Lock()


def _job_update(job_id: str, **patch):
    with _parse_jobs_lock:
        job = _parse_jobs.get(job_id)
        if not job:
            return
        job.update(patch)


def _job_log(job_id: str, message: str):
    now = time.strftime("%H:%M:%S")
    line = f"[{now}] {message}"
    with _parse_jobs_lock:
        job = _parse_jobs.get(job_id)
        if not job:
            return
        logs = job.setdefault("logs", [])
        logs.append(line)
        if len(logs) > 300:
            del logs[:100]


def _run_parse_job(job_id: str, fidelity: str, uploaded_zip_path: str, uploaded_zip_name: str):
    try:
        _job_update(job_id, status="running", progress=1, stage="init", error=None, files={})
        _job_log(job_id, f"Job started (fidelity={fidelity}).")

        # Step 1: model
        _job_update(job_id, stage="model", progress=5)
        _job_log(job_id, "Step 1/3: Generating model.json from MSSM...")
        pipeline_result = run_pipeline(fidelity=fidelity)
        output_path = pipeline_result.get("output_path")
        if not output_path:
            raise RuntimeError("Pipeline did not return output_path")

        model_path = os.path.join(output_path, "model.json")
        if not os.path.exists(model_path):
            raise RuntimeError("model.json not found after pipeline run")

        _job_update(job_id, progress=35)
        _job_log(job_id, "model.json generated.")

        # Step 2: des inputs
        _job_update(job_id, stage="des_inputs", progress=40)
        _job_log(job_id, "Step 2/3: Creating DES inputs from model + machines DB...")
        connection = get_connection()
        if not connection:
            raise RuntimeError("Failed to connect to MySQL database")
        try:
            cursor = connection.cursor()
            machines = fetch_machines(cursor)
        finally:
            connection.close()

        if not machines:
            raise RuntimeError("No machines found in database")

        machine_templates = load_machine_templates(MACHINE_TEMPLATES_PATH)
        machines_list = load_machines_list(MACHINES_LIST_PATH)
        des_inputs = create_des_inputs_from_model_file(
            model_path=model_path,
            machines=machines,
            site_name="Simulation",
            sim_time=DEFAULT_CONFIG["simulation"]["sim_time"],
            machines_with_events=None,
            machine_templates={"machines_list": machines_list},
            telemetry_data=None,
            coordinates_in_meters=True,
        )

        safe_name = "simulation"
        des_inputs_filename = f"{safe_name}_des_inputs.json.gz"
        des_inputs_path = os.path.join(output_path, des_inputs_filename)
        import gzip

        with gzip.open(des_inputs_path, "wb") as f:
            f.write(json.dumps(des_inputs, indent=2).encode("utf-8"))

        _job_update(job_id, progress=60)
        _job_log(job_id, f"DES inputs written: {des_inputs_filename}")

        # Step 3: ledger
        _job_update(job_id, stage="ledger", progress=65)
        _job_log(job_id, "Step 3/3: Parsing raw ZIP and generating events ledger...")

        if not EXECUTE_FILE_PATH or not os.path.exists(EXECUTE_FILE_PATH):
            raise RuntimeError("Parser executable not found. Please configure EXECUTE_FILE_PATH in .env")

        temp_base = TEMP_DIR if TEMP_DIR else None
        if temp_base and not os.path.exists(temp_base):
            os.makedirs(temp_base, exist_ok=True)
        temp_upload_dir = tempfile.mkdtemp(prefix="parse_", dir=temp_base)

        ledger_filename = None
        try:
            if not uploaded_zip_name.lower().endswith(".zip"):
                raise ValueError("Only ZIP files are allowed")

            zip_name = secure_filename(uploaded_zip_name) or "raw.zip"
            zip_path = os.path.join(temp_upload_dir, zip_name)
            _job_log(job_id, f"Staging ZIP: {zip_name}")
            shutil.copy2(uploaded_zip_path, zip_path)

            extract_dir = os.path.join(temp_upload_dir, "e0")
            os.makedirs(extract_dir, exist_ok=True)
            file_paths = []

            _job_log(job_id, "Extracting ZIP...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                valid_files = [
                    zi
                    for zi in zip_ref.infolist()
                    if not zi.is_dir() and not zi.filename.lower().endswith(".zip")
                ]

                for idx, zip_info in enumerate(valid_files):
                    orig_name = os.path.basename(zip_info.filename)
                    _, ext = os.path.splitext(orig_name)
                    short_name = f"f{idx}{ext}"
                    target_path = os.path.join(extract_dir, short_name)
                    with zip_ref.open(zip_info) as src, open(target_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    file_paths.append(target_path)

            if not file_paths:
                raise ValueError("No valid files found in ZIP")

            _job_update(job_id, progress=75)
            _job_log(job_id, f"Running gateway parser on {len(file_paths)} files...")
            parse_result = parse_gateway_files(
                site_name="Simulation",
                file_paths=file_paths,
                parser_exe_path=EXECUTE_FILE_PATH,
                temp_base_dir=temp_base,
            )

            if "error" in parse_result:
                raise RuntimeError(parse_result.get("error") or "Gateway parser failed")

            _job_update(job_id, progress=82)
            _job_log(job_id, "Converting parser output to telemetry...")
            records = process_parser_output(parse_result)
            sample_interval = DEFAULT_CONFIG["data_fetching"]["sample_interval"]
            telemetry_data = convert_imported_records_to_telemetry(
                parse_result,
                records,
                sample_interval=sample_interval,
            )

            _job_log(job_id, "Extracting zones from import...")
            _, zones = extract_zones_from_import(parse_result)

            _job_update(job_id, progress=90)
            _job_log(job_id, "Generating ledger via process_site()...")
            result = process_site(
                cursor=None,
                site_name="Simulation",
                machines=machines,
                output_dir=output_path,
                limit=DEFAULT_CONFIG["data_fetching"]["limit"],
                sample_interval=sample_interval,
                grid_size=DEFAULT_CONFIG["road_detection"]["grid_size"],
                min_density=DEFAULT_CONFIG["road_detection"]["min_density"],
                simplify_epsilon=DEFAULT_CONFIG["road_detection"]["simplify_epsilon"],
                max_node_distance=DEFAULT_CONFIG["road_detection"]["max_node_distance"],
                merge_tolerance=DEFAULT_CONFIG["road_detection"]["merge_tolerance"],
                zone_grid_size=DEFAULT_CONFIG["zone_detection"]["grid_size"],
                zone_min_stops=DEFAULT_CONFIG["zone_detection"]["min_stop_count"],
                sim_time=DEFAULT_CONFIG["simulation"]["sim_time"],
                machine_templates=machine_templates,
                machines_list=machines_list,
                telemetry_data=telemetry_data,
                coordinates_in_meters=True,
                precomputed_zones=zones if zones else None,
                output_base_name=safe_name,
                export_model=False,
                export_simulation=True,
                export_routes_excel=False,
            )

            if result and "ledger" in result and os.path.exists(result["ledger"]):
                ledger_filename = os.path.basename(result["ledger"])
                _job_log(job_id, f"Ledger written: {ledger_filename}")
            else:
                _job_log(job_id, "Ledger not produced by process_site().")

        finally:
            try:
                shutil.rmtree(temp_upload_dir, ignore_errors=True)
            except Exception:
                pass
            try:
                if uploaded_zip_path and os.path.exists(uploaded_zip_path):
                    os.remove(uploaded_zip_path)
            except Exception:
                pass
            try:
                staged_parent = os.path.dirname(uploaded_zip_path) if uploaded_zip_path else None
                if staged_parent and os.path.isdir(staged_parent):
                    shutil.rmtree(staged_parent, ignore_errors=True)
            except Exception:
                pass

        stages = pipeline_result.get("stages", {})
        elapsed = pipeline_result.get("elapsed")
        files = {
            "model": "model.json",
            "des_inputs": des_inputs_filename,
            "ledger": ledger_filename,
        }

        _job_update(
            job_id,
            status="completed",
            stage="done",
            progress=100,
            output_path=output_path,
            stages=stages,
            elapsed_seconds=elapsed,
            files=files,
        )
        _job_log(job_id, "Job completed.")

    except Exception as e:
        _job_update(job_id, status="error", stage="error", progress=100, error=str(e))
        _job_log(job_id, f"ERROR: {e}")



@app.route('/api/simulation/parse', methods=['POST'])
def parse_simulation():
    """
    Unified simulation parsing pipeline:
    1. Generate model.json from MSSM databases (roads_network_pipeline).
    2. Generate DES inputs from the generated model and machines from DB.
    3. Generate events ledger from raw telemetry.

    Request: multipart/form-data with fields:
      - fidelity: string (optional, default "Low")
      - file: ZIP with raw gateway data (REQUIRED).
    """
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "Raw ZIP file is required"}), 400

        fidelity = request.form.get("fidelity", "Low")

        upload_file = request.files["file"]
        if not upload_file.filename or not upload_file.filename.lower().endswith(".zip"):
            return jsonify({"error": "Only ZIP files are allowed"}), 400

        job_id = uuid.uuid4().hex
        temp_base = TEMP_DIR if TEMP_DIR else None
        if temp_base and not os.path.exists(temp_base):
            os.makedirs(temp_base, exist_ok=True)
        staged_dir = tempfile.mkdtemp(prefix="upload_", dir=temp_base)
        staged_zip_name = secure_filename(upload_file.filename) or "raw.zip"
        staged_zip_path = os.path.join(staged_dir, staged_zip_name)
        upload_file.save(staged_zip_path)

        with _parse_jobs_lock:
            _parse_jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "stage": "queued",
                "progress": 0,
                "error": None,
                "files": {},
                "logs": [],
                "created_at": time.time(),
            }
        _job_log(job_id, "API received request. Job queued.")
        _job_log(job_id, f"Staged upload: {staged_zip_name}")

        thread = threading.Thread(
            target=_run_parse_job,
            args=(job_id, fidelity, staged_zip_path, staged_zip_name),
            daemon=True,
        )
        thread.start()

        return jsonify({"success": True, "job_id": job_id}), 202

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/simulation/parse/status/<job_id>', methods=['GET'])
def parse_simulation_status(job_id: str):
    with _parse_jobs_lock:
        job = _parse_jobs.get(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        resp = {k: v for k, v in job.items() if k != "created_at"}
    return jsonify(resp), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"}), 200


@app.route('/api/model/download/<file_type>', methods=['GET'])
def download_model_file(file_type: str):
    """
    Download model-related files generated from MSSM pipeline.

    Files are written directly under OUTPUT_PATH (no subdirectories).

    Supported file types:
        - model  -> model.json
    """
    try:
        valid_types = ['model']
        if file_type not in valid_types:
            return jsonify({"error": f"Invalid file type. Must be one of: {valid_types}"}), 400

        # Files are written directly under OUTPUT_PATH (aligned with run_pipeline)
        if not OUTPUT_PATH:
            return jsonify({"error": "OUTPUT_PATH is not configured on server"}), 500

        filename = "model.json"
        file_path = os.path.join(OUTPUT_PATH, filename)

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found on server"}), 404

        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype="application/json",
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/simulation/download/<file_type>', methods=['GET'])
def download_simulation_file(file_type: str):
    """
    Download files generated by /api/simulation/parse.

    Files are written directly under OUTPUT_PATH (no subdirectories).

    Supported file types:
        - model      -> model.json
        - des_inputs -> simulation_des_inputs.json.gz
        - ledger     -> simulation_ledger.json.gz (if available)
    """
    try:
        valid_types = ['model', 'des_inputs', 'ledger']
        if file_type not in valid_types:
            return jsonify({"error": f"Invalid file type. Must be one of: {valid_types}"}), 400

        if not OUTPUT_PATH:
            return jsonify({"error": "OUTPUT_PATH is not configured on server"}), 500

        # Filenames are global under OUTPUT_PATH; route parameters are ignored.
        safe_name = "simulation"

        if file_type == 'model':
            filename = "model.json"
        elif file_type == 'des_inputs':
            filename = f"{safe_name}_des_inputs.json.gz"
        else:  # ledger
            filename = f"{safe_name}_ledger.json.gz"

        file_path = os.path.join(OUTPUT_PATH, filename)

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found on server"}), 404

        if filename.endswith(".gz"):
            mimetype = "application/gzip"
        else:
            mimetype = "application/json"

        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype=mimetype,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
