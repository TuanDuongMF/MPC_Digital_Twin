# AMT Cycle Productivity Message Reader - Application Description

## Overview

The AMT Cycle Productivity Message Reader is a web application that processes raw mining telemetry data from Autonomous Mining Trucks (AMT) and generates simulation-ready outputs. The application provides a **unified Parse workflow** that:

1. Generates spatial model data from MSSM (MineStar System Management) databases
2. Creates DES (Discrete Event Simulation) inputs from the model and machine data
3. Produces an events ledger from raw gateway message files

### Primary Functions

1. **Parse Simulation Data**: Single workflow that combines model generation, DES inputs creation, and events ledger generation
2. **Model Generation**: Extract roads, zones, and fleet configuration from MSSM SQL Server via `roads_network_pipeline`
3. **DES Inputs**: Build simulation configuration (nodes, roads, zones, haulers, loaders) from model and MySQL machine data
4. **Events Ledger**: Convert raw gateway files (`.gwm`, `.dat`, `.bin`) into chronological event sequences for animation playback
5. **Zone Detection**: Automatically identify load and dump zones from machine movement patterns
6. **Road Network Generation**: Create road network models from telemetry path data

## Technical Architecture

### System Architecture

```
┌─────────────────────┐
│   React Frontend     │  (Port 3000)
│   ParseSimulationForm│
│   - Fidelity select  │
│   - ZIP upload       │
│   - Download buttons │
└──────────┬──────────┘
           │ HTTP/REST API
           │
┌──────────▼──────────┐
│   Flask Backend      │  (Port 5000)
│   POST /api/simulation/parse
│   GET  /api/simulation/download/<file_type>
│   GET  /api/model/download/<file_type>
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼───┐  ┌─────▼─────┐
│ MySQL │  │ MSSM SQL  │
│  DB   │  │  Server   │
└───────┘  └───────────┘
```

### Technology Stack

**Backend:**
- **Framework**: Flask (Python 3.9+)
- **Databases**: MySQL (PyMySQL) for telemetry/machines; MSSM SQL Server for spatial model
- **Data Processing**: Pandas, NumPy, Shapely, scikit-learn
- **External Parser**: GWMReader.exe (binary executable for gateway message parsing)

**Frontend:**
- **Framework**: React 18
- **UI Library**: Bootstrap 5
- **State Management**: React Hooks (useState, useRef)

## Project Structure

```
Simulation/
├── backend/
│   ├── app.py                    # Flask app, API endpoints
│   ├── config.py                 # Centralized configuration (env vars)
│   ├── core/
│   │   ├── gateway_parser_wrapper.py   # GWMReader.exe wrapper
│   │   ├── gateway_data_converter.py   # Parser output → telemetry, zones
│   │   ├── Reader.py, Cycle.py, Segment.py, Zone.py, Routes.py
│   │   └── constants.py
│   ├── roads_network_pipeline/   # MSSM model generation
│   │   ├── run.py                # run_pipeline(fidelity)
│   │   ├── extract.py, preprocess.py, tsm.py
│   │   ├── roads.py, zones.py, fleet.py, productivity.py
│   │   └── ...
│   ├── scripts/
│   │   └── simulation_generator.py    # process_site, create_des_inputs_from_model_file
│   └── simulation_analysis/      # Event conversion
│       ├── gps_to_events_converter.py
│       └── convert_telemetry.py
├── frontend/
│   └── src/
│       ├── App.js
│       └── components/
│           └── ParseSimulationForm.js   # Main UI
└── docs/
    ├── des_inputs_specification.md
    ├── event_generation_algorithm.md
    ├── events_structure_specification.md
    ├── model-structure.md
    └── model_generation_algorithm.md
```

## Parse Workflow (3 Steps)

The application exposes a single **Parse Data** workflow:

### Step 1: Generate Model from MSSM

- Uses `roads_network_pipeline.run_pipeline(fidelity=fidelity)`
- Connects to MSSM SQL Server (credentials from `MSSM_*` env vars)
- Stages: Extract → Preprocess → TSM → Roads → Zones → Fleet → Productivity
- Output: `model.json` (nodes, roads, load zones, dump zones, fleet)

### Step 2: Create DES Inputs

- Loads `model.json` from Step 1
- Fetches machines from MySQL (all machines, no site filter)
- Calls `create_des_inputs_from_model_file()` to build DES configuration
- Output: `simulation_des_inputs.json.gz`

### Step 3: Generate Events Ledger

- Saves uploaded ZIP to temp directory, extracts files
- Runs GWMReader.exe via `parse_gateway_files()`
- Converts parser output to telemetry, extracts zones
- Calls `process_site()` with telemetry and precomputed zones
- Output: `simulation_ledger.json.gz`

All outputs are written to `OUTPUT_PATH` (no per-site subdirectories).

## API Endpoints

### POST `/api/simulation/parse`

Unified parsing pipeline. Runs all 3 steps sequentially.

**Request:** `multipart/form-data`
- `fidelity`: string (optional, default `"Low"`) — Model fidelity for roads_network_pipeline
- `file`: ZIP file with raw gateway data (`.gwm`, `.dat`, `.bin`) — **required**

**Response (200):**
```json
{
  "success": true,
  "output_path": "/path/to/output",
  "stages": {"extract": true, "preprocess": true, ...},
  "elapsed_seconds": 45.2,
  "files": {
    "model": "model.json",
    "des_inputs": "simulation_des_inputs.json.gz",
    "ledger": "simulation_ledger.json.gz"
  }
}
```

**Errors:** 400 (missing file, invalid ZIP), 500 (pipeline/DB failure)

### GET `/api/simulation/download/<file_type>`

Download files generated by `/api/simulation/parse`.

**Parameters:** `file_type` — one of `model`, `des_inputs`, `ledger`

**Files:**
- `model` → `model.json`
- `des_inputs` → `simulation_des_inputs.json.gz`
- `ledger` → `simulation_ledger.json.gz`

### GET `/api/model/download/<file_type>`

Download model file from MSSM pipeline output.

**Parameters:** `file_type` — `model` only → `model.json`

### GET `/api/health`

Health check. Response: `{"status": "ok"}`

## Data Flow

### Parse Flow

```
User selects Fidelity + uploads ZIP
    ↓
POST /api/simulation/parse
    ↓
Step 1: run_pipeline(fidelity) → model.json
    ↓
Step 2: create_des_inputs_from_model_file(model.json, machines) → simulation_des_inputs.json.gz
    ↓
Step 3: parse_gateway_files(ZIP) → telemetry + zones
         process_site(telemetry, zones) → simulation_ledger.json.gz
    ↓
Response: {success, files: {model, des_inputs, ledger}}
    ↓
User downloads via /api/simulation/download/<file_type>
```

## Configuration

### Environment Variables

All configuration via `.env` in `backend/`:

**MySQL (telemetry, machines):**
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`, `DB_CHARSET`
- `DB_SSL_CA`, `DB_SSL_CERT`, `DB_SSL_KEY` (optional, for SSL connections)

**MSSM SQL Server (model generation):**
- `MSSM_SERVER`, `MSSM_USER`, `MSSM_PASSWORD`, `MSSM_DATABASE`

**Paths:**
- `OUTPUT_PATH`: Output directory for model, DES inputs, ledger (required)
- `EXECUTE_FILE_PATH`: Path to GWMReader.exe (required for Step 3)
- `EXAMPLE_JSON_PATH`: Path to exampleJSON (machine templates, etc.)
- `TEMP_DIR`: Optional temp directory for ZIP extraction (short path recommended on Windows)

## Key Components

### Backend

- **app.py**: Flask routes, `parse_simulation()` orchestrates 3-step pipeline
- **roads_network_pipeline/run.py**: `run_pipeline(fidelity)` — MSSM → model.json
- **simulation_generator.py**: `process_site()`, `create_des_inputs_from_model_file()`, `fetch_machines()`
- **gateway_parser_wrapper.py**: `parse_gateway_files()` — wraps GWMReader.exe
- **gateway_data_converter.py**: `process_parser_output()`, `convert_imported_records_to_telemetry()`, `extract_zones_from_import()`

### Frontend

- **ParseSimulationForm.js**: Fidelity dropdown, ZIP drag-and-drop upload, Parse Data button, download buttons (Model, DES Inputs, Events Ledger)

## Business Logic (Unchanged)

Cycle detection, segment classification, zone detection, road network generation, loss analysis, and event generation logic remain as described in the original documentation. The main change is the **unified Parse workflow** replacing separate Import/Export flows.

## Security

- File names sanitized with `secure_filename()`
- Temp directories use random prefixes and are cleaned after processing
- Parser executable path validated before execution
