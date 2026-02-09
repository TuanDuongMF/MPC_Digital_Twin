# AMT Cycle Productivity Message Reader - Application Description

## Overview

The AMT Cycle Productivity Message Reader is a web application designed to process, analyze, and export mining telemetry data from Autonomous Mining Trucks (AMT). The application transforms raw gateway message files into structured simulation data, enabling productivity analysis, cycle optimization, and discrete event simulation (DES) modeling.

### Primary Functions

1. **Data Import**: Parse raw gateway message files (`.gwm`, `.dat`, `.bin`) into structured telemetry data
2. **Data Export**: Generate simulation-ready files (model, DES inputs, events ledger) from database or imported data
3. **Cycle Analysis**: Extract and analyze machine cycles (dump-to-dump operations) with productivity metrics
4. **Zone Detection**: Automatically identify load and dump zones from machine movement patterns
5. **Road Network Generation**: Create road network models from telemetry path data
6. **Loss Analysis**: Calculate productivity losses and efficiency metrics based on ASLR (Autonomous Speed Limit Reason) codes

## Technical Architecture

### System Architecture

The application follows a **client-server architecture** with clear separation of concerns:

```
┌─────────────────┐
│   React Frontend │  (Port 3000)
│   - UI Components│
│   - State Mgmt   │
└────────┬─────────┘
         │ HTTP/REST API
         │
┌────────▼─────────┐
│  Flask Backend    │  (Port 5000)
│  - REST API      │
│  - Business Logic│
│  - Data Processing│
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
┌───▼───┐ ┌──▼────┐
│ MySQL │ │ File │
│  DB   │ │System│
└───────┘ └──────┘
```

### Technology Stack

**Backend:**
- **Framework**: Flask (Python 3.9+)
- **Database**: MySQL (via PyMySQL)
- **Data Processing**: Pandas, NumPy, Shapely, scikit-learn
- **External Parser**: GWMReader.exe (binary executable for gateway message parsing)

**Frontend:**
- **Framework**: React 18
- **UI Library**: Bootstrap 5
- **State Management**: React Hooks (useState, useEffect)
- **HTTP Client**: Fetch API

**Data Processing Libraries:**
- **Spatial Analysis**: Shapely (geometric operations), DBSCAN (clustering)
- **Data Manipulation**: Pandas (DataFrames), NumPy (numerical operations)
- **Time Handling**: Python datetime, GPS epoch calculations

## Project Structure

```
webapp/
├── backend/                    # Flask REST API and processing logic
│   ├── app.py                  # Main Flask application and API endpoints
│   ├── core/                   # Core AMT processing modules
│   │   ├── Reader.py           # Main data parser (CP1/CP2 format)
│   │   ├── Cycle.py            # Cycle object and business logic
│   │   ├── Segment.py          # Segment classification and analysis
│   │   ├── Zone.py             # Zone detection and clustering
│   │   ├── Routes.py           # Route and lap analysis
│   │   ├── AMTCycleProdInfoMessage.py  # Message data structure
│   │   ├── gateway_parser_wrapper.py   # GWMReader.exe wrapper
│   │   ├── gateway_data_converter.py  # Data format conversion
│   │   ├── db_config.py        # Database configuration
│   │   └── constants.py        # Enums and constants
│   ├── scripts/                # Standalone processing scripts
│   │   ├── simulation_generator.py    # Main export logic
│   │   └── config.json         # Configuration file
│   └── simulation_analysis/     # Event conversion modules
│       ├── gps_to_events_converter.py
│       └── event_generator.py
├── frontend/                   # React application
│   ├── src/
│   │   ├── App.js              # Main application component
│   │   ├── components/
│   │   │   ├── SiteList.js      # Site selection component
│   │   │   ├── ExportButton.js # Export functionality
│   │   │   └── ImportButton.js # Import functionality
│   │   └── index.js            # React entry point
│   └── package.json
└── docs/                       # Documentation
    └── import_api.md           # Import API documentation
```

## Core Business Logic

### 1. Data Processing Pipeline

#### Import Flow

```
Raw Gateway Files (.gwm/.dat/.bin)
    ↓
GWMReader.exe Parser
    ↓
JSON Output (CycleProdInfo structure)
    ↓
process_parser_output() → List[Dict] (DB column format)
    ↓
convert_imported_records_to_telemetry() → List[Tuple] (telemetry format)
    ↓
AMTCycleProdInfoReader.parse_cp1_data() → (Cycles, Zones)
```

#### Export Flow

```
Database Telemetry OR Imported Telemetry
    ↓
process_site()
    ↓
├── Road Detection (grid-based clustering)
├── Zone Detection (stop-based clustering)
├── Cycle Analysis (segment classification)
└── Event Generation (GPS to events conversion)
    ↓
Output Files:
├── model.json (road network, nodes, zones)
├── des_inputs.json (simulation configuration)
└── ledger.json (events timeline)
```

### 2. Cycle Detection Logic

**Business Rules:**

1. **Cycle Definition**: A cycle represents a complete dump-to-dump operation
   - Starts when machine leaves dump zone (empty state)
   - Ends when machine returns to dump zone (after loading)

2. **Cycle Identification (CP1 Format)**:
   - Cycles are grouped by `segmentId` (GPS timestamp)
   - Contiguous messages with same `segmentId` form a segment
   - Segments are classified based on payload percentage and next segment payload
   - Cycle ends when segment type transitions from loaded to empty (payload ≤ 50%)

3. **Cycle Identification (CP2 Format)**:
   - Cycles are explicitly identified by `cycleId` field
   - Messages grouped by `cycleId` form complete cycles
   - Simpler classification: empty segment (payload ≤ 50%) followed by loaded segment

4. **Full Cycle Validation**:
   - A cycle is considered "full" only if both `dumpZoneStart` and `dumpZoneEnd` are identified
   - Incomplete cycles (missing zones) are marked as `isFullCycle = False`

### 3. Segment Classification

**Segment Types:**
- `SPOTTING_AT_SOURCE`: Machine reversing with payload transition from empty (≤50%) to loaded (>50%)
- `SPOTTING_AT_SINK`: Machine reversing with payload transition from loaded (>50%) to empty (≤50%)
- `TRAVELLING_EMPTY`: Machine moving with payload ≤ 50%
- `TRAVELLING_FULL`: Machine moving with payload > 50%

**Classification Rules:**

1. **Payload Threshold**: 50% is the critical threshold separating empty and loaded states
2. **Reversing Detection**: Machine is considered reversing if `actualSpeed > 0` and `expectedSpeed > 0` (initially `isReversing = True`)
3. **Next Payload Analysis**: Segment type depends on:
   - Current segment payload
   - Next segment payload (if available)
   - Reversing state
4. **Cycle End Detection**: Cycle ends when:
   - Segment type is `SPOTTING_AT_SINK` (for CP1)
   - Payload transitions from loaded to empty (for CP2)

### 4. Zone Detection

**Load Zone Detection:**
- Extracts GPS points from segments with type `SPOTTING_AT_SOURCE`
- Uses DBSCAN clustering (epsilon = 50 meters) to group points
- Creates convex hull polygons around clustered points
- Minimum 2 points required to form a zone polygon

**Dump Zone Detection:**
- Extracts GPS points from segments with type `SPOTTING_AT_SINK`
- Same clustering and polygon generation logic as load zones

**Business Rules:**
- Zone points are collected from "spotting" segments (reversing at load/dump locations)
- Noise points (DBSCAN label = -1) are filtered out
- Zones with fewer than 2 unique points use a 5-meter buffer around the point
- Zone polygons represent operational areas where machines load or dump material

### 5. Road Network Generation

**Road Detection Algorithm:**

1. **Grid-Based Clustering**:
   - Telemetry points are binned into a grid (default: 5m × 5m cells)
   - Cells with density ≥ `min_density` (default: 3 points) are considered road cells
   - Road cells are converted to nodes

2. **Road Simplification**:
   - Douglas-Peucker algorithm (epsilon = 5.0m) simplifies road paths
   - Reduces computational complexity while preserving road shape

3. **Node and Road Creation**:
   - Nodes represent road intersections and key points
   - Roads connect nodes based on machine travel patterns
   - Road properties include: distance, average efficiency, loss summary

4. **Road Intersection Splitting**:
   - Roads can only share nodes at start or end points
   - If roads share nodes in the middle, they are split at intersection points
   - Overlapping segments are deduplicated and marked with "_Shared" suffix
   - Routes are updated to reference new segment IDs

### 6. Loss Analysis

**Productivity Loss Calculation:**

1. **Loss Bucket Creation**:
   - Segments are divided into "loss buckets" based on ASLR (Autonomous Speed Limit Reason) changes
   - Each bucket represents a continuous period with the same ASLR code
   - Bucket loss = `actualTimeTaken - expectedTimeTaken` (in seconds)

2. **Efficiency Calculation**:
   - Segment efficiency = `(expectedTimeTaken / actualTimeTaken) * 100`
   - Cycle efficiency = weighted average of segment efficiencies
   - Route efficiency = average of all lap efficiencies

3. **Loss Summary Aggregation**:
   - Loss summaries are aggregated by ASLR reason code
   - Summaries track: total loss, count, actual/expected time, efficiency
   - Aggregation occurs at segment → cycle → route levels

**ASLR Reason Categories:**
- **Operations-Office**: Assignment limits, bed down areas
- **Machine Limited**: Base machine speed limits, power limitations
- **Non-Diagnostic Limited**: A-Stop, path avoidance areas
- **Diagnostic**: Health events, system diagnostics

### 7. Event Generation

**GPS to Events Conversion:**

The application converts GPS telemetry data into discrete events for simulation:

1. **State Detection**: Identifies machine states (TRAVEL_LOADED, TRAVEL_UNLOADED, LOADING, DUMPING, etc.)
2. **Event Creation**: Generates state transition events with timestamps
3. **Location Mapping**: Maps GPS coordinates to road network nodes and zones
4. **Event Ledger**: Creates chronological sequence of events for simulation playback

## Data Flow

### Import Data Flow

```
User uploads files → Flask receives multipart/form-data
    ↓
Files saved to temp directory
    ↓
ZIP extraction (if applicable)
    ↓
GWMReader.exe execution (all files in single command)
    ↓
JSON output parsed from stderr
    ↓
process_parser_output() converts to DB format
    ↓
Response: {success, records_count, records[]}
```

### Export Data Flow

```
User selects site → POST /api/export
    ↓
Background thread: process_export()
    ↓
Fetch machines from database
    ↓
Fetch telemetry data (with limit, sample_interval)
    ↓
process_site():
    ├── Detect roads (grid clustering)
    ├── Split roads at intersections
    ├── Detect zones (stop clustering)
    ├── Analyze cycles
    └── Generate events
    ↓
Create output files:
    ├── model.json
    ├── des_inputs.json
    └── ledger.json
    ↓
Status: completed → Files available for download
```

### Import + Export Flow

```
User uploads files with export=true
    ↓
Import processing (same as above)
    ↓
convert_imported_records_to_telemetry()
    ↓
Create machine info from telemetry
    ↓
process_site() with telemetry_data parameter:
    ├── Detect roads (grid clustering)
    ├── Split roads at intersections
    ├── Detect zones (stop clustering)
    ├── Analyze cycles
    └── Generate events
    ↓
Generate simulation files
    ↓
Status: completed → Files available for download
```

## API Endpoints

### Site Management

**GET `/api/sites`**
- Returns list of available sites from database
- Response: `{sites: [{site_name, site_short, site_id}]}`

### Export Endpoints

**POST `/api/export`**
- Starts export process for a site
- Request: `{site_name, config: {limit, sample_interval, grid_size, ...}}`
- Response: `{message: "Export started", site_name}`
- Status: 202 Accepted (async processing)

**GET `/api/export/status/<site_name>`**
- Returns export status and progress
- Response: `{status, progress, message, files: {model, des_inputs, ledger}}`
- Status values: `idle`, `processing`, `completed`, `error`

**GET `/api/export/download/<site_name>/<file_type>`**
- Downloads generated file
- File types: `model`, `des_inputs`, `ledger`
- Returns file as JSON attachment

### Import Endpoints

**POST `/api/import`**
- Imports and parses raw gateway message files
- Request: `multipart/form-data` with `files` and optional `site_name`, `export`
- Supports: single file, multiple files, ZIP archives
- Response: `{success, site_name, files_processed, records_count, records[]}`
- If `export=true`: Returns 202 and starts background export

**GET `/api/import/status/<site_name>`**
- Returns import+export status (when export=true)
- Same format as export status endpoint

**GET `/api/import/download/<site_name>/<file_type>`**
- Downloads exported file from import process
- Same as export download endpoint

### Health Check

**GET `/api/health`**
- Health check endpoint
- Response: `{status: "ok"}`

## Business Rules and Validations

### Import Validations

1. **File Validation**:
   - Files must be provided in request (`files` field required)
   - Supported formats: `.gwm`, `.dat`, `.bin`, or no extension
   - ZIP files are automatically extracted
   - Maximum upload size: 5 GB per request

2. **Parser Validation**:
   - Parser executable (`GWMReader.exe`) must exist at configured path
   - Parser must return exit code 0 for successful parsing
   - JSON output must be valid and parseable

3. **Data Validation**:
   - Each message array must have exactly 24 elements
   - Missing or invalid fields are handled with safe conversion functions
   - Payload values > 200 are decoded using formula: `value - 255`

### Export Validations

1. **Site Validation**:
   - Site name must be provided
   - Site must exist in database
   - At least one machine must be associated with the site

2. **Configuration Validation**:
   - `limit`: Maximum number of telemetry records to process (default: 100,000)
   - `sample_interval`: Time interval between samples in seconds (default: 5)
   - `grid_size`: Road detection grid cell size in meters (default: 5.0)
   - `min_density`: Minimum points per grid cell for road detection (default: 3)
   - `simplify_epsilon`: Road simplification tolerance in meters (default: 5.0)
   - `zone_grid_size`: Zone detection grid size in meters (default: 10.0)
   - `zone_min_stops`: Minimum stops required for zone detection (default: 20)
   - `sim_time`: Simulation time in minutes (default: 480)

3. **Concurrent Export Prevention**:
   - Only one export per site can run simultaneously
   - Attempting to start export while another is processing returns 409 Conflict

### Cycle Processing Rules

1. **Cycle Completeness**:
   - Cycles must have at least one segment
   - Full cycles require both dump zone start and end identification
   - Incomplete cycles are still processed but marked as `isFullCycle = False`

2. **Segment Classification Rules**:
   - Payload threshold of 50% determines empty vs. loaded classification
   - Segment type depends on payload transitions and reversing state
   - Cycle end is detected when payload drops from loaded to empty

3. **Zone Extraction Rules**:
   - Zone points are only extracted from "spotting" segments
   - Load zones: `SPOTTING_AT_SOURCE` segments
   - Dump zones: `SPOTTING_AT_SINK` segments
   - Minimum 2 points required to form a valid zone polygon

4. **Road Intersection Rules**:
   - Roads can only share nodes at their start or end points
   - If roads share nodes in the middle, they are split at those intersection points
   - Overlapping road segments (same start/end nodes) are deduplicated
   - Shared segments are named with "_Shared" suffix (e.g., "Road_1_Shared")
   - Routes are automatically updated to reference new segment IDs

### Data Processing Rules

1. **Coordinate System**:
   - All coordinates are in meters (no conversion needed)
   - GPS coordinates use Easting (X), Northing (Y), Elevation (Z)
   - Time values are in seconds (no millisecond conversion)

2. **Time Handling**:
   - GPS timestamps are converted using GPS epoch (Jan 6, 1980) + leap seconds offset
   - ISO timestamp strings are parsed with timezone support
   - All times are normalized to UTC

3. **Payload Encoding**:
   - Payload percentage range: 0-200 (normal) or >200 (special encoding)
   - Values > 200: decoded as `value - 255`
   - Database uses 255 to represent unknown payload

4. **Loss Calculation Rules**:
   - Loss = actual time - expected time (positive = delay, negative = ahead of schedule)
   - Efficiency = (expected time / actual time) × 100
   - Loss buckets are created when ASLR reason changes
   - Loss summaries aggregate at segment, cycle, and route levels

### Machine Filtering Rules

1. **Event-Based Machine Inclusion**:
   - During simulation generation (both database export and import+export flows), only machines that generate at least one event in the GPS-to-events conversion stage are included in the DES inputs.
   - Machines (typically haulers) that do not produce any events are excluded from the `des_inputs.json` configuration, so they do not appear as active resources in the simulation model.

2. **Business Intent**:
   - Avoids cluttering the simulation with machines that have no operational activity in the selected dataset or time window.
   - Ensures that simulation results and productivity metrics focus only on machines that contributed actual events.

## Key Components

### Backend Components

**AMTCycleProdInfoReader** (`core/Reader.py`):
- Main parser for CP1 and CP2 data formats
- Converts raw message tuples into Cycle and Zone objects
- Handles segment grouping and classification
- Extracts zone points from spotting segments

**Cycle** (`core/Cycle.py`):
- Represents a complete dump-to-dump operation
- Tracks segments, messages, loss summary, efficiency
- Validates cycle completeness (full cycle detection)
- Aggregates segment-level metrics

**Segment** (`core/Segment.py`):
- Represents a portion of a cycle with consistent payload state
- Classifies segment type based on payload and transitions
- Creates loss buckets based on ASLR changes
- Tracks path, distance, and time metrics

**Zone** (`core/Zone.py`):
- Represents load or dump operational areas
- Uses DBSCAN clustering to group GPS points
- Creates convex hull polygons for zone boundaries
- Tracks associated cycle IDs

**GatewayParserWrapper** (`core/gateway_parser_wrapper.py`):
- Wraps GWMReader.exe executable
- Handles file validation and parser execution
- Manages retry logic and error handling
- Parses JSON output from parser stderr

**simulation_generator** (`scripts/simulation_generator.py`):
- Main export processing logic
- Road detection using grid-based clustering
- Road intersection splitting and segment deduplication
- Zone detection using stop-based clustering
- Generates model, DES inputs, and events ledger files

### Frontend Components

**App.js**:
- Main application component
- Manages site list and selection
- Coordinates ExportButton and ImportButton components
- Handles error states and retry logic

**ExportButton.js**:
- Handles export initiation and status polling
- Displays progress and completion status
- Provides file download functionality
- Manages export state (idle, processing, completed, error)

**ImportButton.js**:
- Handles file upload (drag-and-drop or file picker)
- Supports ZIP files and multiple file selection
- Displays upload progress and parsing status
- Optionally triggers export after import
- Polls export status when export=true

**SiteList.js**:
- Displays available sites from database
- Handles site selection
- Shows loading and error states

## Configuration

### Environment Variables

All configuration is managed through `.env` file in `backend/` directory:

**Database Configuration:**
- `DB_HOST`: Database host (default: 192.168.0.18)
- `DB_PORT`: Database port (default: 3306)
- `DB_USER`: Database username (default: dev_user)
- `DB_PASSWORD`: Database password (required)
- `DB_NAME`: Database name (default: speed_efficiency)
- `DB_CHARSET`: Database charset (default: utf8mb4)

**Path Configuration:**
- `OUTPUT_PATH`: Output directory for generated files (default: ../output)
- `EXECUTE_FILE_PATH`: Path to GWMReader.exe parser executable (required for import)
- `EXAMPLE_JSON_PATH`: Path to exampleJSON directory (default: ../exampleJSON)

### Configuration File

`backend/scripts/config.json` contains processing parameters:
- `site`: Site name override
- `data_fetching`: `limit`, `sample_interval`
- `road_detection`: `grid_size`, `min_density`, `simplify_epsilon`
- `zone_detection`: `grid_size`, `min_stop_count`
- `simulation`: `sim_time`

## Performance Considerations

1. **Large File Handling**:
   - Files up to 5 GB are supported
   - Chunked reading (64 MB chunks) prevents memory issues
   - Temporary files are cleaned up after processing

2. **Background Processing**:
   - Export operations run in background threads
   - Status polling allows non-blocking UI updates
   - Multiple exports can be queued (one per site)

3. **Database Optimization**:
   - Telemetry data fetching uses `LIMIT` to control dataset size
   - Sample interval reduces data volume while preserving accuracy
   - Indexed queries for site and machine lookups

4. **Spatial Processing**:
   - Grid-based clustering reduces computational complexity
   - Douglas-Peucker simplification reduces road network size
   - DBSCAN clustering efficiently groups zone points

## Security Considerations

1. **File Upload Security**:
   - File names are sanitized using `secure_filename()` to prevent path traversal
   - Temporary directories use random prefixes
   - All temporary files are cleaned up after processing

2. **Input Validation**:
   - File type validation (extension checking)
   - File size limits prevent resource exhaustion
   - Parser output validation (JSON structure checking)

3. **Error Handling**:
   - Errors are logged but sensitive information is not exposed to clients
   - Parser failures return generic error messages
   - Database connection errors are handled gracefully

## Future Enhancements

Potential areas for future development:

1. **Real-time Processing**: WebSocket support for real-time status updates
2. **Batch Processing**: Support for processing multiple sites simultaneously
3. **Advanced Analytics**: Additional productivity metrics and visualizations
4. **Export Formats**: Support for additional simulation formats
5. **Data Validation**: Enhanced validation rules and error reporting
6. **Performance Optimization**: Parallel processing for large datasets
7. **User Authentication**: Role-based access control
8. **Audit Logging**: Track all import/export operations
