# AMT Cycle Productivity Reader WebApp

Simple web application for exporting model and simulation files from AMT telemetry data.

## Structure

```
webapp/
├── backend/          # Flask RESTful API + All processing logic
│   ├── app.py        # Main Flask application
│   ├── requirements.txt
│   ├── core/         # Core AMT modules
│   │   ├── Reader.py
│   │   ├── Cycle.py
│   │   ├── Segment.py
│   │   ├── db_config.py
│   │   └── ... (other core modules)
│   ├── scripts/      # Scripts for data processing
│   │   ├── simulation_generator.py
│   │   └── config.json
│   └── simulation_analysis/  # Simulation event conversion
│       ├── gps_to_events_converter.py
│       └── ... (other simulation modules)
├── exampleJSON/      # Example data files
├── frontend/         # React application
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── components/
│   │   │   ├── SiteList.js
│   │   │   └── ExportButton.js
│   │   └── index.js
│   └── package.json
└── README.md
```

## Features

- **List Sites**: Display all available sites from database
- **Export Files**: Generate and download model/simulation files for selected site
  - Model file (road network)
  - DES Inputs file (simulation configuration)
  - Events Ledger file (simulation events)
- **Import Raw Data**: Upload and parse raw gateway message files
  - Support for single files, multiple files, or ZIP archives
  - Parse gateway messages (`.gwm`, `.dat`, `.bin`, or files without extension)
  - Returns structured JSON data matching database schema
  - Handles large files (up to 5 GB) efficiently

## Prerequisites

- Python 3.9+
- Node.js 16+ and npm
- MySQL database access (configured in `.env` file)

## Setup

### Backend Setup

1. Navigate to backend directory:
   ```bash
   cd webapp/backend
   ```

2. Create virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   # Copy example file
   cp .env.example .env
   
   # Edit .env file with your database credentials and paths:
   # - Database: DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_CHARSET
   # - OUTPUT_PATH (path to output directory for generated files)
   # - EXECUTE_FILE_PATH (path to GWMReader.exe parser executable)
   # - EXAMPLE_JSON_PATH (path to exampleJSON directory)
   ```

5. Run Flask server:
   ```bash
   python app.py
   ```

   Backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to frontend directory:
   ```bash
   cd webapp/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start development server:
   ```bash
   npm start
   ```

   Frontend will run on `http://localhost:3000` and automatically proxy API requests to backend.

## Usage

### Export Files

1. Start backend server (port 5000)
2. Start frontend server (port 3000)
3. Open browser to `http://localhost:3000`
4. Select a site from the list
5. Click "Export Files" to generate model/simulation files
6. Wait for export to complete (progress bar will show status)
7. Download generated files using the download buttons

### Import Raw Data

1. Click "Import" button (available regardless of site selection)
2. Enter site name (or use default)
3. Select files or ZIP archive containing raw gateway message files
4. Click "Import" to upload and parse files
5. View parsed records in the response (list of dictionaries with database column names)

For detailed Import API documentation, see [docs/import_api.md](docs/import_api.md).

## API Endpoints

### GET `/api/sites`
Get list of available sites.

**Response:**
```json
{
  "sites": [
    {
      "site_name": "BhpEscondida",
      "site_short": "ESC",
      "site_id": 1
    }
  ]
}
```

### POST `/api/export`
Start export process for a site.

**Request:**
```json
{
  "site_name": "BhpEscondida",
  "config": {
    "limit": 100000,
    "sample_interval": 5
  }
}
```

**Response:**
```json
{
  "message": "Export started",
  "site_name": "BhpEscondida"
}
```

### GET `/api/export/status/<site_name>`
Get export status for a site.

**Response:**
```json
{
  "status": "processing",
  "progress": 50,
  "message": "Processing site data...",
  "files": {}
}
```

### GET `/api/export/download/<site_name>/<file_type>`
Download exported file.

**Parameters:**
- `file_type`: One of `model`, `des_inputs`, `ledger`

### POST `/api/import`
Import and parse raw gateway message files.

**Request:** `multipart/form-data`
- `files`: One or more files (or ZIP archive)
- `site_name`: Site name (optional, default: "DefaultSite")

**Response:**
```json
{
  "success": true,
  "site_name": "DefaultSite",
  "files_processed": 34,
  "records_count": 6072,
  "records": [
    {
      "expectedElapsedTime": 526,
      "actualElapsedTime": 600,
      "pathEasting": 9882,
      ...
    }
  ]
}
```

For detailed documentation, see [docs/import_api.md](docs/import_api.md).

## Development

### Backend

- Flask app with CORS enabled for frontend communication
- Uses existing `simulation_generator.py` functions for processing
- Background thread processing for long-running exports
- Status polling for progress updates

### Frontend

- React 18 with functional components and hooks
- Bootstrap 5 for styling
- Real-time status polling during export
- File download functionality

## Notes

- Export process may take several minutes depending on data size
- Generated files are saved to directory specified in `OUTPUT_PATH` environment variable
- Export status is stored in memory (resets on server restart)
- Frontend polls status every 2 seconds during export

## Environment Variables

All configuration is done through `.env` file in `backend/` directory:

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | Database host | 192.168.0.18 |
| `DB_PORT` | Database port | 3306 |
| `DB_USER` | Database username | dev_user |
| `DB_PASSWORD` | Database password | (required) |
| `DB_NAME` | Database name | speed_efficiency |
| `DB_CHARSET` | Database charset | utf8mb4 |
| `OUTPUT_PATH` | Output directory for generated files | ../output |
| `EXECUTE_FILE_PATH` | Path to GWMReader.exe parser executable (required for import) | ../executables |
| `EXAMPLE_JSON_PATH` | Example JSON files directory | ../exampleJSON |

All paths can be relative (to backend directory) or absolute.
