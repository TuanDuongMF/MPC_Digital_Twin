# AMT Cycle Productivity Reader WebApp

Web application for generating simulation files from AMT telemetry data. Provides a unified **Parse Data** workflow that produces model, DES inputs, and events ledger from raw gateway files and MSSM databases.

## Structure

```
Simulation/
├── backend/
│   ├── app.py              # Flask API
│   ├── config.py           # Configuration (env vars)
│   ├── core/               # Gateway parser, data conversion
│   ├── roads_network_pipeline/  # MSSM model generation
│   ├── scripts/            # simulation_generator.py
│   └── simulation_analysis/
├── frontend/
│   └── src/
│       ├── App.js
│       └── components/
│           └── ParseSimulationForm.js
├── docs/                   # Specifications
└── README.md
```

## Features

- **Parse Data**: Single workflow that:
  1. Generates `model.json` from MSSM databases (roads, zones, fleet)
  2. Creates `simulation_des_inputs.json.gz` from model + MySQL machines
  3. Produces `simulation_ledger.json.gz` from raw gateway files in uploaded ZIP
- **Download**: After parsing, download Model, DES Inputs, and Events (Ledger) files
- **Fidelity**: Select model fidelity (Low/High) for roads_network_pipeline

## Prerequisites

- Python 3.9+
- Node.js 16+ and npm
- MySQL database (telemetry, machines)
- MSSM SQL Server access (for model generation)
- GWMReader.exe parser executable

## Setup

### Backend

1. Navigate to backend:
   ```bash
   cd backend
   ```

2. Create virtual environment (optional):
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/Mac: source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure `.env`:
   ```bash
   cp .env.example .env
   # Edit: DB_*, MSSM_*, OUTPUT_PATH, EXECUTE_FILE_PATH, EXAMPLE_JSON_PATH
   ```

5. Run server:
   ```bash
   python app.py
   ```
   Backend: `http://localhost:5000`

### Frontend

1. Navigate to frontend:
   ```bash
   cd frontend
   ```

2. Install and start:
   ```bash
   npm install
   npm start
   ```
   Frontend: `http://localhost:3000` (proxies API to backend)

## Usage

1. Start backend (port 5000) and frontend (port 3000)
2. Open `http://localhost:3000`
3. Select **Fidelity** (Low/High)
4. Upload a **ZIP file** containing raw gateway message files (`.gwm`, `.dat`, `.bin`)
5. Click **Parse Data**
6. After completion, use **Download** buttons for Model, DES Inputs, Events (Ledger)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/simulation/parse` | Run 3-step parse pipeline (fidelity + ZIP required) |
| GET | `/api/simulation/download/<file_type>` | Download model, des_inputs, or ledger |
| GET | `/api/model/download/<file_type>` | Download model.json |
| GET | `/api/health` | Health check |

### POST `/api/simulation/parse`

**Request:** `multipart/form-data`
- `fidelity`: string (optional, default `"Low"`)
- `file`: ZIP with raw gateway data (required)

**Response:**
```json
{
  "success": true,
  "output_path": "...",
  "stages": {...},
  "elapsed_seconds": 45.2,
  "files": {
    "model": "model.json",
    "des_inputs": "simulation_des_inputs.json.gz",
    "ledger": "simulation_ledger.json.gz"
  }
}
```

### GET `/api/simulation/download/<file_type>`

- `file_type`: `model` | `des_inputs` | `ledger`
- Returns file as attachment

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | MySQL connection |
| `DB_SSL_CA`, `DB_SSL_CERT`, `DB_SSL_KEY` | Optional MySQL SSL |
| `MSSM_SERVER`, `MSSM_USER`, `MSSM_PASSWORD`, `MSSM_DATABASE` | MSSM SQL Server |
| `OUTPUT_PATH` | Output directory for generated files |
| `EXECUTE_FILE_PATH` | Path to GWMReader.exe |
| `EXAMPLE_JSON_PATH` | Path to exampleJSON directory |
| `TEMP_DIR` | Optional temp dir for ZIP extraction |

## Documentation

- [APP_DESCRIPTION.md](APP_DESCRIPTION.md) — Full application description
- [docs/des_inputs_specification.md](docs/des_inputs_specification.md) — DES inputs format
- [docs/event_generation_algorithm.md](docs/event_generation_algorithm.md) — Events generation
- [docs/model-structure.md](docs/model-structure.md) — Model JSON structure
