"""
Top-level backend configuration module.

All environment-driven configuration for the backend lives here:
- MySQL connection (DB_*)
- MSSM SQL Server connection (MSSM_*)
- Common paths (OUTPUT_PATH, EXECUTE_FILE_PATH, EXAMPLE_JSON_PATH, TEMP_DIR)

Other modules should import from `backend.config` instead of reading
environment variables directly.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# Load environment variables from backend/.env
_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_ENV_PATH):
    load_dotenv(_ENV_PATH)


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "192.168.0.18"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "dev_user"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "speed_efficiency"),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
    "autocommit": True,
    # Optional SSL parameters (used when MySQL has require_secure_transport=ON)
    "ssl_ca": os.getenv("DB_SSL_CA"),       # e.g. /path/to/ca.pem
    "ssl_cert": os.getenv("DB_SSL_CERT"),   # optional client cert
    "ssl_key": os.getenv("DB_SSL_KEY"),     # optional client key
}


@dataclass
class MSSMConfig:
    server: str
    user: str
    password: str


MSSM_CONFIG = MSSMConfig(
    server=os.getenv("MSSM_SERVER", "localhost"),
    user=os.getenv("MSSM_USER", "sa"),
    password=os.getenv("MSSM_PASSWORD", ""),
)


# Main table name for MySQL telemetry
TABLE_NAME = "amt_cycleprodinfo_handle"

# Connection pool settings
POOL_CONFIG = {
    "pool_name": "amt_pool",
    "pool_size": 5,
    "pool_reset_session": True,
}

# Paths from environment variables
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
EXECUTE_FILE_PATH = os.getenv("EXECUTE_FILE_PATH")
EXAMPLE_JSON_PATH = os.getenv("EXAMPLE_JSON_PATH")
# Temp directory for file processing (use short path to avoid Windows MAX_PATH limit)
TEMP_DIR = os.getenv("TEMP_DIR", "")


__all__ = [
    "DB_CONFIG",
    "TABLE_NAME",
    "POOL_CONFIG",
    "MSSM_CONFIG",
    "OUTPUT_PATH",
    "EXECUTE_FILE_PATH",
    "EXAMPLE_JSON_PATH",
    "TEMP_DIR",
]

