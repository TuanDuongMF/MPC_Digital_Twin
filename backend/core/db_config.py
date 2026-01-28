"""
Database Configuration
MySQL connection settings loaded from environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Load from backend/.env (parent directory of core/)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "192.168.0.18"),
    "port": int(os.getenv("DB_PORT", "3306")),
    "user": os.getenv("DB_USER", "dev_user"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "speed_efficiency"),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
    "autocommit": True,
}

# Main table name
TABLE_NAME = "amt_cycleprodinfo_handle"

# Connection pool settings
POOL_CONFIG = {
    "pool_name": "amt_pool",
    "pool_size": 5,
    "pool_reset_session": True,
}

# Paths from environment variables
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "../output")
EXECUTE_FILE_PATH = os.getenv("EXECUTE_FILE_PATH", "../executables")
EXAMPLE_JSON_PATH = os.getenv("EXAMPLE_JSON_PATH", "../exampleJSON")
# Temp directory for file processing (use short path to avoid Windows MAX_PATH limit)
TEMP_DIR = os.getenv("TEMP_DIR", "")
