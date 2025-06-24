"""Debug script to check settings values"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import Settings

settings = Settings()
print("Database Settings:")
print(f"DATABASE_URL: {settings.DATABASE_URL}")
print(f"ASYNC_DATABASE_URL: {settings.ASYNC_DATABASE_URL}")
print(f"database.DATABASE_URL: {settings.database.DATABASE_URL}")
print(f"database.ASYNC_DATABASE_URL: {settings.database.ASYNC_DATABASE_URL}")
