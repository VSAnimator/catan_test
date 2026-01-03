"""
Pytest configuration for test database setup.
"""
import os
import sys
import pytest
import sqlite3
from pathlib import Path

# Set test database environment variable BEFORE any database imports
# This must happen at the very top, before any modules import api.database
os.environ["USE_TEST_DB"] = "1"

# Test database path
TEST_DB_PATH = Path(__file__).parent.parent / "catan_test.db"


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Set up test database environment and clean up after all tests."""
    # Ensure we're using test database
    os.environ["USE_TEST_DB"] = "1"
    
    yield
    
    # Clean up test database after all tests complete
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
            print(f"\nCleaned up test database: {TEST_DB_PATH}", flush=True)
        except Exception as e:
            print(f"Warning: Could not delete test database: {e}", flush=True)


@pytest.fixture(autouse=True)
def reset_test_db():
    """Reset test database before each test by clearing all data."""
    # Clear connection cache to force reconnection with test DB
    from api.database import _connection_cache, _connection_lock, init_db
    
    # Close any existing connections
    with _connection_lock:
        for cache_key in list(_connection_cache.keys()):
            try:
                _connection_cache[cache_key].close()
            except:
                pass
        _connection_cache.clear()
    
    # Delete test database if it exists to start fresh
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
        except:
            pass
    
    # Initialize database (creates tables)
    init_db()
    
    yield
