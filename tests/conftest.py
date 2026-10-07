"""Fixtures shared by the test files."""

import psycopg
import pytest

from datasheet_rag.indexing.store import connect, create_schema

TEST_SCHEMA = "pytest_tmp"


@pytest.fixture
def conn():
    """A connection to the database of docker-compose.yml, working in a temporary schema.

    The test is skipped when the database is not running. The schema is
    dropped at the end, so the real `chunks` table is not touched.
    """
    try:
        conn = connect()
    except (KeyError, psycopg.OperationalError) as error:  # no .env, or the database is not running
        pytest.skip(f"database not available: {error}")
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {TEST_SCHEMA}")
    conn.execute(f"SET search_path TO {TEST_SCHEMA}, public")  # public holds the vector type
    create_schema(conn)
    yield conn
    conn.execute(f"DROP SCHEMA {TEST_SCHEMA} CASCADE")
    conn.close()
