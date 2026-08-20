#!/usr/bin/env python
"""Interactive DuckDB shell for the YipitData pipeline outputs.

Starts a Python REPL with a read-only connection to the persisted DuckDB
database so you can run SQL interactively.

Usage:
    ./.venv/bin/python -i query_duckdb.py

Once inside, type SQL like:
    con.execute("SELECT * FROM dim_company LIMIT 5").df()
"""
import duckdb

DB_PATH = "data/output/yipitdata.duckdb"
con = duckdb.connect(DB_PATH, read_only=True)

print("=" * 60)
print("YipitData DuckDB - interactive session")
print(f"Connected to: {DB_PATH} (read-only)")
print("Tables:", [t[0] for t in con.execute("show tables").fetchall()])
print()
print("Examples:")
print('  con.execute("SELECT * FROM dim_company LIMIT 5").df()')
print('  con.execute("SELECT count(*) FROM fact_arr_observation").fetchall()')
print("=" * 60)
