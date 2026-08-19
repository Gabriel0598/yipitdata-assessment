#!/usr/bin/env python
"""Main entry point for the YipitData pipeline.

Run:
    python run_pipeline.py [--no-embeddings] [--no-duckdb]

By default it builds the full medallion model, generates embeddings, exports
every modeled table (CSV), writes ``ai_articles_enriched.csv`` and persists a
DuckDB database for hybrid search.
"""
from __future__ import annotations

import argparse

import numpy as np

from src import config
from src import duckdb_store
from src import export, pipeline
from src.embeddings import generate_embeddings, top_similar_for_each


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the YipitData pipeline.")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Skip slow embedding generation (exports still produced).")
    parser.add_argument("--no-duckdb", action="store_true",
                        help="Skip persisting the DuckDB database file.")
    args = parser.parse_args()

    print("[1/4] Loading raw data (bronze) ...")
    result = pipeline.run_pipeline()

    print(f"      bronze articles      : {len(result.bronze_articles)}")
    print(f"      clean articles       : {len(result.silver_articles)}")
    print(f"      valid ARR obs        : {len(result.fact_arr)}")
    print(f"      unmatched companies  : {len(result.unmatched)}")
    print(f"      parse issues logged  : {len(result.parse_errors)}")

    embeddings = None
    top_similar = None
    if not args.no_embeddings:
        print("[2/4] Generating embeddings (all-MiniLM-L6-v2) ...")
        embeddings = generate_embeddings(result.silver_articles)
        print(f"      embeddings shape     : {embeddings.shape}")
        top_similar = top_similar_for_each(result.silver_articles, embeddings)
    else:
        print("[2/4] Skipping embeddings (--no-embeddings)")

    print("[3/4] Exporting modeled tables + ai_articles_enriched.csv ...")
    ai = export.export_all(result, embeddings=embeddings, top_similar=top_similar)
    print(f"      ai_articles_enriched : {len(ai)} rows")
    print(f"      outputs written to   : {config.OUTPUT_DIR}")

    if not args.no_duckdb:
        print("[4/4] Persisting DuckDB database ...")
        duckdb_store.ensure_database(result, embeddings)
        print(f"      duckdb file         : {config.DUCKDB_PATH}")
    else:
        print("[4/4] Skipping DuckDB persistence (--no-duckdb)")

    print("\nDone.")


if __name__ == "__main__":
    main()
