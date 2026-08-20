"""DuckDB integration: load the cleaned/enriched data and support hybrid
search (SQL filters + vector cosine similarity).

Embeddings are stored as DuckDB ``DOUBLE[]`` arrays. Because DuckDB itself
does not (out of the box) ship a vector index here, we implement hybrid lookup
by applying the SQL filters first and then computing cosine similarity against
the filtered subset for the query vector -- a minimal, correct hybrid search.
"""
from __future__ import annotations

import json

import duckdb
import numpy as np

from . import config
from . import embeddings as emb


_DDL = {
    "dim_company": """
        CREATE OR REPLACE TABLE dim_company (
            company_id VARCHAR, company_name VARCHAR, industry VARCHAR,
            founded_year INTEGER, headquarters VARCHAR, employee_count BIGINT,
            is_public BOOLEAN, stock_ticker VARCHAR
        )
    """,
    "dim_article": """
        CREATE OR REPLACE TABLE dim_article (
            article_id VARCHAR, source_article_id VARCHAR, title VARCHAR,
            company_name_raw VARCHAR, company_name_resolved VARCHAR, company_id VARCHAR,
            published_date VARCHAR, published_year INTEGER, published_quarter INTEGER,
            published_month INTEGER, category_raw VARCHAR, category_standardized VARCHAR,
            revenue_raw VARCHAR, revenue_currency VARCHAR, revenue_is_range BOOLEAN,
            arr_usd BIGINT, arr_parse_status VARCHAR, summary VARCHAR, url VARCHAR,
            author VARCHAR, word_count VARCHAR, industry VARCHAR, founded_year INTEGER,
            headquarters VARCHAR, employee_count BIGINT, is_public BOOLEAN,
            stock_ticker VARCHAR, company_age BIGINT, company_size_category VARCHAR,
            embedding DOUBLE[]
        )
    """,
    "fact_arr_observation": """
        CREATE OR REPLACE TABLE fact_arr_observation (
            arr_observation_id VARCHAR, company_id VARCHAR, article_id VARCHAR,
            observation_date VARCHAR, arr_usd BIGINT, arr_raw_value VARCHAR,
            arr_currency VARCHAR, arr_is_range BOOLEAN
        )
    """,
}


def ensure_database(result, embeddings: np.ndarray | None = None) -> duckdb.DuckDBPyConnection:
    """Load results into a persisted DuckDB file and return the connection."""
    con = duckdb.connect(str(config.DUCKDB_PATH))

    # dim_company
    con.execute(_DDL["dim_company"])
    cc = result.companies.copy()
    con.execute("INSERT INTO dim_company SELECT * FROM cc")

    # dim_article + embeddings
    con.execute(_DDL["dim_article"])
    art = result.dim_article.copy()
    if embeddings is not None:
        ids = list(result.dim_article["article_id"])
        id_to_emb = dict(zip(ids, embeddings.tolist()))
        art["embedding"] = art["article_id"].map(lambda a: id_to_emb[a])
        # Store as an array literal string for DuckDB parsing.
        art["embedding"] = art["embedding"].map(lambda v: json.dumps(v))
    else:
        art["embedding"] = None
    con.execute("INSERT INTO dim_article SELECT * FROM art")

    # fact_arr_observation
    con.execute(_DDL["fact_arr_observation"])
    ff = result.fact_arr.copy()
    con.execute("INSERT INTO fact_arr_observation SELECT * FROM ff")

    con.close()
    return duckdb.connect(str(config.DUCKDB_PATH))


def hybrid_search(
    query_text: str,
    top_k: int = 5,
    *,
    published_year_from: int | None = None,
    published_year_to: int | None = None,
    arr_min_usd: float | None = None,
    categories: list[str] | None = None,
    industries: list[str] | None = None,
    db_path: str | None = None,
) -> list[dict]:
    """Hybrid search: apply SQL filters, then rank by cosine similarity.

    Returns a list of dicts with article_id, similarity and the filtered fields.
    """
    con = duckdb.connect(db_path or str(config.DUCKDB_PATH))

    conds = []
    if published_year_from is not None:
        conds.append(f"published_year >= {int(published_year_from)}")
    if published_year_to is not None:
        conds.append(f"published_year <= {int(published_year_to)}")
    if arr_min_usd is not None:
        conds.append(f"arr_usd > {float(arr_min_usd)}")
    if categories:
        conds.append("category_standardized IN (" +
                     ",".join(f"'{c}'" for c in categories) + ")")
    if industries:
        conds.append("industry IN (" +
                     ",".join(f"'{i}'" for i in industries) + ")")

    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    rows = con.execute(
        f"SELECT article_id, title, company_name_resolved, published_year, "
        f"category_standardized, industry, arr_usd, embedding FROM dim_article{where}"
    ).fetchall()

    if not rows:
        con.close()
        return []

    cols = ["article_id", "title", "company_name", "published_year",
            "category", "industry", "arr_usd", "embedding"]

    # Build query vector.
    model = emb._model()
    q = model.encode([query_text], normalize_embeddings=True).ravel()

    results = []
    for r in rows:
        record = dict(zip(cols, r))
        raw_vec = r[7]
        if isinstance(raw_vec, str):
            vec = np.asarray(json.loads(raw_vec), dtype="float32")
        else:
            # DuckDB returns the DOUBLE[] column as a Python list.
            vec = np.asarray(raw_vec, dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            sim = float(q @ (vec / norm))
        else:
            sim = 0.0
        results.append({**{k: record[k] for k in cols if k != "embedding"},
                        "similarity": round(sim, 6)})

    results.sort(key=lambda x: x["similarity"], reverse=True)
    con.close()
    return results[:top_k]


def main() -> None:
    """Demonstrative runner: list tables and show an example query.

    Requires the DuckDB file to already exist (run ``run_pipeline.py`` first).
    """
    print("DuckDB path:", config.DUCKDB_PATH)
    if not config.DUCKDB_PATH.exists():
        print("Database not found. Run `python run_pipeline.py` first.")
        return

    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=True)
    tables = [t[0] for t in con.execute("show tables").fetchall()]
    print("Tables:", tables)

    print("\nLatest ARR per company (top 5):")
    rows = con.execute("""
        WITH ranked AS (
          SELECT company_id, observation_date, arr_usd,
                 ROW_NUMBER() OVER (PARTITION BY company_id
                                    ORDER BY observation_date DESC) rn
          FROM fact_arr_observation
        )
        SELECT company_id, observation_date, arr_usd
        FROM ranked WHERE rn = 1
        ORDER BY arr_usd DESC LIMIT 5
    """).fetchall()
    for r in rows:
        print("  ", r)
    con.close()


if __name__ == "__main__":
    main()
