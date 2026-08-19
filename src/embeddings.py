"""Text embeddings, cosine similarity and the required search functions.

Uses ``sentence-transformers`` with ``all-MiniLM-L6-v2`` (384-dim) over the
article *title + summary* text.

Artifacts (serialized, reusable):
  - ``embeddings.npy``   -> (N, 384) float32 numpy array aligned to
  - ``embedding_ids.json`` -> ordered list of article ids matching rows.

Provides:
  - :func:`generate_embeddings`     -> build + persist embeddings
  - :func:`load_embeddings`         -> reload persisted embeddings
  - :func:`find_similar_articles`   -> cosine search (query, top_k)
  - :func:`top_similar_for_each`    -> top-3 (excluding self) for every article
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config


def _model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(config.EMBEDDINGS_MODEL)


def _text_for(row) -> str:
    title = str(row.get("title") or "")
    summary = str(row.get("summary") or "")
    return f"{title} {summary}".strip()


@staticmethod
def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-12)
    b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-12)
    return a @ b.T


def generate_embeddings(articles: pd.DataFrame, force: bool = False) -> np.ndarray:
    config.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    texts = [_text_for(row) for _, row in articles.iterrows()]
    model = _model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vecs = np.asarray(vecs, dtype="float32").reshape(len(articles), config.EMBEDDING_DIM)
    np.save(config.EMBEDDING_NPY_PATH, vecs)
    with open(config.EMBEDDING_IDS_PATH, "w") as fh:
        json.dump(list(articles["article_id"]), fh)
    return vecs


def load_embeddings() -> tuple[np.ndarray, list[str]]:
    vecs = np.load(config.EMBEDDING_NPY_PATH)
    with open(config.EMBEDDING_IDS_PATH) as fh:
        ids = json.load(fh)
    # maps id -> row index
    return vecs, ids


def find_similar_articles(query_text: str, top_k: int = 5) -> list[dict]:
    """Return the top_k most similar articles to a free-text query.

    Returns a list of {"article_id": str, "similarity": float} sorted desc.
    """
    vecs, ids = load_embeddings()
    model = _model()
    q = model.encode([query_text], normalize_embeddings=True)
    scores = vecs @ q.T
    scores = scores.ravel()
    order = np.argsort(-scores)[:top_k]
    return [{"article_id": ids[i], "similarity": round(float(scores[i]), 6)} for i in order]


def top_similar_for_each(articles: pd.DataFrame, vecs: np.ndarray) -> list[list[str]]:
    """For every article, the top 3 most similar OTHER article ids (excl self)."""
    sim = _cosine(vecs, vecs)
    np.fill_diagonal(sim, -1.0)
    out = []
    ids = list(articles["article_id"])
    for i in range(sim.shape[0]):
        top = np.argsort(-sim[i])[:3]
        out.append([ids[j] for j in top])
    return out
