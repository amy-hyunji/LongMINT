"""Shared OpenAI-compatible embeddings client used by BaseRAG test drivers.

Originally inlined in `tests_subset_combined_baserag.py` and
`tests_wiki_revision_baserag.py`. Extracted so `tests_git_commits_baserag.py`
and `tests_horizonbench_baserag.py` can reuse the same disk-cached client
without duplicating ~100 lines per file.
"""

import hashlib
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from openai import OpenAI


class OpenAIEmbeddingClient:
    """OpenAI-compatible embeddings client (works with Gemini's
    `https://generativelanguage.googleapis.com/v1beta/openai/` endpoint, OpenAI
    proper, vLLM-served embedders, etc.).

    Implements the slice of the SentenceTransformer.encode() interface that
    the BaseRAG scripts use: returns an L2-normalized float32 ndarray of
    shape (n, d).

    Disk cache (`cache_db`, sqlite WAL): keyed by `sha256(model + dim_tag +
    text)`. Across a top_k sweep the same docs/questions get encoded N times;
    the cache makes pass 2..N essentially free, so cost ≈ 1× even when top_k
    values run sequentially.
    """
    def __init__(self, model, base_url, api_key, *, max_concurrency=8,
                 batch_size=50, max_retries=6, output_dim=None, cache_db=None):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_concurrency = max_concurrency
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.output_dim = output_dim
        self._cache_db = cache_db
        self._cache_lock = threading.Lock()
        self._mem_cache = {}
        self._dim_tag = f"d{output_dim}" if output_dim else "default"
        self._key_prefix = f"{model}|{self._dim_tag}|"
        if cache_db:
            os.makedirs(os.path.dirname(cache_db), exist_ok=True)
            with sqlite3.connect(cache_db) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("CREATE TABLE IF NOT EXISTS embeddings ("
                             "k TEXT PRIMARY KEY, v BLOB NOT NULL, dim INTEGER NOT NULL)")

    def _key(self, text):
        return hashlib.sha256((self._key_prefix + text).encode("utf-8")).hexdigest()

    def _cache_get_many(self, keys):
        if not self._cache_db:
            return {k: self._mem_cache.get(k) for k in keys}
        with self._cache_lock, sqlite3.connect(self._cache_db) as conn:
            out = dict.fromkeys(keys, None)
            for chunk_start in range(0, len(keys), 500):
                chunk = keys[chunk_start:chunk_start+500]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT k, v, dim FROM embeddings WHERE k IN ({placeholders})", chunk
                ).fetchall()
                for k, v, dim in rows:
                    out[k] = np.frombuffer(v, dtype=np.float32).reshape(dim)
        return out

    def _cache_put_many(self, kv_pairs):
        if not self._cache_db:
            for k, v in kv_pairs:
                self._mem_cache[k] = v
            return
        with self._cache_lock, sqlite3.connect(self._cache_db) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(k, v, dim) VALUES (?, ?, ?)",
                [(k, v.astype(np.float32).tobytes(), int(v.shape[0])) for k, v in kv_pairs],
            )
            conn.commit()

    def _embed_batch(self, texts):
        kw = {"model": self.model, "input": list(texts)}
        if self.output_dim:
            kw["dimensions"] = self.output_dim
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.embeddings.create(**kw)
                return [np.asarray(d.embedding, dtype=np.float32) for d in resp.data]
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 30) + random.random())
        raise RuntimeError(f"embedding API failed after {self.max_retries} retries: {last_err}")

    def encode(self, texts, *, normalize_embeddings=True, **_ignored):
        keys = [self._key(t) for t in texts]
        cached = self._cache_get_many(keys)
        missing_idx = [i for i, k in enumerate(keys) if cached[k] is None]
        if missing_idx:
            chunks = [missing_idx[i:i+self.batch_size]
                      for i in range(0, len(missing_idx), self.batch_size)]
            new_pairs = []
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
                fut_to_chunk = {ex.submit(self._embed_batch, [texts[j] for j in c]): c
                                for c in chunks}
                for fut in as_completed(fut_to_chunk):
                    chunk = fut_to_chunk[fut]
                    embs = fut.result()
                    for j, e in zip(chunk, embs):
                        cached[keys[j]] = e
                        new_pairs.append((keys[j], e))
            if new_pairs:
                self._cache_put_many(new_pairs)
        out = np.stack([cached[k] for k in keys], axis=0).astype(np.float32)
        if normalize_embeddings:
            out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
        return out


def load_api_key(api_key_file: str, env_var: str = "EMBEDDER_API_KEY") -> str:
    key_path = os.path.expanduser(api_key_file)
    if os.path.exists(key_path):
        return open(key_path).read().strip()
    if os.environ.get(env_var):
        return os.environ[env_var]
    raise SystemExit(
        f"--embedder_provider=openai but no API key: neither "
        f"{key_path} nor ${env_var} is set."
    )
