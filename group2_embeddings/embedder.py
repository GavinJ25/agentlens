"""
Embedding cache for the agent determinacy test suite.

Loads the sentence-transformer model once per model name per process, embeds all RunResult
text outputs, and persists each vector to outputs/embeddings/{run_hash}.npy.
Subsequent calls for the same run hash return the cached array directly
without re-running the model.

All G2 metric modules call load_embeddings() from this module — they never
instantiate the model themselves.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from agent.schema import RunResult

logger = logging.getLogger(__name__)

# Module-level model cache — loaded once per process lifetime.
_model: SentenceTransformer | None = None
_model_name: str | None = None


def _get_model(model_name: str) -> SentenceTransformer:
    """
    Return the sentence-transformer model, loading it on first call.

    The model is cached in module scope so that repeated calls within
    the same process do not reload weights from disk.

    Args:
        model_name: HuggingFace model identifier, e.g.
                    ``sentence-transformers/all-MiniLM-L6-v2``.

    Returns:
        Loaded SentenceTransformer instance.
    """
    global _model, _model_name
    if _model is None or _model_name != model_name:
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        logger.info("Embedding model loaded.")
    return _model


def _run_hash(result: RunResult, model_name: str) -> str:
    """
    Derive a stable cache key from the embedding model name and a
    RunResult's prompt_id, run_index, and text content.

    Args:
        result:     RunResult instance to hash.
        model_name: Embedding model identifier from config.

    Returns:
        16-character hex prefix of the SHA-256 digest — sufficient for
        cache key uniqueness within a test suite run.
    """
    payload = f"{model_name}:{result.prompt_id}:{result.run_index}:{result.text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _embeddings_dir(cfg: dict[str, Any]) -> Path:
    """
    Resolve the outputs/embeddings/ directory from config.

    Args:
        cfg: Full config dict loaded from config.yaml.

    Returns:
        Path to the embeddings cache directory.
    """
    return Path(cfg["output"]["dir"]) / "embeddings"


def embed_run(result: RunResult, cfg: dict[str, Any]) -> np.ndarray:
    """
    Return the embedding vector for a single RunResult.

    Loads from the .npy cache if available; otherwise computes the
    embedding and writes it to disk before returning.

    Args:
        result: RunResult whose ``text`` field will be embedded.
        cfg:    Full config dict loaded from config.yaml.

    Returns:
        1-D float32 numpy array of the embedding vector.
    """
    emb_dir = _embeddings_dir(cfg)
    emb_dir.mkdir(parents=True, exist_ok=True)

    model_name: str = cfg["embeddings"]["model"]
    cache_enabled: bool = cfg["embeddings"].get("cache", True)
    run_key = _run_hash(result, model_name)
    cache_path = emb_dir / f"{run_key}.npy"

    if cache_enabled and cache_path.exists():
        logger.debug("Embedding cache hit: %s", cache_path)
        return np.load(str(cache_path))
    model = _get_model(model_name)

    vector: np.ndarray = model.encode(
        result.text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if cache_enabled:
        np.save(str(cache_path), vector)
        logger.debug("Embedding saved to %s", cache_path)

    return vector


def load_embeddings(
    results: list[RunResult],
    cfg: dict[str, Any],
) -> np.ndarray:
    """
    Embed all RunResult objects for one prompt and return a stacked matrix.

    Each row in the returned matrix corresponds to one run, in the same
    order as the input list. Embeddings are loaded from cache where possible.

    Args:
        results: List of RunResult objects for a single prompt_id.
        cfg:     Full config dict loaded from config.yaml.

    Returns:
        2-D float32 numpy array of shape (n_runs, embedding_dim).

    Raises:
        ValueError: If results is empty.
    """
    if not results:
        raise ValueError("load_embeddings received an empty results list.")

    vectors = [embed_run(r, cfg) for r in results]
    matrix = np.stack(vectors, axis=0)

    logger.debug(
        "Embeddings loaded | prompt_id=%s shape=%s",
        results[0].prompt_id,
        matrix.shape,
    )
    return matrix