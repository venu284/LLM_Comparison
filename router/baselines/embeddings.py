"""Prompt embeddings for the kNN baseline (S2.5), from a pinned sentence-transformers model.

Model: `sentence-transformers/all-MiniLM-L6-v2` at revision `MINILM_REVISION`. It is loaded only
from the local Hugging Face cache (`local_files_only`), never downloaded here, and every file the
model reads is checked against `MINILM_SHA256` first; a missing or changed file stops the run.

The pin lives here rather than in `scripts/fetch_data.py`: that script pins dataset files and row
extracts, and has no model-repository kind yet. Adding one (and the matching `external/CHECKSUMS`
lines) is proposed for the reviewer; until then the cache must already hold this revision
(`huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --revision <MINILM_REVISION>`).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_SHA256 = {
    "1_Pooling/config.json": "4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23",
    "config.json": "953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41",
    "config_sentence_transformers.json": "061ca9d39661d6c6d6de5ba27f79a1cd5770ea247f8d46412a68a498dc5ac9f3",
    "model.safetensors": "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db",
    "modules.json": "84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf",
    "sentence_bert_config.json": "fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    "tokenizer_config.json": "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}


class EmbeddingPinError(RuntimeError):
    pass


def snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(MINILM_REPO, revision=MINILM_REVISION, local_files_only=True))
    except Exception as e:      # not in the cache
        raise EmbeddingPinError(f"{MINILM_REPO}@{MINILM_REVISION} is not in the local cache") from e


def verify(path: Path | None = None) -> Path:
    path = snapshot_dir() if path is None else path
    for rel, want in MINILM_SHA256.items():
        f = path / rel
        if not f.is_file():
            raise EmbeddingPinError(f"missing {rel}")
        got = hashlib.sha256(f.read_bytes()).hexdigest()
        if got != want:
            raise EmbeddingPinError(f"{rel}: sha256 {got} != pinned {want}")
    return path


@lru_cache(maxsize=1)
def _model():
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(verify()), device="cpu", local_files_only=True)


_cache: dict[str, np.ndarray] = {}


def embed(texts: Sequence[str]) -> np.ndarray:
    """Unit-norm embeddings, one row per text (cached by text)."""
    todo = sorted({t for t in texts if t not in _cache})
    if todo:
        vecs = _model().encode(todo, batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True)
        _cache.update(zip(todo, np.asarray(vecs, dtype=np.float64)))
    return np.stack([_cache[t] for t in texts]) if texts else np.zeros((0, 384))
