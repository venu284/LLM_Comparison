"""Prompt embeddings for the kNN baseline (S2.5) and the S4 difficulty predictor, from a pinned
sentence-transformers model.

Model: `sentence-transformers/all-MiniLM-L6-v2` at revision `MINILM_REVISION`. The pin lives in
`scripts/fetch_data.py` (`MODELS["all-MiniLM-L6-v2"]`), which downloads only the listed files into
`external/models/all-MiniLM-L6-v2/` and records their sha256 in `external/CHECKSUMS`; the revision
and hashes here must equal that pin (tested). Fetch it with
`pipeline/venv/bin/python scripts/fetch_data.py`.

Loading is offline and from that directory only. Before the model is built, `verify` checks that
the directory holds exactly the pinned files (so no pickle-based checkpoint can be picked up) and
that each one's sha256 is the pinned one; a missing, extra or changed file stops the run. The
weights are safetensors, and no repository code is run (`trust_remote_code=False`).
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from router.settings import EXTERNAL_DIR

MINILM_NAME = "all-MiniLM-L6-v2"
MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MINILM_DIR = EXTERNAL_DIR / "models" / MINILM_NAME
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
_IGNORED = {".DS_Store"}


class EmbeddingPinError(RuntimeError):
    pass


def verify(path: Path | None = None) -> Path:
    """The model directory, after checking it holds exactly the pinned files with their sha256."""
    path = MINILM_DIR if path is None else Path(path)
    if not path.is_dir():
        raise EmbeddingPinError(f"{path} is missing; run scripts/fetch_data.py")
    present = {p.relative_to(path).as_posix() for p in path.rglob("*")
               if p.is_file() and p.name not in _IGNORED}
    extra = sorted(present - set(MINILM_SHA256))
    if extra:
        raise EmbeddingPinError(f"unpinned files in {path}: {extra}")
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
    return SentenceTransformer(str(verify()), device="cpu", local_files_only=True,
                               trust_remote_code=False, model_kwargs={"use_safetensors": True})


_cache: dict[str, np.ndarray] = {}


def embed(texts: Sequence[str]) -> np.ndarray:
    """Unit-norm embeddings, one row per text (cached by text)."""
    todo = sorted({t for t in texts if t not in _cache})
    if todo:
        vecs = _model().encode(todo, batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True)
        _cache.update(zip(todo, np.asarray(vecs, dtype=np.float64)))
    return np.stack([_cache[t] for t in texts]) if texts else np.zeros((0, 384))
