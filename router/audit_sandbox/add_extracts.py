"""Add new `scripts/fetch_data.py` extracts to `external/CHECKSUMS` without re-streaming the
existing ones (S1.7 checkpoint 2).

For each named extract not yet in CHECKSUMS: check that the pinned per-file hashes are the Hub's at
the pinned revision (as `--write-checksums` does), build it, and add its SHA-256 and `# extract:`
line. Existing entries are kept as they are. Afterwards `fetch_data.py --verify-only` must pass.

    pipeline/venv/bin/python -m router.audit_sandbox.add_extracts bigcodebench_tests [...]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import fetch_data as fd  # noqa: E402


def main(names: list[str]) -> int:
    from huggingface_hub import HfApi

    dest = fd.EXTERNAL_DIR
    entries, _ = fd.parse_checksums(fd.COMMITTED_CHECKSUMS.read_text())
    api = HfApi()
    for name in names:
        ex = fd.EXTRACTS[name]
        info = api.dataset_info(ex.repo_id, revision=ex.revision, files_metadata=True)
        if info.sha != ex.revision:
            raise SystemExit(f"{ex.repo_id}@{ex.revision} resolved to {info.sha}")
        hub = {s.rfilename: fd.hub_expected(s) for s in info.siblings}
        bad = [p for p, h in ex.files.items() if hub.get(p) != h]
        if bad:
            raise SystemExit(f"{name}: pinned hash differs from the Hub for {bad}")
        print(f"building extract {name}")
        fd.build_extract(dest, name)
        entries[fd.extract_path(name)] = fd.sha256_file(dest / fd.extract_path(name))
    fd.COMMITTED_CHECKSUMS.write_text(fd.format_checksums(entries))
    print(f"CHECKSUMS now lists {len(entries)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
