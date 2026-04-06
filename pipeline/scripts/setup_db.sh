#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is required. Set it in pipeline/.env or the shell environment."
  exit 1
fi

python db/seed_tasks.py

python - <<'PY'
import os
from urllib.parse import urlparse

import psycopg2

url = os.environ["DATABASE_URL"]
parsed = urlparse(url)
db_name = parsed.path.lstrip("/")
admin_url = parsed._replace(path="/postgres").geturl()

conn = psycopg2.connect(admin_url)
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    if cur.fetchone() is None:
        cur.execute(f'CREATE DATABASE "{db_name}"')
        print(f"Created database: {db_name}")
    else:
        print(f"Database already exists: {db_name}")
conn.close()
PY

python db/migrate.py
