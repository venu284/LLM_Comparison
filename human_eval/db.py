"""Database access for the human evaluation module.

Reuses the `runs`, `tasks`, `models` and `human_votes` tables already defined in
`pipeline/db/schema.sql`. No schema change is required for Phase 7.

`human_votes.winner` is VARCHAR(10), which cannot hold a model name, so the
winner is stored as the side that won -- 'a', 'b' or 'tie' -- and resolved back
to a model through `model_a` / `model_b` on read.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"

WINNER_A = "a"
WINNER_B = "b"
WINNER_TIE = "tie"
VALID_WINNERS = {WINNER_A, WINNER_B, WINNER_TIE}


def load_database_url() -> Optional[str]:
    """Read DATABASE_URL, loading pipeline/.env if it has not been loaded."""
    url = os.getenv("DATABASE_URL")
    if url:
        return url.strip() or None

    env_path = PIPELINE_DIR / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except ModuleNotFoundError:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("DATABASE_URL="):
                    return line.split("=", 1)[1].strip() or None

    url = os.getenv("DATABASE_URL")
    return url.strip() if url else None


@contextmanager
def connect() -> Iterator["object"]:
    import psycopg2

    url = load_database_url()
    if not url:
        raise RuntimeError("DATABASE_URL is not set; cannot reach the results database")

    connection = psycopg2.connect(url, connect_timeout=20)
    try:
        yield connection
    finally:
        connection.close()


def fetch_comparable_tasks() -> List[Dict[str, object]]:
    """Tasks with at least two models that produced extractable code."""
    query = """
        SELECT t.task_id, t.category, t.difficulty, t.title, COUNT(*) AS n_models
        FROM runs r
        JOIN tasks t ON t.id = r.task_id
        WHERE r.extracted_code IS NOT NULL AND r.extracted_code <> ''
        GROUP BY t.task_id, t.category, t.difficulty, t.title
        HAVING COUNT(*) >= 2
        ORDER BY t.task_id
    """
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(query)
        return [
            {
                "task_id": row[0],
                "category": row[1],
                "difficulty": row[2],
                "title": row[3],
                "n_models": row[4],
            }
            for row in cursor.fetchall()
        ]


def fetch_submissions(task_id: str) -> List[Dict[str, str]]:
    """Every model's extracted code for one task."""
    query = """
        SELECT m.name, r.extracted_code
        FROM runs r
        JOIN tasks t ON t.id = r.task_id
        JOIN models m ON m.id = r.model_id
        WHERE t.task_id = %s AND r.extracted_code IS NOT NULL AND r.extracted_code <> ''
        ORDER BY m.name
    """
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(query, (task_id,))
        return [{"model_name": row[0], "code": row[1]} for row in cursor.fetchall()]


def fetch_task_prompt(task_id: str) -> Optional[str]:
    query = "SELECT title FROM tasks WHERE task_id = %s"
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(query, (task_id,))
        row = cursor.fetchone()
        return row[0] if row else None


def record_vote(
    task_id: str, model_a: str, model_b: str, winner: str, evaluator_id: str
) -> int:
    """Insert one vote. `winner` must be 'a', 'b' or 'tie'."""
    if winner not in VALID_WINNERS:
        raise ValueError(f"winner must be one of {sorted(VALID_WINNERS)}, got {winner!r}")

    query = """
        INSERT INTO human_votes (task_id, model_a, model_b, winner, evaluator_id)
        VALUES (
            (SELECT id FROM tasks WHERE task_id = %s),
            (SELECT id FROM models WHERE name = %s),
            (SELECT id FROM models WHERE name = %s),
            %s, %s
        )
        RETURNING id
    """
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(query, (task_id, model_a, model_b, winner, evaluator_id))
        vote_id = cursor.fetchone()[0]
        connection.commit()
        return int(vote_id)


def record_votes_bulk(
    rows: Sequence[Tuple[str, str, str, str, str]]
) -> int:
    """Insert many votes as (task_id, model_a, model_b, winner, evaluator_id)."""
    query = """
        INSERT INTO human_votes (task_id, model_a, model_b, winner, evaluator_id)
        VALUES (
            (SELECT id FROM tasks WHERE task_id = %s),
            (SELECT id FROM models WHERE name = %s),
            (SELECT id FROM models WHERE name = %s),
            %s, %s
        )
    """
    with connect() as connection:
        cursor = connection.cursor()
        cursor.executemany(query, rows)
        connection.commit()
        return cursor.rowcount


def fetch_votes() -> List[Tuple[str, str, str]]:
    """All stored votes as (model_a, model_b, winner_model_name) triples."""
    query = """
        SELECT ma.name, mb.name, v.winner
        FROM human_votes v
        JOIN models ma ON ma.id = v.model_a
        JOIN models mb ON mb.id = v.model_b
    """
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute(query)
        triples = []
        for name_a, name_b, winner in cursor.fetchall():
            if winner == WINNER_A:
                resolved = name_a
            elif winner == WINNER_B:
                resolved = name_b
            else:
                resolved = WINNER_TIE
            triples.append((name_a, name_b, resolved))
        return triples


def clear_votes(evaluator_prefix: Optional[str] = None) -> int:
    """Delete votes. With a prefix, only matching evaluator_ids are removed.

    Used to reset synthetic votes without touching real human ones.
    """
    with connect() as connection:
        cursor = connection.cursor()
        if evaluator_prefix:
            cursor.execute(
                "DELETE FROM human_votes WHERE evaluator_id LIKE %s",
                (f"{evaluator_prefix}%",),
            )
        else:
            cursor.execute("DELETE FROM human_votes")
        connection.commit()
        return cursor.rowcount


def vote_counts() -> Dict[str, int]:
    with connect() as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM human_votes")
        total = cursor.fetchone()[0]
        cursor.execute(
            "SELECT COUNT(*) FROM human_votes WHERE evaluator_id LIKE 'synthetic_%'"
        )
        synthetic = cursor.fetchone()[0]
        return {"total": int(total), "synthetic": int(synthetic), "human": int(total - synthetic)}
