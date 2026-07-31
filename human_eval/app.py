#!/usr/bin/env python3
"""Blind pairwise voting interface for Phase 7.

    cd human_eval && python app.py --port 8000

Built on the standard library only. The Phase 2 framework named FastAPI, but
this is a research instrument used by a handful of evaluators, not the Phase 8
demonstration application, and `http.server` removes an install step from every
participant's setup. psycopg2 -- already a pipeline dependency -- is the only
third-party import.

Blinding: model identity is never sent to the browser. The client receives
opaque side labels 'a' and 'b', and which model sits on which side is chosen at
random per comparison, so position bias does not attach to a model.
"""

from __future__ import annotations

import argparse
import json
import random
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import db

# Server-side map from an opaque comparison id to the models behind each side.
# Keeping it on the server is what makes the blinding real: a determined voter
# cannot read the answer out of the page source.
_PENDING: Dict[str, Tuple[str, str, str]] = {}
_MAX_PENDING = 5000

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 7 - Blind Code Comparison</title>
<style>
 :root{--bg:#ffffff;--fg:#1a1a1a;--muted:#666;--line:#e2e2e2;--card:#fafafa;--accent:#4C72B0}
 @media (prefers-color-scheme:dark){:root{--bg:#16181d;--fg:#e8e8e8;--muted:#9aa0a6;--line:#2c2f36;--card:#1e2127;--accent:#7aa2d6}}
 *{box-sizing:border-box}
 body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
      font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 header{max-width:1200px;margin:0 auto 18px}
 h1{font-size:19px;margin:0 0 4px}
 .meta{color:var(--muted);font-size:13px}
 .grid{max-width:1200px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:16px}
 @media (max-width:860px){.grid{grid-template-columns:1fr}}
 .card{border:1px solid var(--line);border-radius:8px;background:var(--card);overflow:hidden}
 .card h2{font-size:13px;margin:0;padding:9px 12px;border-bottom:1px solid var(--line);
          text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
 pre{margin:0;padding:12px;overflow-x:auto;font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
     max-height:60vh;white-space:pre}
 .actions{max-width:1200px;margin:18px auto 0;display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
 button{font:inherit;padding:10px 18px;border-radius:7px;border:1px solid var(--line);
        background:var(--card);color:var(--fg);cursor:pointer}
 button.primary{background:var(--accent);color:#fff;border-color:transparent}
 button:hover{filter:brightness(1.08)} button:disabled{opacity:.5;cursor:not-allowed}
 #status{max-width:1200px;margin:14px auto 0;text-align:center;color:var(--muted);font-size:13px;min-height:20px}
</style></head><body>
<header>
  <h1>Blind Code Comparison</h1>
  <div class="meta" id="taskmeta">Loading...</div>
</header>
<div class="grid">
  <div class="card"><h2>Solution A</h2><pre id="codeA"></pre></div>
  <div class="card"><h2>Solution B</h2><pre id="codeB"></pre></div>
</div>
<div class="actions">
  <button class="primary" onclick="vote('a')">A is better</button>
  <button onclick="vote('tie')">Tie / can't tell</button>
  <button class="primary" onclick="vote('b')">B is better</button>
  <button onclick="load()">Skip</button>
</div>
<div id="status"></div>
<script>
let current=null;
const evaluator = localStorage.getItem('evaluator_id') ||
  (()=>{const v='human_'+Math.random().toString(36).slice(2,8);localStorage.setItem('evaluator_id',v);return v;})();

async function load(){
  document.getElementById('status').textContent='';
  const r = await fetch('/api/pair');
  if(!r.ok){document.getElementById('status').textContent='Error loading comparison.';return;}
  const d = await r.json(); current = d.comparison_id;
  document.getElementById('taskmeta').textContent =
    `${d.task_id} - ${d.title} (${d.category}, ${d.difficulty}) - evaluator ${evaluator}`;
  document.getElementById('codeA').textContent = d.code_a;
  document.getElementById('codeB').textContent = d.code_b;
  window.scrollTo(0,0);
}
async function vote(w){
  if(!current) return;
  const buttons=[...document.querySelectorAll('button')]; buttons.forEach(b=>b.disabled=true);
  const r = await fetch('/api/vote',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({comparison_id:current,winner:w,evaluator_id:evaluator})});
  const d = await r.json().catch(()=>({}));
  document.getElementById('status').textContent = r.ok ? `Recorded. Total votes: ${d.total_votes}` : ('Error: '+(d.error||'failed'));
  buttons.forEach(b=>b.disabled=false);
  if(r.ok) load();
}
load();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # noqa: A002
        return  # keep the console readable during a session

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path

        if route in ("/", "/index.html"):
            self._send_html(PAGE)
            return

        if route == "/api/pair":
            try:
                self._send_json(self._build_pair())
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, 500)
            return

        if route == "/api/results":
            try:
                self._send_json(self._results())
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)}, 500)
            return

        self._send_json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/vote":
            self._send_json({"error": "not found"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")

            comparison_id = payload.get("comparison_id")
            winner = payload.get("winner")
            evaluator_id = (payload.get("evaluator_id") or "anonymous")[:50]

            entry = _PENDING.pop(comparison_id, None)
            if entry is None:
                self._send_json({"error": "unknown or expired comparison"}, 400)
                return
            if winner not in db.VALID_WINNERS:
                self._send_json({"error": f"invalid winner {winner!r}"}, 400)
                return

            task_id, model_a, model_b = entry
            db.record_vote(task_id, model_a, model_b, winner, evaluator_id)
            self._send_json({"ok": True, "total_votes": db.vote_counts()["total"]})
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, 500)

    def _build_pair(self) -> dict:
        tasks = self.server.tasks  # type: ignore[attr-defined]
        if not tasks:
            raise RuntimeError("no tasks with two or more extractable solutions")

        task = random.choice(tasks)
        submissions = db.fetch_submissions(str(task["task_id"]))
        if len(submissions) < 2:
            raise RuntimeError(f"task {task['task_id']} has fewer than two solutions")

        first, second = random.sample(submissions, 2)
        # Randomize which model is shown on the left, independently of anything
        # about the models themselves.
        if random.random() < 0.5:
            first, second = second, first

        comparison_id = secrets.token_urlsafe(16)
        if len(_PENDING) > _MAX_PENDING:
            _PENDING.clear()
        _PENDING[comparison_id] = (
            str(task["task_id"]),
            first["model_name"],
            second["model_name"],
        )

        return {
            "comparison_id": comparison_id,
            "task_id": task["task_id"],
            "title": task["title"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "code_a": first["code"],
            "code_b": second["code"],
        }

    def _results(self) -> dict:
        import bradley_terry

        votes = db.fetch_votes()
        counts = db.vote_counts()
        if len(votes) < 10:
            return {"counts": counts, "message": "need at least 10 votes to fit"}

        result = bradley_terry.fit(votes)
        return {
            "counts": counts,
            "converged": result.converged,
            "ranking": [
                {"model": model, "rating": round(rating, 1)}
                for model, rating in result.ranking()
            ],
            "warning": (
                "Includes synthetic votes; not a valid H3 test."
                if counts["synthetic"]
                else None
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 blind voting interface")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    tasks = db.fetch_comparable_tasks()
    print(f"Loaded {len(tasks)} tasks with two or more extractable solutions")
    print(f"Existing votes: {db.vote_counts()}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.tasks = tasks  # type: ignore[attr-defined]
    print(f"Voting interface on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
