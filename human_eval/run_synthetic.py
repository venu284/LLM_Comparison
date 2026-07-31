#!/usr/bin/env python3
"""End-to-end validation of the Phase 7 pipeline using synthetic votes.

    python run_synthetic.py                 # in memory, touches nothing
    python run_synthetic.py --store         # also write to human_votes
    python run_synthetic.py --clear         # remove synthetic rows only

Every generated row carries evaluator_id 'synthetic_NN', so `--clear` removes
them without touching votes cast by real people.

These numbers validate the machinery. They are NOT a test of H3 -- the votes
are generated from the automated pass/fail results, so agreement with those
results is guaranteed by construction rather than discovered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HUMAN_EVAL_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = HUMAN_EVAL_DIR.parent / "pipeline"
for candidate in (HUMAN_EVAL_DIR, PIPELINE_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import bradley_terry  # noqa: E402
import db  # noqa: E402
from synthetic_votes import (  # noqa: E402
    agreement_with_pass_rate,
    generate_votes,
    recovery_check,
    votes_to_triples,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 synthetic validation")
    parser.add_argument("--votes", type=int, default=1200)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--store", action="store_true", help="Write votes to human_votes")
    parser.add_argument("--clear", action="store_true", help="Delete synthetic votes and exit")
    parser.add_argument("--output", type=str, default="", help="Write results JSON here")
    args = parser.parse_args()

    if args.clear:
        removed = db.clear_votes("synthetic")
        print(f"Removed {removed} synthetic votes. Counts now: {db.vote_counts()}")
        return

    from dotenv import load_dotenv

    load_dotenv(PIPELINE_DIR / ".env")

    from analysis.data import load_experiment_data

    data = load_experiment_data()
    pass_rates = data.overall_pass_rates()

    print(f"Loaded {len(data.runs)} runs from {data.source}")
    print(f"Generating {args.votes} synthetic votes (noise={args.noise}, seed={args.seed})")

    votes = generate_votes(
        data.runs, n_votes=args.votes, noise=args.noise, seed=args.seed, models=data.models
    )
    triples = votes_to_triples(votes)
    print(f"Generated {len(votes)} votes across {len(data.models)} models")

    result = bradley_terry.fit(triples, models=data.models, bootstrap=args.bootstrap)

    print(f"\nBradley-Terry fit: converged={result.converged} in {result.iterations} iterations")
    print(f"log-likelihood={result.log_likelihood:.1f}\n")
    print(f"{'model':16}{'rating':>9}{'95% CI':>20}{'auto pass rate':>16}")
    for model, rating in result.ranking():
        interval = (
            f"[{result.ci_lower[model]:.0f}, {result.ci_upper[model]:.0f}]"
            if result.ci_lower
            else "n/a"
        )
        print(f"{model:16}{rating:>9.1f}{interval:>20}{pass_rates.get(model, 0) * 100:>15.1f}%")

    recovery = recovery_check(result.ranking(), votes)
    agreement = agreement_with_pass_rate(result.ranking(), pass_rates)

    print(f"\nEstimator check (fit vs observed win rates): rho={recovery['spearman_rho']:.3f}")
    print("  -> confirms Bradley-Terry reproduces the ordering in the votes it was given.")
    print(f"\nSimulated-preference vs automated pass rate: rho={agreement['spearman_rho']:.3f}")
    print("  -> this is the shape of the H3 computation, but it is CIRCULAR here:")
    print("     the votes were generated from these same pass rates. Not evidence.")

    stored = 0
    if args.store:
        rows = [
            (
                vote.task_id,
                vote.model_a,
                vote.model_b,
                db.WINNER_A if vote.winner == vote.model_a else db.WINNER_B,
                vote.evaluator_id,
            )
            for vote in votes
        ]
        stored = db.record_votes_bulk(rows)
        print(f"\nStored {stored} synthetic votes. Counts: {db.vote_counts()}")
        print("Undo with: python run_synthetic.py --clear")
    else:
        print("\nNot stored (pass --store to write to human_votes).")

    if args.output:
        payload = {
            "synthetic": True,
            "not_an_h3_test": True,
            "n_votes": len(votes),
            "noise": args.noise,
            "seed": args.seed,
            "converged": result.converged,
            "iterations": result.iterations,
            "log_likelihood": result.log_likelihood,
            "ranking": [
                {
                    "model": model,
                    "rating": rating,
                    "ci_lower": result.ci_lower.get(model),
                    "ci_upper": result.ci_upper.get(model),
                    "auto_pass_rate": float(pass_rates.get(model, 0.0)),
                }
                for model, rating in result.ranking()
            ],
            "estimator_check": recovery,
            "agreement_with_pass_rate_CIRCULAR": agreement,
            "stored_rows": stored,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
