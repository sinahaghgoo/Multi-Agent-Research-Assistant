"""Run the evaluation set across one or more modes and save results.

    uv run python scripts/run_evaluation.py --modes multi_agent single_agent --no-web

Writes per-run reports/traces under runs/eval/ and a results JSON + CSV, and prints an
aggregate table. Automated metrics only; use --manual to also emit a claim-support review CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from research_assistant.config import Settings
from research_assistant.engine import ResearchAssistant
from research_assistant.evaluation import (
    aggregate,
    evaluate_report,
    load_eval_questions,
    manual_review_rows,
)
from research_assistant.report import save_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=["multi_agent", "single_agent"])
    parser.add_argument("--model", default=None, help="override chat model")
    parser.add_argument("--no-web", action="store_true", help="disable web search (default on)")
    parser.add_argument("--only", nargs="*", help="run only these question ids")
    parser.add_argument("--manual", action="store_true", help="also write a manual claim-support review CSV")
    parser.add_argument("--out", default="runs/eval", help="output directory")
    args = parser.parse_args()

    settings = Settings.from_env().with_overrides(
        chat_model=args.model, web_enabled=False if args.no_web else None
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions = load_eval_questions(settings.data_dir / "eval" / "questions.json")
    if args.only:
        questions = [q for q in questions if q.id in set(args.only)]
    if not args.no_web:
        print("note: web enabled; results depend on live search and are less reproducible.")

    print(f"Loading assistant (model={settings.chat_model}, web={'off' if args.no_web else 'on'}) …")
    assistant = ResearchAssistant(settings)
    print(f"Index: {assistant.index_summary()}\n")

    results, manual_rows = [], []
    for mode in args.modes:
        for q in questions:
            if q.needs_web and args.no_web:
                print(f"  skip {q.id} [{mode}] (needs web, web disabled)")
                continue
            print(f"  run {q.id} [{mode}] …", end="", flush=True)
            out = assistant.run(q.question, mode=mode)
            save_run(out.report, out.events, out_dir, slug=f"{mode}-{q.id}")
            result = evaluate_report(out.report, q)
            results.append(result)
            manual_rows.extend(manual_review_rows(out.report, q.id))
            print(
                f" {result.status} · {result.duration_s:.0f}s · coverage {result.finding_coverage:.0%} · "
                f"citations {result.citation_validity:.0%} valid · {result.n_evidence} evidence"
            )

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    results_path = out_dir / f"results-{stamp}.json"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": settings.chat_model,
        "web_enabled": not args.no_web,
        "per_mode_aggregate": {
            mode: aggregate([r for r in results if r.mode == mode]) for mode in args.modes
        },
        "results": [r.to_dict() for r in results],
    }
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = out_dir / f"results-{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].to_dict())) if results else None
        if writer:
            writer.writeheader()
            for r in results:
                row = r.to_dict()
                for k, v in row.items():
                    if isinstance(v, list):
                        row[k] = "; ".join(map(str, v))
                writer.writerow(row)

    print("\n=== Aggregate by mode (automated metrics) ===")
    for mode in args.modes:
        agg = payload["per_mode_aggregate"][mode]
        if agg:
            print(f"\n[{mode}]")
            for k, v in agg.items():
                print(f"  {k:24s} {v}")

    if args.manual and manual_rows:
        manual_path = out_dir / f"manual-review-{stamp}.csv"
        with manual_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(manual_rows[0])))
            writer.writeheader()
            for row in manual_rows:
                d = asdict(row)
                d["evidence_ids"] = "; ".join(d["evidence_ids"])
                writer.writerow(d)
        print(f"\nManual claim-support template -> {manual_path} (fill the 'supported' column by hand)")

    print(f"\nResults: {results_path}\n         {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
