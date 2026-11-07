"""Render a validated Report as Markdown, and persist run artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .schemas import Report
from .text import truncate
from .trace import TraceEvent


def _finding_line(statement: str, ids: list[str], invalid: list[str]) -> str:
    cite = ""
    if ids:
        cite = " " + " ".join(f"[{i}]" for i in ids)
    if invalid:
        cite += " _(unverified: " + ", ".join(invalid) + ")_"
    if not ids and not invalid:
        cite = " _(no citation)_"
    return f"- {statement.rstrip('.')}.{cite}"


def report_to_markdown(report: Report, *, include_quotes: bool = True) -> str:
    r = report.run
    c = report.citation_check
    lines = [
        f"# Research report: {report.question}",
        "",
        f"*Mode: {r.mode} · Model: {r.model} · "
        f"Web: {'on' if r.web_enabled else 'off'} · Status: {r.status} · {r.duration_s:.0f}s · "
        f"{r.llm_calls} LLM calls · {r.tool_calls} tool calls*",
        "",
        "## Answer",
        "",
        report.answer or "_No answer produced._",
        "",
        "## Key findings",
        "",
    ]
    lines += [_finding_line(f.statement, f.evidence_ids, f.invalid_evidence_ids) for f in report.key_findings] or ["_None._"]

    if report.conflicts:
        lines += ["", "## Conflicting or time-sensitive evidence", ""]
        lines += [_finding_line(f.statement, f.evidence_ids, f.invalid_evidence_ids) for f in report.conflicts]

    lines += ["", "## Limitations and open questions", ""]
    lines += [f"- {x}" for x in report.limitations] or ["- None stated."]
    for q in report.open_questions:
        lines.append(f"- *Open question:* {q}")

    lines += ["", "## Evidence", ""]
    if report.evidence:
        for e in report.evidence:
            lines.append(f"**[{e.id}]** {e.source_id} — {e.location}  ")
            lines.append(f"*Note ({e.recorded_by}):* {e.note}  ")
            if include_quotes:
                lines.append(f"> {truncate(e.quote, 600)}")
            lines.append("")
    else:
        lines.append("_No evidence recorded._")

    lines += ["## Sources", ""]
    for s in report.sources:
        bits = [f"**{s.id}** — {s.title}"]
        if s.url:
            bits.append(f"<{s.url}>")
        meta = []
        if s.published:
            meta.append(f"published {s.published}")
        if s.kind == "web" and s.retrieved:
            meta.append(f"retrieved {s.retrieved}")
        if s.license:
            meta.append(f"license: {truncate(s.license, 120)}")
        if s.flags:
            meta.append("flags: " + ", ".join(s.flags))
        line = " ".join(bits)
        if meta:
            line += "  \n  " + "; ".join(meta)
        lines.append(f"- {line}")

    lines += [
        "",
        "## Citation check (automated)",
        "",
        f"- Citation validity: {c.valid_citations}/{c.total_citations} references resolve to "
        f"recorded evidence ({c.validity_rate:.0%}).",
        f"- Findings without a valid citation: {c.uncited_findings}.",
        f"- Answer contains inline citations: {'yes' if c.answer_has_citations else 'no'}.",
    ]
    if c.invalid_ids:
        lines.append(f"- Invalid citation IDs removed/flagged: {', '.join(c.invalid_ids)}.")
    lines.append(
        "\n> Validity means a cited ID maps to evidence recorded this run. It does **not** verify "
        "that the evidence supports the claim — see the evaluation for sampled support checks."
    )
    if r.errors:
        lines += ["", "## Run warnings", ""] + [f"- {truncate(e, 240)}" for e in r.errors]
    return "\n".join(lines) + "\n"


def trace_to_markdown(events: list[TraceEvent]) -> str:
    lines = ["# Execution trace", "", "| # | t (s) | agent | kind | name | detail |", "|--|--|--|--|--|--|"]
    for e in events:
        detail = e.detail.get("summary") or e.detail.get("args") or ""
        detail = str(detail).replace("|", "\\|").replace("\n", " ")
        dur = f" · {e.duration_s:.1f}s" if e.duration_s else ""
        lines.append(f"| {e.seq} | {e.t:.1f} | {e.agent} | {e.kind} | {e.name}{dur} | {truncate(detail, 160)} |")
    return "\n".join(lines) + "\n"


def save_run(
    report: Report, events: list[TraceEvent], runs_dir: Path, slug: str = "run"
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = runs_dir / f"{stamp}-{report.run.mode}-{slug}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report_to_markdown(report), encoding="utf-8")
    (out / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    (out / "trace.md").write_text(trace_to_markdown(events), encoding="utf-8")
    (out / "trace.jsonl").write_text(
        "\n".join(e.model_dump_json() for e in events), encoding="utf-8"
    )
    return out
