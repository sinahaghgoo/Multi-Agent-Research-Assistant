"""Command-line interface.

research-assistant prepare                 # build the local corpus index
research-assistant doctor                  # check Ollama, models and corpus
research-assistant examples                # list example questions
research-assistant ask "question" [opts]   # run the assistant
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import Settings
from .examples import EXAMPLES, EXAMPLES_BY_ID
from .report import report_to_markdown, save_run, trace_to_markdown
from .trace import TraceEvent


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", help="Ollama chat model (default from config/.env)")
    p.add_argument("--no-web", action="store_true", help="disable web search for this run")
    p.add_argument("--base-url", help="Ollama base URL")


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env()
    return s.with_overrides(
        chat_model=getattr(args, "model", None),
        ollama_base_url=getattr(args, "base_url", None),
        web_enabled=False if getattr(args, "no_web", False) else None,
    )


def _live_printer():
    def printer(event: TraceEvent) -> None:
        if event.kind in {"decision", "tool_call", "tool_result", "guard", "error", "node"}:
            print(f"  {event.one_line()}", file=sys.stderr)

    return printer


def cmd_prepare(args: argparse.Namespace) -> int:
    from .corpus import prepare_corpus

    settings = _settings(args)
    print(f"Preparing corpus in {settings.corpus_dir} ...")
    counts = prepare_corpus(settings.corpus_dir)
    for source_id, n in counts.items():
        print(f"  {source_id}: {n} passages")
    print(f"Total: {sum(counts.values())} passages -> {settings.corpus_dir / 'passages.jsonl'}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from .corpus import load_passages
    from .models import check_ollama

    settings = _settings(args)
    ok = True
    status = check_ollama(settings)
    if status.reachable:
        print(f"[ok] Ollama reachable at {settings.ollama_base_url} ({len(status.models)} models)")
        for model, label in [
            (settings.chat_model, "chat"),
            (settings.embedding_model, "embedding"),
        ]:
            if status.has(model):
                print(f"[ok] {label} model '{model}' available")
            else:
                print(f"[!!] {label} model '{model}' missing -> ollama pull {model}")
                ok = ok and label != "chat"
    else:
        print(f"[!!] Ollama not reachable at {settings.ollama_base_url}: {status.detail}")
        ok = False
    passages_file = settings.corpus_dir / "passages.jsonl"
    if passages_file.exists():
        print(f"[ok] corpus index: {len(load_passages(settings.corpus_dir))} passages")
    else:
        print("[!!] corpus not prepared -> run: research-assistant prepare")
        ok = False
    print("\nReady." if ok else "\nSome checks failed; see above.")
    return 0 if ok else 1


def cmd_examples(args: argparse.Namespace) -> int:
    for e in EXAMPLES:
        web = " [needs web]" if e.needs_web else ""
        print(f"- {e.id}{web}\n    {e.question}\n    {e.hint}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .engine import build_assistant

    if args.question in EXAMPLES_BY_ID:
        args.question = EXAMPLES_BY_ID[args.question].question
    settings = _settings(args)
    try:
        assistant = build_assistant(settings)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Index: {assistant.index_summary()}", file=sys.stderr)
    print(f"Running [{args.mode}]: {args.question}\n", file=sys.stderr)
    out = assistant.run(args.question, mode=args.mode, on_event=_live_printer() if args.trace else None)
    markdown = report_to_markdown(out.report)
    if args.output:
        Path(args.output).write_text(markdown, encoding="utf-8")
        print(f"\nReport written to {args.output}", file=sys.stderr)
    else:
        print("\n" + markdown)
    if args.save:
        saved = save_run(out.report, out.events, settings.runs_dir, slug=args.mode)
        print(f"Run artifacts saved to {saved}", file=sys.stderr)
    if args.show_trace:
        print("\n" + trace_to_markdown(out.events))
    r = out.report.run
    c = out.report.citation_check
    print(
        f"\n[{r.status}] {r.duration_s:.0f}s · {r.llm_calls} LLM calls · {r.tool_calls} tool calls · "
        f"citations {c.valid_citations}/{c.total_citations} valid",
        file=sys.stderr,
    )
    return 0 if r.status != "failed" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-assistant", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="download and index the local corpus")
    _add_common(p_prepare)
    p_prepare.set_defaults(func=cmd_prepare)

    p_doctor = sub.add_parser("doctor", help="check environment readiness")
    _add_common(p_doctor)
    p_doctor.set_defaults(func=cmd_doctor)

    p_examples = sub.add_parser("examples", help="list example questions")
    p_examples.set_defaults(func=cmd_examples)

    p_ask = sub.add_parser("ask", help="answer a research question")
    p_ask.add_argument("question", help="the research question, or an example id")
    p_ask.add_argument("--mode", choices=["multi_agent", "single_agent"], default="multi_agent")
    p_ask.add_argument("--trace", action="store_true", help="stream agent activity to stderr")
    p_ask.add_argument(
        "--show-trace", action="store_true", help="print the full trace table after the report"
    )
    p_ask.add_argument("--save", action="store_true", help="save report + trace under runs/")
    p_ask.add_argument("-o", "--output", help="write the report Markdown to this file")
    _add_common(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
