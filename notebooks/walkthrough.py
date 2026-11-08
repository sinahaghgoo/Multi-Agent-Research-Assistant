# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Local Multi-Agent Research Assistant — Walkthrough
#
# This notebook runs the assistant **on a real local model via Ollama** (no paid APIs) and shows:
#
# 1. the task and the example documents,
# 2. the workflow,
# 3. a full run on a representative question (trace, evidence, citations, report),
# 4. evaluation results and the multi-agent vs. single-agent comparison, and
# 5. a discussed failure case.
#
# > **Not mocked.** Every run below calls the local model. Execute top-to-bottom from a clean kernel;
# > outputs are produced live. Requires Ollama running with the configured models
# > (`research-assistant doctor`). Set `RA_CHAT_MODEL` to pick the model (default `qwen3.5:9b`;
# > `qwen2.5:7b` is a faster fallback).

# %%
import sys
from pathlib import Path

# Make the package importable whether run from repo root or notebooks/.
ROOT = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from research_assistant.config import Settings
from research_assistant.models import check_ollama

settings = Settings.from_env()
status = check_ollama(settings)
print("Ollama reachable:", status.reachable)
print("Chat model:", settings.chat_model, "| available:", status.has(settings.chat_model))
print("Embedding model:", settings.embedding_model, "| available:", status.has(settings.embedding_model))
assert status.reachable, "Start Ollama with `ollama serve` and pull the models (see README)."

# %% [markdown]
# ## 1. The task and the documents
#
# The assistant answers research questions about **AI governance** over a small, licensed corpus: the
# **EU AI Act** (Regulation (EU) 2024/1689) and two **NIST** AI Risk Management Framework documents.
# Each document is parsed into citable passages — the EU AI Act by Article/Recital/Annex, the NIST
# PDFs by page — so every quoted fact keeps an identifiable location.

# %%
import json

from research_assistant.corpus import load_passages, local_sources

manifest = json.loads((settings.corpus_dir / "manifest.json").read_text())
for src in manifest["sources"]:
    print(f"- {src['short_title']}: {src['title'][:70]}…")
    print(f"    published {src['published']} · {src['license'][:60]}…")

passages = load_passages(settings.corpus_dir)
sources = local_sources(settings.corpus_dir)
print(f"\nTotal passages: {len(passages)} across {len(sources)} documents")
# Show one real passage with its citable location.
art53 = next(p for p in passages if p.id.startswith("eu-ai-act:art-53"))
print(f"\nExample passage [{art53.id}] — {art53.location}:\n{art53.text[:400]}…")

# %% [markdown]
# ## 2. The workflow
#
# A **supervisor** picks the next step (delegate to a web researcher, delegate to a document analyst,
# or finish). Two workers do the research; a **writer** composes a cited report and a deterministic
# **finalize** step validates every citation. The model only chooses *what next* and *tool arguments*;
# routing, budgets, guards, evidence quoting and citation validation are deterministic code.

# %%
from research_assistant.graph import build_multi_agent_graph

graph = build_multi_agent_graph()
try:
    print(graph.get_graph().draw_ascii())
except Exception:
    print("Nodes:", list(graph.get_graph().nodes))

# %% [markdown]
# ## 3. A full run (offline, multi-agent)
#
# We run a question answerable entirely from the local corpus, with web search disabled for
# reproducibility. Building the index embeds the corpus once (then it is cached on disk).

# %%
import time

from research_assistant.engine import ResearchAssistant

assistant = ResearchAssistant(settings.with_overrides(web_enabled=False))
print("Index:", assistant.index_summary())

question = (
    "What transparency and documentation obligations does the EU AI Act place on "
    "providers of general-purpose AI models?"
)
t0 = time.time()
out = assistant.run(question, mode="multi_agent")
print(f"Completed in {time.time() - t0:.0f}s · status={out.report.run.status}")

# %% [markdown]
# ### Execution trace
# Every decision, tool call and retrieved source is recorded.

# %%
import pandas as pd

trace_df = pd.DataFrame(
    {
        "t": round(e.t, 1),
        "agent": e.agent,
        "kind": e.kind,
        "name": e.name,
        "detail": str(e.detail.get("summary") or e.detail.get("args") or "")[:90],
    }
    for e in out.events
)
trace_df

# %% [markdown]
# ### Evidence, citations and the report
# Evidence quotes are copied verbatim from the corpus; the analyst only writes the note.

# %%
for e in out.report.evidence:
    print(f"[{e.id}] {e.source_id} — {e.location}")
    print(f"     note: {e.note}")
    print(f"     quote: {e.quote[:160]}…\n")

c = out.report.citation_check
print(f"Citation validity: {c.valid_citations}/{c.total_citations} resolve to recorded evidence "
      f"({c.validity_rate:.0%}); {c.uncited_findings} uncited findings.")

# %%
from IPython.display import Markdown

from research_assistant.report import report_to_markdown

Markdown(report_to_markdown(out.report))

# %% [markdown]
# ## 4. Evaluation
#
# A small **grounded** evaluation set pairs each question with expected sources and findings (verified
# by hand against the corpus). We evaluate both the **multi-agent** workflow and the **single-agent**
# baseline on the offline questions.
#
# **Honest labelling:** the checks below are *automated and deterministic* — citation validity,
# expected-source recall, expected-finding coverage (keyword match), and a *lexical grounding proxy*
# (word overlap between a claim and its cited evidence; a weak signal, not entailment). Whether a
# claim is *truly supported* is left to a human review step (shown at the end). The model scoring its
# own output is not treated as validation.

# %% [markdown]
# The full 3-question × 2-mode sweep takes several minutes, so it is run once via
# `scripts/run_evaluation.py` and the results are committed to `data/eval/results-latest.json`.
# We **load** those results here for the aggregate comparison, then run one **live** single-agent
# pass below so this notebook still exercises the baseline on the real model.

# %%
eval_results = json.loads((settings.data_dir / "eval" / "results-latest.json").read_text())
print(f"Loaded evaluation: model={eval_results['model']}, web={eval_results['web_enabled']}")
summary = pd.DataFrame(eval_results["per_mode_aggregate"]).T
summary[["completion_rate", "mean_citation_validity", "mean_finding_coverage",
         "mean_grounding_proxy", "mean_duration_s", "mean_llm_calls"]]

# %% [markdown]
# **Reading the comparison.** The multi-agent workflow completed and validly cited *all* questions;
# the single-agent baseline (same tools, same empty-evidence safeguard) completed fewer and covered
# less on average. But it is not uniformly worse — on the EU-vs-NIST comparison question the single
# agent actually covered more — so the win is **reliability and source breadth**, at the cost of more
# LLM calls. Small-n on a local 9B model: directional, not a benchmark.

# %% [markdown]
# ### Live single-agent run (same question as Section 3)
# One agent with the same tools and budget. Compare its evidence and citations to the multi-agent run.

# %%
single = assistant.run(question, mode="single_agent")
sr = single.report
print(f"single-agent: status={sr.run.status} · {sr.run.duration_s:.0f}s · {sr.run.llm_calls} LLM calls · "
      f"{len(sr.evidence)} evidence · citations {sr.citation_check.valid_citations}/"
      f"{sr.citation_check.total_citations} valid")

# %% [markdown]
# ### Manual claim-support review (the honest check)
# Automated validity only proves a citation *resolves*. Whether evidence *supports* a claim is for a
# human to judge — `scripts/run_evaluation.py --manual` writes one row per finding to a CSV with the
# `supported` column left blank. Here are the rows for the multi-agent showcase run, with the lexical
# grounding proxy as a hint (not a verdict).

# %%
from research_assistant.evaluation import manual_review_rows

review_df = pd.DataFrame(
    {
        "finding": row.finding[:70],
        "evidence": "; ".join(row.evidence_ids),
        "grounding_proxy": row.grounding_proxy,
        "supported (human fills in)": row.supported or "?",
    }
    for row in manual_review_rows(out.report, "gpai-obligations")
)
review_df

# %% [markdown]
# ## 5. A failure case
#
# Early runs surfaced a real failure with `qwen3.5:9b`: the document analyst **searched but never
# recorded evidence**, and the supervisor then "finished" — producing an empty, failed report.
#
# **Fixes applied** (both in the repo):
# 1. the analyst prompt now states that searching alone is not enough and it must call
#    `record_evidence` before finishing; and
# 2. a supervisor **guard** forces re-analysis (up to a retry budget) when `finish` is chosen with
#    zero recorded evidence, instead of emitting a hollow report.
#
# The lesson: when a local model misbehaves, back it with a deterministic guard rather than trusting a
# prompt tweak. We can see the guard in action if a run ever records nothing — the run status becomes
# `failed` only after the retry budget is spent, and the limitations section says so explicitly.

# %%
# The run's own self-report of limitations and any warnings (deterministic, not model-written):
for lim in out.report.limitations:
    print("-", lim)
if out.report.run.errors:
    print("\nWarnings:")
    for err in out.report.run.errors:
        print("-", err[:120])

# %% [markdown]
# ---
# **Takeaway.** The assistant produces a structured, source-backed report with an auditable trace,
# preserves source identifiers end to end, copies evidence verbatim, and validates citations in code.
# The multi-agent design earns its keep on multi-part questions; a single agent or plain RAG is
# preferable for simple lookups — quantified by the baseline above.
