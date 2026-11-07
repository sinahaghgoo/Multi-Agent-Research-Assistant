# Handover

## What was built

A **local multi-agent research assistant** (no paid APIs) that answers research questions over an
AI-governance corpus and returns a structured, source-backed report with an inspectable execution
trace. A LangGraph **supervisor** coordinates two specialised agents — a **web researcher** and a
**document analyst** — through genuine tool calling, then a **writer** and a deterministic
**finaliser** produce and validate the report. A **single-agent baseline** shares the same tools for
comparison.

- **Stack:** Python 3.12, LangGraph, langchain-ollama, Ollama (`qwen3.5:9b` default, `qwen2.5:7b`
  fallback), `qwen3-embedding:0.6b`, hybrid BM25+embedding retrieval, trafilatura/pypdf, ddgs,
  Streamlit, uv.
- **Corpus:** EU AI Act + two NIST AI RMF documents → 762 citable passages with provenance,
  checksums, and licences recorded.

## What was actually tested (real execution, not mocked)

- **Real local-model runs** on both models (web off): `qwen2.5:7b` → 7/7 valid citations;
  `qwen3.5:9b` → 9/9 valid citations with inline answer citations, accurate reports grounded in the
  EU AI Act (Article 53, Annex XII, Article 55).
- **Evaluation** (3 offline questions × 2 modes, `qwen3.5:9b`): multi-agent 3/3 completed, 100%
  citation validity, 0.67 finding coverage; single-agent 2/3 completed, 0.42 coverage, fewer LLM
  calls. Results in `data/eval/results-latest.json`. Honest and mixed (single agent out-covered on
  one question; multi-agent is more reliable).
- **Walkthrough notebook** executed from a clean kernel with outputs retained
  (`notebooks/walkthrough.ipynb`).
- **Web path** validated end-to-end: a live web-enabled run fetched two official EU Commission
  GPAI-guidelines pages, analysed them alongside the local corpus, and completed with 17/17 valid
  citations (174 s). `fetch_page` URL/SSRF guards are also unit-tested.
- **Test suite:** 45 tests pass (no Ollama required — scripted fake model). `ruff` clean.
- **Clean install:** `uv sync` + `research-assistant doctor` + `pytest` all green in a fresh venv.

## Exact demo commands

```bash
ollama pull qwen3.5:9b && ollama pull qwen3-embedding:0.6b
uv sync
uv run research-assistant doctor
uv run research-assistant prepare
uv run research-assistant ask gpai-obligations --mode multi_agent --trace
uv run --extra ui streamlit run streamlit_app/app.py          # demo UI
uv run python scripts/run_evaluation.py --no-web --manual       # evaluation
```

## Remaining limitations

- A 9B local model is capable but slow (one cross-document question took ~400 s) and occasionally
  under-covers or, before the guards were added, failed to record evidence. The deterministic guards
  and validation exist precisely because the model is not fully reliable.
- `json_schema`/`json_mode` structured output is unavailable for some Ollama models (HTTP 501); the
  writer uses the `function_calling` method, which works across models.
- Web search via `ddgs` is unofficial and can rate-limit; the assistant runs fully offline without it.
- The grounding metric is a lexical proxy; true claim-support is a manual human check (template
  provided). Evaluation is small-n — directional, not a benchmark.

## Accurate CV bullet

> **Built a local multi-agent research assistant (Python, LangGraph, Ollama) in which a supervisor
> coordinates specialised agents through tool calling to retrieve web sources, analyse documents, and
> synthesise structured, source-backed reports — with end-to-end source tracking, deterministic
> citation validation, bounded/recoverable execution, a single-agent baseline, and an automated +
> manual evaluation.**
