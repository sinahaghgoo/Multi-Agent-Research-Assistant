# Local Multi-Agent Research Assistant

A research assistant that runs **entirely on local models** (via [Ollama](https://ollama.com)) with
**no paid APIs**. A supervisor coordinates specialised agents through genuine tool calling to
retrieve web sources, analyse documents, and synthesise a **structured, source-backed report** with
an inspectable execution trace.

The bundled example corpus is **AI governance** — the EU AI Act and two NIST AI Risk Management
Framework documents — so the demo is useful to a recruiter in a few minutes and the answers are
checkable against primary sources.

> **Why this exists.** It demonstrates agentic workflow design (LangGraph), honest source tracking
> and citation validation, retrieval over real documents, and disciplined engineering (tests,
> evaluation, a single-agent baseline) — on a laptop, for free.

---

## What it does (concrete example)

```bash
research-assistant ask gpai-obligations --mode multi_agent
```

> **Q:** *What transparency and documentation obligations does the EU AI Act place on providers of
> general-purpose AI models?*
>
> **Answer.** The EU AI Act requires providers of general-purpose AI models to draw up and maintain
> technical documentation, provide information to downstream providers who integrate the model, put
> in place a copyright policy, and publish a summary of training content … Providers of models with
> systemic risk face additional obligations and may rely on codes of practice. [E2] [E4] [E5]
>
> **Evidence.** [E2] EU AI Act, Article 53 (Obligations for providers of general-purpose AI models);
> [E4] Annex XII; [E5] Article 55 (…systemic risk). **7/7 citations valid.**

Every run returns:

- a concise **answer** and **key findings**, each tied to evidence IDs;
- **evidence** passages copied verbatim from identified sources (never paraphrased by the model);
- **limitations, conflicts, and open questions**;
- a **Markdown report** you can export; and
- an **execution trace** of every decision, tool call, and retrieved source.

---

## Architecture

A **supervisor** makes the routing decisions; everything safety- or budget-related is deterministic
Python. Two focused workers do the research; a writer and a deterministic finaliser produce the
report.

```mermaid
flowchart TD
    START([research question]) --> SUP{supervisor<br/>picks next step}
    SUP -->|delegate_web_research| WEB[web researcher<br/>web_search · fetch_page]
    SUP -->|delegate_document_analysis| DOC[document analyst<br/>search_documents · record_evidence]
    SUP -->|finish_research| WRITER[writer<br/>structured JSON report]
    WEB --> SUP
    DOC --> SUP
    WRITER --> FINAL[finalize<br/>citation validation · render]
    FINAL --> END([report + trace])

    subgraph Evidence store (LangGraph state)
        S[sources] --- E[evidence] --- U[allowed URLs]
    end
    DOC -.records.-> E
    WEB -.adds.-> S
```

**Who decides what.** The model decides *only*: which step to take next and the task text
(supervisor), which tools to call with which arguments (workers), and the report wording plus which
evidence to cite (writer). **Deterministic code** handles routing, all budgets and guards, tool
execution, ID assignment, evidence quoting, citation validation, and rendering.

| Agent | Responsibility | Tools | Stops when |
|---|---|---|---|
| **Supervisor** | Pick the next step; enforce stopping conditions | `delegate_web_research`, `delegate_document_analysis`, `finish_research` | evidence suffices, turn/time/evidence limit, or no useful step remains |
| **Web researcher** | Find & fetch public sources | `web_search`, `fetch_page` | answered, tool budget, or deadline |
| **Document analyst** | Retrieve passages & record evidence | `search_documents`, `record_evidence` | answered, tool budget, or deadline |
| **Writer** | Compose a cited report from evidence only | structured JSON output (no tools) | one call |
| **Finalize** | Validate citations, render Markdown | — (deterministic) | — |

**Workflow state** (LangGraph `TypedDict`): `question`, `sources`, `evidence`, `allowed_urls`,
`seen_passage_ids`, `turn`, `web_rounds`, `decisions`, `worker_reports`, `draft`, `report`.
Source and passage IDs are preserved end to end, so every claim traces back to an identifiable
source.

### Guardrails & honesty

- **No invented sources or quotes.** Evidence quotes are copied verbatim from stored passages; the
  model can only cite passages it actually retrieved (`record_evidence` rejects unseen IDs).
- **Citation *validity* ≠ *support*.** The finaliser deterministically checks that every citation
  resolves to recorded evidence. Whether the evidence *supports* the claim is a separate question,
  assessed by a lexical grounding proxy plus a **manual review** step (see Evaluation).
- **Untrusted web content.** Fetched pages are wrapped as `<untrusted_content>`, scanned for
  prompt-injection patterns, and never treated as instructions. `fetch_page` only accepts URLs
  returned by `web_search`, and an SSRF guard blocks private/loopback/metadata addresses.
- **Bounded execution.** Per-worker tool budgets, a supervisor turn limit, a whole-run deadline,
  and recoverable error handling (retries, graceful degradation) guarantee termination.

---

## Technology stack

| Choice | Why |
|---|---|
| **Python + LangGraph** | Explicit graph state and deterministic transitions; the control flow is auditable, not hidden in an agent framework's prompt. |
| **Ollama** | Free, local, OpenAI-free tool calling; easy model swaps. |
| **qwen3.5:9b** (default) / **qwen2.5:7b** (fallback) | Strong tool-calling in a size that fits 18 GB RAM (see Models). |
| **Hybrid retrieval (BM25 + embeddings, RRF)** | BM25 nails exact legal references ("Article 53", "10^25 FLOP"); embeddings catch paraphrases. Reciprocal rank fusion needs no tuning. ~800 passages → a NumPy matrix, no vector DB. |
| **qwen3-embedding:0.6b** | Small, multilingual, instruction-aware local embeddings (1024-dim). |
| **trafilatura / pypdf** | Robust main-text extraction from HTML and PDF. |
| **ddgs** | Keyless web search (unofficial; optional — the assistant runs fully offline without it). |
| **pydantic** | Typed records and JSON-schema-constrained report generation. |
| **uv** | Fast, reproducible dependency management (`uv.lock`). |

---

## Models & hardware

Developed and tested on an **Apple M3 Pro, 18 GB RAM, macOS**.

| Config | Model | Download | Notes |
|---|---|---|---|
| **Recommended** | `qwen3.5:9b` (Q4) | ~6.6 GB | Apache-2.0. Tools + 256K context. Thinking disabled for faster tool loops. |
| **Lighter fallback** | `qwen2.5:7b` | ~4.7 GB | Apache-2.0. Faster per run; strong tool calling. Set `RA_CHAT_MODEL=qwen2.5:7b`. |
| **Embeddings** | `qwen3-embedding:0.6b` | ~0.6 GB | Apache-2.0. 1024-dim, instruction-aware. |

A single 9B model plus the embedder fits comfortably in 18 GB. **Expected limitation:** a 9B model
occasionally omits inline citations in prose or needs the budget/guard nudges; the fallback 7B is
faster but slightly weaker at synthesis. Larger models (27B+) are not recommended on 18 GB.

---

## Installation & quickstart

**Prerequisites:** [Ollama](https://ollama.com/download) running locally, and
[uv](https://docs.astral.sh/uv/).

```bash
# 1. Pull the models
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:0.6b

# 2. Install
uv sync

# 3. Check the environment (Ollama, models, corpus)
uv run research-assistant doctor

# 4. Build the local corpus index (downloads originals, extracts passages)
uv run research-assistant prepare

# 5. Ask
uv run research-assistant examples
uv run research-assistant ask gpai-obligations --mode multi_agent --trace
```

Configuration is via environment variables or a `.env` file — see [`.env.example`](.env.example).
For a fully offline run, add `--no-web` (or set `RA_WEB_ENABLED=false`).

### Interfaces

| Interface | Command |
|---|---|
| **CLI** | `uv run research-assistant ask "<question>" [--mode multi_agent\|single_agent] [--no-web] [--trace] [--save] [-o report.md]` |
| **Streamlit demo** | `uv run --extra ui streamlit run streamlit_app/app.py` |
| **Walkthrough notebook** | `notebooks/walkthrough.ipynb` (executed, outputs retained) |
| **Python API** | `from research_assistant.engine import build_assistant` |

---

## Data provenance & licensing

The example corpus uses public, reusable documents. [`data/corpus/manifest.json`](data/corpus/manifest.json)
records each source's URL, retrieval date, SHA-256 checksum, and licence. Raw originals are
re-downloadable and **not committed**; the extracted passages (`passages.jsonl`) are.

| Source | Publisher | Reuse |
|---|---|---|
| EU AI Act — Regulation (EU) 2024/1689 | EU (EUR-Lex) | Reusable under the EUR-Lex legal notice (Commission Decision 2011/833/EU). Only the Official Journal text is authentic. |
| NIST AI RMF 1.0 (AI 100-1) | NIST | U.S. Government work; public information, credit requested. |
| NIST GenAI Profile (AI 600-1) | NIST | U.S. Government work; public information, credit requested. |

Web pages fetched at runtime are the user's responsibility and are treated as untrusted evidence.

---

## Evaluation

A small grounded evaluation set ([`data/eval/questions.json`](data/eval/questions.json)) pairs each
question with expected sources and expected findings, verified by hand against the corpus.

```bash
uv run python scripts/run_evaluation.py --modes multi_agent single_agent --no-web --manual
```

**Automated, deterministic checks** (no model judging itself): citation validity, expected-source
recall, expected-finding coverage, and a **lexical grounding proxy** (word overlap between a claim
and its cited evidence — a weak signal, not entailment). **Operational:** completion, latency, LLM
and tool calls. **Manual:** a generated CSV where a human marks whether each sampled claim is truly
supported by its evidence — the honest claim-support check. The same model scoring its own output
is **not** presented as independent validation.

> Measured results and the multi-agent-vs-single-agent comparison are in
> `notebooks/walkthrough.ipynb` and committed under `runs/eval/`. _(This section is updated with
> each evaluation run; numbers are reproducible with the command above.)_

### Single-agent baseline

The same tools and a comparable total tool budget are given to **one** agent (`--mode single_agent`).
The comparison shows where a supervisor helps (focused sub-tasks, cleaner evidence) and where it
adds overhead without improving the result — reported honestly rather than assumed.

---

## Design trade-offs & future work

- **Supervisor vs. single agent.** Multiple agents add LLM calls and latency. The split pays off
  for multi-part questions and keeping web vs. document work separate; for simple lookups a single
  agent is as good and cheaper. The baseline quantifies this.
- **Local models are the constraint.** A 9B model is capable but slower and occasionally sloppy with
  inline citations — hence the deterministic validation and guards rather than trusting the model.
- **No vector DB.** At ~800 passages a NumPy matrix is simplest and fastest; a larger corpus would
  warrant FAISS/Qdrant and incremental indexing.
- **Future:** per-claim NLI-based support scoring; reranking; multi-hop sub-questions; richer
  conflict detection across sources; caching of web fetches.

---

## Development

```bash
uv run pytest            # 43 tests, no Ollama required (scripted fake model)
uv run ruff check .      # lint
uv run ruff format .     # format
```

Tests cover retrieval, tool guards, agent loops, graph termination and supervisor guards, citation
validation, and report rendering. Tests needing a live model are marked `@pytest.mark.ollama` and
skipped by default.

## Repository layout

```
src/research_assistant/   # package: graph, agents, tools, retrieval, corpus, evaluation, report
data/corpus/              # manifest + extracted passages (raw originals re-downloaded)
data/eval/                # grounded evaluation questions
scripts/run_evaluation.py # evaluation runner
streamlit_app/            # demo UI
notebooks/                # executed walkthrough
tests/                    # pytest suite
```

## License

MIT (code). Corpus documents retain their own licences (see above).
