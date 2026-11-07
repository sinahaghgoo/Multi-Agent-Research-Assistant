# Interview guide

A cheat sheet for talking through this project: how it works, why it's built this way, how evidence
integrity is enforced, and when a simpler design would be better. Everything here matches the code.

---

## 30-second summary

> A local, API-free research assistant. A LangGraph **supervisor** decides which specialised agent
> runs next — a **web researcher** (search + fetch) or a **document analyst** (retrieve + record
> evidence) — then a **writer** composes a structured report and a **deterministic finaliser**
> validates every citation. It runs on Ollama (`qwen3.5:9b`) over an AI-governance corpus (EU AI Act
> + NIST). The design point is *trustworthy* agentic output: source IDs are preserved end to end,
> evidence quotes are copied verbatim, and citation validity is checked in code, not by the model.

---

## How it works (the data flow)

1. **Supervisor** sees the current state (question, recorded evidence, prior steps) and calls exactly
   one tool: delegate to web research, delegate to document analysis, or finish. This is the only
   place the "what next" decision is made.
2. **Workers** run a bounded tool-calling loop. The document analyst calls `search_documents` then
   `record_evidence`; the web researcher calls `web_search` then `fetch_page`. Each tool call is
   executed by deterministic code, and failures come back as error strings the model can react to.
3. Control returns to the supervisor after each worker. It re-decides until evidence is sufficient or
   a limit is hit.
4. **Writer** receives only the recorded evidence (numbered E1, E2, …) and returns JSON constrained
   to a schema: answer, key findings with evidence IDs, conflicts, limitations, open questions.
5. **Finalize** (no model) validates citations, strips/flags invalid ones, adds deterministic
   limitations, and renders Markdown.

**Model vs. code.** The model decides: next step + task (supervisor), tool arguments (workers), and
report wording + citations (writer). Code owns: routing, budgets, guards, tool execution, ID
assignment, evidence quoting, validation, and rendering. I can point to the exact functions.

---

## Why this architecture

- **Why LangGraph, not a single prompt or a prompt-chained agent?** I wanted the control flow to be
  *explicit and inspectable*: a typed state object, deterministic transitions, and a trace of every
  decision. That makes failures debuggable and the system testable without a model.
- **Why a supervisor + workers, not one agent?** Separating "web gathering" from "document analysis"
  keeps each agent's context small and its tool set focused, which small local models handle better.
  It also gives a natural place to enforce ordering (fetched pages must be analysed before they count
  as evidence).
- **Why so many deterministic guards?** Local 9B models are capable but not reliable enough to trust
  with stopping conditions or citation integrity. The guards (turn/evidence/time limits, forced
  re-analysis when a round records nothing, no-finish-before-evidence) make termination and honesty
  properties of the *system*, not of the model.
- **Why hybrid retrieval?** Legal text needs exact matches ("Article 53", "10^25 FLOP") where BM25
  wins, and conceptual paraphrases where embeddings win. Reciprocal rank fusion combines them with
  no weight tuning.

---

## How evidence integrity is enforced

This is the part I'm most deliberate about:

- **No invented sources or quotes.** `record_evidence` copies the quote verbatim from the stored
  passage and rejects any passage ID the agent hasn't actually retrieved this run. The model writes
  only the *note*, never the quote.
- **IDs preserved end to end.** `source → passage → evidence` IDs are stable; the local corpus uses
  legal locations (`eu-ai-act:art-53:1`). Every finding traces back to an identifiable source.
- **Validity vs. support — kept separate and labelled.** The finaliser checks that each citation
  *resolves* to recorded evidence (deterministic, 100% automatable). Whether the evidence *supports*
  the claim is assessed separately by (a) a lexical grounding proxy and (b) a human filling a
  generated review CSV. I never call the model scoring its own output "validation".
- **Untrusted web content.** Fetched text is wrapped in `<untrusted_content>`, scanned for injection
  patterns, and used only as evidence. `fetch_page` only accepts URLs from `web_search`, and an SSRF
  guard blocks private/loopback/cloud-metadata addresses.

---

## A real failure case (and the fix)

In an early run, `qwen3.5:9b`'s analyst **searched but never recorded evidence**, and the supervisor
then "finished" while claiming evidence existed → an empty, failed report. Two fixes:

1. **Prompt:** the analyst is told explicitly that searching alone is not enough and it must call
   `record_evidence` before finishing.
2. **Guard:** the supervisor now *forces re-analysis* (up to a retry budget) when a `finish` is
   chosen with zero recorded evidence, instead of emitting a hollow report.

This is the pattern I lean on: when a model misbehaves, add a deterministic guard rather than hope a
better prompt fixes it. The failure is also documented in the walkthrough notebook.

---

## When a simpler solution is better

Be honest about this in interviews:

- **Single-agent baseline.** For straightforward, single-document lookups, one agent with the same
  tools matches the multi-agent quality at lower latency and fewer LLM calls. The repo ships this
  baseline (`--mode single_agent`) and the evaluation reports where multiple agents help and where
  they're just overhead.
- **Plain RAG.** If you only need "retrieve top-k and answer with citations", skip the agents
  entirely — a single retrieval + generation call is cheaper and more predictable. The agentic design
  earns its keep when the task needs *decisions*: whether to go to the web, which source to search,
  whether enough evidence has been gathered.
- **No vector DB.** At ~800 passages, a NumPy matrix beats standing up FAISS/Qdrant. I'd only add one
  past tens of thousands of passages or with incremental indexing needs.

---

## Likely questions & crisp answers

- *"How do you stop it looping?"* Per-worker tool-call budgets, a supervisor turn limit, and a
  whole-run deadline — all deterministic. The writer always runs within a reserved time slice.
- *"How do you know a citation is real?"* The finaliser resolves every cited ID against recorded
  evidence and flags any that don't resolve; invalid inline citations are marked in the output.
- *"Is the self-check trustworthy?"* The automated check only proves *validity* (the ID exists).
  Support is judged by a human via the review CSV; I don't dress up model self-scoring as validation.
- *"What breaks on bigger data / web?"* Retrieval moves to a real vector store; web fetches need
  caching and politeness; the grounding proxy should become an NLI model. All are noted as future
  work.
- *"Why local models?"* Cost, privacy, and to show the engineering holds up under real constraints —
  the guards and validation matter *more* with a weaker model, not less.
