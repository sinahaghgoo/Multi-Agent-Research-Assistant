"""Streamlit demo for the multi-agent research assistant.

Run with:  uv run --extra ui streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import time

import streamlit as st

from research_assistant.config import Settings
from research_assistant.engine import ResearchAssistant
from research_assistant.examples import EXAMPLES
from research_assistant.models import check_ollama
from research_assistant.report import report_to_markdown
from research_assistant.trace import TraceEvent

st.set_page_config(page_title="Local Research Assistant", page_icon="🔎", layout="wide")

KIND_ICON = {
    "node": "⚙️",
    "decision": "🧭",
    "llm": "🧠",
    "tool_call": "🔧",
    "tool_result": "📄",
    "guard": "🛡️",
    "error": "❌",
}


@st.cache_resource(show_spinner="Loading corpus and building the index …")
def get_assistant(chat_model: str, web_enabled: bool) -> ResearchAssistant:
    settings = Settings.from_env().with_overrides(chat_model=chat_model, web_enabled=web_enabled)
    return ResearchAssistant(settings)


def sidebar() -> tuple[str, str, bool]:
    st.sidebar.title("🔎 Research Assistant")
    st.sidebar.caption("Local multi-agent research over an AI-governance corpus (EU AI Act + NIST).")
    settings = Settings.from_env()
    status = check_ollama(settings)
    if status.reachable:
        st.sidebar.success(f"Ollama online · {len(status.models)} models")
    else:
        st.sidebar.error("Ollama not reachable. Start it with `ollama serve`.")

    model = st.sidebar.text_input("Chat model", value=settings.chat_model)
    mode = st.sidebar.radio(
        "Mode",
        ["multi_agent", "single_agent"],
        format_func=lambda m: {
            "multi_agent": "Multi-agent (supervisor)",
            "single_agent": "Single-agent baseline",
        }[m],
    )
    web = st.sidebar.toggle("Enable web search", value=settings.web_enabled)
    st.sidebar.caption(
        "Multi-agent: a supervisor delegates to a web researcher and a document analyst, "
        "then a writer composes a cited report. Single-agent: one agent with the same tools."
    )
    return model, mode, web


def render_trace_live(container, events: list[TraceEvent]) -> None:
    lines = []
    for e in events:
        icon = KIND_ICON.get(e.kind, "•")
        detail = e.detail.get("summary") or e.detail.get("args") or ""
        lines.append(f"{icon} `{e.t:5.1f}s` **{e.agent}** · {e.name} — {str(detail)[:110]}")
    container.markdown("\n\n".join(lines[-40:]) or "_waiting …_")


def main() -> None:
    model, mode, web = sidebar()

    st.title("Ask a research question")
    st.caption(
        "The assistant answers from local AI-governance documents (and the web, if enabled), "
        "returning a structured, source-backed report with an inspectable execution trace."
    )

    example_labels = ["— pick an example —"] + [f"{e.question}" for e in EXAMPLES]
    picked = st.selectbox("Example questions", example_labels, index=0)
    default_q = "" if picked == example_labels[0] else picked
    question = st.text_area(
        "Your question",
        value=default_q,
        height=90,
        placeholder="e.g. What documentation must providers of general-purpose AI models keep?",
    )

    if not st.button("Run research", type="primary", disabled=not question.strip()):
        st.stop()

    try:
        assistant = get_assistant(model, web)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    st.caption(f"Index: {assistant.index_summary()}")
    trace_box = st.expander("Execution trace (live)", expanded=True)
    trace_container = trace_box.empty()
    events: list[TraceEvent] = []
    progress = st.progress(0.0, text="Starting …")
    start = time.time()

    def on_event(e: TraceEvent) -> None:
        events.append(e)
        render_trace_live(trace_container, events)
        progress.progress(min(0.95, len(events) / 40), text=f"{e.agent} · {e.name}")

    with st.spinner(f"Researching with {model} …"):
        out = assistant.run(question, mode=mode, on_event=on_event)
    progress.progress(1.0, text=f"Done in {time.time() - start:.0f}s")

    report = out.report
    r, c = report.run, report.citation_check
    cols = st.columns(5)
    cols[0].metric("Status", r.status)
    cols[1].metric("Duration", f"{r.duration_s:.0f}s")
    cols[2].metric("Evidence", len(report.evidence))
    cols[3].metric("Citations valid", f"{c.valid_citations}/{c.total_citations}")
    cols[4].metric("LLM · tools", f"{r.llm_calls} · {r.tool_calls}")

    tab_report, tab_evidence, tab_sources, tab_trace = st.tabs(
        ["📝 Report", "🔖 Evidence", "📚 Sources", "🧭 Full trace"]
    )
    with tab_report:
        st.markdown(_report_body(report))
        st.download_button(
            "Download report (Markdown)",
            report_to_markdown(report),
            file_name="report.md",
            mime="text/markdown",
        )
    with tab_evidence:
        if not report.evidence:
            st.info("No evidence was recorded.")
        for e in report.evidence:
            st.markdown(f"**[{e.id}]** `{e.source_id}` — {e.location}  \n*{e.recorded_by}:* {e.note}")
            st.caption(e.quote[:700])
    with tab_sources:
        for s in report.sources:
            flags = f" · flags: {', '.join(s.flags)}" if s.flags else ""
            link = f"[{s.url}]({s.url})" if s.url else ""
            st.markdown(f"**{s.id}** — {s.title} {link}  \n`{s.kind}`{flags}")
    with tab_trace:
        st.dataframe(
            [
                {
                    "#": e.seq,
                    "t": e.t,
                    "agent": e.agent,
                    "kind": e.kind,
                    "name": e.name,
                    "detail": str(e.detail.get("summary") or e.detail.get("args") or "")[:120],
                }
                for e in out.events
            ],
            use_container_width=True,
            hide_index=True,
        )


def _report_body(report) -> str:
    """Report markdown without the leading H1 (Streamlit shows the question as the page title)."""
    md = report_to_markdown(report)
    return md.split("\n", 2)[2] if md.startswith("# ") else md


if __name__ == "__main__":
    main()
