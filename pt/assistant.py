"""Agentic assistant harness for bounded process-tracing research tasks.

This module is intentionally narrow: it routes workspace-agent work through
``llm_client`` and returns typed artifacts. It must not import Codex or Claude
Code SDKs directly; those backends are selected by model string and owned by
``llm_client``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from dotenv import load_dotenv
from llm_client import call_llm_structured, render_prompt
from pydantic import BaseModel, Field

PROMPTS_DIR = Path(__file__).parent / "prompts"
SOURCE_PACKET_PROMPT = PROMPTS_DIR / "assistant_source_packet.yaml"
DEFAULT_ASSISTANT_MODEL = os.getenv("PT_ASSISTANT_MODEL", "codex")
DEFAULT_ASSISTANT_MAX_BUDGET = float(
    os.getenv("PT_ASSISTANT_MAX_BUDGET", os.getenv("PT_MAX_BUDGET", "1.0"))
)
SOURCE_PACKET_TASK = "process_tracing.assistant.source_packet"
SOURCE_PACKET_PROMPT_REF = "pt.assistant.source_packet.v1"

_secrets = Path.home() / ".secrets" / "api_keys.env"
if _secrets.exists():
    load_dotenv(_secrets)
load_dotenv(override=True)


class AssistantError(Exception):
    """Raised when assistant task setup or execution fails."""


class SourceCandidate(BaseModel):
    """One proposed source for a process-tracing source packet."""

    title: str = Field(description="Human-readable source title.")
    source_kind: str = Field(
        description="Broad source kind, such as primary legal text, proclamation, memoir, newspaper, archive, secondary narrative, or historiography."
    )
    date_coverage: str = Field(description="Date or period covered by the source.")
    locator: str | None = Field(
        default=None,
        description="URL, citation, archive locator, file path, or null when not yet known.",
    )
    provenance_note: str = Field(description="Where this source comes from and why it is admissible.")
    reliability_note: str = Field(description="Biases, limitations, authorship issues, or source-production risks.")
    expected_observability: str = Field(
        description="What traces this source genre should and should not reveal for the research question."
    )
    relevance_to_question: str = Field(description="Why this source helps discriminate among rival explanations.")


class RivalInterpretation(BaseModel):
    """A rival scholarly or source-grounded interpretation to preserve."""

    interpretation: str = Field(description="The rival interpretation or mechanism.")
    supporting_sources: list[str] = Field(
        default_factory=list,
        description="Source titles or source classes that support or represent this interpretation.",
    )
    discriminating_implication: str = Field(
        description="What evidence would distinguish this interpretation from important rivals."
    )


class SourceGap(BaseModel):
    """A missing source class or trace that limits the packet."""

    missing_source_class: str = Field(description="Concrete source class still missing.")
    why_it_matters: str = Field(description="How the gap could cap inference quality.")
    expected_location: str = Field(description="Where an analyst or agent should look next.")
    priority: Literal["high", "medium", "low"] = Field(description="Collection priority.")


class SourcePacketDraft(BaseModel):
    """Typed draft emitted by the assistant before Slice 1 finalizes the packet contract."""

    case_name: str = Field(description="Case or event being studied.")
    research_question: str = Field(description="Process-tracing research question.")
    focal_window: str = Field(description="Focal temporal window for the outcome and proximate traces.")
    outcome: str = Field(description="Outcome the source packet is designed to explain.")
    source_candidates: list[SourceCandidate] = Field(
        min_length=1,
        description="Candidate sources to include in the packet.",
    )
    rival_interpretations: list[RivalInterpretation] = Field(
        default_factory=list,
        description="Rival interpretations the packet should preserve before testing.",
    )
    observability_notes: list[str] = Field(
        default_factory=list,
        description="Cross-source notes about what the corpus can and cannot reveal.",
    )
    known_gaps: list[SourceGap] = Field(
        default_factory=list,
        description="Missing source classes or traces that still limit inference.",
    )
    proposed_next_steps: list[str] = Field(
        default_factory=list,
        description="Concrete next actions for improving the packet or running the pipeline.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Caveats that should cap claims if not resolved.",
    )


class AssistantRunMetadata(BaseModel):
    """Metadata needed to audit and rerun one assistant task."""

    backend_model: str = Field(description="Agent backend model string passed to llm_client.")
    execution_mode: Literal["workspace_agent"] = Field(description="llm_client execution mode.")
    task: str = Field(description="llm_client task tag.")
    trace_id: str = Field(description="llm_client trace id.")
    max_budget: float = Field(gt=0, description="Per-call max budget passed to llm_client.")
    cwd: str = Field(description="Workspace directory passed to the assistant backend.")
    prompt_ref: str = Field(description="Prompt/spec reference for observability.")
    output_path: str = Field(description="Artifact JSON path.")
    context_paths: list[str] = Field(description="Context files provided to the assistant.")
    created_at: str = Field(description="UTC ISO timestamp when the artifact was written.")


class SourcePacketAssistantArtifact(BaseModel):
    """Persisted assistant result for source-packet drafting."""

    metadata: AssistantRunMetadata
    draft: SourcePacketDraft


def _is_supported_agent_model(model: str) -> bool:
    """Return whether ``model`` names a supported llm_client workspace-agent backend."""

    lower = model.strip().lower()
    return (
        lower == "codex"
        or lower.startswith("codex/")
        or lower.startswith("codex-")
        or lower == "claude-code"
        or lower.startswith("claude-code/")
        or "-codex" in lower
    )


def _read_context_files(paths: list[Path]) -> tuple[str, list[str]]:
    """Read assistant context files into one prompt block."""

    chunks: list[str] = []
    resolved_paths: list[str] = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise AssistantError(f"context file not found: {path}")
        text = resolved.read_text(encoding="utf-8")
        if not text.strip():
            raise AssistantError(f"context file is empty: {path}")
        resolved_paths.append(str(resolved))
        chunks.append(f"## Context file: {resolved}\n\n{text.strip()}")
    return "\n\n---\n\n".join(chunks), resolved_paths


def _build_messages(
    *,
    case_name: str,
    research_question: str | None,
    context_block: str,
) -> list[dict[str, str]]:
    """Render the source-packet assistant prompt."""

    research_question_block = (
        research_question.strip()
        if research_question and research_question.strip()
        else "Infer the research question from the context and state it explicitly."
    )
    messages = render_prompt(
        SOURCE_PACKET_PROMPT,
        case_name=case_name.strip(),
        research_question_block=research_question_block,
        context_block=context_block,
    )
    return [{"role": str(m["role"]), "content": str(m["content"])} for m in messages]


def draft_source_packet(
    *,
    case_name: str,
    context_paths: list[Path],
    output_path: Path,
    research_question: str | None = None,
    model: str = DEFAULT_ASSISTANT_MODEL,
    trace_id: str | None = None,
    max_budget: float = DEFAULT_ASSISTANT_MAX_BUDGET,
    cwd: Path | None = None,
    timeout: int = 600,
) -> SourcePacketAssistantArtifact:
    """Run the workspace-agent assistant and persist a source-packet draft."""

    if not case_name.strip():
        raise ValueError("case_name must be non-empty")
    if not _is_supported_agent_model(model):
        raise ValueError(
            "assistant model must be a Codex or Claude Code backend accepted by "
            "llm_client, e.g. 'codex' or 'claude-code'"
        )
    if max_budget <= 0:
        raise ValueError("max_budget must be greater than 0")
    if not context_paths:
        raise ValueError("at least one context file is required")

    context_block, resolved_context_paths = _read_context_files(context_paths)
    messages = _build_messages(
        case_name=case_name,
        research_question=research_question,
        context_block=context_block,
    )
    resolved_cwd = (cwd or Path.cwd()).expanduser().resolve()
    if not resolved_cwd.is_dir():
        raise AssistantError(f"cwd is not a directory: {resolved_cwd}")
    resolved_output = output_path.expanduser().resolve()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_trace_id = trace_id or f"pt-assistant-source-packet-{uuid4().hex[:12]}"

    try:
        draft, _result = call_llm_structured(
            model,
            messages,
            SourcePacketDraft,
            timeout=timeout,
            num_retries=0,
            execution_mode="workspace_agent",
            task=SOURCE_PACKET_TASK,
            trace_id=resolved_trace_id,
            max_budget=max_budget,
            prompt_ref=SOURCE_PACKET_PROMPT_REF,
            cwd=str(resolved_cwd),
            working_directory=str(resolved_cwd),
            sandbox_mode="workspace-write",
            approval_policy="never",
            codex_transport=os.getenv("PT_ASSISTANT_CODEX_TRANSPORT", "auto"),
            agent_hard_timeout=timeout,
        )
    except Exception as exc:
        raise AssistantError(f"source-packet assistant failed: {exc}") from exc

    artifact = SourcePacketAssistantArtifact(
        metadata=AssistantRunMetadata(
            backend_model=model,
            execution_mode="workspace_agent",
            task=SOURCE_PACKET_TASK,
            trace_id=resolved_trace_id,
            max_budget=max_budget,
            cwd=str(resolved_cwd),
            prompt_ref=SOURCE_PACKET_PROMPT_REF,
            output_path=str(resolved_output),
            context_paths=resolved_context_paths,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        draft=draft,
    )
    resolved_output.write_text(
        json.dumps(artifact.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return artifact


def _source_packet_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register source-packet subcommand arguments."""

    parser = subparsers.add_parser(
        "source-packet",
        help="Draft a process-tracing source packet through llm_client workspace_agent",
    )
    parser.add_argument("--case-name", required=True, help="Case name, e.g. '18 Brumaire'")
    parser.add_argument(
        "--context",
        action="append",
        required=True,
        help="Context file to provide to the assistant. Repeat for multiple files.",
    )
    parser.add_argument("--output", required=True, help="Path to write artifact JSON")
    parser.add_argument(
        "--research-question",
        default=None,
        help="Optional pinned research question. If omitted, the assistant infers it from context.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_ASSISTANT_MODEL,
        help="Agent backend model, e.g. 'codex' or 'claude-code'.",
    )
    parser.add_argument(
        "--trace-id",
        default=None,
        help="Optional llm_client trace id. Defaults to a generated trace.",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=DEFAULT_ASSISTANT_MAX_BUDGET,
        help="Per-call max budget passed to llm_client.",
    )
    parser.add_argument(
        "--cwd",
        default=".",
        help="Workspace directory passed to the assistant backend.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Agent hard timeout in seconds.",
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for assistant tasks."""

    parser = argparse.ArgumentParser(description="Process-tracing assistant tasks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _source_packet_parser(subparsers)
    args = parser.parse_args(argv)

    if args.command == "source-packet":
        try:
            artifact = draft_source_packet(
                case_name=args.case_name,
                research_question=args.research_question,
                context_paths=[Path(p) for p in args.context],
                output_path=Path(args.output),
                model=args.model,
                trace_id=args.trace_id,
                max_budget=args.max_budget,
                cwd=Path(args.cwd),
                timeout=args.timeout,
            )
        except (AssistantError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Source-packet draft: {artifact.metadata.output_path}")
        print(f"Trace: {artifact.metadata.trace_id}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
