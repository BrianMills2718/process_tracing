"""Local interactive trace-execution host state and stage runner.

The host is a thin orchestration layer over the existing pipeline passes. It
creates a run record, executes one stage at a time, persists typed artifacts,
and keeps the run state readable over JSON APIs and the workbench UI.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from pt.apply_refinement import apply_refinement
from pt.bayesian import run_bayesian_update
from pt.pass_absence import run_absence
from pt.pass_extract import run_extract
from pt.pass_hypothesize import run_hypothesize
from pt.pass_refine import run_refine
from pt.pass_synthesize import run_synthesize
from pt.pass_test import run_test
from pt.report import generate_report
from pt.pipeline import _source_text_sha256
from pt.schemas import (
    AbsenceResult,
    BayesianResult,
    ExtractionResult,
    HypothesisSpace,
    ProcessTracingResult,
    RefinementResult,
    SynthesisResult,
    TestingResult,
)
from pt.source_packet import SourcePacket, load_source_packet
from pt.source_coverage import build_source_coverage

StageId = Literal["setup", "extract", "hypothesize", "test", "absence", "update", "synthesize", "refine"]
RunStatus = Literal["ready", "running", "complete", "failed"]
StageStatus = Literal["ready", "blocked", "running", "complete", "failed", "skipped"]

STAGE_ORDER: list[StageId] = [
    "setup",
    "extract",
    "hypothesize",
    "test",
    "absence",
    "update",
    "synthesize",
    "refine",
]

STAGE_GUIDES: dict[StageId, dict[str, Any]] = {
    "setup": {
        "purpose": "Validate inputs and record the run configuration.",
        "consumes": ["input_path", "source_packet_path", "theories_path"],
        "produces": ["run_config.json"],
        "audit_questions": [
            "Are the corpus, source packet, and theories wired to the same case?",
        ],
        "tooltip": "Setup pins the run configuration; it is not evidence.",
    },
    "extract": {
        "purpose": "Turn the input text into typed causal objects.",
        "consumes": ["source text", "source packet context"],
        "produces": ["ExtractionResult"],
        "audit_questions": [
            "Are extracted evidence items traceable to the input text?",
        ],
        "tooltip": "Extraction should reflect the text, not the packet metadata.",
    },
    "hypothesize": {
        "purpose": "Build rival explanations for the pinned question.",
        "consumes": ["ExtractionResult", "theories", "research question"],
        "produces": ["HypothesisSpace"],
        "audit_questions": [
            "Are rival mechanisms distinct and tied to the same question?",
        ],
        "tooltip": "Hypotheses should compete, not restate the same mechanism.",
    },
    "test": {
        "purpose": "Assign coherent likelihood vectors to each evidence item.",
        "consumes": ["ExtractionResult", "HypothesisSpace"],
        "produces": ["TestingResult"],
        "audit_questions": [
            "Does every evidence item cover every hypothesis?",
        ],
        "tooltip": "The matrix is the core diagnostic readout.",
    },
    "absence": {
        "purpose": "Record missing traces that matter for the current claims.",
        "consumes": ["ExtractionResult", "HypothesisSpace", "TestingResult"],
        "produces": ["AbsenceResult"],
        "audit_questions": [
            "Is the absence actually informative for the claim scope?",
        ],
        "tooltip": "Absence should cap claims only when the missing trace matters.",
    },
    "update": {
        "purpose": "Convert the test matrix into Bayesian support.",
        "consumes": ["TestingResult", "priors"],
        "produces": ["BayesianResult"],
        "audit_questions": [
            "Are the posterior shifts traceable to the evidence matrix?",
        ],
        "tooltip": "Support is comparative, not a truth probability.",
    },
    "synthesize": {
        "purpose": "Write the analytical narrative and caveats.",
        "consumes": ["ExtractionResult", "HypothesisSpace", "TestingResult", "BayesianResult", "AbsenceResult"],
        "produces": ["SynthesisResult", "ProcessTracingResult", "report.html"],
        "audit_questions": [
            "Does the narrative match the typed evidence trail?",
        ],
        "tooltip": "Synthesis should explain the result, not invent it.",
    },
    "refine": {
        "purpose": "Re-read the text and apply a bounded refinement delta.",
        "consumes": ["ProcessTracingResult", "source text"],
        "produces": ["RefinementResult", "updated ProcessTracingResult"],
        "audit_questions": [
            "Does the second reading change inference only when rerun?",
        ],
        "tooltip": "Refinement is a second reading, not a shortcut around review.",
    },
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class TraceRunRequest(BaseModel):
    """Request payload for creating a host run."""

    input_path: str = Field(description="Path to the input text for the run.")
    source_packet_path: str | None = Field(default=None, description="Optional source packet JSON.")
    research_question: str | None = Field(default=None, description="Optional pinned research question.")
    theories_path: str | None = Field(default=None, description="Optional theories text file.")
    model: str | None = Field(default=None, description="Optional model override for LLM-backed stages.")
    refine: bool = Field(default=True, description="Whether the run should include the refine stage.")
    max_budget: float | None = Field(default=None, description="Optional budget hint for the run.")


class StageGuide(BaseModel):
    """Human-readable guide for one stage."""

    stage_id: StageId
    purpose: str
    consumes: list[str]
    produces: list[str]
    audit_questions: list[str]
    tooltip: str


class StageArtifact(BaseModel):
    """Persisted typed artifact for a stage."""

    artifact_id: str
    stage_id: StageId
    schema_name: str
    path: str
    summary: str
    provenance: dict[str, Any] = Field(default_factory=dict)


class StageExecution(BaseModel):
    """Status and outputs for one stage execution."""

    stage_id: StageId
    status: StageStatus
    started_at: str | None = None
    completed_at: str | None = None
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)
    summary: str | None = None
    error: str | None = None
    artifacts: list[StageArtifact] = Field(default_factory=list)


class TraceRun(BaseModel):
    """Durable state for one interactive host run."""

    run_id: str
    case_name: str
    input_path: str
    source_packet_path: str | None = None
    theories_path: str | None = None
    research_question: str | None = None
    model: str | None = None
    refine: bool = True
    max_budget: float | None = None
    output_dir: str
    status: RunStatus = "ready"
    current_stage: str = "extract"
    created_at: str = Field(default_factory=_utcnow)
    updated_at: str = Field(default_factory=_utcnow)
    stages: list[StageExecution] = Field(default_factory=list)
    guides: dict[str, StageGuide] = Field(default_factory=dict)
    result_path: str | None = None
    report_path: str | None = None
    source_packet_summary: dict[str, Any] | None = None


class TraceHostError(Exception):
    """Raised when a run or stage cannot be created or advanced."""


class StageOrderError(TraceHostError):
    """Raised when a stage is requested before its prerequisites are ready."""


def build_stage_guides() -> dict[str, StageGuide]:
    """Return typed guides for every supported stage."""

    return {
        stage_id: StageGuide(stage_id=stage_id, **payload)
        for stage_id, payload in STAGE_GUIDES.items()
    }


class TraceHostStore:
    """Filesystem-backed run store and stage runner."""

    def __init__(self, repo_root: Path, output_root: Path | None = None) -> None:
        self.repo_root = repo_root
        self.output_root = output_root or (repo_root / "output" / "workbench_runs")
        self.output_root.mkdir(parents=True, exist_ok=True)

    def create_run(self, request: TraceRunRequest) -> TraceRun:
        """Create a new run and persist the setup state."""

        input_path = self._resolve_repo_path(request.input_path)
        if not input_path.is_file():
            raise TraceHostError(f"input file not found: {request.input_path}")
        source_packet = self._load_packet(request.source_packet_path)
        theories_path = self._resolve_repo_path(request.theories_path) if request.theories_path else None
        if theories_path is not None and not theories_path.is_file():
            raise TraceHostError(f"theories file not found: {request.theories_path}")
        run_id = self._make_run_id()
        output_dir = self.output_root / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        case_name = source_packet.case_name if source_packet is not None else input_path.stem
        run = TraceRun(
            run_id=run_id,
            case_name=case_name,
            input_path=self._relativize(input_path),
            source_packet_path=self._relativize(self._resolve_repo_path(request.source_packet_path))
            if request.source_packet_path
            else None,
            theories_path=self._relativize(theories_path) if theories_path is not None else None,
            research_question=request.research_question,
            model=request.model,
            refine=request.refine,
            max_budget=request.max_budget,
            output_dir=self._relativize(output_dir),
            status="ready",
            current_stage="extract",
            guides=build_stage_guides(),
            source_packet_summary=(
                source_packet.to_summary(request.source_packet_path).model_dump()
                if source_packet is not None
                else None
            ),
            stages=self._initial_stages(request.refine),
        )
        self._write_json(output_dir / "run_config.json", request.model_dump())
        self._write_json(output_dir / "run.json", run.model_dump())
        if source_packet is not None and request.source_packet_path is not None:
            self._write_json(output_dir / "source_packet.json", source_packet.model_dump())
        if theories_path is not None:
            self._write_text(output_dir / "theories.txt", theories_path.read_text(encoding="utf-8"))
        self._write_text(output_dir / "input_text.txt", input_path.read_text(encoding="utf-8"))
        return run

    def get_run(self, run_id: str) -> TraceRun:
        """Load a run from disk."""

        path = self._run_dir(run_id) / "run.json"
        if not path.is_file():
            raise TraceHostError(f"run not found: {run_id}")
        return TraceRun.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def run_stage(self, run_id: str, stage_id: StageId, *, force: bool = False) -> dict[str, Any]:
        """Execute one stage and persist the typed artifact."""

        run = self.get_run(run_id)
        self._validate_stage_order(run, stage_id)
        stage = self._get_stage(run, stage_id)
        if stage.status == "complete" and not force:
            return {
                "ok": True,
                "run": self._serialize_run(run),
                "stage": stage.model_dump(),
                "artifacts": [artifact.model_dump() for artifact in stage.artifacts],
            }
        if stage_id == "setup":
            raise StageOrderError("setup is completed when the run is created")
        stage.started_at = _utcnow()
        stage.status = "running"
        stage.error = None
        self._save_run(run)
        try:
            artifact = self._execute_stage(run, stage_id)
            stage.completed_at = _utcnow()
            stage.status = "complete"
            stage.artifacts = [artifact]
            stage.summary = artifact.summary
            stage.output_refs = [Path(artifact.path).name]
            if not stage.input_refs:
                stage.input_refs = self._default_inputs_for_stage(run, stage_id)
            self._advance_run_state(run, stage_id)
            self._save_run(run)
            return {"ok": True, "run": self._serialize_run(run), "stage": stage.model_dump(), "artifacts": [artifact.model_dump()]}
        except Exception as exc:
            stage.completed_at = _utcnow()
            stage.status = "failed"
            stage.error = f"{exc.__class__.__name__}: {exc}"
            run.status = "failed"
            run.current_stage = stage_id
            run.updated_at = _utcnow()
            self._save_run(run)
            raise

    def _execute_stage(self, run: TraceRun, stage_id: StageId) -> StageArtifact:
        output_dir = self._run_dir(run.run_id)
        text = self._read_text(self._resolve_repo_path(run.input_path))
        packet = self._load_packet(run.source_packet_path)
        source_context = packet.to_prompt_context() if packet is not None else None
        if stage_id == "extract":
            extraction = run_extract(
                text,
                model=run.model,
                source_packet_context=source_context,
                trace_id=run.run_id,
            )
            self._write_json(output_dir / "extraction.json", extraction.model_dump())
            return self._artifact(
                stage_id,
                "ExtractionResult",
                output_dir / "extraction.json",
                f"{len(extraction.events)} events, {len(extraction.evidence)} evidence items, {len(extraction.mechanisms)} mechanisms",
                trace_id=run.run_id,
            )
        if stage_id == "hypothesize":
            extraction = self._load_json_model(output_dir / "extraction.json", ExtractionResult)
            hypothesis_space = run_hypothesize(
                extraction,
                model=run.model,
                theories=self._read_text(self._resolve_repo_path(run.theories_path)) if run.theories_path else None,
                research_question=run.research_question,
                source_packet_context=source_context,
                trace_id=run.run_id,
            )
            self._write_json(output_dir / "hypothesis_space.json", hypothesis_space.model_dump())
            return self._artifact(
                stage_id,
                "HypothesisSpace",
                output_dir / "hypothesis_space.json",
                f"{len(hypothesis_space.hypotheses)} hypotheses",
                trace_id=run.run_id,
            )
        if stage_id == "test":
            extraction = self._load_json_model(output_dir / "extraction.json", ExtractionResult)
            hypothesis_space = self._load_json_model(output_dir / "hypothesis_space.json", HypothesisSpace)
            testing = run_test(extraction, hypothesis_space, model=run.model, trace_id=run.run_id)
            self._write_json(output_dir / "testing.json", testing.model_dump())
            return self._artifact(
                stage_id,
                "TestingResult",
                output_dir / "testing.json",
                f"{len(testing.evidence_likelihoods)} likelihood vectors",
                trace_id=run.run_id,
            )
        if stage_id == "absence":
            extraction = self._load_json_model(output_dir / "extraction.json", ExtractionResult)
            hypothesis_space = self._load_json_model(output_dir / "hypothesis_space.json", HypothesisSpace)
            testing = self._load_json_model(output_dir / "testing.json", TestingResult)
            absence = run_absence(extraction, hypothesis_space, testing, model=run.model, trace_id=run.run_id)
            self._write_json(output_dir / "absence.json", absence.model_dump())
            damaging = sum(1 for item in absence.evaluations if item.severity == "damaging")
            return self._artifact(
                stage_id,
                "AbsenceResult",
                output_dir / "absence.json",
                f"{len(absence.evaluations)} absence findings, {damaging} damaging",
                trace_id=run.run_id,
            )
        if stage_id == "update":
            hypothesis_space = self._load_json_model(output_dir / "hypothesis_space.json", HypothesisSpace)
            testing = self._load_json_model(output_dir / "testing.json", TestingResult)
            bayesian = run_bayesian_update(
                testing,
                [hypothesis.id for hypothesis in hypothesis_space.hypotheses],
            )
            self._write_json(output_dir / "bayesian.json", bayesian.model_dump())
            return self._artifact(
                stage_id,
                "BayesianResult",
                output_dir / "bayesian.json",
                f"Top support {bayesian.posteriors[0].final_posterior:.3f}" if bayesian.posteriors else "No posteriors",
                trace_id=run.run_id,
            )
        if stage_id == "synthesize":
            extraction = self._load_json_model(output_dir / "extraction.json", ExtractionResult)
            hypothesis_space = self._load_json_model(output_dir / "hypothesis_space.json", HypothesisSpace)
            testing = self._load_json_model(output_dir / "testing.json", TestingResult)
            absence = self._load_json_model(output_dir / "absence.json", AbsenceResult)
            bayesian = self._load_json_model(output_dir / "bayesian.json", BayesianResult)
            synthesis = run_synthesize(
                extraction,
                hypothesis_space,
                testing,
                bayesian,
                absence,
                model=run.model,
                trace_id=run.run_id,
            )
            self._write_json(output_dir / "synthesis.json", synthesis.model_dump())
            result = self._build_result(run, extraction, hypothesis_space, testing, absence, bayesian, synthesis, packet)
            self._write_json(output_dir / "result.json", result.model_dump())
            self._write_text(output_dir / "report.html", generate_report(result))
            run.result_path = self._relativize(output_dir / "result.json")
            run.report_path = self._relativize(output_dir / "report.html")
            return self._artifact(
                stage_id,
                "ProcessTracingResult",
                output_dir / "result.json",
                f"{len(synthesis.verdicts)} verdicts, report written",
                trace_id=run.run_id,
            )
        if stage_id == "refine":
            if not run.refine:
                raise TraceHostError("refine stage is disabled for this run")
            extraction = self._load_json_model(output_dir / "extraction.json", ExtractionResult)
            hypothesis_space = self._load_json_model(output_dir / "hypothesis_space.json", HypothesisSpace)
            testing = self._load_json_model(output_dir / "testing.json", TestingResult)
            absence = self._load_json_model(output_dir / "absence.json", AbsenceResult)
            bayesian = self._load_json_model(output_dir / "bayesian.json", BayesianResult)
            synthesis = self._load_json_model(output_dir / "synthesis.json", SynthesisResult)
            refinement = run_refine(
                text,
                extraction,
                hypothesis_space,
                bayesian,
                absence,
                synthesis,
                model=run.model,
                trace_id=run.run_id,
            )
            self._write_json(output_dir / "refinement.json", refinement.model_dump())
            extraction, hypothesis_space = apply_refinement(
                extraction,
                hypothesis_space,
                refinement,
                verbose=False,
            )
            testing = run_test(extraction, hypothesis_space, model=run.model, trace_id=run.run_id)
            absence = run_absence(extraction, hypothesis_space, testing, model=run.model, trace_id=run.run_id)
            bayesian = run_bayesian_update(
                testing,
                [hypothesis.id for hypothesis in hypothesis_space.hypotheses],
            )
            synthesis = run_synthesize(
                extraction,
                hypothesis_space,
                testing,
                bayesian,
                absence,
                model=run.model,
                trace_id=run.run_id,
            )
            self._write_json(output_dir / "extraction.json", extraction.model_dump())
            self._write_json(output_dir / "hypothesis_space.json", hypothesis_space.model_dump())
            self._write_json(output_dir / "testing.json", testing.model_dump())
            self._write_json(output_dir / "absence.json", absence.model_dump())
            self._write_json(output_dir / "bayesian.json", bayesian.model_dump())
            self._write_json(output_dir / "synthesis.json", synthesis.model_dump())
            result = self._build_result(run, extraction, hypothesis_space, testing, absence, bayesian, synthesis, packet, refinement)
            self._write_json(output_dir / "result.json", result.model_dump())
            self._write_text(output_dir / "report.html", generate_report(result))
            run.result_path = self._relativize(output_dir / "result.json")
            run.report_path = self._relativize(output_dir / "report.html")
            return self._artifact(
                stage_id,
                "RefinementResult",
                output_dir / "refinement.json",
                f"{len(refinement.new_evidence)} new evidence, {len(refinement.hypothesis_refinements)} hypothesis refinements",
                trace_id=run.run_id,
            )
        raise TraceHostError(f"unsupported stage: {stage_id}")

    def _build_result(
        self,
        run: TraceRun,
        extraction: ExtractionResult,
        hypothesis_space: HypothesisSpace,
        testing: TestingResult,
        absence: AbsenceResult,
        bayesian: BayesianResult,
        synthesis: SynthesisResult,
        packet: SourcePacket | None,
        refinement: RefinementResult | None = None,
    ) -> ProcessTracingResult:
        source_summary = packet.to_summary(run.source_packet_path) if packet is not None else None
        text = self._read_text(self._resolve_repo_path(run.input_path))
        source_coverage = build_source_coverage(packet, text, extraction) if packet is not None else None
        return ProcessTracingResult(
            source_text_sha256=_source_text_sha256(text),
            extraction=extraction,
            hypothesis_space=hypothesis_space,
            testing=testing,
            absence=absence,
            bayesian=bayesian,
            synthesis=synthesis,
            source_packet=source_summary,
            source_coverage=source_coverage,
            refinement=refinement,
            is_refined=refinement is not None,
        )

    def _artifact(
        self,
        stage_id: StageId,
        schema_name: str,
        path: Path,
        summary: str,
        *,
        trace_id: str,
    ) -> StageArtifact:
        return StageArtifact(
            artifact_id=f"artifact_{stage_id}_{uuid4().hex[:6]}",
            stage_id=stage_id,
            schema_name=schema_name,
            path=self._relativize(path),
            summary=summary,
            provenance={"trace_id": trace_id},
        )

    def _validate_stage_order(self, run: TraceRun, stage_id: StageId) -> None:
        if stage_id == "setup":
            return
        required = STAGE_ORDER[: STAGE_ORDER.index(stage_id)]
        done = {stage.stage_id for stage in run.stages if stage.status == "complete"}
        missing = [item for item in required if item != "setup" and item not in done]
        if missing:
            raise StageOrderError(f"stage '{stage_id}' requires completed {missing[-1]} stage")

    def _advance_run_state(self, run: TraceRun, completed_stage: StageId) -> None:
        run.updated_at = _utcnow()
        if completed_stage == "refine":
            run.status = "complete"
            run.current_stage = "complete"
            return
        next_stage = self._next_stage(completed_stage, run.refine)
        run.current_stage = next_stage or "complete"
        run.status = "complete" if next_stage is None else "ready"
        self._refresh_stage_statuses(run)

    def _refresh_stage_statuses(self, run: TraceRun) -> None:
        completed = {stage.stage_id for stage in run.stages if stage.status == "complete"}
        running = {stage.stage_id for stage in run.stages if stage.status == "running"}
        for stage in run.stages:
            if stage.stage_id == "setup":
                continue
            if stage.stage_id in running:
                continue
            if stage.stage_id in completed:
                continue
            if stage.stage_id == "refine" and not run.refine:
                stage.status = "skipped"
                continue
            prev = self._previous_stage(stage.stage_id)
            stage.status = "ready" if prev in completed else "blocked"

    def _default_inputs_for_stage(self, run: TraceRun, stage_id: StageId) -> list[str]:
        mapping = {
            "extract": [run.input_path, run.source_packet_path or "source_packet:none"],
            "hypothesize": ["extraction.json", run.theories_path or "theories:none"],
            "test": ["extraction.json", "hypothesis_space.json"],
            "absence": ["extraction.json", "hypothesis_space.json", "testing.json"],
            "update": ["testing.json"],
            "synthesize": ["extraction.json", "hypothesis_space.json", "testing.json", "absence.json", "bayesian.json"],
            "refine": ["result.json", run.input_path],
        }
        return mapping.get(stage_id, [])

    def _initial_stages(self, refine: bool) -> list[StageExecution]:
        stages: list[StageExecution] = []
        stages.append(
            StageExecution(
                stage_id="setup",
                status="complete",
                completed_at=_utcnow(),
                input_refs=[],
                output_refs=["run_config.json"],
                summary="Run configuration recorded.",
                artifacts=[],
            )
        )
        for stage_id in STAGE_ORDER[1:]:
            stages.append(
                StageExecution(
                    stage_id=stage_id,
                    status="blocked" if stage_id != "extract" else "ready",
                    input_refs=[],
                    output_refs=[],
                    summary=None,
                )
            )
        if not refine:
            for stage in stages:
                if stage.stage_id == "refine":
                    stage.status = "skipped"
        return stages

    def _next_stage(self, stage_id: StageId, refine: bool) -> StageId | None:
        index = STAGE_ORDER.index(stage_id)
        for candidate in STAGE_ORDER[index + 1 :]:
            if candidate == "refine" and not refine:
                continue
            return candidate
        return None

    def _previous_stage(self, stage_id: StageId) -> StageId | None:
        index = STAGE_ORDER.index(stage_id)
        return STAGE_ORDER[index - 1] if index > 0 else None

    def _run_dir(self, run_id: str) -> Path:
        return self.output_root / run_id

    def _save_run(self, run: TraceRun) -> None:
        path = self._run_dir(run.run_id) / "run.json"
        run.updated_at = _utcnow()
        self._write_json(path, run.model_dump())

    def _serialize_run(self, run: TraceRun) -> dict[str, Any]:
        return json.loads(run.model_dump_json())

    def _get_stage(self, run: TraceRun, stage_id: StageId) -> StageExecution:
        stage = next((item for item in run.stages if item.stage_id == stage_id), None)
        if stage is None:
            raise TraceHostError(f"unknown stage: {stage_id}")
        return stage

    def _load_packet(self, source_packet_path: str | None) -> SourcePacket | None:
        if source_packet_path is None:
            return None
        path = self._resolve_repo_path(source_packet_path)
        return load_source_packet(path)

    def _load_json_model(self, path: Path, model: type[BaseModel]) -> Any:
        return model.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def _resolve_repo_path(self, path: str | Path | None) -> Path:
        if path is None:
            raise TraceHostError("path is required")
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.repo_root / candidate
        return candidate.resolve()

    def _relativize(self, path: Path | None) -> str:
        if path is None:
            raise TraceHostError("path is required")
        try:
            return str(path.resolve().relative_to(self.repo_root))
        except ValueError:
            return str(path.resolve())

    def _read_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _make_run_id(self) -> str:
        return f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:4]}"
