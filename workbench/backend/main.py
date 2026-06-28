"""FastAPI backend for the process-tracing step-by-step workbench.

Each pipeline pass is exposed as a separate POST endpoint so the
frontend can run them one at a time, inspect output, edit hypotheses,
and continue.  Session state is kept in-memory.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pt.bayesian import run_bayesian_update
from pt.pass_absence import run_absence
from pt.pass_extract import run_extract
from pt.pass_hypothesize import run_hypothesize
from pt.pass_partition import run_partition
from pt.pass_synthesize import run_synthesize
from pt.pass_test import run_test
from pt.report import generate_report
from pt.schemas import (
    AbsenceResult,
    BayesianResult,
    ExtractionResult,
    Hypothesis,
    HypothesisSpace,
    PartitionAudit,
    ProcessTracingResult,
    SynthesisResult,
    TestingResult,
)

app = FastAPI(title="Process Tracing Workbench", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────

class _Session(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    session_id: str
    text: str
    model: str
    extraction: ExtractionResult | None = None
    hypothesis_space: HypothesisSpace | None = None
    partition_audit: PartitionAudit | None = None
    testing_result: TestingResult | None = None
    absence_result: AbsenceResult | None = None
    bayesian_result: BayesianResult | None = None
    synthesis_result: SynthesisResult | None = None
    status: str = "ready"
    current_pass: str | None = None
    error: str | None = None

_sessions: dict[str, _Session] = {}


def _get(session_id: str) -> _Session:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return _sessions[session_id]


def _reset_downstream(s: _Session, from_pass: str) -> None:
    order = ["extract", "hypothesize", "partition", "test", "absence", "bayesian", "synthesize"]
    idx = order.index(from_pass)
    if idx <= order.index("hypothesize"):
        s.partition_audit = None
    if idx <= order.index("partition"):
        s.testing_result = None
        s.absence_result = None
        s.bayesian_result = None
        s.synthesis_result = None
    if idx <= order.index("test"):
        s.absence_result = None
        s.bayesian_result = None
        s.synthesis_result = None
    if idx <= order.index("bayesian"):
        s.synthesis_result = None


# ── Request / response models ─────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    text: str
    model: str = "gemini/gemini-2.5-flash"


class SessionSummary(BaseModel):
    session_id: str
    model: str
    status: str
    current_pass: str | None
    error: str | None
    passes_complete: list[str]


class HypothesisEditRequest(BaseModel):
    hypotheses: list[dict]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _session_summary(s: _Session) -> SessionSummary:
    done = []
    if s.extraction:        done.append("extract")
    if s.hypothesis_space:  done.append("hypothesize")
    if s.partition_audit:   done.append("partition")
    if s.testing_result:    done.append("test")
    if s.absence_result:    done.append("absence")
    if s.bayesian_result:   done.append("bayesian")
    if s.synthesis_result:  done.append("synthesize")
    return SessionSummary(
        session_id=s.session_id,
        model=s.model,
        status=s.status,
        current_pass=s.current_pass,
        error=s.error,
        passes_complete=done,
    )


# ── Session management ────────────────────────────────────────────────────────

@app.post("/api/sessions", response_model=SessionSummary)
def create_session(req: CreateSessionRequest) -> SessionSummary:
    sid = str(uuid.uuid4())[:8]
    s = _Session(session_id=sid, text=req.text, model=req.model)
    _sessions[sid] = s
    return _session_summary(s)


@app.get("/api/sessions/{session_id}", response_model=SessionSummary)
def get_session(session_id: str) -> SessionSummary:
    return _session_summary(_get(session_id))


@app.get("/api/sessions")
def list_sessions() -> list[SessionSummary]:
    return [_session_summary(s) for s in _sessions.values()]


# ── Pass 1: Extract ───────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/extract")
def run_pass_extract(session_id: str) -> dict:
    s = _get(session_id)
    s.status = "running"; s.current_pass = "extract"; s.error = None
    try:
        s.extraction = run_extract(s.text, model=s.model)
        _reset_downstream(s, "extract")
        s.status = "ready"; s.current_pass = None
        return s.extraction.model_dump()
    except Exception as e:
        s.status = "error"; s.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Pass 2: Hypothesize ───────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/hypothesize")
def run_pass_hypothesize(session_id: str) -> dict:
    s = _get(session_id)
    if not s.extraction:
        raise HTTPException(status_code=400, detail="Run extract first")
    s.status = "running"; s.current_pass = "hypothesize"; s.error = None
    try:
        s.hypothesis_space = run_hypothesize(s.extraction, model=s.model)
        _reset_downstream(s, "hypothesize")
        s.status = "ready"; s.current_pass = None
        return s.hypothesis_space.model_dump()
    except Exception as e:
        s.status = "error"; s.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Edit hypotheses ───────────────────────────────────────────────────────────

@app.put("/api/sessions/{session_id}/hypotheses")
def edit_hypotheses(session_id: str, req: HypothesisEditRequest) -> dict:
    s = _get(session_id)
    if not s.hypothesis_space:
        raise HTTPException(status_code=400, detail="No hypotheses to edit")
    hyps = [Hypothesis.model_validate(h) for h in req.hypotheses]
    s.hypothesis_space = HypothesisSpace(
        research_question=s.hypothesis_space.research_question,
        hypotheses=hyps,
    )
    _reset_downstream(s, "hypothesize")
    return s.hypothesis_space.model_dump()


# ── Pass 2.5: Partition audit ─────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/partition")
def run_pass_partition(session_id: str) -> dict:
    s = _get(session_id)
    if not s.hypothesis_space:
        raise HTTPException(status_code=400, detail="Run hypothesize first")
    s.status = "running"; s.current_pass = "partition"; s.error = None
    try:
        s.partition_audit = run_partition(s.hypothesis_space, model=s.model)
        _reset_downstream(s, "partition")
        s.status = "ready"; s.current_pass = None
        return s.partition_audit.model_dump()
    except Exception as e:
        s.status = "error"; s.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Pass 3: Test ──────────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/test")
def run_pass_test(session_id: str) -> dict:
    s = _get(session_id)
    if not s.hypothesis_space or not s.extraction:
        raise HTTPException(status_code=400, detail="Run extract and hypothesize first")
    s.status = "running"; s.current_pass = "test"; s.error = None
    try:
        s.testing_result = run_test(s.extraction, s.hypothesis_space, model=s.model)
        _reset_downstream(s, "test")
        s.status = "ready"; s.current_pass = None
        return s.testing_result.model_dump()
    except Exception as e:
        s.status = "error"; s.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Pass 3b: Absence ──────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/absence")
def run_pass_absence(session_id: str) -> dict:
    s = _get(session_id)
    if not s.hypothesis_space or not s.extraction or not s.testing_result:
        raise HTTPException(status_code=400, detail="Run extract, hypothesize, and test first")
    s.status = "running"; s.current_pass = "absence"; s.error = None
    try:
        s.absence_result = run_absence(
            s.extraction, s.hypothesis_space, s.testing_result, model=s.model
        )
        s.synthesis_result = None
        s.status = "ready"; s.current_pass = None
        return s.absence_result.model_dump()
    except Exception as e:
        s.status = "error"; s.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Pass 3.5: Bayesian update ─────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/bayesian")
def run_pass_bayesian(session_id: str) -> dict:
    s = _get(session_id)
    if not s.testing_result:
        raise HTTPException(status_code=400, detail="Run test first")
    s.status = "running"; s.current_pass = "bayesian"; s.error = None
    try:
        s.bayesian_result = run_bayesian_update(s.testing_result)
        s.synthesis_result = None
        s.status = "ready"; s.current_pass = None
        return s.bayesian_result.model_dump()
    except Exception as e:
        s.status = "error"; s.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Pass 4: Synthesize ────────────────────────────────────────────────────────

@app.post("/api/sessions/{session_id}/synthesize")
def run_pass_synthesize(session_id: str) -> dict:
    s = _get(session_id)
    if not s.bayesian_result or not s.hypothesis_space or not s.extraction or not s.testing_result:
        raise HTTPException(status_code=400, detail="Complete passes 1-3.5 first")
    if not s.absence_result:
        raise HTTPException(status_code=400, detail="Run absence pass first")
    s.status = "running"; s.current_pass = "synthesize"; s.error = None
    try:
        s.synthesis_result = run_synthesize(
            s.extraction,
            s.hypothesis_space,
            s.testing_result,
            s.bayesian_result,
            s.absence_result,
            model=s.model,
        )
        s.status = "ready"; s.current_pass = None
        return s.synthesis_result.model_dump()
    except Exception as e:
        s.status = "error"; s.error = str(e)
        raise HTTPException(status_code=500, detail=str(e))


# ── Full result + report ──────────────────────────────────────────────────────

def _build_result(s: _Session) -> ProcessTracingResult:
    assert s.extraction and s.hypothesis_space and s.testing_result
    assert s.absence_result and s.bayesian_result and s.synthesis_result
    return ProcessTracingResult(
        extraction=s.extraction,
        hypothesis_space=s.hypothesis_space,
        testing=s.testing_result,
        absence=s.absence_result,
        bayesian=s.bayesian_result,
        synthesis=s.synthesis_result,
        partition_audit=s.partition_audit,
    )


@app.get("/api/sessions/{session_id}/result")
def get_result(session_id: str) -> dict:
    s = _get(session_id)
    if not s.synthesis_result:
        raise HTTPException(status_code=400, detail="Pipeline not complete")
    return _build_result(s).model_dump()


@app.get("/api/sessions/{session_id}/report")
def get_report(session_id: str) -> dict:
    s = _get(session_id)
    if not s.synthesis_result:
        raise HTTPException(status_code=400, detail="Pipeline not complete")
    return {"html": generate_report(_build_result(s))}
