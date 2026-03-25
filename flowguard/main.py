"""
flowguard/api/main.py
─────────────────────
FastAPI application.

ENDPOINT MAP
════════════
POST /parse
    IN : NLPParseRequest  { raw_text, language, context? }
    OUT: ScoreRequest     { obligations[], cash_position }

POST /score
    IN : ScoreRequest     { obligations[], cash_position, scenario_label? }
    OUT: EngineResult     { decisions[], days_to_zero, summary_narrative, ... }

POST /narrate
    IN : NLPNarrateRequest { engine_result, output_language, channel }
    OUT: NarrateResponse   { text, whatsapp_preview }

POST /pipeline
    IN : NLPParseRequest   (same as /parse)
    OUT: PipelineResponse  { score_request, engine_result, narrative }
    ↑ This is the ONE-SHOT endpoint for WhatsApp bots.
      Human text IN → full narrated result OUT.

GET  /health
    OUT: { status, engine_version, spacy_available, st_available }

POST /email
    IN : EmailRequest      { decision_id, engine_result, sender_name, proposed_date? }
    OUT: EmailResponse     { subject, body, tone }

POST /whatif
    IN : WhatIfRequest     { base_score_request, what_if_text }
    OUT: WhatIfResponse    { original, modified, delta_narrative }

POST /audit/{run_id}
    IN : run_id (path param)
    OUT: AuditEntry        { input_hash, decisions_hashes[], computed_at }

Run with:
    uvicorn flowguard.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Project root for serving static files
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from .models import (
    ActionTag,
    CashPosition,
    DecisionRecord,
    EngineResult,
    NLPNarrateRequest,
    NLPParseRequest,
    Obligation,
    ObligationCategory,
    ScoreRequest,
    UserProfile,
)
from .scorer import run_engine
from .parser import (
    ST_AVAILABLE,
    SPACY_AVAILABLE,
    classify_intent,
    draft_negotiation_email,
    extract_what_if_params,
    infer_category,
    infer_flexibility,
    infer_penalty_rate,
    narrate_result,
    narrate_whatsapp_preview,
    parse_text_to_obligations,
)
from .groq_client import (
    groq_parse_input,
    groq_rewrite_cot,
    groq_draft_email,
    is_groq_available,
    get_groq_status,
)
from .database import (
    init_db,
    SessionLocal,
    upsert_obligation,
    store_engine_run,
    get_all_obligations,
    get_run_history,
    delete_obligation,
    get_db_status,
    get_user_profile,
    update_user_profile,
    record_transaction,
    get_transactions,
    get_tx_summary,
    delete_transaction,
    TxMedium,
    TX_PREFIX,
)
from .file_ingest import (
    import_csv,
    import_pdf,
    import_image,
    get_import_capabilities,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENGINE_VERSION = "2.0.0"

# ─────────────────────────────────────────────
# Audit Store (abstracted for PostgreSQL migration)
# ─────────────────────────────────────────────

from abc import ABC, abstractmethod


class AuditStore(ABC):
    """Abstract audit store. Swap InMemoryAuditStore for PostgresAuditStore in prod."""

    @abstractmethod
    def store(self, run_id: str, entry: dict) -> None: ...

    @abstractmethod
    def get(self, run_id: str) -> Optional[dict]: ...

    @abstractmethod
    def list_ids(self) -> list[str]: ...


class InMemoryAuditStore(AuditStore):
    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def store(self, run_id: str, entry: dict) -> None:
        self._store[run_id] = entry

    def get(self, run_id: str) -> Optional[dict]:
        return self._store.get(run_id)

    def list_ids(self) -> list[str]:
        return list(self._store.keys())


_audit_store = InMemoryAuditStore()


# ─────────────────────────────────────────────
# EXTRA REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class NarrateResponse(BaseModel):
    text: str
    whatsapp_preview: str


class PipelineResponse(BaseModel):
    """One-shot: raw text → full narrated result (for WhatsApp bot)."""
    intent: str
    score_request: ScoreRequest
    engine_result: EngineResult
    narrative: str
    whatsapp_preview: str


class EmailRequest(BaseModel):
    decision_index: int = Field(
        ..., description="Index into engine_result.decisions list (0-based)."
    )
    engine_result: EngineResult
    sender_name: str = Field("The Management")
    proposed_date: Optional[date] = Field(None)


class EmailResponse(BaseModel):
    subject: str
    body: str
    tone: str


class WhatIfRequest(BaseModel):
    """
    Scenario analysis.
    Modifies the cash_position (adds an expected inflow) then re-runs the engine.
    """
    base_score_request: ScoreRequest
    what_if_text: str = Field(
        ...,
        description="Natural language what-if. "
                    "E.g. 'What if Kapoor pays me 30000 tomorrow?'"
    )


class WhatIfResponse(BaseModel):
    what_if_text: str
    extracted_params: dict
    original_result: EngineResult
    modified_result: EngineResult
    delta_narrative: str


class AuditEntry(BaseModel):
    run_id: str
    input_hash: str
    computed_at: datetime
    obligation_count: int
    decision_hashes: list[str]


class HealthResponse(BaseModel):
    status: str
    engine_version: str
    spacy_available: bool
    sentence_transformers_available: bool
    timestamp: datetime


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="FlowGuard — Obligation Priority Engine",
    description=(
        "Deterministic cash flow prioritisation for Indian MSMEs.\n\n"
        "**Architecture**:\n"
        "- Engine (scorer.py) is pure math — no LLM, no randomness.\n"
        "- NLP layer (parser.py) converts human text ↔ engine data.\n"
        "- `/pipeline` is the single entry point for WhatsApp bots.\n\n"
        "**Recommended NLP stack**: spaCy + sentence-transformers.\n"
        "See `/health` to confirm which libraries are loaded."
    ),
    version=ENGINE_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _store_audit(result: EngineResult) -> None:
    _audit_store.store(result.run_id, {
        "run_id":           result.run_id,
        "input_hash":       hashlib.sha256(
            json.dumps([d.input_hash for d in result.decisions], sort_keys=True).encode()
        ).hexdigest()[:16],
        "computed_at":      result.computed_at.isoformat(),
        "obligation_count": len(result.decisions),
        "decision_hashes":  [d.input_hash for d in result.decisions],
    })


def _try_groq_parse(raw_text: str, ref: date) -> tuple[Optional[ScoreRequest], dict]:
    """Attempt to parse input via Groq LLM (Layer 2).
    Returns (ScoreRequest | None, metadata_dict).
    metadata_dict always has: intent, bot_reply, filter_query
    """
    meta = {"intent": "STATUS", "bot_reply": "", "filter_query": {}}
    if not is_groq_available():
        return None, meta

    groq_result = groq_parse_input(raw_text, reference_date=ref)
    if groq_result is None:
        return None, meta

    meta["intent"]       = groq_result.get("intent", "STATUS")
    meta["bot_reply"]    = groq_result.get("bot_reply", "")
    meta["filter_query"] = groq_result.get("filter_query", {})

    logger.info("Groq intent=%s, %d obligations",
                meta["intent"], len(groq_result.get("obligations", [])))

    # Only build ScoreRequest for INGEST/STATUS intents that carry obligations
    obligations_raw = groq_result.get("obligations", [])
    if not obligations_raw:
        return None, meta

    obligations: list[Obligation] = []
    for ob_raw in obligations_raw:
        try:
            cat_str  = ob_raw.get("category", "OTHER").upper()
            flex_str = ob_raw.get("flexibility", "NEGOTIABLE").upper()
            due      = ob_raw.get("due_date", (ref + timedelta(days=7)).isoformat())
            desc     = ob_raw.get("description", ob_raw.get("counterparty_name", "Unknown"))

            category   = cat_str if cat_str in [e.value for e in ObligationCategory] else infer_category(desc)
            flexibility = flex_str if flex_str in ("FIXED", "NEGOTIABLE", "DEFERRABLE") else infer_flexibility(desc, category)
            penalty_rate = infer_penalty_rate(category)

            ob = Obligation(
                obligation_id=hashlib.sha256(
                    f"{ob_raw.get('counterparty_name', 'Unknown')}|{ob_raw.get('amount_inr', 0)}|{due}".encode()
                ).hexdigest()[:16],
                counterparty_name=ob_raw.get("counterparty_name", "Unknown"),
                description=desc[:120],
                amount_inr=float(ob_raw.get("amount_inr", 0)),
                penalty_rate_annual_pct=penalty_rate,
                due_date=date.fromisoformat(due),
                max_deferral_days=7,
                category=category,
                flexibility=flexibility,
                relationship_score=50.0,
                is_recurring=category in ("SALARY", "RENT", "UTILITY", "SECURED_LOAN"),
                source_hash=hashlib.sha256(raw_text.encode()).hexdigest()[:16],
                parse_confidence=0.85,
            )
            obligations.append(ob)
        except Exception as e:
            logger.warning("Groq obligation validation failed: %s", e)
            continue

    if not obligations:
        return None, meta

    cash_inr = float(groq_result.get("cash_balance_inr", 0))
    cash_position = CashPosition(
        available_cash_inr=cash_inr if cash_inr > 0 else 0.0,
        as_of_date=ref,
        cash_is_verified=False,
    )
    return ScoreRequest(obligations=obligations, cash_position=cash_position), meta


def _parse_raw_to_score_request(req: NLPParseRequest) -> tuple[ScoreRequest, dict]:
    """Convert NLPParseRequest → (ScoreRequest, groq_meta).

    Pipeline: Groq (llama3-8b) → Pydantic validate → fallback to regex parser.
    groq_meta always contains: intent, bot_reply, filter_query.
    """
    ref = date.today()
    meta = {"intent": "STATUS", "bot_reply": "", "filter_query": {}}

    # Layer 2: Try Groq parsing first
    groq_req, meta = _try_groq_parse(req.raw_text, ref)
    if groq_req is not None and groq_req.obligations:
        logger.info("Using Groq-parsed input (%d obligations, intent=%s)",
                    len(groq_req.obligations), meta["intent"])
        return groq_req, meta

    # Fallback: regex parser (existing deterministic path)
    logger.info("Groq unavailable or failed — using regex parser")
    obligation_dicts, cash_inr = parse_text_to_obligations(req.raw_text, ref)

    if not obligation_dicts:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any obligations from the provided text. "
                   "Try: 'GST 20000 due friday, rent 25000 due 5th, cash 60000'",
        )

    obligations: list[Obligation] = []
    parse_errors: list[str] = []
    for i, ob_dict in enumerate(obligation_dicts):
        try:
            obligations.append(Obligation(**ob_dict))
        except Exception as e:
            parse_errors.append(f"Segment {i+1}: {str(e)[:120]}")

    if not obligations:
        raise HTTPException(
            status_code=422,
            detail=f"All segments failed validation: {parse_errors}"
        )

    if parse_errors:
        logger.warning("Partial parse errors: %s", parse_errors)

    cash_position = CashPosition(
        available_cash_inr=cash_inr if cash_inr > 0 else 0.0,
        as_of_date=ref,
        cash_is_verified=False,
    )
    return ScoreRequest(obligations=obligations, cash_position=cash_position), meta


def _build_delta_narrative(
    original: EngineResult,
    modified: EngineResult,
    params: dict,
) -> str:
    """Plain English comparison of two engine runs."""
    lines = [
        f"If {params.get('counterparty', 'your customer')} pays "
        f"₹{params.get('inflow_amount', 0):,.0f} in {params.get('inflow_day_offset', '?')} day(s):"
    ]
    orig_shortfall = original.cash_shortfall_inr
    new_shortfall  = modified.cash_shortfall_inr

    if new_shortfall < orig_shortfall:
        lines.append(
            f"✅ Shortfall drops from ₹{orig_shortfall:,.0f} to ₹{new_shortfall:,.0f}."
        )
    if original.days_to_zero and not modified.days_to_zero:
        lines.append("✅ Cash no longer runs out in the next 30 days.")
    elif modified.days_to_zero and modified.days_to_zero > (original.days_to_zero or 0):
        lines.append(
            f"✅ Days-to-zero improves from {original.days_to_zero} to {modified.days_to_zero} days."
        )

    # Check if any action changed
    orig_map = {d.obligation_id: d.action for d in original.decisions}
    for d in modified.decisions:
        orig_action = orig_map.get(d.obligation_id)
        if orig_action and orig_action != d.action:
            lines.append(
                f"🔄 {d.counterparty_name}: action changes from "
                f"{orig_action.value} → {d.action.value}."
            )

    if len(lines) == 1:
        lines.append("No material change in payment priority order.")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health():
    """Check engine status and NLP library availability."""
    resp = HealthResponse(
        status="ok",
        engine_version=ENGINE_VERSION,
        spacy_available=SPACY_AVAILABLE,
        sentence_transformers_available=ST_AVAILABLE,
        timestamp=datetime.utcnow(),
    )
    # Attach Groq status as extra info (won't break HealthResponse schema)
    resp_dict = resp.model_dump()
    resp_dict["groq"] = get_groq_status()
    return resp_dict


@app.post(
    "/parse",
    response_model=ScoreRequest,
    tags=["NLP"],
    summary="Human text → structured ScoreRequest",
    description=(
        "**INPUT**: Raw human text (English / Hinglish / Tamil-English mix).\n\n"
        "**OUTPUT**: A `ScoreRequest` with validated `Obligation` objects and `CashPosition`.\n\n"
        "This is the parsing-only endpoint. To go all the way to a decision, use `/pipeline`.\n\n"
        "**Example input**:\n"
        "```\n"
        "GST 20000 due Friday, rent 25000 due 5th, supplier payment 80000 by Thursday. "
        "My cash is 1 lakh.\n"
        "```"
    ),
)
async def parse(req: NLPParseRequest) -> ScoreRequest:
    score_req, _ = _parse_raw_to_score_request(req)  # discard meta, return only ScoreRequest
    return score_req


@app.post(
    "/score",
    response_model=EngineResult,
    tags=["Engine"],
    summary="Structured obligations → scored decisions",
    description=(
        "**INPUT**: `ScoreRequest` with fully-structured `Obligation` objects.\n\n"
        "**OUTPUT**: `EngineResult` with ranked `DecisionRecord` list, "
        "Days-to-Zero, and shortfall.\n\n"
        "**DETERMINISTIC**: Same input always produces identical output.\n\n"
        "The narrative fields (`summary_narrative`, `whatsapp_summary`) are "
        "populated in this endpoint — no separate narrate call needed "
        "if you already have structured data."
    ),
)
async def score(req: ScoreRequest) -> EngineResult:
    try:
        result = run_engine(
            obligations=req.obligations,
            cash_position=req.cash_position,
            scenario_label=req.scenario_label,
        )
        result.summary_narrative  = narrate_result(result, channel="web")
        result.whatsapp_summary   = narrate_whatsapp_preview(result)
        _store_audit(result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post(
    "/narrate",
    response_model=NarrateResponse,
    tags=["NLP"],
    summary="EngineResult → human-readable text",
    description=(
        "**INPUT**: An `EngineResult` (from `/score`) + output preferences.\n\n"
        "**OUTPUT**: Formatted narrative text.\n\n"
        "- `channel='whatsapp'` — emoji-rich, bullet points, under 300-char preview\n"
        "- `channel='web'`      — full markdown with COT and sub-scores\n"
        "- `channel='voice'`    — TTS-friendly, no emoji, flowing sentences\n\n"
        "Language codes: `en`, `hi`, `ta`, `te`  (non-English is partial in v1.0)"
    ),
)
async def narrate(req: NLPNarrateRequest) -> NarrateResponse:
    text    = narrate_result(req.engine_result, channel=req.channel, language=req.output_language)
    preview = narrate_whatsapp_preview(req.engine_result)
    return NarrateResponse(text=text, whatsapp_preview=preview)


@app.post(
    "/pipeline",
    response_model=PipelineResponse,
    tags=["WhatsApp Bot"],
    summary="ONE-SHOT: Human text → full narrated decisions",
    description=(
        "**This is the primary endpoint for WhatsApp bots.**\n\n"
        "**INPUT**: `NLPParseRequest` — the raw WhatsApp message text.\n\n"
        "**OUTPUT**: `PipelineResponse` with:\n"
        "- `intent` — what the user was asking for\n"
        "- `score_request` — the structured data extracted\n"
        "- `engine_result` — the full deterministic decision set\n"
        "- `narrative` — human-readable text to send back to WhatsApp\n"
        "- `whatsapp_preview` — under-300-char preview\n\n"
        "**Example WhatsApp input**:\n"
        "```\n"
        "gst 20k friday rent 25k monday supplier 80k thursday cash 1 lakh\n"
        "```\n\n"
        "The WhatsApp bot should POST this text here and send back `narrative`."
    ),
)
async def pipeline(req: NLPParseRequest):
    """
    Intent-routed pipeline:
      INGEST  → record obligations, return affirmative bot_reply.
      STATUS  → run the full scoring engine, return decisions + narrative.
      FILTER  → query transactions/obligations DB, return matching records as JSON.
    """
    ref = date.today()

    # ── Run Groq to classify intent and extract data ──
    groq_req, meta = _try_groq_parse(req.raw_text, ref)
    intent      = meta.get("intent", "STATUS")
    bot_reply   = meta.get("bot_reply", "")
    filter_q    = meta.get("filter_query", {})

    # ── INGEST: just store and return quick affirmation ──────────────────
    if intent == "INGEST":
        stored_obs = []
        if groq_req and groq_req.obligations:
            db = SessionLocal()
            try:
                for ob in groq_req.obligations:
                    ob_dict = {
                        "counterparty_name": ob.counterparty_name,
                        "amount_inr":        ob.amount_inr,
                        "due_date":          ob.due_date,
                        "category":          ob.category if isinstance(ob.category, str) else ob.category.value,
                        "flexibility":       ob.flexibility if isinstance(ob.flexibility, str) else ob.flexibility.value,
                        "description":       ob.description,
                        "obligation_id":     ob.obligation_id,
                    }
                    row, is_new = upsert_obligation(db, ob_dict, "CHAT", None)
                    stored_obs.append({
                        "ref": ob.obligation_id,
                        "counterparty": ob.counterparty_name,
                        "amount": ob.amount_inr,
                        "new": is_new,
                    })
            finally:
                db.close()

        return {
            "intent":    "INGEST",
            "bot_reply": bot_reply or f"✅ Recorded {len(stored_obs)} obligation(s).",
            "stored":    stored_obs,
        }

    # ── FILTER: build DB query and return results ────────────────────────
    if intent == "FILTER":
        from datetime import date as _date
        db = SessionLocal()
        try:
            fq = filter_q or {}
            # Try transactions table first
            party   = fq.get("counterparty_name")
            medium  = fq.get("medium")
            dirn    = fq.get("direction")
            df_from = _date.fromisoformat(fq["date_from"]) if fq.get("date_from") else None
            df_to   = _date.fromisoformat(fq["date_to"])   if fq.get("date_to")   else None

            txns = get_transactions(
                db,
                direction=dirn,
                medium=medium,
                from_date=df_from,
                to_date=df_to,
                limit=200,
            )
            # Further filter by counterparty name if specified
            if party:
                party_norm = party.lower().strip()
                txns = [t for t in txns if party_norm in (t.counterparty or "").lower()]

            # Also query obligations if category filter specified
            obs_rows = []
            if fq.get("category") or (party and not txns):
                obs_rows = get_all_obligations(
                    db,
                    category=fq.get("category"),
                    active_only=False,
                )
                if party:
                    p = party.lower().strip()
                    obs_rows = [o for o in obs_rows if p in o.counterparty_name.lower()]

        finally:
            db.close()

        return {
            "intent":      "FILTER",
            "bot_reply":   bot_reply or "Fetching matching records…",
            "filter_used": fq,
            "transactions": [t.to_dict() for t in txns],
            "obligations":  [o.to_dict() for o in obs_rows],
            "total_found":  len(txns) + len(obs_rows),
        }

    # ── STATUS: full scoring engine ──────────────────────────────────────
    # Reuse groq_req if Groq already parsed obligations (avoids a second Groq call).
    # Fall back to regex parser if groq_req has no obligations.
    if groq_req and groq_req.obligations:
        score_req = groq_req
    else:
        try:
            score_req, _ = _parse_raw_to_score_request(req)
        except HTTPException:
            # No obligations extractable at all — return the bot_reply alone
            return {
                "intent":    "STATUS",
                "bot_reply": bot_reply or "Sure! Analysing your current cash flow…",
                "note":      "No obligations found in current session. Please ingest some data first.",
            }

    result = run_engine(
        obligations=score_req.obligations,
        cash_position=score_req.cash_position,
    )

    # COT rewriting via llama-3.3-70b
    for decision in result.decisions:
        facts = {
            "counterparty_name": decision.counterparty_name,
            "amount_inr": decision.amount_inr,
            "due_date": decision.due_date.isoformat(),
            "consequence_score": decision.consequence_score,
            "score_band": decision.score_band.value,
            "action": decision.action.value,
            "penalty_per_day_inr": decision.penalty_per_day_inr,
            "cot_reason": decision.cot_reason,
            "cot_tradeoff": decision.cot_tradeoff,
            "cot_downstream": decision.cot_downstream,
        }
        rewrite = groq_rewrite_cot(facts)
        if rewrite:
            decision.cot_reason     = rewrite.get("cot_reason", decision.cot_reason)
            decision.cot_tradeoff   = rewrite.get("cot_tradeoff", decision.cot_tradeoff)
            decision.cot_downstream = rewrite.get("cot_downstream", decision.cot_downstream)

    channel   = "whatsapp"
    narrative = narrate_result(result, channel=channel, language=req.language)
    preview   = narrate_whatsapp_preview(result)

    result.summary_narrative = narrative
    result.whatsapp_summary  = preview
    _store_audit(result)

    return {
        "intent":          "STATUS",
        "bot_reply":       bot_reply or "Sure! Analysing your current cash flow…",
        "score_request":   score_req,
        "engine_result":   result,
        "narrative":       narrative,
        "whatsapp_preview": preview,
    }


@app.post(
    "/email",
    response_model=EmailResponse,
    tags=["NLP"],
    summary="Draft a negotiation email for a deferred obligation",
    description=(
        "**INPUT**: An `engine_result`, the index of the decision to draft for, "
        "sender name, and optional proposed payment date.\n\n"
        "**OUTPUT**: Subject line + email body (tone set by relationship_score).\n\n"
        "Tone bands:\n"
        "- relationship 80-100 → Warm & apologetic\n"
        "- relationship 40-79  → Professional & neutral\n"
        "- relationship 0-39   → Firm & brief\n\n"
        "**Note**: The engine decides the tone.  The NLP layer writes the prose."
    ),
)
async def email(req: EmailRequest) -> EmailResponse:
    decisions = req.engine_result.decisions
    if req.decision_index < 0 or req.decision_index >= len(decisions):
        raise HTTPException(
            status_code=422,
            detail=f"decision_index {req.decision_index} out of range "
                   f"(0–{len(decisions)-1}).",
        )
    d    = decisions[req.decision_index]
    body = draft_negotiation_email(d, req.sender_name, req.proposed_date)
    subject = (
        f"Payment Rescheduling — {d.counterparty_name} — "
        f"₹{d.amount_inr:,.0f}"
    )
    return EmailResponse(subject=subject, body=body, tone=d.email_tone.value)


@app.post(
    "/whatif",
    response_model=WhatIfResponse,
    tags=["Engine"],
    summary="Scenario analysis: what if I receive an extra inflow?",
    description=(
        "**INPUT**: A base `ScoreRequest` + a natural-language what-if question.\n\n"
        "**OUTPUT**: Original result, modified result, and a delta narrative.\n\n"
        "The what-if question is parsed to extract:\n"
        "- `inflow_amount` — how much cash arrives\n"
        "- `inflow_day_offset` — when it arrives (days from today)\n"
        "- `counterparty` — who is paying\n\n"
        "The engine then re-runs with the updated `expected_inflows` "
        "and returns both results for comparison.\n\n"
        "**Example**: `'What if Kapoor pays me 30000 tomorrow?'`"
    ),
)
async def whatif(req: WhatIfRequest) -> WhatIfResponse:
    # Run original
    original = run_engine(
        obligations=req.base_score_request.obligations,
        cash_position=req.base_score_request.cash_position,
    )

    # Extract what-if params
    params = extract_what_if_params(req.what_if_text)
    if params["inflow_amount"] <= 0:
        raise HTTPException(
            status_code=422,
            detail="Could not extract an inflow amount from the what-if text.",
        )

    # Build modified cash position
    modified_cash = req.base_score_request.cash_position.model_copy(deep=True)
    day_key = str(params["inflow_day_offset"])
    existing = modified_cash.expected_inflows.get(day_key, 0.0)
    modified_cash.expected_inflows[day_key] = existing + params["inflow_amount"]

    # Re-run engine
    modified = run_engine(
        obligations=req.base_score_request.obligations,
        cash_position=modified_cash,
    )

    delta = _build_delta_narrative(original, modified, params)
    _store_audit(original)
    _store_audit(modified)

    return WhatIfResponse(
        what_if_text=req.what_if_text,
        extracted_params=params,
        original_result=original,
        modified_result=modified,
        delta_narrative=delta,
    )


@app.get(
    "/audit/{run_id}",
    response_model=AuditEntry,
    tags=["Audit"],
    summary="Retrieve the audit trail for a past run",
    description=(
        "Every `/score`, `/pipeline`, and `/whatif` call stores an audit entry.\n\n"
        "The `input_hash` is a SHA-256 of the obligation inputs — "
        "allows judges to verify the same input always produces the same output.\n\n"
        "In production, replace the in-memory store with SQLite or PostgreSQL."
    ),
)
async def audit(run_id: str) -> AuditEntry:
    entry = _audit_store.get(run_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found.")
    return AuditEntry(**entry)


@app.get("/audit", response_model=list[str], tags=["Audit"])
async def list_audits():
    """List all run IDs in the audit store."""
    return _audit_store.list_ids()


# ─────────────────────────────────────────────
# DATABASE INIT ON STARTUP
# ─────────────────────────────────────────────

@app.on_event("startup")
def startup_event():
    init_db()
    logger.info("FlowGuard database initialized")


# ─────────────────────────────────────────────
# FILE UPLOAD ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/upload/csv", tags=["File Import"])
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file of obligations.
    
    The CSV should have columns like: counterparty/vendor/party, amount/amt, due_date/due.
    Columns are auto-mapped. If headers are non-standard, Groq will attempt to parse.
    Duplicate obligations (same counterparty + amount + due_date) are merged, not duplicated.
    """
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    result = import_csv(content, file.filename or "upload.csv")
    return result.to_dict()


@app.post("/upload/pdf", tags=["File Import"])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF invoice/bill.
    
    Text is extracted via pdfplumber, then parsed with Groq (or regex fallback)
    to identify obligations. Tables within the PDF are also extracted.
    """
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    result = import_pdf(content, file.filename or "upload.pdf")
    return result.to_dict()


@app.post("/upload/image", tags=["File Import"])
async def upload_image(file: UploadFile = File(...)):
    """Upload an invoice/bill image for OCR.
    
    The image is processed with Tesseract OCR (grayscale + sharpen),
    then the extracted text is parsed with Groq to identify obligations.
    Supports: .jpg, .png, .webp, .bmp, .tiff
    """
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    result = import_image(content, file.filename or "upload.jpg")
    return result.to_dict()


# ─────────────────────────────────────────────
# OBLIGATION MANAGEMENT ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/obligations", tags=["Data"])
async def list_obligations(category: Optional[str] = None, active_only: bool = True):
    """List all stored obligations, optionally filtered by category."""
    db = SessionLocal()
    try:
        rows = get_all_obligations(db, category=category, active_only=active_only)
        return {"count": len(rows), "obligations": [r.to_dict() for r in rows]}
    finally:
        db.close()


@app.delete("/obligations/{obligation_id}", tags=["Data"])
async def remove_obligation(obligation_id: str):
    """Delete an obligation by its dedup ID."""
    db = SessionLocal()
    try:
        deleted = delete_obligation(db, obligation_id)
        if not deleted:
            raise HTTPException(404, f"Obligation '{obligation_id}' not found")
        return {"deleted": obligation_id}
    finally:
        db.close()


@app.get("/history", tags=["Data"])
async def run_history(limit: int = 20):
    """List recent engine runs with decisions."""
    db = SessionLocal()
    try:
        runs = get_run_history(db, limit=limit)
        return {
            "count": len(runs),
            "runs": [
                {
                    "run_id": r.run_id,
                    "as_of_date": r.as_of_date.isoformat() if r.as_of_date else None,
                    "available_cash_inr": r.available_cash_inr,
                    "total_obligations": r.total_obligations,
                    "cash_shortfall": r.cash_shortfall,
                    "num_obligations": r.num_obligations,
                    "source_type": r.source_type,
                    "computed_at": r.computed_at.isoformat() if r.computed_at else None,
                }
                for r in runs
            ],
        }
    finally:
        db.close()


@app.get("/profile", tags=["Profile"])
async def get_profile():
    """Fetch the business and personal profile."""
    db = SessionLocal()
    try:
        profile = get_user_profile(db)
        if not profile:
            return {}
        return profile.to_dict()
    finally:
        db.close()


@app.post("/profile", tags=["Profile"])
async def save_profile(profile_data: UserProfile):
    """Create or update the business and personal profile."""
    db = SessionLocal()
    try:
        profile = update_user_profile(db, profile_data.dict(exclude_unset=True))
        return profile.to_dict()
    finally:
        db.close()


# ─────────────────────────────────────────────
# TRANSACTIONS
# ─────────────────────────────────────────────

class TransactionRequest(BaseModel):
    medium: str               # TxMedium value, e.g. "UPI", "BANK_CHEQUE"
    direction: str            # "IN" or "OUT"
    amount_inr: float
    txn_date: str             # ISO date YYYY-MM-DD
    counterparty: Optional[str] = None
    description: Optional[str] = None
    notes: Optional[str] = None
    external_ref: Optional[str] = None   # UPI txn ID / cheque no / receipt no


@app.post("/transactions", tags=["Transactions"])
async def add_transaction(req: TransactionRequest):
    """
    Record a cash movement (IN or OUT).
    The ref_id is auto-built from the medium prefix + external_ref.
    Re-submitting the same transaction (same ref_id) is silently ignored.
    """
    from datetime import date as _date
    try:
        txn_date = _date.fromisoformat(req.txn_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="txn_date must be YYYY-MM-DD")

    if req.direction.upper() not in ("IN", "OUT"):
        raise HTTPException(status_code=422, detail="direction must be 'IN' or 'OUT'")

    db = SessionLocal()
    try:
        row, is_new = record_transaction(
            db=db,
            medium=req.medium,
            direction=req.direction,
            amount_inr=req.amount_inr,
            txn_date=txn_date,
            counterparty=req.counterparty,
            description=req.description,
            notes=req.notes,
            external_ref=req.external_ref,
            source_type="API",
        )
        return {
            "ref_id":     row.ref_id,
            "is_new":     is_new,
            "duplicate":  not is_new,
            "transaction": row.to_dict(),
        }
    finally:
        db.close()


@app.get("/transactions", tags=["Transactions"])
async def list_transactions(
    direction: Optional[str] = None,
    medium: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: int = 200,
):
    """List all transactions with optional filters."""
    from datetime import date as _date
    fd = _date.fromisoformat(from_date) if from_date else None
    td = _date.fromisoformat(to_date) if to_date else None
    db = SessionLocal()
    try:
        rows = get_transactions(db, direction=direction, medium=medium,
                                from_date=fd, to_date=td, limit=limit)
        return {"count": len(rows), "transactions": [r.to_dict() for r in rows]}
    finally:
        db.close()


@app.get("/transactions/summary", tags=["Transactions"])
async def transaction_summary(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    """Cash flow summary: total IN, total OUT, net, broken down by medium."""
    from datetime import date as _date
    fd = _date.fromisoformat(from_date) if from_date else None
    td = _date.fromisoformat(to_date) if to_date else None
    db = SessionLocal()
    try:
        return get_tx_summary(db, from_date=fd, to_date=td)
    finally:
        db.close()


@app.delete("/transactions/{ref_id}", tags=["Transactions"])
async def remove_transaction(ref_id: str):
    """Delete a single transaction record."""
    db = SessionLocal()
    try:
        deleted = delete_transaction(db, ref_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Transaction '{ref_id}' not found")
        return {"deleted": True, "ref_id": ref_id}
    finally:
        db.close()


@app.get("/transactions/mediums", tags=["Transactions"])
async def list_mediums():
    """Return all valid payment mediums and their ref-id prefixes."""
    return {
        "mediums": [
            {"medium": m.value, "prefix": TX_PREFIX[m], "description": _MEDIUM_DESC.get(m.value, "")}
            for m in TxMedium
        ]
    }

_MEDIUM_DESC = {
    "UPI":           "UPI payment (PhonePe, GPay, Paytm, etc.)",
    "BANK_CHEQUE":   "Physical or clearing cheque",
    "RECEIPT":       "Paper / digital receipt",
    "LIQUID_CASH":   "Physical cash transaction",
    "BANK_TRANSFER": "NEFT / RTGS / IMPS bank transfer",
    "ONLINE":        "Online gateway (Razorpay, Stripe, Paytm PG)",
    "DEMAND_DRAFT":  "Demand Draft",
    "AUTO":          "Auto-generated (no external reference)",
}


# ─────────────────────────────────────────────
# UI — CHAT INTERFACE
# ─────────────────────────────────────────────

@app.get("/chat", include_in_schema=False)
async def chat_ui():
    """Serve the FlowGuard chat UI."""
    chat_path = _PROJECT_ROOT / "chat.html"
    if not chat_path.exists():
        raise HTTPException(status_code=404, detail="chat.html not found.")
    return FileResponse(str(chat_path), media_type="text/html")


# ─────────────────────────────────────────────
# GLOBAL ERROR HANDLER
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": str(exc),
            "hint":   "Check server logs for full traceback.",
        },
    )
