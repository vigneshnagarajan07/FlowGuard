"""
flowguard/engine/models.py
──────────────────────────
All Pydantic models used across the FlowGuard engine.
These are the canonical data contracts.  The NLP layer converts human
text INTO these models; the engine writes ONTO these models; the NLP
layer converts them BACK to human text.

No LLM logic here.  Pure data.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────

class ObligationCategory(str, Enum):
    """
    Category determines the type_ceiling (Step 2) and domain floors (Step 3).
    Mirrors the FlowGuard obligation-weight table from the spec.
    """
    STATUTORY   = "STATUTORY"    # GST, TDS, PF, ESI — criminal risk possible
    SECURED_LOAN = "SECURED_LOAN" # Bank EMI, NBFC — NPA / credit score
    SALARY       = "SALARY"       # Employee wages — Labour Court exposure
    RENT         = "RENT"         # Premises lease — eviction risk
    UTILITY      = "UTILITY"      # Power, water, internet
    TRADE_PAYABLE= "TRADE_PAYABLE" # Supplier invoices, vendor bills
    OTHER        = "OTHER"        # Petty cash, misc


class SourceType(str, Enum):
    """Where this obligation was ingested from."""
    CHAT   = "CHAT"       # Typed text input
    CSV    = "CSV"        # CSV file import
    PDF    = "PDF"        # PDF invoice
    IMAGE  = "IMAGE"      # OCR from photo
    API    = "API"        # Direct API call


class FlexibilityLevel(str, Enum):
    FIXED        = "FIXED"        # Cannot move (GST, court order)
    NEGOTIABLE   = "NEGOTIABLE"   # Can discuss with counterparty
    DEFERRABLE   = "DEFERRABLE"   # Can push without meaningful penalty


class ActionTag(str, Enum):
    PAY            = "PAY"
    DEFER          = "DEFER"
    NEGOTIATE      = "NEGOTIATE"
    ESCALATE       = "ESCALATE"   # Edge: legal deadline < 24 h, cash insufficient


class ScoreBand(str, Enum):
    LOW      = "LOW"      # 0 – 20
    MEDIUM   = "MEDIUM"   # 21 – 39
    HIGH     = "HIGH"     # 40 – 69
    CRITICAL = "CRITICAL" # 70 – 100


class EmailTone(str, Enum):
    WARM_APOLOGETIC      = "WARM_APOLOGETIC"       # relationship_score 80-100
    PROFESSIONAL_NEUTRAL = "PROFESSIONAL_NEUTRAL"  # 40-79
    FIRM_BRIEF           = "FIRM_BRIEF"             # 0-39


# ─────────────────────────────────────────────
# OBLIGATION  (input to the engine)
# ─────────────────────────────────────────────

class Obligation(BaseModel):
    """
    One financial obligation.  The NLP layer fills this from human text.
    The engine never modifies it — it only reads it.
    """

    # ── identity ──────────────────────────────
    obligation_id: str = Field(
        ...,
        description="Unique ID.  SHA-256 of (counterparty + amount + due_date) "
                    "computed by the NLP layer for deduplication."
    )
    counterparty_name: str           = Field(..., description="Who you owe money to.")
    description: str                 = Field(..., description="Human-readable label.")

    # ── money ─────────────────────────────────
    amount_inr: float                = Field(..., gt=0, description="Amount in ₹.")
    penalty_rate_annual_pct: float   = Field(
        0.0, ge=0, le=200,
        description="Annual % penalty if not paid on time.  "
                    "GST=18, PF=12, typical EMI=24, salary=0."
    )

    # ── time ──────────────────────────────────
    due_date: date
    max_deferral_days: int           = Field(
        0, ge=0,
        description="Maximum days the obligation can be deferred "
                    "before the penalty rate kicks in at full force. "
                    "0 means it is already past or immovable."
    )

    # ── classification ────────────────────────
    category: ObligationCategory
    flexibility: FlexibilityLevel

    # ── relationship ──────────────────────────
    relationship_score: float        = Field(
        50.0, ge=0, le=100,
        description="0=new vendor, 100=critical long-term partner. "
                    "Derived from payment history + tenure."
    )

    # ── cascade ───────────────────────────────
    blocks_other_obligation_ids: list[str] = Field(
        default_factory=list,
        description="IDs of obligations that CANNOT be paid if this one is missed "
                    "(e.g. bank freeze cascades into salary failure)."
    )

    # ── partial payment ───────────────────────
    partial_payment_pct: float       = Field(
        0.0, ge=0, le=1,
        description="If > 0, the solver may allocate amount × partial_payment_pct "
                    "when cash is insufficient for the full amount. 0 = full-or-nothing."
    )

    # ── parse quality ─────────────────────────
    parse_confidence: float          = Field(
        1.0, ge=0, le=1,
        description="NLP parser confidence in this extraction. "
                    "1.0 = structured input, <0.5 = vague/inferred."
    )

    # ── metadata ──────────────────────────────
    is_recurring: bool               = Field(False)
    source_hash: Optional[str]       = Field(None, description="SHA-256 for dedup.")
    notes: Optional[str]             = Field(None)

    # ── validation ────────────────────────────
    @field_validator("amount_inr")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount_inr must be positive.")
        return round(v, 2)

    @field_validator("due_date")
    @classmethod
    def due_date_not_ancient(cls, v: date) -> date:
        if v.year < 2000:
            raise ValueError("due_date looks wrong — year before 2000.")
        return v


class CashPosition(BaseModel):
    """Current liquidity state of the business."""
    available_cash_inr: float        = Field(..., ge=0)
    as_of_date: date                 = Field(default_factory=date.today)

    # Expected inflows keyed by day-offset from as_of_date (e.g. "3" = in 3 days)
    expected_inflows: dict[str, float] = Field(
        default_factory=dict,
        description="Keys are day-offsets as strings (e.g. '3', '7').  "
                    "Values are INR amounts."
    )

    cash_is_verified: bool           = Field(
        False,
        description="True if cash balance was verified via bank API / Account Aggregator. "
                    "False if self-reported (manual entry). Affects confidence scoring."
    )


# ─────────────────────────────────────────────
# ENGINE OUTPUTS
# ─────────────────────────────────────────────

class SubScores(BaseModel):
    """
    Intermediate Step-1 sub-scores.  All in [0, 1].
    Exposed so the NLP layer can narrate exactly why a decision was made.
    """
    P: float = Field(..., ge=0, le=1, description="Penalty score — financial cost of deferral")
    U: float = Field(..., ge=0, le=1, description="Urgency score — time proximity to due date")
    L: float = Field(..., ge=0, le=1, description="Legal score — statutory / court risk")
    C: float = Field(..., ge=0, le=1, description="Contagion score — cascade failure fraction")
    R: float = Field(..., ge=0, le=1, description="Relationship score — counterparty importance")
    KA: float = Field(0.0, ge=0, le=1, description="Cash Absorption — fraction of available cash consumed")
    F: float = Field(..., ge=0, le=1, description="Flexibility discount — deferral headroom (lower is better)")
    blended: float = Field(..., ge=0, le=1, description="Weighted blend before type ceiling")
    type_ceiling: float = Field(..., ge=0, le=1)


class DecisionRecord(BaseModel):
    """
    One engine decision for one obligation.
    This is what the NLP layer narrates back to the user.
    """
    obligation_id: str
    obligation_description: str
    counterparty_name: str
    amount_inr: float
    due_date: date

    # ── scoring ───────────────────────────────
    consequence_score: float         = Field(..., ge=0, le=100)
    score_band: ScoreBand
    sub_scores: SubScores

    # ── action ────────────────────────────────
    action: ActionTag
    action_date: Optional[date]      = Field(None, description="Recommended date to execute.")
    cash_allocated_inr: float        = Field(0.0, description="Cash assigned by solver.")

    # ── penalty exposure ──────────────────────
    penalty_per_day_inr: float       = Field(0.0, description="Daily cost of deferral.")
    penalty_if_deferred_full_inr: float = Field(
        0.0, description="Total penalty if deferred to max_deferral_days."
    )

    # ── explainability ────────────────────────
    reasoning_key: str               = Field(
        ..., description="Machine-readable key passed to NLP layer for narration."
    )
    confidence: float                = Field(..., ge=0, le=1)
    confidence_basis: list[str]      = Field(default_factory=list)
    cot_action: str                  = Field("", description="Chain-of-thought part 1: action")
    cot_reason: str                  = Field("", description="COT part 2: reason")
    cot_tradeoff: str                = Field("", description="COT part 3: trade-off")
    cot_downstream: str              = Field("", description="COT part 4: downstream effect")

    # ── email ─────────────────────────────────
    email_tone: EmailTone
    negotiation_email_draft: Optional[str] = Field(None)

    # ── audit ─────────────────────────────────
    input_hash: str                  = Field("", description="SHA-256 of obligation inputs.")
    computed_at: datetime            = Field(default_factory=datetime.utcnow)


class EngineResult(BaseModel):
    """
    Full output of one engine run.  This is the response body from POST /score.
    """
    run_id: str
    as_of_date: date
    available_cash_inr: float
    total_obligations_inr: float
    cash_shortfall_inr: float        = Field(
        0.0, description="Positive = deficit.  0 = fully solvent."
    )

    decisions: list[DecisionRecord]  = Field(
        ..., description="Sorted by consequence_score DESC."
    )

    days_to_zero: Optional[int]      = Field(
        None,
        description="Days until cash hits zero in the 30-day projection.  "
                    "None = solvent for 30 days."
    )
    deficit_on_crisis_day_inr: Optional[float] = Field(None)

    # ── summary for NLP narration ─────────────
    summary_narrative: str           = Field(
        "", description="Human-readable paragraph (filled by NLP layer)."
    )
    whatsapp_summary: str            = Field(
        "", description="Under-300-char version for WhatsApp preview."
    )

    computed_at: datetime            = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────
# API REQUEST / RESPONSE ENVELOPES
# ─────────────────────────────────────────────

class ScoreRequest(BaseModel):
    """POST /score — what the API receives."""
    obligations: list[Obligation]    = Field(..., min_length=1)
    cash_position: CashPosition
    scenario_label: Optional[str]    = Field(None, description="e.g. 'What if Kapoor pays ₹30k tomorrow?'")


class NLPParseRequest(BaseModel):
    """POST /parse — raw human text → ScoreRequest."""
    raw_text: str                    = Field(..., min_length=1, max_length=4000)
    language: str                    = Field("en", description="ISO 639-1 code.")
    context: Optional[str]           = Field(
        None, description="Previous conversation turn for multi-turn flows."
    )


class NLPNarrateRequest(BaseModel):
    """POST /narrate — EngineResult → human-readable text."""
    engine_result: EngineResult
    output_language: str             = Field("en")
    channel: str                     = Field(
        "whatsapp", description="'whatsapp' | 'web' | 'voice'"
    )


class UserProfile(BaseModel):
    """Business and personal profile details."""
    full_name: Optional[str] = None
    business_name: Optional[str] = None
    industry: Optional[str] = None
    business_description: Optional[str] = None
    gstin: Optional[str] = None
    annual_turnover: Optional[str] = None

