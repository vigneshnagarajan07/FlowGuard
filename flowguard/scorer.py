"""
flowguard/engine/scorer.py
──────────────────────────
The deterministic Consequence Score engine.  This is the CORE of FlowGuard.
All downstream LLM narration, FILTER responses, and payment recommendations
are DRIVEN by these scores.  The LLM merely explains — never overrides.

IRON RULE: No LLM, no randomness, no network calls.
Same inputs → identical outputs, every run.

Formula (4 steps):
  Step 1  blended = clamp(0.22·P + 0.22·U + 0.22·L + 0.14·C + 0.10·R + 0.10·KA − 0.07·F, 0, 1)
  Step 2  pre_floor_cs = type_ceiling(category) × blended × 100
  Step 3  cs = max(pre_floor_cs, domain_floor(category))
  Step 3b overdue multiplier: if past due, cs × 1.3 (capped at 100)
  Step 3c recurrence boost: +5 cs if is_recurring (capped at 100)

Sub-scores:
  P   = Penalty score        — financial cost of deferral
  U   = Urgency score        — time proximity to due date
  L   = Legal score          — statutory / court risk level
  C   = Contagion score      — cascade failure fraction
  R   = Relationship score   — counterparty importance
  KA  = Cash Absorption      — fraction of available cash this obligation consumes
  F   = Flexibility discount — deferral headroom (negative contribution)

Edge cases handled:
  • due_date in the past        → urgency = 1.0 + 1.3× overdue multiplier
  • zero cash                   → KA = 1.0 for all; all actions default to NEGOTIATE/ESCALATE
  • cascading failures          → contagion computed from graph traversal
  • identical CS ties           → tie-broken by amount DESC (bigger bill first)
  • GST/statutory < 24 h        → score pinned to 95 minimum (CRITICAL band)
  • amount > 10× available cash → KA = 1.0 (confidence penalty also applied)
  • max_deferral_days = 0       → flexibility forced to FIXED regardless of input
  • is_recurring = True         → +5 CS bonus (recurring bills must stay current)
  • DEFER/NEGOTIATE action_date → explicitly computed from due_date and deferral window
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from .models import (
    ActionTag,
    CashPosition,
    DecisionRecord,
    EmailTone,
    EngineResult,
    FlexibilityLevel,
    Obligation,
    ObligationCategory,
    ScoreBand,
    SubScores,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# CONFIG LOADER  (falls back to hardcoded defaults)
# ─────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

def _load_config() -> dict:
    """Load config.json if it exists, else return empty dict (use defaults)."""
    try:
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Failed to load config.json: %s — using defaults.", e)
    return {}

_CFG = _load_config()
_ENGINE_CFG = _CFG.get("engine", {})


def _enum_dict(raw: dict, enum_cls: type) -> dict:
    """Convert {"STATUTORY": 1.0} → {ObligationCategory.STATUTORY: 1.0}."""
    return {enum_cls(k): v for k, v in raw.items()}


# ─────────────────────────────────────────────
# CONSTANTS  (loaded from config.json, fallback to hardcoded)
# ─────────────────────────────────────────────

# Step 2 — type ceilings
_DEFAULT_TYPE_CEILING = {
    "STATUTORY": 1.00, "SECURED_LOAN": 0.95, "SALARY": 0.90,
    "RENT": 0.80, "UTILITY": 0.75, "TRADE_PAYABLE": 0.60, "OTHER": 0.40,
}
TYPE_CEILING: dict[ObligationCategory, float] = _enum_dict(
    _ENGINE_CFG.get("type_ceiling", _DEFAULT_TYPE_CEILING), ObligationCategory
)

# Step 3 — domain floors (law, not math)
_DEFAULT_DOMAIN_FLOOR = {
    "STATUTORY": 40.0, "SECURED_LOAN": 35.0, "SALARY": 35.0,
    "RENT": 20.0, "UTILITY": 10.0, "TRADE_PAYABLE": 5.0, "OTHER": 0.0,
}
DOMAIN_FLOOR: dict[ObligationCategory, float] = _enum_dict(
    _ENGINE_CFG.get("domain_floor", _DEFAULT_DOMAIN_FLOOR), ObligationCategory
)

# Edge: statutory obligation due within 24 hours → pin CS ≥ 95
STATUTORY_URGENT_FLOOR = _ENGINE_CFG.get("statutory_urgent_floor", 95.0)
STATUTORY_URGENT_HOURS = _ENGINE_CFG.get("statutory_urgent_hours", 24)

# Flexibility multipliers
_DEFAULT_FLEX = {"FIXED": 0.0, "NEGOTIABLE": 0.5, "DEFERRABLE": 1.0}
FLEXIBILITY_DISCOUNT: dict[FlexibilityLevel, float] = _enum_dict(
    _ENGINE_CFG.get("flexibility_discount", _DEFAULT_FLEX), FlexibilityLevel
)

# Sub-score weights  (now 7 components — updated from v1 config)
_weights = _ENGINE_CFG.get("sub_score_weights", {})
W_P  = _weights.get("P",  0.22)   # Penalty score
W_U  = _weights.get("U",  0.22)   # Urgency
W_L  = _weights.get("L",  0.22)   # Legal risk
W_C  = _weights.get("C",  0.14)   # Contagion
W_R  = _weights.get("R",  0.10)   # Relationship
W_KA = _weights.get("KA", 0.10)   # Cash absorption (NEW)
W_F  = _weights.get("F",  0.07)   # Flexibility discount (reduced slightly)

# Overdue severity multiplier (NEW)
OVERDUE_MULTIPLIER = _ENGINE_CFG.get("overdue_multiplier", 1.30)

# Recurrence CS bonus (NEW): recurring bills must stay current
RECURRENCE_BONUS = _ENGINE_CFG.get("recurrence_bonus", 5.0)

# Score band thresholds
_band_map = {"CRITICAL": ScoreBand.CRITICAL, "HIGH": ScoreBand.HIGH,
             "MEDIUM": ScoreBand.MEDIUM, "LOW": ScoreBand.LOW}
_raw_bands = _ENGINE_CFG.get("score_bands", [[70, "CRITICAL"], [40, "HIGH"],
                                              [21, "MEDIUM"], [0, "LOW"]])
BANDS = [(b[0], _band_map[b[1]]) for b in _raw_bands]

# Email tone thresholds
_tone_map = {"WARM_APOLOGETIC": EmailTone.WARM_APOLOGETIC,
             "PROFESSIONAL_NEUTRAL": EmailTone.PROFESSIONAL_NEUTRAL,
             "FIRM_BRIEF": EmailTone.FIRM_BRIEF}
_raw_tones = _ENGINE_CFG.get("email_thresholds", [[80, "WARM_APOLOGETIC"],
                                                   [40, "PROFESSIONAL_NEUTRAL"],
                                                   [0, "FIRM_BRIEF"]])
EMAIL_THRESHOLDS = [(t[0], _tone_map[t[1]]) for t in _raw_tones]

# Penalty score cap for normalisation (₹5 lakh)
PENALTY_CAP_INR = _ENGINE_CFG.get("penalty_cap_inr", 500_000)

# Urgency: full score when due today, linear decay to 0 at 90 days
URGENCY_ZERO_DAYS = _ENGINE_CFG.get("urgency_zero_days", 90)

# Contagion freeze thresholds
CONTAGION_FREEZE_THRESHOLD = _ENGINE_CFG.get("contagion_freeze_threshold", 0.60)
CONTAGION_FREEZE_SCORE = _ENGINE_CFG.get("contagion_freeze_score", 0.85)

# Days-to-zero projection window (configurable, default 30)
DAYS_TO_ZERO_WINDOW = _ENGINE_CFG.get("days_to_zero_window", 30)


# ─────────────────────────────────────────────
# SUB-SCORE CALCULATORS
# ─────────────────────────────────────────────

def _penalty_score(ob: Obligation, today: date) -> float:
    """
    P ∈ [0, 1].
    Penalty in absolute ₹ if deferred to max_deferral_days.
    Normalised against PENALTY_CAP_INR.

    Edge: if max_deferral_days == 0, penalty is the per-day rate × 1 day
          (captures the "it's due today" scenario).
    """
    defer_days = max(ob.max_deferral_days, 1)
    daily_penalty = (ob.amount_inr * ob.penalty_rate_annual_pct / 100) / 365
    total_penalty = daily_penalty * defer_days
    # Also consider: penalty could equal the principal (e.g. 100% surcharge)
    # Cap at the full obligation amount as well
    total_penalty = min(total_penalty, ob.amount_inr)
    return min(total_penalty / PENALTY_CAP_INR, 1.0)


def _urgency_score(ob: Obligation, today: date) -> float:
    """
    U ∈ [0, 1].
    1 if due today or overdue.  Linear decay to 0 at URGENCY_ZERO_DAYS.

    Edge: if due_date < today (overdue) → U = 1.0 always.
    """
    days_left = (ob.due_date - today).days
    if days_left <= 0:
        return 1.0
    return max(0.0, 1.0 - days_left / URGENCY_ZERO_DAYS)


def _legal_score(ob: Obligation) -> float:
    """
    L ∈ [0, 1].
    Binary + graded by category.

    STATUTORY:    1.0  (criminal prosecution possible — GST, TDS, PF)
    SECURED_LOAN: 0.6  (civil suit, NPA, SARFAESI action)
    SALARY:       0.5  (Labour Court, ID Act)
    RENT:         0.3  (eviction — civil)
    UTILITY:      0.1  (service disconnection only)
    TRADE_PAYABLE: 0.05 (MSME SAMADHAAN possible but rarely pursued)
    OTHER:        0.0
    """
    legal_map = {
        ObligationCategory.STATUTORY:    1.00,
        ObligationCategory.SECURED_LOAN: 0.60,
        ObligationCategory.SALARY:       0.50,
        ObligationCategory.RENT:         0.30,
        ObligationCategory.UTILITY:      0.10,
        ObligationCategory.TRADE_PAYABLE: 0.05,
        ObligationCategory.OTHER:        0.00,
    }
    return legal_map[ob.category]


def _contagion_score(
    ob: Obligation,
    all_obligations: list[Obligation],
    total_cash: float,
) -> float:
    """
    C ∈ [0, 1].
    Fraction of REMAINING obligations that would also fail if THIS one
    is not paid (cascade graph traversal).

    Special case: STATUTORY / SECURED_LOAN non-payment can trigger a
    bank-account freeze → ALL obligations become unpayable → C = 1.0.

    Edge: single-obligation scenario → C = 0.0 (no cascade possible).
    """
    if len(all_obligations) <= 1:
        return 0.0

    # Hard cascade: bank account freeze risk
    if ob.category in (ObligationCategory.STATUTORY, ObligationCategory.SECURED_LOAN):
        if ob.amount_inr > total_cash * CONTAGION_FREEZE_THRESHOLD:
            # Missing a large statutory payment when cash is tight → freeze risk
            return CONTAGION_FREEZE_SCORE

    # Graph traversal: follow blocks_other_obligation_ids
    id_to_ob = {o.obligation_id: o for o in all_obligations}
    blocked_ids: set[str] = set()
    frontier = list(ob.blocks_other_obligation_ids)
    while frontier:
        nxt = frontier.pop()
        if nxt in blocked_ids or nxt not in id_to_ob:
            continue
        blocked_ids.add(nxt)
        frontier.extend(id_to_ob[nxt].blocks_other_obligation_ids)

    other_count = len(all_obligations) - 1
    if other_count == 0:
        return 0.0
    return min(len(blocked_ids) / other_count, 1.0)


def _relationship_score_normalised(ob: Obligation) -> float:
    """R ∈ [0, 1].  relationship_score is already [0, 100] → divide by 100."""
    return ob.relationship_score / 100.0


def _cash_absorption_score(
    ob: Obligation,
    available_cash: float,
) -> float:
    """
    KA ∈ [0, 1].  NEW in v2.
    Measures how much of available cash this single obligation would consume.
    High KA = paying this nearly empties the tank, leaving zero buffer.

    KA = 0.0  if obligation is tiny relative to cash (< 10% of available)
    KA = 1.0  if amount >= available cash (paying this = zero balance or overdraft)

    Edge: if available_cash == 0, KA = 1.0 for all (nothing is affordable).
    """
    if available_cash <= 0:
        return 1.0
    ratio = ob.amount_inr / available_cash
    # Smooth sigmoid-style: below 0.1 ratio → ~0; at ratio=1 → ~1
    return _clamp(ratio, 0.0, 1.0)


def _flexibility_score(ob: Obligation) -> float:
    """
    F ∈ [0, 1].  The DISCOUNT factor.

    Edge: if max_deferral_days == 0, flexibility is forced to FIXED
          regardless of what the user said.  Can't defer what's already due.
    """
    if ob.max_deferral_days == 0:
        return FLEXIBILITY_DISCOUNT[FlexibilityLevel.FIXED]
    return FLEXIBILITY_DISCOUNT[ob.flexibility]


# ─────────────────────────────────────────────
# CORE FORMULA
# ─────────────────────────────────────────────

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _score_band(cs: float) -> ScoreBand:
    for threshold, band in BANDS:
        if cs >= threshold:
            return band
    return ScoreBand.LOW


def _email_tone(relationship_score: float) -> EmailTone:
    for threshold, tone in EMAIL_THRESHOLDS:
        if relationship_score >= threshold:
            return tone
    return EmailTone.FIRM_BRIEF


def compute_consequence_score(
    ob: Obligation,
    all_obligations: list[Obligation],
    cash_position: CashPosition,
) -> tuple[float, SubScores]:
    """
    Returns (final_cs, sub_scores).

    Step 1: blended sub-score (7 components)
    Step 2: multiply by type_ceiling
    Step 3: apply domain floor
    Step 3b: overdue multiplier (1.3× if past due)
    Step 3c: recurrence bonus (+5 if is_recurring)
    Step 3d: statutory urgent pin (< 24 h)
    """
    today = cash_position.as_of_date
    available = cash_position.available_cash_inr

    P  = _penalty_score(ob, today)
    U  = _urgency_score(ob, today)
    L  = _legal_score(ob)
    C  = _contagion_score(ob, all_obligations, available)
    R  = _relationship_score_normalised(ob)
    KA = _cash_absorption_score(ob, available)
    F  = _flexibility_score(ob)

    blended = _clamp(
        W_P * P + W_U * U + W_L * L + W_C * C + W_R * R + W_KA * KA - W_F * F
    )
    ceiling = TYPE_CEILING[ob.category]
    pre_floor_cs = ceiling * blended * 100

    # Step 3: domain floor
    floor = DOMAIN_FLOOR[ob.category]
    cs = max(pre_floor_cs, floor)

    # Step 3b: overdue multiplier — past-due obligations are MORE urgent
    days_left = (ob.due_date - today).days
    if days_left < 0:
        cs = min(cs * OVERDUE_MULTIPLIER, 100.0)

    # Step 3c: recurrence boost — recurring bills must stay current
    if ob.is_recurring:
        cs = min(cs + RECURRENCE_BONUS, 100.0)

    # Step 3d: statutory + due within 24 h → hard pin
    if ob.category == ObligationCategory.STATUTORY:
        hours_left = days_left * 24
        if hours_left <= STATUTORY_URGENT_HOURS:
            cs = max(cs, STATUTORY_URGENT_FLOOR)

    cs = _clamp(cs, 0.0, 100.0)

    sub_scores = SubScores(
        P=round(P, 4), U=round(U, 4), L=round(L, 4),
        C=round(C, 4), R=round(R, 4), F=round(F, 4),
        blended=round(blended, 4),
        type_ceiling=ceiling,
        KA=round(KA, 4),
    )
    return round(cs, 2), sub_scores



# ─────────────────────────────────────────────
# CONSTRAINT SOLVER  (greedy, deterministic)
# ─────────────────────────────────────────────

def _days_to_zero(
    cash: float,
    inflows: dict[str, float],
    outflows_by_day: dict[int, float],
    projection_days: int = DAYS_TO_ZERO_WINDOW,
) -> tuple[Optional[int], Optional[float]]:
    """
    Project forward `projection_days` days (configurable, default from config).
    Returns (crisis_day, deficit_amount) or (None, None) if solvent.
    inflows keys are string day-offsets from cash_position.as_of_date.
    """
    running = cash
    for day in range(1, projection_days + 1):
        running += inflows.get(str(day), 0.0)
        running -= outflows_by_day.get(day, 0.0)
        if running < 0:
            return day, round(running, 2)
    return None, None


def _compute_daily_outflows(
    decisions: list[DecisionRecord],
    today: date,
    window: int = DAYS_TO_ZERO_WINDOW,
) -> dict[int, float]:
    outflows: dict[int, float] = {}
    for d in decisions:
        if d.action == ActionTag.PAY and d.action_date:
            offset = (d.action_date - today).days
            if 0 < offset <= window:
                outflows[offset] = outflows.get(offset, 0.0) + d.amount_inr
    return outflows


def _confidence(
    ob: Obligation,
    cs: float,
    sub_scores: SubScores,
    cash_coverage: float,   # available_cash / amount
    cash_is_verified: bool = False,
) -> tuple[float, list[str]]:
    """
    Confidence ∈ [0, 1].  Penalised by:
    • estimated (not verified) penalty rate
    • very low cash coverage (paying this empties the tank)
    • low urgency score (far-future deadline → less certain)
    • unverified (self-reported) cash balance
    • low parse confidence (NLP extraction uncertainty)
    """
    base = 0.85
    basis: list[str] = []

    if ob.penalty_rate_annual_pct > 0:
        basis.append(f"penalty_rate_{ob.penalty_rate_annual_pct:.0f}pct_annual")
    else:
        base -= 0.05
        basis.append("penalty_rate_not_provided")

    if cash_coverage < 1.2:
        base -= 0.08
        basis.append("cash_covers_barely_over_obligation")
    if cash_coverage < 0.5:
        base -= 0.07
        basis.append("cash_insufficient_for_obligation")

    if sub_scores.U < 0.3:
        base -= 0.05
        basis.append("low_urgency_far_deadline")

    if ob.relationship_score >= 80:
        basis.append(f"relationship_score_{ob.relationship_score:.0f}_strong")

    # Phase 1 addition: penalise unverified (self-reported) cash
    if not cash_is_verified:
        base -= 0.10
        basis.append("cash_balance_unverified_self_reported")

    # Phase 1 addition: penalise low parse confidence
    if ob.parse_confidence < 0.5:
        base -= 0.10
        basis.append(f"parse_confidence_low_{ob.parse_confidence:.2f}")
    elif ob.parse_confidence < 0.8:
        base -= 0.05
        basis.append(f"parse_confidence_moderate_{ob.parse_confidence:.2f}")

    return round(_clamp(base), 3), basis


# ─────────────────────────────────────────────
# COT  (chain-of-thought — keys for NLP layer)
# ─────────────────────────────────────────────

def _build_cot(
    ob: Obligation,
    cs: float,
    sub_scores: SubScores,
    action: ActionTag,
    action_date: Optional[date],
    penalty_per_day: float,
    available_cash: float = 0.0,
    today: Optional[date] = None,
) -> dict[str, str]:
    """
    Returns the 4-part COT as plain English strings.
    Enhanced v2: includes all sub-score drivers, cash impact, and recurrence context.
    These are ALREADY human-readable — the NLP layer can pass them
    through unchanged or rephrase them for the output channel.
    """
    band   = _score_band(cs)
    _today = today or date.today()
    days_left = (ob.due_date - _today).days if ob.due_date else 0

    act = f"{action.value} ₹{ob.amount_inr:,.0f} to {ob.counterparty_name}"
    if action_date:
        act += f" by {action_date.strftime('%d %b %Y')}"

    # ── Reason: surface the dominant drivers ──
    drivers = []
    if sub_scores.L >= 0.5:
        drivers.append("carries LEGAL risk (prosecution/court exposure)")
    if days_left < 0:
        drivers.append(f"OVERDUE by {abs(days_left)} day(s)")
    elif sub_scores.U >= 0.7:
        drivers.append(f"due in {days_left} day(s) (URGENT)")
    if sub_scores.P >= 0.4:
        drivers.append(f"daily penalty ₹{penalty_per_day:,.0f}")
    if sub_scores.C >= 0.5:
        drivers.append("failure cascades to other obligations")
    if sub_scores.KA >= 0.7:
        pct = sub_scores.KA * 100
        drivers.append(f"consumes ~{pct:.0f}% of available cash")
    if ob.is_recurring:
        drivers.append("recurring bill — missing this harms credit history")
    if not drivers:
        drivers.append(f"CS {cs:.1f} ({band.value})")

    reason = f"CS {cs:.1f} ({band.value}) | " + " | ".join(drivers) + "."

    # ── Sub-score breakdown for explainability ──
    reason += (
        f"  [P={sub_scores.P:.2f} U={sub_scores.U:.2f} L={sub_scores.L:.2f}"
        f" C={sub_scores.C:.2f} R={sub_scores.R:.2f} KA={sub_scores.KA:.2f} F={sub_scores.F:.2f}]"
    )

    # ── Tradeoff: daily burn rate ──
    if penalty_per_day > 0:
        tradeoff = (
            f"Deferring costs ₹{penalty_per_day:,.0f}/day. "
            f"After {ob.max_deferral_days or 1} day(s): ₹{penalty_per_day * max(ob.max_deferral_days,1):,.0f} total."
        )
    else:
        tradeoff = "No direct penalty, but relationship score and supply chain continuity at risk."

    # ── Downstream: cash impact ──
    remaining_after = max(available_cash - ob.amount_inr, 0)
    if action == ActionTag.PAY:
        downstream = (
            f"Paying leaves ₹{remaining_after:,.0f} available. "
            "Clears this obligation entirely and preserves creditworthiness."
        )
    elif action == ActionTag.ESCALATE:
        downstream = (
            "Immediate escalation needed — seek emergency credit or overdraft. "
            "Missing this payment could trigger legal action or bank-account freeze."
        )
    elif action == ActionTag.NEGOTIATE:
        downstream = (
            f"Negotiating a revised date frees ₹{ob.amount_inr:,.0f} for higher-priority "
            "obligations. Relationship score must be ≥ 50 to succeed."
        )
    else:
        downstream = (
            f"Deferring to {action_date.strftime('%d %b') if action_date else 'a later date'} "
            f"frees ₹{ob.amount_inr:,.0f} for higher-priority obligations today."
        )

    return {
        "cot_action":     act,
        "cot_reason":     reason,
        "cot_tradeoff":   tradeoff,
        "cot_downstream": downstream,
    }



def _input_hash(ob: Obligation) -> str:
    payload = json.dumps({
        "id":     ob.obligation_id,
        "amount": ob.amount_inr,
        "due":    ob.due_date.isoformat(),
        "cat":    ob.category.value,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def run_engine(
    obligations: list[Obligation],
    cash_position: CashPosition,
    scenario_label: Optional[str] = None,
) -> EngineResult:
    """
    Main entry point.  Takes obligations + cash, returns EngineResult.

    Pure deterministic — no I/O, no LLM.  Call from the API layer.
    """
    if not obligations:
        raise ValueError("obligations list is empty.")

    today = cash_position.as_of_date
    available = cash_position.available_cash_inr
    total_owed = sum(o.amount_inr for o in obligations)
    shortfall = max(0.0, total_owed - available)

    # ── Step A: score every obligation ────────
    scored: list[tuple[float, SubScores, Obligation]] = []
    for ob in obligations:
        cs, subs = compute_consequence_score(ob, obligations, cash_position)
        scored.append((cs, subs, ob))

    # ── Step B: sort DESC by CS, tie-break by amount DESC ─────
    scored.sort(key=lambda x: (x[0], x[2].amount_inr), reverse=True)

    # ── Step C: greedy allocation ──────────────
    remaining_cash = available
    decisions: list[DecisionRecord] = []

    for cs, subs, ob in scored:
        penalty_per_day = (ob.amount_inr * ob.penalty_rate_annual_pct / 100) / 365
        penalty_if_deferred = penalty_per_day * max(ob.max_deferral_days, 1)

        # Determine action and smart action_date
        if remaining_cash >= ob.amount_inr:
            action = ActionTag.PAY
            allocated = ob.amount_inr
            remaining_cash -= allocated
            action_date = today  # pay today
        elif ob.partial_payment_pct > 0 and remaining_cash >= ob.amount_inr * ob.partial_payment_pct:
            # Partial payment: allocate the allowed fraction
            action = ActionTag.NEGOTIATE
            allocated = round(ob.amount_inr * ob.partial_payment_pct, 2)
            remaining_cash -= allocated
            action_date = today
        elif ob.category == ObligationCategory.STATUTORY and ob.max_deferral_days == 0:
            # Cannot defer a statutory obligation due today with no cash → ESCALATE
            action = ActionTag.ESCALATE
            allocated = 0.0
            action_date = today
        elif ob.flexibility == FlexibilityLevel.FIXED:
            action = ActionTag.ESCALATE
            allocated = 0.0
            action_date = today
        elif ob.flexibility == FlexibilityLevel.NEGOTIABLE:
            action = ActionTag.NEGOTIATE
            allocated = 0.0
            # Smart action_date: suggest the midpoint of deferral window or due date
            deferral_window = max(ob.max_deferral_days, 1)
            action_date = min(
                ob.due_date,
                date.fromordinal(today.toordinal() + deferral_window // 2 + 1),
            )
        else:
            action = ActionTag.DEFER
            allocated = 0.0
            # Smart action_date: latest safe date (due_date or max_deferral_days out)
            deferral_window = max(ob.max_deferral_days, 1)
            action_date = min(
                ob.due_date,
                date.fromordinal(today.toordinal() + deferral_window),
            )

        cash_coverage = available / ob.amount_inr if ob.amount_inr > 0 else 99.0
        confidence, basis = _confidence(
            ob, cs, subs, cash_coverage,
            cash_is_verified=cash_position.cash_is_verified,
        )

        cot = _build_cot(
            ob, cs, subs, action, action_date, penalty_per_day,
            available_cash=available,
            today=today,
        )

        decisions.append(DecisionRecord(
            obligation_id=ob.obligation_id,
            obligation_description=ob.description,
            counterparty_name=ob.counterparty_name,
            amount_inr=ob.amount_inr,
            due_date=ob.due_date,
            consequence_score=cs,
            score_band=_score_band(cs),
            sub_scores=subs,
            action=action,
            action_date=action_date,
            cash_allocated_inr=allocated,
            penalty_per_day_inr=round(penalty_per_day, 2),
            penalty_if_deferred_full_inr=round(penalty_if_deferred, 2),
            reasoning_key=f"{action.value.lower()}_{ob.category.value.lower()}_cs{int(cs)}",
            confidence=confidence,
            confidence_basis=basis,
            email_tone=_email_tone(ob.relationship_score),
            input_hash=_input_hash(ob),
            **cot,
        ))

    # ── Step D: days-to-zero projection ───────────────────────────────
    daily_outflows = _compute_daily_outflows(decisions, today)
    crisis_day, deficit = _days_to_zero(
        available, cash_position.expected_inflows, daily_outflows
    )

    # ── Step E: aggregate penalty exposure ─────────────────────────────
    # Total daily burn rate if ALL deferred obligations keep accruing penalty
    deferred_decisions = [d for d in decisions if d.action in (ActionTag.DEFER, ActionTag.NEGOTIATE)]
    total_penalty_exposure = sum(d.penalty_per_day_inr for d in deferred_decisions)

    result = EngineResult(
        run_id=str(uuid.uuid4())[:8],
        as_of_date=today,
        available_cash_inr=available,
        total_obligations_inr=round(total_owed, 2),
        cash_shortfall_inr=round(shortfall, 2),
        decisions=decisions,
        days_to_zero=crisis_day,
        deficit_on_crisis_day_inr=deficit,
    )

    # Attach penalty exposure as extra metadata (not in Pydantic model — safe to add)
    object.__setattr__(result, "total_penalty_exposure_per_day_inr",
                       round(total_penalty_exposure, 2))

    return result
