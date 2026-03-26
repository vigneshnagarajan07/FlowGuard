"""
flowguard/file_ingest.py
────────────────────────
Unified file import module.

Pipeline:
  OCR   → PaddleOCR  (no Tesseract dependency)
  PDF   → pdfplumber (text) + Camelot (tables)
  CSV   → Pandas
  LLM   → Groq (LLaMA 3) — structures raw text into canonical format
  Valid → Pydantic — validates every parsed obligation before DB insert

All methods:
  1. Hash file content → check file_imports table → skip if already imported
  2. Extract raw text / structured data (OCR / PDF / CSV)
  3. Route through Groq (Layer 2) for structured extraction
  4. Validate each obligation via Pydantic model
  5. Compute dedup obligation_id for each obligation
  6. UPSERT into database
  7. Return results + dedup report
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# OPTIONAL IMPORTS (graceful degradation)
# ─────────────────────────────────────────────

# PDF: pdfplumber (text) + camelot (tables)
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pdfplumber not installed — PDF text extraction disabled")

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logger.warning("camelot-py not installed — PDF table extraction disabled")

# OCR: Groq Vision API (no local dependencies required)
# pypdfium2 is already installed (comes with pdfplumber) — used to render PDF pages to images
try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False

OCR_AVAILABLE = True  # Always available via Groq Vision when API key is set


# CSV: Pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("pandas not installed — using stdlib csv fallback")

# PIL still useful for image pre-processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


from .database import (
    SessionLocal,
    compute_obligation_id,
    compute_file_hash,
    upsert_obligation,
    check_file_imported,
    record_file_import,
    record_transaction,
    generate_txn_ref_id,
)
from .groq_client import groq_parse_input, groq_vision_ocr, is_groq_available
from .parser import parse_text_to_obligations


# ─────────────────────────────────────────────
# PYDANTIC VALIDATION MODEL
# ─────────────────────────────────────────────

class ValidatedObligation(BaseModel):
    """Pydantic model that validates every parsed obligation before DB insert."""
    counterparty_name: str = Field(..., min_length=1)
    amount_inr: float = Field(..., gt=0)
    due_date: date
    category: str = Field(default="OTHER")
    description: str = Field(default="")
    flexibility: str = Field(default="DEFERRABLE")
    penalty_rate_annual_pct: float = Field(default=0.0)
    max_deferral_days: int = Field(default=0)
    relationship_score: float = Field(default=50.0)
    is_recurring: bool = Field(default=False)
    parse_confidence: float = Field(default=1.0)
    notes: Optional[str] = None

    @validator("category", pre=True, always=True)
    def normalize_category(cls, v):
        if not v:
            return "OTHER"
        return v.upper().strip()

    @validator("due_date", pre=True)
    def parse_due_date(cls, v):
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%Y/%m/%d"):
                try:
                    return datetime.strptime(v.strip(), fmt).date()
                except ValueError:
                    continue
            # Try ISO as last resort
            try:
                return date.fromisoformat(v.strip())
            except ValueError:
                pass
        return date.today()

    class Config:
        extra = "allow"  # Allow extra fields to pass through


class ValidatedTransaction(BaseModel):
    """Pydantic model for transaction data parsed from files."""
    medium: str = Field(default="AUTO")
    direction: str = Field(default="OUT")  # IN | OUT
    amount_inr: float = Field(..., gt=0)
    txn_date: date
    counterparty: Optional[str] = None
    description: Optional[str] = None
    external_ref: Optional[str] = None

    @validator("direction", pre=True, always=True)
    def normalize_direction(cls, v):
        if not v:
            return "OUT"
        return v.upper().strip()

    @validator("txn_date", pre=True)
    def parse_txn_date(cls, v):
        if isinstance(v, date):
            return v
        if isinstance(v, str):
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"):
                try:
                    return datetime.strptime(v.strip(), fmt).date()
                except ValueError:
                    continue
            try:
                return date.fromisoformat(v.strip())
            except ValueError:
                pass
        return date.today()

    class Config:
        extra = "allow"


# ─────────────────────────────────────────────
# RESULT MODEL
# ─────────────────────────────────────────────

class ImportResult:
    """Standardized result from any file import operation."""

    def __init__(self, filename: str, file_type: str):
        self.filename = filename
        self.file_type = file_type
        self.success = False
        self.error: Optional[str] = None
        self.raw_text: Optional[str] = None
        self.obligations: list[dict] = []
        self.transactions: list[dict] = []
        self.new_count = 0
        self.updated_count = 0
        self.skipped_duplicate = False
        self.file_hash: Optional[str] = None
        self.validation_errors: list[str] = []

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "success": self.success,
            "error": self.error,
            "obligations_found": len(self.obligations),
            "obligations_new": self.new_count,
            "obligations_updated": self.updated_count,
            "skipped_duplicate": self.skipped_duplicate,
            "obligations": self.obligations,
            "transactions_found": len(self.transactions),
            "validation_errors": self.validation_errors[:10],  # Cap at 10
        }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _parse_amount(val: str) -> Optional[float]:
    """Parse Indian-style amounts: '1,50,000' or '150000' or '1.5L'."""
    val = val.strip().replace("₹", "").replace("Rs.", "").replace("Rs", "").replace(",", "").strip()
    if not val:
        return None

    m = re.match(r"^(\d+(?:\.\d+)?)\s*(l|lakh|lac|k|cr|crore)?$", val, re.IGNORECASE)
    if m:
        num = float(m.group(1))
        suffix = (m.group(2) or "").lower()
        multiplier = {
            "l": 100000, "lakh": 100000, "lac": 100000,
            "k": 1000, "cr": 10000000, "crore": 10000000,
        }.get(suffix, 1)
        return num * multiplier

    try:
        return float(val)
    except ValueError:
        return None


def _validate_and_store_obligations(
    db, obligations_raw: list[dict], source_type: str,
    file_hash: str, result: ImportResult
):
    """Validate each obligation with Pydantic, compute dedup key, and UPSERT."""
    for raw_ob in obligations_raw:
        try:
            validated = ValidatedObligation(**raw_ob)
            ob = validated.dict()
            ob["obligation_id"] = compute_obligation_id(
                ob["counterparty_name"], ob["amount_inr"], ob["due_date"]
            )
            row, is_new = upsert_obligation(db, ob, source_type, file_hash)
            result.obligations.append(ob)
            if is_new:
                result.new_count += 1
            else:
                result.updated_count += 1
        except Exception as e:
            result.validation_errors.append(
                f"Validation failed for {raw_ob.get('counterparty_name', '?')}: {e}"
            )
            logger.warning("Obligation validation failed: %s — %s", raw_ob, e)


def _validate_and_store_transactions(
    db, transactions_raw: list[dict], source_type: str,
    file_hash: str, result: ImportResult
):
    """Validate each transaction with Pydantic, generate ref_id, and store."""
    for raw_tx in transactions_raw:
        try:
            validated = ValidatedTransaction(**raw_tx)
            row, is_new = record_transaction(
                db=db,
                medium=validated.medium,
                direction=validated.direction,
                amount_inr=validated.amount_inr,
                txn_date=validated.txn_date,
                counterparty=validated.counterparty,
                description=validated.description,
                external_ref=validated.external_ref,
                source_type=source_type,
                source_file_hash=file_hash,
            )
            result.transactions.append(row.to_dict())
        except Exception as e:
            result.validation_errors.append(f"Transaction validation: {e}")
            logger.warning("Transaction validation failed: %s — %s", raw_tx, e)


# ─────────────────────────────────────────────
# GROQ LLM STRUCTURING PROMPT
# ─────────────────────────────────────────────

_GROQ_FILE_PROMPT = """You are a financial document parser. Extract ALL obligations AND transactions from the text below.

Return valid JSON with this exact structure:
{
  "obligations": [
    {
      "counterparty_name": "...",
      "amount_inr": 50000,
      "due_date": "YYYY-MM-DD",
      "category": "STATUTORY|TRADE_PAYABLE|SALARY|SECURED_LOAN|UNSECURED_LOAN|RENT|UTILITY|INSURANCE|OTHER",
      "description": "...",
      "flexibility": "FIXED|NEGOTIABLE|DEFERRABLE"
    }
  ],
  "transactions": [
    {
      "medium": "UPI|BANK_CHEQUE|RECEIPT|LIQUID_CASH|BANK_TRANSFER|ONLINE|DEMAND_DRAFT|AUTO",
      "direction": "IN|OUT",
      "amount_inr": 50000,
      "txn_date": "YYYY-MM-DD",
      "counterparty": "...",
      "description": "...",
      "external_ref": "UPI txn ID / cheque no / receipt no / null"
    }
  ]
}

Rules:
- Dates must be ISO format YYYY-MM-DD. If year is missing, use 2026.
- Amounts in INR. Convert lakhs (L) = 100000, crores (Cr) = 10000000.
- For transactions, detect the medium from keywords (UPI, NEFT, cheque, cash, etc.)
- For external_ref, extract any reference/transaction/receipt/cheque number if visible.
- If no transactions are found, return empty transactions array.
- If no obligations are found, return empty obligations array.
- Return ONLY the JSON object, no markdown, no explanation.

Document text:
"""


def _groq_structure_file(text: str) -> Optional[dict]:
    """Use Groq LLM to structure raw text into obligations + transactions."""
    if not is_groq_available():
        return None

    import json
    try:
        from .groq_client import _client, _MODEL

        response = _client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise financial document parser. Output only valid JSON."},
                {"role": "user", "content": _GROQ_FILE_PROMPT + text[:6000]},
            ],
            temperature=0.1,
            max_tokens=4000,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        return parsed

    except Exception as e:
        logger.warning("Groq file structuring failed: %s", e)
        return None


# ─────────────────────────────────────────────
# 1. CSV IMPORT — Pandas
# ─────────────────────────────────────────────

# Column mapping (case-insensitive)
_COL_MAP = {
    "counterparty": "counterparty_name", "party": "counterparty_name",
    "vendor": "counterparty_name", "supplier": "counterparty_name",
    "name": "counterparty_name", "payee": "counterparty_name",
    "amount": "amount_inr", "amount_inr": "amount_inr",
    "amt": "amount_inr", "value": "amount_inr", "total": "amount_inr",
    "payment": "amount_inr",
    "due_date": "due_date", "due": "due_date", "date": "due_date",
    "deadline": "due_date", "pay_by": "due_date",
    "category": "category", "type": "category", "cat": "category",
    "description": "description", "desc": "description",
    "details": "description", "notes": "description",
    # Transaction columns
    "medium": "medium", "mode": "medium", "payment_mode": "medium",
    "payment_method": "medium",
    "direction": "direction", "txn_type": "direction",
    "reference": "external_ref", "ref": "external_ref",
    "ref_id": "external_ref", "txn_id": "external_ref",
    "cheque_no": "external_ref", "upi_id": "external_ref",
}


def import_csv(content: bytes, filename: str = "upload.csv") -> ImportResult:
    """Import obligations/transactions from a CSV or XLSX file using Pandas."""
    result = ImportResult(filename, "CSV")
    file_hash = compute_file_hash(content)
    result.file_hash = file_hash

    db = SessionLocal()
    try:
        existing = check_file_imported(db, file_hash)
        if existing:
            result.skipped_duplicate = True
            result.success = True
            result.error = f"File already imported on {existing.imported_at}"
            return result

        is_excel = filename.lower().endswith(('.xlsx', '.xls'))

        if PANDAS_AVAILABLE:
            if is_excel:
                df = pd.read_excel(io.BytesIO(content))
                result.file_type = "EXCEL"
            else:
                text = content.decode("utf-8-sig")
                df = pd.read_csv(io.StringIO(text))

            # Normalize column names
            rename_map = {}
            for col in df.columns:
                cleaned = col.strip().lower().replace(" ", "_")
                if cleaned in _COL_MAP:
                    rename_map[col] = _COL_MAP[cleaned]
            df = df.rename(columns=rename_map)

            has_party = "counterparty_name" in df.columns
            has_amount = "amount_inr" in df.columns

            # Smart fallback: use description as counterparty if no party column
            if not has_party and "description" in df.columns:
                df["counterparty_name"] = df["description"]
                has_party = True

            if not has_party or not has_amount:
                # Fallback: send raw text to Groq for structuring
                raw_text = df.to_string() if PANDAS_AVAILABLE else content.decode("utf-8-sig", errors="replace")
                if is_groq_available():
                    logger.info("CSV columns unrecognized, routing to Groq LLM")
                    structured = _groq_structure_file(raw_text)
                    if structured:
                        _validate_and_store_obligations(
                            db, structured.get("obligations", []),
                            "CSV", file_hash, result
                        )
                        _validate_and_store_transactions(
                            db, structured.get("transactions", []),
                            "CSV", file_hash, result
                        )
                        result.success = True
                        record_file_import(db, file_hash, filename, "CSV", len(content),
                                           len(result.obligations), result.new_count,
                                           result.updated_count, raw_text[:2000])
                        return result
                result.error = (
                    "CSV/Excel must have columns for counterparty (e.g. 'vendor', 'party', 'name') "
                    "and amount (e.g. 'amount', 'amt', 'total'). "
                    f"Found columns: {list(df.columns)}"
                )
                return result

            # Detect if this is a transaction file (has direction/medium columns)
            is_txn_file = "direction" in df.columns or "medium" in df.columns

            for _, row_data in df.iterrows():
                row_dict = {}
                for col in df.columns:
                    val = row_data[col]
                    if pd.notna(val):
                        row_dict[col] = str(val).strip() if not isinstance(val, (int, float)) else val

                # Parse amount
                amt_raw = row_dict.get("amount_inr", "")
                amt = _parse_amount(str(amt_raw)) if isinstance(amt_raw, str) else float(amt_raw) if amt_raw else None
                if not amt or amt <= 0:
                    continue
                row_dict["amount_inr"] = amt

                # Parse date
                date_field = row_dict.get("due_date", "")
                # Pandas may have parsed it already
                if hasattr(date_field, "date"):
                    row_dict["due_date"] = date_field.date()
                elif isinstance(date_field, str) and date_field:
                    parsed_d = None
                    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y"):
                        try:
                            parsed_d = datetime.strptime(date_field.strip(), fmt).date()
                            break
                        except ValueError:
                            continue
                    row_dict["due_date"] = parsed_d or date.today()
                else:
                    row_dict["due_date"] = date.today()

                if not row_dict.get("counterparty_name"):
                    row_dict["counterparty_name"] = "Unknown"

                if is_txn_file:
                    row_dict.setdefault("txn_date", row_dict.get("due_date", date.today()))
                    row_dict.setdefault("counterparty", row_dict.get("counterparty_name", "Unknown"))
                    _validate_and_store_transactions(db, [row_dict], "CSV", file_hash, result)
                else:
                    row_dict.setdefault("category", "OTHER")
                    _validate_and_store_obligations(db, [row_dict], "CSV", file_hash, result)

        else:
            # Fallback: stdlib csv + Groq (pandas not available)
            import csv
            text = content.decode("utf-8-sig", errors="replace")
            reader = csv.DictReader(io.StringIO(text))
            if is_groq_available():
                structured = _groq_structure_file(text)
                if structured:
                    _validate_and_store_obligations(
                        db, structured.get("obligations", []),
                        "CSV", file_hash, result
                    )
                    _validate_and_store_transactions(
                        db, structured.get("transactions", []),
                        "CSV", file_hash, result
                    )

        result.success = True
        # Safe preview: Excel files never have a `text` variable
        if is_excel:
            preview_text = df.to_string()[:2000] if PANDAS_AVAILABLE and 'df' in dir() else ""
        else:
            preview_text = (text[:2000] if 'text' in dir() and text else content.decode("utf-8-sig", errors="replace")[:2000])
        record_file_import(db, file_hash, filename, result.file_type, len(content),
                           len(result.obligations), result.new_count,
                           result.updated_count, preview_text)

    except Exception as e:
        result.error = str(e)
        logger.exception("CSV import failed: %s", e)
    finally:
        db.close()

    return result


# ─────────────────────────────────────────────
# 2. PDF IMPORT — pdfplumber + Camelot
# ─────────────────────────────────────────────

def import_pdf(content: bytes, filename: str = "upload.pdf") -> ImportResult:
    """Extract text + tables from PDF → Groq LLM → validate → store."""
    result = ImportResult(filename, "PDF")

    if not PDF_AVAILABLE:
        result.error = "pdfplumber not installed. Run: pip install pdfplumber"
        return result

    file_hash = compute_file_hash(content)
    result.file_hash = file_hash

    db = SessionLocal()
    try:
        existing = check_file_imported(db, file_hash)
        if existing:
            result.skipped_duplicate = True
            result.success = True
            result.error = f"File already imported on {existing.imported_at}"
            return result

        pages_text = []

        # 1. pdfplumber: extract text + embedded tables
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text.strip())

                tables = page.extract_tables()
                for table in tables:
                    for row in table:
                        if row:
                            pages_text.append(" | ".join(str(c or "") for c in row))

        # 2. Camelot: additional table extraction (if available)
        if CAMELOT_AVAILABLE:
            try:
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    cam_tables = camelot.read_pdf(tmp_path, pages="all", flavor="stream")
                    for table in cam_tables:
                        df = table.df
                        for _, row in df.iterrows():
                            row_str = " | ".join(str(v) for v in row.values if v)
                            if row_str.strip():
                                pages_text.append(row_str)
                finally:
                    os.unlink(tmp_path)
            except Exception as e:
                logger.warning("Camelot extraction failed (non-critical): %s", e)

        full_text = "\n".join(pages_text)
        result.raw_text = full_text

        # Scanned PDF fallback: if pdfplumber found no text, render pages and use Groq Vision OCR
        if not full_text.strip() and PDFIUM_AVAILABLE and is_groq_available():
            logger.info("PDF has no extractable text — trying Groq Vision OCR on rendered pages")
            vision_texts = []
            try:
                doc = pdfium.PdfDocument(io.BytesIO(content))
                for page_idx in range(min(len(doc), 10)):  # cap at 10 pages
                    page = doc[page_idx]
                    bitmap = page.render(scale=2)  # 2x scale for better OCR
                    pil_image = bitmap.to_pil()
                    # Convert to JPEG bytes
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format="JPEG", quality=90)
                    page_text = groq_vision_ocr(img_buffer.getvalue(), mime_type="image/jpeg")
                    if page_text:
                        vision_texts.append(f"--- Page {page_idx + 1} ---\n{page_text}")
                doc.close()
            except Exception as e:
                logger.warning("PDF Vision OCR failed: %s", e)

            if vision_texts:
                full_text = "\n\n".join(vision_texts)
                result.raw_text = full_text
                logger.info("Groq Vision OCR extracted %d chars from scanned PDF", len(full_text))

        if not full_text.strip():
            result.error = "No text could be extracted from the PDF (tried text + Vision OCR)"
            return result

        # 3. Route through Groq LLM for structuring
        structured = _groq_structure_file(full_text)
        if structured:
            _validate_and_store_obligations(
                db, structured.get("obligations", []),
                "PDF", file_hash, result
            )
            _validate_and_store_transactions(
                db, structured.get("transactions", []),
                "PDF", file_hash, result
            )
        else:
            # Fallback: regex parser (obligations only)
            parsed_obs, _ = parse_text_to_obligations(full_text)
            _validate_and_store_obligations(db, parsed_obs, "PDF", file_hash, result)

        result.success = True
        record_file_import(db, file_hash, filename, "PDF", len(content),
                           len(result.obligations), result.new_count,
                           result.updated_count, full_text[:5000])

    except Exception as e:
        result.error = str(e)
        logger.exception("PDF import failed: %s", e)
    finally:
        db.close()

    return result


# ─────────────────────────────────────────────
# 3. OCR IMAGE IMPORT — PaddleOCR
# ─────────────────────────────────────────────

def import_image(content: bytes, filename: str = "upload.jpg") -> ImportResult:
    """OCR via Groq Vision API → Groq LLM → validate → store."""
    result = ImportResult(filename, "IMAGE")

    if not is_groq_available():
        result.error = "Groq API key not configured. Set GROQ_API_KEY in .env to enable image OCR."
        return result

    file_hash = compute_file_hash(content)
    result.file_hash = file_hash

    db = SessionLocal()
    try:
        existing = check_file_imported(db, file_hash)
        if existing:
            result.skipped_duplicate = True
            result.success = True
            result.error = f"File already imported on {existing.imported_at}"
            return result

        # Detect MIME type from filename extension
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else "jpg"
        mime_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "webp": "image/webp",
            "bmp": "image/bmp", "tiff": "image/tiff",
        }
        mime_type = mime_map.get(ext, "image/jpeg")

        # Optionally compress large images before sending to Groq
        image_bytes = content
        if PIL_AVAILABLE and len(content) > 4_000_000:  # > 4MB
            try:
                img = Image.open(io.BytesIO(content)).convert("RGB")
                img.thumbnail((2048, 2048))  # cap resolution
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85)
                image_bytes = buf.getvalue()
                mime_type = "image/jpeg"
                logger.info("Image compressed: %d → %d bytes", len(content), len(image_bytes))
            except Exception as e:
                logger.warning("Image compression failed, using original: %s", e)

        # Run Groq Vision OCR
        raw_text = groq_vision_ocr(image_bytes, mime_type)

        if not raw_text or not raw_text.strip():
            result.error = "Groq Vision OCR returned no text from this image."
            return result

        result.raw_text = raw_text
        logger.info("Vision OCR extracted %d chars from %s", len(raw_text), filename)

        # Route through Groq LLM for structuring
        structured = _groq_structure_file(raw_text)
        if structured:
            _validate_and_store_obligations(
                db, structured.get("obligations", []),
                "IMAGE", file_hash, result
            )
            _validate_and_store_transactions(
                db, structured.get("transactions", []),
                "IMAGE", file_hash, result
            )
        else:
            # Fallback: regex parser
            parsed_obs, _ = parse_text_to_obligations(raw_text)
            _validate_and_store_obligations(db, parsed_obs, "IMAGE", file_hash, result)

        result.success = True
        record_file_import(db, file_hash, filename, "IMAGE", len(content),
                           len(result.obligations), result.new_count,
                           result.updated_count, raw_text[:5000])

    except Exception as e:
        result.error = str(e)
        logger.exception("Image import failed: %s", e)
    finally:
        db.close()

    return result


# ─────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────

def get_import_capabilities() -> dict:
    """Return which file import features are available."""
    groq_ok = is_groq_available()
    return {
        "csv": True,
        "csv_engine": "pandas" if PANDAS_AVAILABLE else "stdlib",
        "pdf_text": PDF_AVAILABLE,
        "pdf_tables": CAMELOT_AVAILABLE,
        "pdf_scanned_ocr": PDF_AVAILABLE and PDFIUM_AVAILABLE and groq_ok,
        "ocr": groq_ok,
        "ocr_engine": "groq-vision (llama-3.2-11b-vision-preview)" if groq_ok else "none",
        "llm": groq_ok,
        "llm_model": "groq-llama-3.1-8b-instant",
        "validation": "pydantic",
    }
