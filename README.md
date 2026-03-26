# FlowGuard 🛡️
### AI-Powered Cash Flow Priority Engine for Indian MSMEs

> **Stop guessing which bill to pay first. FlowGuard tells you exactly — with regulatory context, penalty calculations, and plain-English reasoning.**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203-FF6B35)](https://console.groq.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)

---

## 🚀 What It Does

FlowGuard is a conversational CFO advisor for Indian small businesses. You describe your bills and cash balance in plain English — it tells you what to pay, what to defer, and what will hurt you most if ignored.

| Intent | Example Input | What Happens |
|---|---|---|
| **INGEST** | *"GST ₹42k due 20th, EMI ₹28.5k due 25th, cash 1.5L"* | Extracts & stores all obligations |
| **STATUS** | *"What should I pay first?"* | Scores every obligation, returns PAY/DEFER/NEGOTIATE/ESCALATE |
| **FILTER** | *"Show all statutory dues"* | Queries DB, returns matching records |
| **File Upload** | Excel / PDF / Image | Extracts obligations via OCR + LLM |

---

## 🏗️ Architecture

```
chat.html (UI)
    │
FastAPI (main.py) — port 8000
    ├── /pipeline       ← NL text → INGEST / STATUS / FILTER
    ├── /upload/csv     ← Excel / CSV ingestion
    ├── /upload/pdf     ← PDF (text + scanned via Vision OCR)
    ├── /upload/image   ← JPG/PNG → Groq Vision OCR
    ├── /profile        ← Business onboarding
    └── /              ← Serves chat.html (no Live Server needed)
         │
    ┌────┴────────────────────────────┐
    │                                 │
groq_client.py              scorer.py (deterministic)
    │                                 │
  Groq API                    7-factor consequence score
  ├── llama-3.1-8b-instant    (pure math — no LLM, reproducible)
  ├── llama-3.3-70b-versatile
  └── llama-4-scout-17b (Vision OCR)
         │
    SQLite (flowguard.db)
    ├── obligations
    ├── transactions
    ├── file_imports (dedup by SHA256)
    └── user_profile
```

---

## ⚡ Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/vigneshnagarajan07/FlowGuard.git
cd FlowGuard
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac
pip install -r flowguard/requirements.txt
```

### 2. Set API Key
Create `.env` in the project root:
```env
GROQ_API_KEY=gsk_your_key_here
```
Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run
```bash
uvicorn flowguard.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open
Navigate to **http://localhost:8000** — the onboarding modal will appear.

---

## 🌟 Key Features

### 🤖 Conversational AI (3 LLMs)
| Model | Role |
|---|---|
| `llama-3.1-8b-instant` | Parses NL input → structured JSON obligations |
| `llama-3.3-70b-versatile` | COT narration — India-specific regulatory advice |
| `llama-4-scout-17b` (Vision) | OCR — extracts text from invoices & scanned PDFs |

### 📊 Deterministic Scoring Engine
- 7-factor consequence score: penalty rate, due date proximity, relationship score, statutory flag, flexibility, recurring flag, amount weight
- **Same input always = same output** (auditable, reproducible)
- Actions: `PAY` · `DEFER` · `NEGOTIATE` · `ESCALATE`

### 📁 File Ingestion Pipeline
```
Excel/CSV → pandas → column normalisation → Groq LLM (unrecognised columns)
PDF (text) → pdfplumber → Groq LLM → Pydantic → DB
PDF (scan) → pypdfium2 renders pages → Groq Vision → Groq LLM → DB
Image/PNG → Pillow converts → Groq Vision OCR → Groq LLM → DB
```
- SHA256 dedup — same file never imported twice
- Pydantic validation on every obligation before DB write

### 🏢 Business Profile Context
Onboarding dialog captures: Name · Business · Industry · GSTIN · Turnover.  
Every AI request is prefixed with this context so advice is industry-specific.

### 💬 Stats in Every Response
```json
{
  "stats": {
    "total_amount_due_inr": 135500,
    "cash_balance_inr": 150000,
    "estimated_shortfall_inr": 0,
    "critical_count": 1,
    "days_to_critical": 20
  }
}
```

---

## 🇮🇳 Indian Regulatory Context Built-In

| Category | Risk if Deferred |
|---|---|
| GST / TDS / PF | 18% annual interest + ₹100/day + prosecution |
| EMI / Bank Loan | NPA in 90 days, credit score, asset seizure |
| Salary | Labour Court complaint, ₹25,000 penalty |
| Rent | Eviction, security deposit forfeiture |
| Supplier | Supply stop, credit line revoked |
| Utility | Disconnection after 15-30 days |

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status + model info |
| `POST` | `/pipeline` | Main chat endpoint — INGEST / STATUS / FILTER |
| `POST` | `/upload/csv` | Upload Excel / CSV |
| `POST` | `/upload/pdf` | Upload PDF |
| `POST` | `/upload/image` | Upload image (Groq Vision OCR) |
| `GET/POST` | `/profile` | Business profile |
| `GET` | `/obligations` | List all obligations |
| `GET` | `/transactions` | List all transactions |
| `POST` | `/whatif` | Scenario analysis |
| `GET` | `/` | Chat UI (chat.html) |

---

## 🗂️ Project Structure

```
FlowGuard/
├── chat.html                  # Frontend chat UI
├── .env                       # GROQ_API_KEY (not committed)
├── flowguard/
│   ├── main.py                # FastAPI app + all endpoints
│   ├── groq_client.py         # All LLM calls (text + vision)
│   ├── scorer.py              # Deterministic scoring engine
│   ├── file_ingest.py         # Excel/PDF/Image ingestion pipeline
│   ├── database.py            # SQLAlchemy ORM + SQLite
│   ├── models.py              # Pydantic models
│   ├── parser.py              # Regex fallback parser
│   ├── requirements.txt
│   └── flowguard.db           # SQLite database (auto-created)
└── README.md
```

---

## 🔧 Optional Dependencies

| Package | Feature | Install |
|---|---|---|
| `camelot-py[cv]` | PDF table extraction | Needs Ghostscript binary |
| `spacy` | Better NER | `python -m spacy download en_core_web_sm` |
| `sentence-transformers` | Intent classification | Auto-downloads model |

---

## 👥 Team
Built for hackathon presentation — FlowGuard v2.0.0

---

*FlowGuard uses Groq's ultra-fast LPU inference for real-time financial analysis.*
