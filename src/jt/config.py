"""Paths and data-governance settings."""
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RUNS = ROOT / "runs"
MODELS = ROOT / "models"

# Endpoint settings and API keys come from ./.env only (copy .env.example and fill in your values).
ENV = dotenv_values(ROOT / ".env")

# Prepared MIMIC-CXR files (credentialed, not distributed): radiologist_test.jsonl and silver_validate_noleak.jsonl,
# one record per study with "study_id", "section" (report text) and "labels" (finding -> value).
CXR_SOURCE = Path(ENV.get("CXR_SOURCE_DIR") or "path/to/prepared/mimic-cxr")

# Credentialed data (PhysioNet DUA; PhysioNet LLM policy of 24 Sep 2025) may only reach these hosts, and never a
# model whose name starts with a denied prefix (e.g. gateway models that are forwarded to an outside provider).
CREDENTIALED_HOSTS = {h.strip() for h in ENV.get("MIMIC_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",") if h.strip()}
DENY_PREFIXES = tuple(p.strip() for p in (ENV.get("RESTRICTED_DENY_MODEL_PREFIXES") or "external-").split(",") if p.strip())

