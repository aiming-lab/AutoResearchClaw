"""Release-grade machine-readable artifact contracts (v2).

This module defines the *machine-readable contracts* extracted from the
governance-layer skills (anti-hallucination audit, clean-room provenance,
two-model review). It deliberately contains only the decidable subset:

  - claims.json           — claim ledger with provenance pointers
  - citations.json        — citation instances with claim-support mapping
  - critique.json         — Socratic critic findings (recommend-only)
  - critique_resolution.json — resolution status per critique finding
  - deai_audit.json       — de-AI prose audit (recommend-only, never applied)
  - truth_audit.json      — frozen paper hash + claims digest
  - run_manifest.json     — run-level manifest (models, roles, final stage)
  - attempts/attempt_log.jsonl — append-only per-stage attempt log
  - cost_log.jsonl        — per-stage cost accounting

Design rules (do not weaken):
  1. Gates read artifacts, never prose.
  2. citation existence != citation support (two independent facts).
  3. Truth audit freezes the paper BEFORE any de-AI/prose pass; the de-AI
     audit is recommend-only. If prose edits are adopted, the truth audit
     must be re-run — enforced via paper_sha256 invariance.
  4. Failed/degraded attempts are recorded, never deleted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "2.0"

#: Claim types that MUST carry provenance for release (user decision:
#: limited to four classes for v2; plain background prose is exempt).
CLAIM_TYPES: tuple[str, ...] = ("quantitative", "comparative", "result", "citation")

#: Citation instance roles. "claim_support" requires supported_claim_id +
#: support_excerpt; "background" requires only a resolved record.
CITATION_ROLES: tuple[str, ...] = ("claim_support", "background", "unmapped")

#: Valid resolution states for critique findings.
RESOLUTION_STATES: tuple[str, ...] = ("fixed", "rebutted", "accepted-risk", "unresolved")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Hashing / normalization
# ---------------------------------------------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def normalize_paper_text(text: str) -> str:
    """Whitespace-insensitive normalization so cosmetic reflow does not
    change the frozen paper hash, while any content change does."""
    return re.sub(r"\s+", " ", text).strip()


def paper_sha256(text: str) -> str:
    return sha256_text(normalize_paper_text(text))


#: Matches numeric tokens including decimals, percentages, thousands
#: separators, and scientific notation. Deliberately deterministic — no LLM.
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"      # 1,234 or 1,234.5
    r"|(?<![\w.])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?"          # 12.34, 1.2e-3
    r"|(?<![\w.])[-+]?\d+(?:[eE][-+]?\d+)?"               # 42, 1e5
)


def extract_numbers(text: str) -> list[float]:
    """Deterministically extract numeric values from free text.

    Used to close the quantitative/comparative claim loop: a claim whose
    ``values`` list is empty but whose text contains numbers must still be
    grounded in those numbers, not waved through as a generic result claim.
    Percent signs are ignored for the value itself (0.5% -> 0.5); callers
    compare against evidence with a relative tolerance.
    """
    out: list[float] = []
    for tok in _NUMBER_RE.findall(text or ""):
        cleaned = tok.replace(",", "")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


def numbers_match(a: float, b: float, *, rel_tol: float = 1e-3) -> bool:
    if round(a, 6) == round(b, 6):
        return True
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom <= rel_tol


def claims_digest(claims: list[dict[str, Any]]) -> str:
    """Digest over the semantic content of the claim ledger.

    Covers claim id, text, type, and evidence pointers — NOT free-form
    notes — so audits can attach commentary without breaking invariance.
    """
    core = []
    for c in sorted(claims, key=lambda c: str(c.get("id", ""))):
        core.append(
            {
                "id": c.get("id"),
                "text": normalize_paper_text(str(c.get("text", ""))),
                "type": c.get("type"),
                "status": c.get("status"),
                "evidence": [
                    {"path": e.get("path"), "sha256": e.get("sha256")}
                    for e in (c.get("evidence") or [])
                    if isinstance(e, dict)
                ],
            }
        )
    return sha256_text(json.dumps(core, sort_keys=True, ensure_ascii=False))


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp", prefix=path.name + "_")
    os.close(fd)
    try:
        Path(tmp).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Canonical paper resolution
# ---------------------------------------------------------------------------

_PAPER_CANDIDATES: tuple[str, ...] = (
    "stage-23/paper_final_verified.md",
    "stage-22/paper_final.md",
    "deliverables/paper_final.md",
)


def canonical_paper_path(run_dir: Path) -> Path | None:
    """The single verified source used by all release audits."""
    for rel in _PAPER_CANDIDATES:
        p = run_dir / rel
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


# ---------------------------------------------------------------------------
# Evidence index (clean-room provenance closure)
# ---------------------------------------------------------------------------

_EVIDENCE_GLOBS: tuple[str, ...] = (
    "stage-14*/experiment_summary.json",
    "experiment_summary_best.json",
    "stage-13*/refinement_log.json",
    "stage-12*/runs/*.json",
)


def collect_evidence_index(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Map of run-relative path -> {sha256, numeric_values}.

    Only artifacts produced by THIS run are legal provenance targets.
    Evidence pointers into skill text, memory, or external files are
    orphans by construction.
    """
    index: dict[str, dict[str, Any]] = {}
    for pattern in _EVIDENCE_GLOBS:
        for path in sorted(run_dir.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(run_dir).as_posix()
            digest = sha256_file(path)
            values: list[float] = []
            data = read_json(path)
            if data is not None:
                values = _collect_numeric_values(data)
            index[rel] = {"sha256": digest, "numeric_values": values}
    # Attempt log is also legal evidence (e.g. for method claims).
    attempt_log = run_dir / "attempts" / "attempt_log.jsonl"
    if attempt_log.is_file():
        rel = attempt_log.relative_to(run_dir).as_posix()
        index[rel] = {"sha256": sha256_file(attempt_log), "numeric_values": []}
    return index


def _collect_numeric_values(data: Any, _depth: int = 0) -> list[float]:
    if _depth > 8:
        return []
    out: list[float] = []
    if isinstance(data, bool):
        return out
    if isinstance(data, (int, float)):
        out.append(round(float(data), 4))
    elif isinstance(data, dict):
        for v in data.values():
            out.extend(_collect_numeric_values(v, _depth + 1))
    elif isinstance(data, list):
        for v in data[:200]:
            out.extend(_collect_numeric_values(v, _depth + 1))
    return out


def match_value_to_evidence(
    value: float, index: dict[str, dict[str, Any]], *, rel_tol: float = 1e-3
) -> str | None:
    """Return the evidence path whose numeric values contain *value*."""
    for rel, meta in index.items():
        for v in meta.get("numeric_values", ()):
            if v == round(value, 4):
                return rel
            denom = max(abs(v), abs(value), 1e-12)
            if abs(v - value) / denom <= rel_tol:
                return rel
    return None


def numbers_in_artifact(path: Path) -> list[float]:
    """Deterministically extract every number from an evidence artifact.

    JSON artifacts are walked structurally (numeric leaves); everything else
    is treated as text. No LLM. Used by release_check to confirm that a
    claimed matched_value actually occurs in the evidence file, not merely
    in claims.json.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    # Try structured JSON first (covers experiment_summary.json, run files).
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return extract_numbers(raw)
    nums = _collect_numeric_values(data)
    # Also scan any string leaves for embedded numbers (e.g. "F1=0.947").
    nums.extend(extract_numbers(raw))
    return nums


# ---------------------------------------------------------------------------
# Citation instance extraction
# ---------------------------------------------------------------------------

_MD_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*\d{4}[A-Za-z0-9_-]*$")


def extract_citation_instances(text: str) -> list[dict[str, Any]]:
    """Extract every citation *instance* (occurrence) with local context.

    Each instance is a distinct support obligation: the same bib key cited
    in two sentences must be supported at both locations.
    """
    instances: list[dict[str, Any]] = []
    stripped = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    def _context(pos: int) -> str:
        lo = max(0, pos - 240)
        hi = min(len(stripped), pos + 240)
        return normalize_paper_text(stripped[lo:hi])

    for m in re.finditer(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]+)\}", stripped):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                instances.append(
                    {"cite_key": key, "offset": m.start(), "context": _context(m.start())}
                )
    for m in re.finditer(r"\[([^\[\]]{4,300})\]", stripped):
        parts = [p.strip() for p in re.split(r"[,;]", m.group(1))]
        if parts and all(_MD_KEY_RE.fullmatch(p) for p in parts if p):
            for key in parts:
                if key:
                    instances.append(
                        {"cite_key": key, "offset": m.start(), "context": _context(m.start())}
                    )
    for i, inst in enumerate(instances):
        inst["instance_id"] = f"cit-{i:04d}"
    return instances


# ---------------------------------------------------------------------------
# Attempt log (append-only; failed attempts are first-class citizens)
# ---------------------------------------------------------------------------

def append_attempt(
    run_dir: Path,
    *,
    run_id: str,
    stage: int,
    stage_name: str,
    status: str,
    decision: str = "",
    error: str | None = None,
    elapsed_sec: float | None = None,
    artifacts: tuple[str, ...] = (),
    kind: str = "stage_execution",
) -> dict[str, Any]:
    log_path = run_dir / "attempts" / "attempt_log.jsonl"
    prior = 0
    if log_path.is_file():
        try:
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("stage") == stage and e.get("kind") == kind:
                    prior += 1
        except OSError:
            pass
    entry = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "run_id": run_id,
        "stage": stage,
        "stage_name": stage_name,
        "attempt": prior + 1,
        "attempt_id": f"stage{stage:02d}-a{prior + 1}",
        "status": status,
        "decision": decision,
        "error": (error or "")[:2000] or None,
        "elapsed_sec": round(elapsed_sec, 2) if elapsed_sec is not None else None,
        "artifacts": list(artifacts),
        "timestamp": utcnow_iso(),
    }
    append_jsonl(log_path, entry)
    return entry


def last_cumulative_cost(run_dir: Path) -> float:
    """Return the most recent cumulative_usd recorded in cost_log.jsonl.

    Used to compute per-stage deltas so summing rows never double-counts.
    """
    log_path = run_dir / "cost_log.jsonl"
    if not log_path.is_file():
        return 0.0
    last = 0.0
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            cum = entry.get("cumulative_usd")
            if isinstance(cum, (int, float)):
                last = float(cum)
    except OSError:
        return last
    return last


def append_cost_entry(
    run_dir: Path,
    *,
    stage: int,
    stage_name: str,
    model: str,
    attempt_id: str,
    cost_usd: float | None,
    cumulative_usd: float | None = None,
    elapsed_sec: float | None = None,
) -> None:
    """Append one cost row.

    ``cost_usd`` is the PER-STAGE delta (safe to sum across rows).
    ``cumulative_usd`` is the running total at this point (take the max/last,
    never sum). Keeping both lets consumers pick the correct aggregation.
    """
    append_jsonl(
        run_dir / "cost_log.jsonl",
        {
            "schema_version": SCHEMA_VERSION,
            "timestamp": utcnow_iso(),
            "stage": stage,
            "stage_name": stage_name,
            "model": model,
            "attempt_id": attempt_id,
            "cost_usd": cost_usd,
            "cumulative_usd": cumulative_usd,
            "elapsed_sec": round(elapsed_sec, 2) if elapsed_sec is not None else None,
        },
    )


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

def write_run_manifest(
    run_dir: Path,
    *,
    run_id: str,
    pipeline_version: str,
    expected_final_stage: int,
    writer_model: str,
    critic_model: str,
    critic_source: str,
    sectional_writer_model: str = "",
    sectional_critic_model: str = "",
    external_review_path: str = "",
    sandbox: dict[str, Any] | None = None,
    environment_policy: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``run_manifest.json`` at the run root.

    Named distinctly from ``deliverables/manifest.json`` on purpose.
    The reviewer block is what the reviewer_isolation release gate reads:
    the critic must be a different machine model, or an external reviewer
    (human / separate agent) with its own artifact — and it must never
    share the writer's conversational context.
    """
    key_artifacts: dict[str, str | None] = {}
    for rel in (
        "stage-24/claims.json",
        "stage-24/citations.json",
        "stage-24/truth_audit.json",
        "stage-25/deai_audit.json",
        "stage-15/critique.json",
        "attempts/attempt_log.jsonl",
    ):
        p = run_dir / rel
        if p.is_file():
            key_artifacts[rel] = sha256_file(p)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "pipeline_version": pipeline_version,
        "expected_final_stage": expected_final_stage,
        "reviewer": {
            "writer_model": writer_model,
            "critic_model": critic_model,
            "critic_source": critic_source,  # "model" | "external" | "none"
            "sectional_writer_model": sectional_writer_model,
            "sectional_critic_model": sectional_critic_model,
            "external_review_path": external_review_path,
            "shared_context": False,
        },
        "sandbox": sandbox or {},
        "environment_policy": environment_policy or {},
        "artifact_digests": key_artifacts,
        "generated": utcnow_iso(),
    }
    if extra:
        manifest.update(extra)
    write_json_atomic(run_dir / "run_manifest.json", manifest)
    return manifest
