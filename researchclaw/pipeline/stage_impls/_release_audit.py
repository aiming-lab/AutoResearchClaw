"""Stages 24-25: Release Audit (v2).

Stage 24 TRUTH_AUDIT — builds the machine-readable claim ledger
(claims.json), maps citation instances to claims (citations.json),
resolves Socratic critique findings (critique_resolution.json), and
freezes the paper hash + claims digest (truth_audit.json).

Stage 25 DEAI_AUDIT — recommend-only prose audit (deai_audit.json).
It NEVER modifies the paper. It verifies the paper hash is unchanged
since the truth audit; if prose edits were adopted in between, the stage
fails and the truth audit must be re-run first.

Ordering is a hard invariant: truth before prose. Do not reorder, and do
not turn the de-AI audit into an auto-rewriter.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.prompts import PromptManager
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.pipeline._helpers import StageResult, _safe_json_loads, _utcnow_iso
from researchclaw.pipeline import release_artifacts as ra

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claim extraction prompts
# ---------------------------------------------------------------------------

_CLAIM_EXTRACT_SYSTEM = """You are a truth auditor for a research pipeline. \
Extract checkable claims from the paper. Output STRICT JSON only (no prose): \
{"claims": [{"text": "<verbatim sentence>", "type": "quantitative|comparative|result|citation", \
"values": [<numbers appearing in the claim>], "cited_keys": ["<bib keys cited in the claim, if any>"]}]}
Rules:
- quantitative: states a specific measured number (metric, percentage, count of trials).
- comparative: claims X outperforms/underperforms Y.
- result: a contribution/finding statement without a specific number.
- citation: attributes a statement to a cited work.
- Only include claims that could in principle be checked against experiment artifacts or sources.
- Copy sentences verbatim. Do not paraphrase. Maximum 60 claims."""

_CITATION_MAP_SYSTEM = """You map citation instances to the claims they support. \
Output STRICT JSON only: {"mappings": [{"instance_id": "...", "role": "claim_support|background", \
"supported_claim_id": "<claim id or null>", "support_excerpt": "<the sentence the citation supports>"}]}
Rules:
- role=claim_support when the citation backs a specific factual claim; include the claim id and the exact sentence.
- role=background for scene-setting/related-work citations.
- Citation existence is NOT citation support. When unsure, use background."""

_RESOLUTION_SYSTEM = """You judge whether a paper addresses critique findings. \
Output STRICT JSON only: {"resolutions": [{"finding_id": "...", "resolution": "fixed|rebutted|unresolved", \
"note": "<one sentence of evidence from the paper>"}]}
Be strict: 'fixed' requires visible evidence in the paper text; 'rebutted' requires an explicit limitation/justification."""

_DEAI_SYSTEM = """You are a prose auditor detecting AI-generated stylistic tics in a research paper. \
You are RECOMMEND-ONLY: you never rewrite the paper. Output STRICT JSON only: \
{"suggestions": [{"span": "<verbatim excerpt>", "issue": "<what reads as AI-generated>", \
"suggested_rewrite": "<optional shorter human alternative>", "risk": "style_only|touches_claim"}]}
Rules:
- Mark risk=touches_claim if the excerpt contains a number, comparison, or cited statement.
- Never suggest changing any numeric value, claim meaning, or citation.
- Maximum 40 suggestions."""


# ---------------------------------------------------------------------------
# Stage 24: Truth Audit
# ---------------------------------------------------------------------------

def _execute_truth_audit(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    paper_path = ra.canonical_paper_path(run_dir)
    if paper_path is None:
        return StageResult(
            stage=Stage.TRUTH_AUDIT,
            status=StageStatus.FAILED,
            artifacts=(),
            error="Truth audit: no canonical paper artifact found.",
            decision="retry",
        )
    paper_text = paper_path.read_text(encoding="utf-8")
    paper_hash = ra.paper_sha256(paper_text)
    evidence_index = ra.collect_evidence_index(run_dir)

    # ---- 1. Claim extraction -------------------------------------------
    claims: list[dict[str, Any]] = []
    extraction_method = "none"
    if llm is not None:
        extraction_prompts = (
            paper_text[:60000],
            (
                "Extract at least one checkable claim from this paper if any "
                "scientific claim, number, comparison, or citation appears. "
                "Return STRICT JSON matching the schema.\n\nPAPER:\n"
                + paper_text[:30000]
            ),
        )
        for extraction_prompt in extraction_prompts:
            raw = _chat_json(llm, _CLAIM_EXTRACT_SYSTEM, extraction_prompt)
            candidates = raw.get("claims") if isinstance(raw, dict) else None
            if not isinstance(candidates, list):
                continue
            extraction_method = "llm"
            claims = []
            for i, c in enumerate(candidates[:60]):
                if not isinstance(c, dict) or not str(c.get("text", "")).strip():
                    continue
                ctype = str(c.get("type", "")).strip()
                if ctype not in ra.CLAIM_TYPES:
                    ctype = "result"
                claims.append(
                    {
                        "id": f"clm-{i:04d}",
                        "text": str(c.get("text", "")).strip(),
                        "type": ctype,
                        "values": [
                            float(v)
                            for v in (c.get("values") or [])
                            if isinstance(v, (int, float)) and not isinstance(v, bool)
                        ],
                        "cited_keys": [
                            str(k) for k in (c.get("cited_keys") or []) if str(k).strip()
                        ],
                        "evidence": [],
                        "status": "pending",
                    }
                )
            if claims:
                break

    if llm is not None and not claims:
        claims = _fallback_extract_claims(paper_text)
        if claims:
            extraction_method = "deterministic_fallback"

    if llm is not None and not claims:
        instances = ra.extract_citation_instances(paper_text)
        for inst in instances:
            inst["role"] = "unmapped"
            inst["supported_claim_id"] = None
            inst["support_excerpt"] = ""
        ra.write_json_atomic(
            stage_dir / "claims.json",
            {
                "schema_version": ra.SCHEMA_VERSION,
                "paper_path": str(paper_path.relative_to(run_dir)),
                "extraction_method": extraction_method,
                "claims": [],
                "counts": {
                    "total": 0,
                    "unsupported": 0,
                    "by_type": {t: 0 for t in ra.CLAIM_TYPES},
                },
                "generated": _utcnow_iso(),
            },
        )
        ra.write_json_atomic(
            stage_dir / "citations.json",
            {
                "schema_version": ra.SCHEMA_VERSION,
                "paper_path": str(paper_path.relative_to(run_dir)),
                "existence_report": "stage-23/verification_report.json"
                if (run_dir / "stage-23" / "verification_report.json").is_file()
                else None,
                "instances": [
                    {
                        "instance_id": i["instance_id"],
                        "cite_key": i["cite_key"],
                        "role": i["role"],
                        "supported_claim_id": i["supported_claim_id"],
                        "support_excerpt": i["support_excerpt"],
                        "context": i["context"][:400],
                    }
                    for i in instances
                ],
                "counts": {
                    "total": len(instances),
                    "claim_support": 0,
                    "background": 0,
                    "unmapped": len(instances),
                },
                "generated": _utcnow_iso(),
            },
        )
        ra.write_json_atomic(
            stage_dir / "critique_resolution.json",
            {
                "schema_version": ra.SCHEMA_VERSION,
                "critique_path": None,
                "resolutions": [],
                "generated": _utcnow_iso(),
            },
        )
        ra.write_json_atomic(
            stage_dir / "truth_audit.json",
            {
                "schema_version": ra.SCHEMA_VERSION,
                "paper_path": str(paper_path.relative_to(run_dir)),
                "paper_sha256": paper_hash,
                "claims_digest": ra.claims_digest([]),
                "extraction_method": extraction_method,
                "llm_available": True,
                "counts": {
                    "total": 0,
                    "unsupported": 0,
                    "by_type": {t: 0 for t in ra.CLAIM_TYPES},
                },
                "error": "claim extraction returned no claims while LLM was available",
                "generated": _utcnow_iso(),
            },
        )
        return StageResult(
            stage=Stage.TRUTH_AUDIT,
            status=StageStatus.FAILED,
            artifacts=(
                "claims.json",
                "citations.json",
                "critique_resolution.json",
                "truth_audit.json",
            ),
            evidence_refs=(
                "stage-24/claims.json",
                "stage-24/citations.json",
                "stage-24/critique_resolution.json",
                "stage-24/truth_audit.json",
            ),
            error="Truth audit: claim extraction returned no claims while LLM was available.",
            decision="retry",
        )

    # Build run-internal citation evidence: a citation-type claim is only
    # supported if its cited keys are VERIFIED (stage-23), and we attach the
    # verification report as a sha256-pinned evidence pointer. cited_keys
    # alone (parametric assertion) is NOT support.
    _verif_rel = "stage-23/verification_report.json"
    _verif_path = run_dir / _verif_rel
    _verified_keys: set[str] = set()
    _verif_sha: str | None = None
    if _verif_path.is_file():
        _verif_sha = ra.sha256_file(_verif_path)
        _vdata = ra.read_json(_verif_path) or {}
        _results = _vdata.get("results") if isinstance(_vdata, dict) else None
        if isinstance(_results, list):
            for _r in _results:
                if isinstance(_r, dict) and str(_r.get("status", "")).lower() in (
                    "verified",
                    "ok",
                    "suspicious",
                ):
                    _k = str(_r.get("cite_key") or _r.get("key") or "").strip()
                    if _k:
                        _verified_keys.add(_k)

    # ---- 2. Provenance closure: match values to run-internal evidence --
    unsupported = 0
    for claim in claims:
        if claim["type"] in ("quantitative", "comparative"):
            # A quant/comparative claim MUST close a numeric loop. If the
            # extractor gave no values, deterministically pull numbers from the
            # claim text itself — a numeric claim with no numbers, or numbers
            # that match no run-internal evidence, is unsupported. It must NOT
            # fall through to the generic result-claim evidence path.
            values = list(claim["values"])
            if not values:
                values = ra.extract_numbers(claim["text"])
                claim["values"] = values
            if not values:
                claim["status"] = "unsupported"
            else:
                matched_all = True
                for value in values:
                    src = ra.match_value_to_evidence(value, evidence_index)
                    if src is None:
                        matched_all = False
                    else:
                        claim["evidence"].append(
                            {
                                "path": src,
                                "sha256": evidence_index[src]["sha256"],
                                "matched_value": value,
                            }
                        )
                claim["status"] = (
                    "supported" if matched_all and claim["evidence"] else "unsupported"
                )
        elif claim["type"] == "citation":
            # Supported only if every cited key is verified AND we can pin the
            # verification report as run-internal evidence.
            cited = [k for k in claim["cited_keys"] if k]
            if (
                cited
                and _verif_sha
                and all(k in _verified_keys for k in cited)
            ):
                claim["evidence"].append(
                    {"path": _verif_rel, "sha256": _verif_sha, "verified_keys": cited}
                )
                claim["status"] = "supported"
            else:
                claim["status"] = "unsupported"
        else:
            # result/contribution claims: point at the attempt log + best summary
            for rel in ("attempts/attempt_log.jsonl", "experiment_summary_best.json"):
                if rel in evidence_index:
                    claim["evidence"].append(
                        {"path": rel, "sha256": evidence_index[rel]["sha256"]}
                    )
            claim["status"] = "supported" if claim["evidence"] else "unsupported"
        if claim["status"] == "unsupported":
            unsupported += 1

    claims_payload = {
        "schema_version": ra.SCHEMA_VERSION,
        "paper_path": str(paper_path.relative_to(run_dir)),
        "extraction_method": extraction_method,
        "claims": claims,
        "counts": {
            "total": len(claims),
            "unsupported": unsupported,
            "by_type": {
                t: sum(1 for c in claims if c["type"] == t) for t in ra.CLAIM_TYPES
            },
        },
        "generated": _utcnow_iso(),
    }
    ra.write_json_atomic(stage_dir / "claims.json", claims_payload)

    # ---- 3. Citation instances + support mapping -----------------------
    instances = ra.extract_citation_instances(paper_text)
    if llm is not None and instances and claims:
        claim_digest_for_llm = json.dumps(
            [{"id": c["id"], "text": c["text"][:200]} for c in claims],
            ensure_ascii=False,
        )
        inst_for_llm = json.dumps(
            [
                {
                    "instance_id": i["instance_id"],
                    "cite_key": i["cite_key"],
                    "context": i["context"][:400],
                }
                for i in instances[:120]
            ],
            ensure_ascii=False,
        )
        raw = _chat_json(
            llm,
            _CITATION_MAP_SYSTEM,
            f"CLAIMS:\n{claim_digest_for_llm}\n\nCITATION INSTANCES:\n{inst_for_llm}",
        )
        mapping_by_id: dict[str, dict[str, Any]] = {}
        if isinstance(raw, dict) and isinstance(raw.get("mappings"), list):
            for m in raw["mappings"]:
                if isinstance(m, dict) and m.get("instance_id"):
                    mapping_by_id[str(m["instance_id"])] = m
        valid_claim_ids = {c["id"] for c in claims}
        for inst in instances:
            m = mapping_by_id.get(inst["instance_id"], {})
            role = str(m.get("role", "unmapped"))
            if role not in ra.CITATION_ROLES:
                role = "unmapped"
            claim_id = m.get("supported_claim_id")
            if role == "claim_support" and claim_id not in valid_claim_ids:
                role = "unmapped"
            inst["role"] = role
            inst["supported_claim_id"] = claim_id if role == "claim_support" else None
            inst["support_excerpt"] = (
                str(m.get("support_excerpt", ""))[:500] if role == "claim_support" else ""
            )
    else:
        for inst in instances:
            inst["role"] = "unmapped"
            inst["supported_claim_id"] = None
            inst["support_excerpt"] = ""

    verification = ra.read_json(run_dir / "stage-23" / "verification_report.json") or {}
    citations_payload = {
        "schema_version": ra.SCHEMA_VERSION,
        "paper_path": str(paper_path.relative_to(run_dir)),
        "existence_report": "stage-23/verification_report.json"
        if verification
        else None,
        "instances": [
            {
                "instance_id": i["instance_id"],
                "cite_key": i["cite_key"],
                "role": i["role"],
                "supported_claim_id": i["supported_claim_id"],
                "support_excerpt": i["support_excerpt"],
                "context": i["context"][:400],
            }
            for i in instances
        ],
        "counts": {
            "total": len(instances),
            "claim_support": sum(1 for i in instances if i["role"] == "claim_support"),
            "background": sum(1 for i in instances if i["role"] == "background"),
            "unmapped": sum(1 for i in instances if i["role"] == "unmapped"),
        },
        "generated": _utcnow_iso(),
    }
    ra.write_json_atomic(stage_dir / "citations.json", citations_payload)

    # ---- 4. Critique resolution (Socratic critic gate) ------------------
    critique = ra.read_json(run_dir / "stage-15" / "critique.json") or {}
    findings = critique.get("findings") if isinstance(critique, dict) else None
    resolutions: list[dict[str, Any]] = []
    if isinstance(findings, list) and findings:
        serious = [
            f
            for f in findings
            if isinstance(f, dict) and str(f.get("severity", "")).upper() in ("P0", "P1")
        ]
        if llm is not None and serious:
            raw = _chat_json(
                llm,
                _RESOLUTION_SYSTEM,
                "FINDINGS:\n"
                + json.dumps(serious, ensure_ascii=False)
                + "\n\nPAPER:\n"
                + paper_text[:50000],
            )
            by_id = {}
            if isinstance(raw, dict) and isinstance(raw.get("resolutions"), list):
                for r in raw["resolutions"]:
                    if isinstance(r, dict) and r.get("finding_id"):
                        by_id[str(r["finding_id"])] = r
            for f in serious:
                fid = str(f.get("id", ""))
                r = by_id.get(fid, {})
                res = str(r.get("resolution", "unresolved"))
                if res not in ra.RESOLUTION_STATES:
                    res = "unresolved"
                resolutions.append(
                    {
                        "finding_id": fid,
                        "severity": str(f.get("severity", "")).upper(),
                        "resolution": res,
                        "note": str(r.get("note", ""))[:500],
                    }
                )
        else:
            for f in serious:
                resolutions.append(
                    {
                        "finding_id": str(f.get("id", "")),
                        "severity": str(f.get("severity", "")).upper(),
                        "resolution": "unresolved",
                        "note": "No LLM available to judge resolution — failing closed.",
                    }
                )
    ra.write_json_atomic(
        stage_dir / "critique_resolution.json",
        {
            "schema_version": ra.SCHEMA_VERSION,
            "critique_path": "stage-15/critique.json" if findings else None,
            "resolutions": resolutions,
            "generated": _utcnow_iso(),
        },
    )

    # ---- 5. Freeze -------------------------------------------------------
    digest = ra.claims_digest(claims)
    ra.write_json_atomic(
        stage_dir / "truth_audit.json",
        {
            "schema_version": ra.SCHEMA_VERSION,
            "paper_path": str(paper_path.relative_to(run_dir)),
            "paper_sha256": paper_hash,
            "claims_digest": digest,
            "extraction_method": extraction_method,
            "llm_available": llm is not None,
            "counts": claims_payload["counts"],
            "generated": _utcnow_iso(),
        },
    )

    artifacts = (
        "claims.json",
        "citations.json",
        "critique_resolution.json",
        "truth_audit.json",
    )
    # Demo-pipeline leniency: the stage completes even with an empty ledger
    # (no LLM), but release_check fails closed on empty/unsupported claims.
    return StageResult(
        stage=Stage.TRUTH_AUDIT,
        status=StageStatus.DONE,
        artifacts=artifacts,
        evidence_refs=tuple(f"stage-24/{a}" for a in artifacts),
    )


# ---------------------------------------------------------------------------
# Stage 25: De-AI Audit (recommend-only)
# ---------------------------------------------------------------------------

#: Deterministic stylistic-tic patterns (governance layer owns the taste;
#: this list only produces *recommendations*, never a gate).
_DEAI_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bdelve(?:s|d)?\b", "'delve' is a well-known AI-generated tic"),
    (r"\bIt is worth noting that\b", "hedging filler common in AI prose"),
    (r"\bIn conclusion,", "formulaic closer"),
    (r"\bFurthermore,\s", "chained formal connectives read as generated"),
    (r"\bMoreover,\s", "chained formal connectives read as generated"),
    (r"\bplays a (?:crucial|pivotal|vital) role\b", "stock intensifier phrase"),
    (r"\bunderscore(?:s|d)? the importance\b", "stock emphasis phrase"),
    (r"\bcomprehensive(?:ly)?\b", "overused breadth adjective"),
)


def _execute_deai_audit(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    truth = ra.read_json(run_dir / "stage-24" / "truth_audit.json")
    if not isinstance(truth, dict) or not truth.get("paper_sha256"):
        return StageResult(
            stage=Stage.DEAI_AUDIT,
            status=StageStatus.FAILED,
            artifacts=(),
            error="De-AI audit requires a completed truth audit (stage-24/truth_audit.json).",
            decision="retry",
        )

    paper_path = ra.canonical_paper_path(run_dir)
    if paper_path is None:
        return StageResult(
            stage=Stage.DEAI_AUDIT,
            status=StageStatus.FAILED,
            artifacts=(),
            error="De-AI audit: no canonical paper artifact found.",
            decision="retry",
        )
    paper_text = paper_path.read_text(encoding="utf-8")
    current_hash = ra.paper_sha256(paper_text)
    frozen_hash = str(truth.get("paper_sha256"))

    if current_hash != frozen_hash:
        # The paper changed after the truth audit. Prose edits invalidate the
        # frozen claim ledger — re-run stage 24 (and stage 23 if citations
        # were touched) before auditing style. Fail closed.
        return StageResult(
            stage=Stage.DEAI_AUDIT,
            status=StageStatus.FAILED,
            artifacts=(),
            error=(
                "Paper hash changed since truth audit "
                f"({frozen_hash[:12]}… → {current_hash[:12]}…). "
                "Re-run TRUTH_AUDIT before the de-AI audit."
            ),
            decision="retry",
        )

    suggestions: list[dict[str, Any]] = []
    stripped = re.sub(r"```.*?```", "", paper_text, flags=re.DOTALL)
    for pattern, issue in _DEAI_PATTERNS:
        for m in re.finditer(pattern, stripped, flags=re.IGNORECASE):
            lo = max(0, m.start() - 80)
            hi = min(len(stripped), m.end() + 80)
            span = ra.normalize_paper_text(stripped[lo:hi])
            touches_claim = bool(re.search(r"\d|\\cite|\[[A-Za-z]+\d{4}", span))
            suggestions.append(
                {
                    "source": "heuristic",
                    "span": span[:300],
                    "issue": issue,
                    "suggested_rewrite": "",
                    "risk": "touches_claim" if touches_claim else "style_only",
                }
            )
            if len(suggestions) >= 60:
                break

    if llm is not None:
        raw = _chat_json(llm, _DEAI_SYSTEM, paper_text[:60000])
        if isinstance(raw, dict) and isinstance(raw.get("suggestions"), list):
            for s in raw["suggestions"][:40]:
                if not isinstance(s, dict):
                    continue
                risk = str(s.get("risk", "style_only"))
                if risk not in ("style_only", "touches_claim"):
                    risk = "touches_claim"  # unknown → conservative
                suggestions.append(
                    {
                        "source": "llm",
                        "span": str(s.get("span", ""))[:300],
                        "issue": str(s.get("issue", ""))[:300],
                        "suggested_rewrite": str(s.get("suggested_rewrite", ""))[:500],
                        "risk": risk,
                    }
                )

    ra.write_json_atomic(
        stage_dir / "deai_audit.json",
        {
            "schema_version": ra.SCHEMA_VERSION,
            "recommend_only": True,
            "applied": False,
            "paper_path": str(paper_path.relative_to(run_dir)),
            "paper_sha256": current_hash,
            "truth_audit_sha256": frozen_hash,
            "hash_invariant_ok": True,
            "suggestions": suggestions,
            "counts": {
                "total": len(suggestions),
                "touches_claim": sum(
                    1 for s in suggestions if s["risk"] == "touches_claim"
                ),
            },
            "rework_rule": (
                "If suggestions are adopted: edits touching citation instances or "
                "claim spans require re-running stages 23+24; style-only edits "
                "require re-running stage 24. Never edit automatically."
            ),
            "generated": _utcnow_iso(),
        },
    )

    return StageResult(
        stage=Stage.DEAI_AUDIT,
        status=StageStatus.DONE,
        artifacts=("deai_audit.json",),
        evidence_refs=("stage-25/deai_audit.json",),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_json(
    llm: LLMClient, system: str, user: str, *, model: str | None = None
) -> Any:
    """One stateless JSON-mode call. The audit stages never share the
    writer's conversational context — every call starts fresh."""
    try:
        resp = llm.chat(
            [{"role": "user", "content": user}],
            system=system,
            json_mode=True,
            model=model,
            strip_thinking=True,
        )
        return _loads_json_repaired(resp.content, {})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Release audit LLM call failed: %s", exc)
        return {}


def _loads_json_repaired(text: str, default: Any) -> Any:
    """Parse JSON from LLM output with bounded, deterministic repair.

    This is intentionally conservative: it only removes common transport/
    formatting noise (markdown fences, line comments, trailing commas, prose
    around a single JSON object/array). It never fabricates missing fields.
    """
    for candidate in _json_candidates(text):
        stripped = _strip_markdown_fence(candidate.strip())
        try:
            return json.loads(stripped)
        except (TypeError, json.JSONDecodeError):
            pass
        repaired = _strip_json_comments(stripped)
        repaired = _strip_trailing_commas(repaired)
        try:
            return json.loads(repaired)
        except (TypeError, json.JSONDecodeError):
            continue
    return _safe_json_loads(text, default)


def _json_candidates(text: str) -> list[str]:
    raw = text or ""
    candidates = [raw]
    fenced = _strip_markdown_fence(raw.strip())
    if fenced != raw:
        candidates.append(fenced)
    embedded = _extract_first_json_value(raw)
    if embedded:
        candidates.append(embedded)
    if embedded:
        unfenced = _strip_markdown_fence(embedded.strip())
        if unfenced != embedded:
            candidates.append(unfenced)
    # Preserve order while dropping exact duplicates.
    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _strip_markdown_fence(text: str) -> str:
    m = re.fullmatch(r"\s*```[A-Za-z0-9_-]*\s*\n(.*?)\n?\s*```\s*", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _strip_json_comments(text: str) -> str:
    out: list[str] = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_first_json_value(text: str) -> str | None:
    starts = [i for i, ch in enumerate(text) if ch in "{["]
    for start in starts:
        opener = text[start]
        closer = "}" if opener == "{" else "]"
        stack = [closer]
        in_string = False
        escape = False
        for idx in range(start + 1, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch in "{[":
                stack.append("}" if ch == "{" else "]")
            elif ch in "}]":
                if not stack or ch != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    return text[start : idx + 1]
        # Try the next possible opening brace/bracket.
    return None


_RESULT_KEYWORDS = re.compile(
    r"\b("
    r"achiev(?:e|ed|es)|accuracy|auc|detect(?:ion|ed|s)?|f1|false positive|"
    r"fpr|improv(?:e|ed|es|ement)|latency|outperform(?:s|ed)?|precision|"
    r"recall|reduc(?:e|ed|es|tion)|result(?:s)?|throughput|tpr"
    r")\b",
    re.IGNORECASE,
)

_COMPARISON_KEYWORDS = re.compile(
    r"\b(outperform(?:s|ed)?|underperform(?:s|ed)?|better|worse|higher|lower|"
    r"improv(?:e|ed|es)|reduc(?:e|ed|es))\b",
    re.IGNORECASE,
)


def _fallback_extract_claims(paper_text: str, *, limit: int = 60) -> list[dict[str, Any]]:
    """Deterministically extract checkable claims from paper prose.

    The fallback exists only to recover from malformed LLM JSON. It copies
    verbatim paper sentences and leaves evidence empty; normal provenance
    closure below decides whether each claim can be supported.
    """
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sentence in _candidate_claim_sentences(paper_text):
        normalized = ra.normalize_paper_text(sentence)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values = ra.extract_numbers(normalized)
        if values:
            ctype = "comparative" if _COMPARISON_KEYWORDS.search(normalized) else "quantitative"
        elif _RESULT_KEYWORDS.search(normalized):
            ctype = "result"
        else:
            continue
        claims.append(
            {
                "id": f"clm-{len(claims):04d}",
                "text": normalized,
                "type": ctype,
                "values": values,
                "cited_keys": _claim_cited_keys(normalized),
                "evidence": [],
                "status": "pending",
            }
        )
        if len(claims) >= limit:
            break
    return claims


def _candidate_claim_sentences(paper_text: str) -> list[str]:
    text = re.sub(r"```.*?```", " ", paper_text or "", flags=re.DOTALL)
    sections = _preferred_sections(text)
    search_text = "\n".join(sections) if sections else text
    sentences: list[str] = []
    for para in re.split(r"\n\s*\n", search_text):
        para = para.strip()
        if not para or para.startswith("|"):
            continue
        para = re.sub(r"^#{1,6}\s+", "", para)
        for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", para):
            cleaned = ra.normalize_paper_text(part)
            if 20 <= len(cleaned) <= 700:
                sentences.append(cleaned)
    return sentences


def _preferred_sections(text: str) -> list[str]:
    current_name = ""
    current_lines: list[str] = []
    sections: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", line)
        if m:
            if current_lines:
                sections.append((current_name, "\n".join(current_lines)))
            current_name = m.group(1).strip().lower()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_name, "\n".join(current_lines)))
    wanted = ("abstract", "result", "discussion", "conclusion")
    selected = [body for name, body in sections if any(w in name for w in wanted)]
    return selected


def _claim_cited_keys(text: str) -> list[str]:
    keys: list[str] = []
    for m in re.finditer(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]+)\}", text):
        keys.extend(k.strip() for k in m.group(1).split(",") if k.strip())
    for m in re.finditer(r"\[([^\[\]]{4,300})\]", text):
        parts = [p.strip() for p in re.split(r"[,;]", m.group(1))]
        if parts and all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*\d{4}[A-Za-z0-9_-]*", p) for p in parts if p):
            keys.extend(p for p in parts if p)
    return keys
