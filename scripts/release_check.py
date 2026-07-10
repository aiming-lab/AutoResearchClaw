#!/usr/bin/env python3
"""Read-only release readiness check for ResearchClaw run directories (v2).

This checker is intentionally stricter than the demo pipeline. It fails closed
when required safety metadata is missing.

v2 gates (in addition to v1):
  - run_manifest.json drives the expected final stage (no hardcoded 23)
  - reviewer_isolation: critic model != writer model, or external reviewer
  - claims_provenance_closure: every release-scoped claim points at
    run-internal evidence (path + sha256), no orphans
  - citation_support: citation existence != citation support; every
    claim_support instance names its claim + excerpt; no unmapped instances
  - claims_digest_invariance: the paper hash frozen by the truth audit is
    unchanged after the (recommend-only) de-AI audit
  - critique_resolution: every P0/P1 Socratic finding has a resolution
  - no_real_data upgraded from warning to error (waiver file possible)

POLICY (do not weaken): rule changes to this file must land in their own
commit, never in the same commit as generated run artifacts. Weakening a
gate to make a particular run pass is a process violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml


EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_DEGRADED = 2

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

#: Claim types that MUST carry provenance (mirrors release_artifacts.CLAIM_TYPES).
RELEASE_CLAIM_TYPES = ("quantitative", "comparative", "result", "citation")
RESOLUTION_OK = ("fixed", "rebutted", "accepted-risk")

_PLACEHOLDER_PAPER_MARKERS = (
    "No content generated.",
    "# Skipped Stage",
)

_CITATION_EVIDENCE_PATH = "stage-23/verification_report.json"


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    path: str = ""


class ReleaseChecker:
    def __init__(self, run_dir: Path, *, quality_threshold: float, allow_suspicious: bool) -> None:
        self.run_dir = run_dir
        self.quality_threshold = quality_threshold
        self.allow_suspicious = allow_suspicious
        self.findings: list[Finding] = []

    def run(self) -> int:
        if not self.run_dir.exists() or not self.run_dir.is_dir():
            self.error("run_dir_missing", f"Run directory does not exist: {self.run_dir}")
            return self.exit_code()

        summary = self.read_json("pipeline_summary.json", required=True)
        quality = self.read_json("stage-20/quality_report.json", required=True)
        fabrication = self.read_json("stage-20/fabrication_flags.json", required=True)
        verification = self.read_json("stage-23/verification_report.json", required=True)
        manifest = self.read_json("run_manifest.json", required=True)

        expected_final = self.expected_final_stage(manifest)

        self.check_summary(summary, expected_final)
        self.check_degradation_signal(quality)
        self.check_quality_report(quality)
        self.check_experiment_contract()
        self.check_fabrication_flags(fabrication)
        self.check_paper_artifacts()
        self.check_citations(verification)
        self.check_sandbox_metadata()
        self.check_environment_metadata()
        self.check_deliverables(summary, expected_final)
        self.check_compile_status()
        self.check_canonical_source()

        # ---- v2 gates ----
        claims = self.read_json("stage-24/claims.json", required=True)
        citations = self.read_json("stage-24/citations.json", required=True)
        truth = self.read_json("stage-24/truth_audit.json", required=True)
        deai = self.read_json("stage-25/deai_audit.json", required=True)

        self.check_reviewer_isolation(manifest)
        self.check_claims_provenance(claims)
        self.check_citation_support(citations, claims)
        self.check_digest_invariance(truth, deai, claims)
        self.check_critique_resolution()
        self.check_cost_log()

        return self.exit_code()

    def expected_final_stage(self, manifest: dict[str, Any] | None) -> int | None:
        """Manifest-driven final stage (v2). No hardcoded stage number:
        adding pipeline stages must not silently invalidate this check."""
        if not manifest:
            return None
        value = int_value(manifest.get("expected_final_stage"))
        if value is None:
            self.error(
                "manifest_final_stage_missing",
                "run_manifest.json lacks expected_final_stage.",
                "run_manifest.json",
            )
        return value

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------

    def check_summary(
        self, summary: dict[str, Any] | None, expected_final: int | None
    ) -> None:
        if not summary:
            return
        final_stage = summary.get("final_stage")
        final_status = summary.get("final_status")
        if expected_final is not None and (
            final_stage != expected_final or final_status != "done"
        ):
            self.error(
                "incomplete_run",
                "Release check applies only to complete runs; expected "
                f"final_stage={expected_final} and final_status='done' "
                f"(got final_stage={final_stage!r}, final_status={final_status!r}).",
                "pipeline_summary.json",
            )
        if int_value(summary.get("stages_failed")) != 0:
            self.error(
                "failed_stages",
                f"Pipeline summary reports failed stages: {summary.get('stages_failed')!r}.",
                "pipeline_summary.json",
            )
        if bool(summary.get("degraded")):
            self.error(
                "degraded_summary",
                "Pipeline summary reports degraded=true.",
                "pipeline_summary.json",
            )

    def check_degradation_signal(self, quality: dict[str, Any] | None) -> None:
        signal_path = self.run_dir / "degradation_signal.json"
        if not signal_path.exists():
            return
        signal = self.read_json("degradation_signal.json", required=False)
        signal_time = parse_time(signal.get("generated") if isinstance(signal, dict) else None)
        quality_time = parse_time(quality.get("generated") if isinstance(quality, dict) else None)
        if signal_time and quality_time and signal_time < quality_time:
            code = "stale_degradation_signal"
            msg = (
                "Stale degradation_signal.json exists. Release mode treats any degradation signal as a blocker; "
                "timestamp only classifies this as stale residue."
            )
        else:
            code = "degradation_signal"
            msg = "degradation_signal.json exists. Release mode treats any degradation signal as a blocker."
        self.error(code, msg, "degradation_signal.json")

    def check_quality_report(self, quality: dict[str, Any] | None) -> None:
        if not quality:
            return
        score = float_value(quality.get("score_1_to_10"))
        if score is None:
            self.error("quality_score_missing", "Quality report has no numeric score_1_to_10.", "stage-20/quality_report.json")
        elif score < self.quality_threshold:
            self.error(
                "quality_below_threshold",
                f"Quality score {score:.2f} is below release threshold {self.quality_threshold:.2f}.",
                "stage-20/quality_report.json",
            )
        verdict = str(quality.get("verdict", "")).strip().lower()
        if verdict in {"revise", "reject", "failed", "fail"}:
            self.error(
                "quality_verdict_not_release_ready",
                f"Quality verdict is not release-ready: {verdict!r}.",
                "stage-20/quality_report.json",
            )

    def check_experiment_contract(self) -> None:
        contract_path = find_experiment_contract(self.run_dir)
        if contract_path is None:
            self.error(
                "experiment_contract_missing",
                "stage-09/experiment_contract.yaml is required for release checks.",
                "stage-09/experiment_contract.yaml",
            )
            return
        try:
            payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            self.error(
                "experiment_contract_invalid",
                f"Cannot parse experiment contract: {exc}",
                relpath(contract_path, self.run_dir),
            )
            return
        if not isinstance(payload, dict):
            self.error(
                "experiment_contract_invalid",
                "experiment contract root must be an object.",
                relpath(contract_path, self.run_dir),
            )
            return
        claim_scope = str(payload.get("claim_scope") or "").strip()
        dataset_origin = str(payload.get("dataset_origin") or "").strip()
        if claim_scope == "research_release" and dataset_origin == "synthetic":
            waiver = self.read_json(
                "waivers/synthetic_research_release.json", required=False
            )
            reason = str((waiver or {}).get("reason", "")).strip()
            approved_by = str((waiver or {}).get("approved_by", "")).strip()
            if waiver and reason and approved_by:
                self.warning(
                    "synthetic_research_release_waived",
                    f"research_release + synthetic waived by {approved_by!r}: {reason[:200]}",
                    "waivers/synthetic_research_release.json",
                )
            else:
                self.error(
                    "synthetic_research_release_blocked",
                    "experiment_contract.yaml declares claim_scope=research_release "
                    "with dataset_origin=synthetic and no valid waiver.",
                    relpath(contract_path, self.run_dir),
                )

    def check_fabrication_flags(self, fabrication: dict[str, Any] | None) -> None:
        if not fabrication:
            return
        if bool(fabrication.get("fabrication_suspected")):
            self.error(
                "fabrication_suspected",
                "fabrication_flags.json reports fabrication_suspected=true.",
                "stage-20/fabrication_flags.json",
            )
        if fabrication.get("has_real_data") is False:
            # v2: upgraded from warning to error. A paper with no real
            # experiment data must not silently pass release. A simulation-
            # only or theory run can be released via an explicit, signed,
            # machine-readable waiver.
            waiver = self.read_json("waivers/no_real_data.json", required=False)
            reason = str((waiver or {}).get("reason", "")).strip()
            approved_by = str((waiver or {}).get("approved_by", "")).strip()
            if waiver and reason and approved_by:
                self.warning(
                    "no_real_data_waived",
                    f"has_real_data=false waived by {approved_by!r}: {reason[:200]}",
                    "waivers/no_real_data.json",
                )
            else:
                self.error(
                    "no_real_data",
                    "fabrication_flags.json reports has_real_data=false and no "
                    "valid waiver (waivers/no_real_data.json with non-empty "
                    "'reason' and 'approved_by') exists.",
                    "stage-20/fabrication_flags.json",
                )

    def check_paper_artifacts(self) -> None:
        paper_paths = (
            "stage-22/paper_final.md",
            "stage-23/paper_final_verified.md",
            "deliverables/paper_final.md",
            "stage-22/paper.tex",
            "deliverables/paper.tex",
        )
        found = False
        for rel in paper_paths:
            text = self.read_text(rel).strip()
            if not text:
                continue
            found = True
            # v2: a placeholder is not a paper. Non-empty is necessary but
            # not sufficient.
            for marker in _PLACEHOLDER_PAPER_MARKERS:
                if marker in text and len(text) < 2000:
                    self.error(
                        "paper_artifact_placeholder",
                        f"Paper artifact {rel} contains placeholder marker {marker!r}.",
                        rel,
                    )
        if not found:
            self.error(
                "paper_artifact_missing",
                "No non-empty paper artifact was found; expected paper_final.md or paper.tex.",
                "stage-22/paper_final.md",
            )

    def check_citations(self, verification: dict[str, Any] | None) -> None:
        cited_keys = self.collect_citation_keys()
        bib_keys = self.collect_bib_keys()

        if cited_keys and not bib_keys:
            self.error(
                "cited_keys_without_verified_bib",
                f"Paper cites {len(cited_keys)} key(s), but no verified bibliography entries were found.",
                "stage-23/references_verified.bib",
            )
        missing = sorted(cited_keys - bib_keys)
        if missing:
            self.error(
                "missing_verified_bib_entries",
                "Cited keys missing from verified bibliography: " + ", ".join(missing[:20]),
                "stage-23/references_verified.bib",
            )

        if not verification:
            return
        summary = verification.get("summary")
        if not isinstance(summary, dict):
            summary = verification
        hallucinated = int_value(summary.get("hallucinated"))
        suspicious = int_value(summary.get("suspicious"))
        if hallucinated is None:
            self.error("verification_hallucinated_missing", "Verification report has no summary.hallucinated count.", "stage-23/verification_report.json")
        elif hallucinated > 0:
            self.error(
                "hallucinated_citations",
                f"Verification report has hallucinated citations: {hallucinated}.",
                "stage-23/verification_report.json",
            )
        if suspicious is None:
            self.error("verification_suspicious_missing", "Verification report has no summary.suspicious count.", "stage-23/verification_report.json")
        elif suspicious > 0 and not self.allow_suspicious:
            self.error(
                "suspicious_citations",
                f"Verification report has suspicious citations: {suspicious}.",
                "stage-23/verification_report.json",
            )
        if cited_keys:
            total = int_value(summary.get("total"))
            if total == 0:
                self.error(
                    "verification_total_zero_with_citations",
                    "Verification report says total=0, but the paper contains citation keys.",
                    "stage-23/verification_report.json",
                )

    def check_sandbox_metadata(self) -> None:
        metadata = self.find_metadata(("sandbox_metadata.json", "runtime_safety.json", "release_safety.json", "stage-12/sandbox_metadata.json", "stage-13/sandbox_metadata.json"))
        if metadata is None:
            self.error(
                "sandbox_metadata_missing",
                "Sandbox backend metadata is missing; failing closed until requested/actual backend is persisted.",
            )
            return
        sandbox = nested_dict(metadata, "sandbox") or metadata
        requested = lower_str(first_present(sandbox, ("requested_backend", "requested_mode", "experiment_mode", "requested")))
        actual = lower_str(first_present(sandbox, ("actual_backend", "backend", "actual")))
        fallback = bool(first_present(sandbox, ("fallback_used", "unsafe_fallback", "docker_fallback"), default=False))
        if not actual:
            self.error("sandbox_actual_backend_missing", "Sandbox metadata lacks actual backend.", metadata_path(metadata))
        if requested == "docker" and actual in {"subprocess", "experiment_sandbox", "sandbox"}:
            self.error("docker_fell_back_to_subprocess", "Requested Docker mode but actual backend is subprocess.", metadata_path(metadata))
        if fallback:
            self.error("unsafe_sandbox_fallback", "Sandbox metadata reports fallback_used=true.", metadata_path(metadata))

    def check_environment_metadata(self) -> None:
        metadata = self.find_metadata(("environment_policy.json", "runtime_safety.json", "release_safety.json", "stage-12/environment_policy.json", "stage-13/environment_policy.json"))
        if metadata is None:
            self.error(
                "environment_metadata_missing",
                "Sandbox environment-policy metadata is missing; failing closed until env policy is persisted.",
            )
            return
        env_policy = nested_dict(metadata, "environment_policy") or nested_dict(metadata, "environment") or metadata
        policy = lower_str(first_present(env_policy, ("policy", "name", "mode")))
        inherit_all = bool(first_present(env_policy, ("inherit_all", "inherits_host_env"), default=False))
        if inherit_all or policy in {"inherit_all", "host", "os.environ", "full"}:
            self.error("unsafe_environment_policy", f"Unsafe environment policy: {policy or 'inherit_all=true'}.", metadata_path(metadata))
        if not policy and "allowlist" not in env_policy:
            self.error("environment_policy_missing", "Environment metadata lacks policy or allowlist.", metadata_path(metadata))

    def check_deliverables(
        self, summary: dict[str, Any] | None, expected_final: int | None
    ) -> None:
        manifest = self.run_dir / "deliverables" / "manifest.json"
        if not manifest.exists():
            self.error("deliverables_manifest_missing", "deliverables/manifest.json is missing.", "deliverables/manifest.json")
            return
        # Determine whether this run is actually clean (complete, not degraded,
        # no failed stages). A non-clean run's deliverables must be explicitly
        # flagged not_release_ready and must NOT advertise release_ready=true.
        run_incomplete = bool(
            summary
            and expected_final is not None
            and (
                summary.get("final_stage") != expected_final
                or summary.get("final_status") != "done"
            )
        )
        run_degraded = bool(summary and summary.get("degraded")) or (
            self.run_dir / "degradation_signal.json"
        ).exists()
        run_failed = bool(summary and int_value(summary.get("stages_failed")) not in (0, None))
        run_not_clean = run_incomplete or run_degraded or run_failed

        if run_incomplete:
            self.error(
                "deliverables_for_incomplete_run",
                f"Deliverables exist for a run that did not complete Stage {expected_final}.",
                "deliverables/manifest.json",
            )

        manifest_data = self.read_json("deliverables/manifest.json", required=False)
        if isinstance(manifest_data, dict):
            claims_release_ready = manifest_data.get("release_ready") is True
            marked_not_ready = manifest_data.get("not_release_ready") is True
            if run_not_clean:
                # A failed/degraded/incomplete run must not produce deliverables
                # that look release-ready. Require an explicit not_release_ready
                # marker, and reject any release_ready=true claim.
                if claims_release_ready or not marked_not_ready:
                    self.error(
                        "deliverables_not_flagged_not_release_ready",
                        "Run is degraded/failed/incomplete, but deliverables/manifest.json does not "
                        "explicitly set not_release_ready=true (or falsely claims release_ready=true).",
                        "deliverables/manifest.json",
                    )
            if claims_release_ready:
                self.warning(
                    "manifest_release_ready_untrusted",
                    "manifest release_ready=true is not sufficient for pass; release_check remains authoritative.",
                    "deliverables/manifest.json",
                )

    def check_compile_status(self) -> None:
        has_tex = any(
            (self.run_dir / rel).exists()
            for rel in ("stage-22/paper.tex", "deliverables/paper.tex")
        )
        if not has_tex:
            return
        compile_status = self.read_json("stage-22/compile_status.json", required=False)
        if isinstance(compile_status, dict):
            if compile_status.get("success") is not True:
                if (
                    lower_str(compile_status.get("status")) == "toolchain_missing"
                    or compile_status.get("tooling_available") is False
                ):
                    self.error(
                        "compile_toolchain_missing",
                        "LaTeX compile toolchain is unavailable; install pdflatex/TeX Live or compile in CI.",
                        "stage-22/compile_status.json",
                    )
                else:
                    self.error(
                        "compile_failed",
                        "compile_status.json does not report success=true.",
                        "stage-22/compile_status.json",
                    )
            return
        # v2: the pipeline now always writes compile_status.json (success or
        # failure). Inferring success from compilation_quality.json was a
        # fail-open shortcut and has been removed.
        self.error(
            "compile_status_missing",
            "No machine-readable compile evidence (stage-22/compile_status.json) found for paper.tex.",
            "stage-22/compile_status.json",
        )

    def check_canonical_source(self) -> None:
        metadata = self.find_metadata(("canonical_source.json", "stage-23/canonical_source.json", "stage-22/canonical_source.json"))
        if metadata is None:
            self.error(
                "canonical_source_metadata_missing",
                "Canonical source provenance metadata is missing; cannot confirm Markdown and LaTeX share one verified source.",
            )
            return
        source_id = first_present(metadata, ("source_id", "canonical_source", "path"))
        md_source = first_present(metadata, ("markdown_source_id", "paper_final_md_source", "markdown_source"))
        tex_source = first_present(metadata, ("latex_source_id", "paper_tex_source", "tex_source"))
        if source_id and md_source and tex_source and (md_source != source_id or tex_source != source_id):
            self.error("mixed_canonical_sources", "Markdown and LaTeX provenance do not point to the same canonical source.", metadata_path(metadata))
        has_tex = any(
            (self.run_dir / rel).exists()
            for rel in ("stage-22/paper.tex", "deliverables/paper.tex")
        )
        for path_key, hash_key, label in (
            ("markdown_path", "markdown_sha256", "Markdown"),
            ("latex_path", "latex_sha256", "LaTeX"),
        ):
            rel = str(metadata.get(path_key) or "").strip()
            expected = str(metadata.get(hash_key) or "").strip()
            if not rel or not expected:
                if label == "LaTeX" and not has_tex:
                    continue
                self.error(
                    "canonical_source_hash_missing",
                    f"{label} canonical source metadata must include both {path_key} and {hash_key}.",
                    metadata_path(metadata),
                )
                continue
            if rel.startswith(("/", "..")) or ".." in rel.split("/"):
                self.error(
                    "canonical_source_path_invalid",
                    f"{label} canonical source path is outside the run directory.",
                    metadata_path(metadata),
                )
                continue
            actual_path = self.run_dir / rel
            if not actual_path.is_file() or sha256_of_file(actual_path) != expected:
                self.error(
                    "canonical_source_hash_mismatch",
                    f"{label} canonical source hash does not match {rel}.",
                    metadata_path(metadata),
                )

    # ------------------------------------------------------------------
    # v2 gates
    # ------------------------------------------------------------------

    def check_reviewer_isolation(self, manifest: dict[str, Any] | None) -> None:
        """Two-model review is only meaningful with real isolation:
        different model (or an external reviewer with its own artifact),
        and no shared conversational context with the writer."""
        if not manifest:
            return
        reviewer = nested_dict(manifest, "reviewer") or {}
        writer = lower_str(reviewer.get("writer_model"))
        critic = lower_str(reviewer.get("critic_model"))
        source = lower_str(reviewer.get("critic_source"))

        if bool(reviewer.get("shared_context")):
            self.error(
                "reviewer_shared_context",
                "run_manifest reviewer.shared_context=true — the critic saw the writer's context.",
                "run_manifest.json",
            )

        if source == "external":
            ext_rel = str(reviewer.get("external_review_path") or "").strip()
            ext_ok = False
            if ext_rel and not ext_rel.startswith(("/", "..")) and ".." not in ext_rel.split("/"):
                ext_path = self.run_dir / ext_rel
                ext_ok = ext_path.is_file() and ext_path.stat().st_size > 0
            if not ext_ok:
                self.error(
                    "external_review_artifact_missing",
                    "critic_source=external requires reviewer.external_review_path "
                    "pointing at an existing, non-empty review artifact inside the run directory.",
                    "run_manifest.json",
                )
        elif source == "model":
            if not critic:
                self.error(
                    "critic_model_missing",
                    "critic_source=model but no critic model is declared.",
                    "run_manifest.json",
                )
            elif critic == writer:
                self.error(
                    "reviewer_not_isolated",
                    f"Critic model equals writer model ({critic!r}); two-model review requires distinct models.",
                    "run_manifest.json",
                )
        else:
            self.error(
                "reviewer_isolation_undeclared",
                f"No isolated reviewer declared (critic_source={source or 'none'!r}). "
                "Configure llm.critic_model or attach an external review.",
                "run_manifest.json",
            )

        critique = self.read_json("stage-15/critique.json", required=True)
        if critique:
            if lower_str(critique.get("critic_source")) == "none":
                self.error(
                    "critique_without_critic",
                    "stage-15/critique.json was produced without an isolated critic.",
                    "stage-15/critique.json",
                )
            if bool(critique.get("shared_context")):
                self.error(
                    "critique_shared_context",
                    "stage-15/critique.json reports shared_context=true.",
                    "stage-15/critique.json",
                )

    def check_claims_provenance(self, claims_data: dict[str, Any] | None) -> None:
        """Clean-room closure: every release-scoped claim must point at
        evidence produced by THIS run (existing path, matching sha256)."""
        if not claims_data:
            return
        claims = claims_data.get("claims")
        if not isinstance(claims, list) or not claims:
            self.error(
                "claims_empty",
                "stage-24/claims.json contains no claims; the truth audit did not run effectively.",
                "stage-24/claims.json",
            )
            return
        unsupported = 0
        orphans = 0
        disallowed_evidence = 0
        supported_without_evidence = 0
        numeric_unclosed = 0
        numeric_evidence_value_missing = 0
        artifact_numbers_cache: dict[str, list[float]] = {}
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            ctype = str(claim.get("type", ""))
            if ctype not in RELEASE_CLAIM_TYPES:
                continue
            if str(claim.get("status", "")) != "supported":
                unsupported += 1
                continue
            # A claim is only "supported" for release if it carries at least
            # one VALID evidence pointer (exists in run_dir, sha256 matches).
            # This closes two escapes:
            #   (a) status=supported with an empty evidence list, and
            #   (b) citation-type claims marked supported merely because
            #       cited_keys is non-empty (no run-internal evidence).
            evidence = claim.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                supported_without_evidence += 1
                continue
            valid_pointers = 0
            # matched_values that are ALSO independently found in their own
            # evidence artifact's content (not merely asserted in claims.json).
            evidence_backed_values: list[float] = []
            claimed_but_absent = False
            for ev in evidence:
                if not isinstance(ev, dict):
                    orphans += 1
                    continue
                rel = str(ev.get("path", ""))
                if not rel or rel.startswith(("/", "..")) or ".." in rel.split("/"):
                    orphans += 1
                    continue
                if not is_allowed_claim_evidence_path(rel, ctype):
                    disallowed_evidence += 1
                    continue
                target = self.run_dir / rel
                if not target.is_file():
                    orphans += 1
                    continue
                recorded = str(ev.get("sha256") or "")
                # sha256 is mandatory: an unpinned pointer is not verifiable.
                if not recorded or sha256_of_file(target) != recorded:
                    orphans += 1
                    continue
                valid_pointers += 1
                mv = ev.get("matched_value")
                if isinstance(mv, (int, float)) and not isinstance(mv, bool):
                    # Independently verify: the claimed matched_value MUST occur
                    # in the evidence artifact's real content (deterministic
                    # extraction), not just in claims.json.
                    if rel not in artifact_numbers_cache:
                        artifact_numbers_cache[rel] = numbers_in_artifact(target)
                    art_numbers = artifact_numbers_cache[rel]
                    if any(numbers_close(float(mv), an) for an in art_numbers):
                        evidence_backed_values.append(float(mv))
                    else:
                        claimed_but_absent = True
            if valid_pointers == 0:
                supported_without_evidence += 1
                continue
            # Numeric closure for quantitative/comparative claims: a generic
            # pointer (e.g. attempt_log with no numbers) is NOT enough. At
            # least one evidence pointer must carry a matched_value that both
            # (a) corresponds to a number stated in the claim (values field or
            # numbers parsed from claim text), AND (b) is deterministically
            # present in that evidence artifact's real content.
            if ctype in ("quantitative", "comparative"):
                claim_numbers = [
                    float(v)
                    for v in (claim.get("values") or [])
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                claim_numbers += numbers_from_text(str(claim.get("text", "")))
                closed = any(
                    any(numbers_close(mv, cn) for cn in claim_numbers)
                    for mv in evidence_backed_values
                )
                if not closed:
                    numeric_unclosed += 1
                    # If the claim named a matched_value that simply isn't in
                    # the evidence file, surface that specific fabrication.
                    if claimed_but_absent and not evidence_backed_values:
                        numeric_evidence_value_missing += 1
        if unsupported:
            self.error(
                "claims_unsupported",
                f"{unsupported} release-scoped claim(s) have no supporting run-internal evidence.",
                "stage-24/claims.json",
            )
        if supported_without_evidence:
            self.error(
                "claims_supported_without_evidence",
                f"{supported_without_evidence} claim(s) are marked supported but carry no "
                "valid run-internal evidence pointer (empty evidence, or all pointers orphaned/unpinned). "
                "citation-type claims must also point at run-internal evidence, not just cited_keys.",
                "stage-24/claims.json",
            )
        if numeric_evidence_value_missing:
            self.error(
                "claims_numeric_evidence_value_missing",
                f"{numeric_evidence_value_missing} quantitative/comparative claim(s) declare a matched_value "
                "in claims.json that does NOT occur in the referenced evidence artifact's real content "
                "(deterministic extraction). A matched_value asserted only in claims.json is not evidence.",
                "stage-24/claims.json",
            )
        if disallowed_evidence:
            self.error(
                "claims_disallowed_evidence_path",
                f"{disallowed_evidence} claim evidence pointer(s) target files that are inside "
                "the run directory but outside the release evidence allowlist. Evidence must "
                "come from stage-12 run files, stage-13 refinement logs, stage-14 summaries, "
                "experiment_summary_best.json, attempts/attempt_log.jsonl, or citation "
                "verification for citation claims.",
                "stage-24/claims.json",
            )
        if numeric_unclosed:
            self.error(
                "claims_numeric_not_closed",
                f"{numeric_unclosed} quantitative/comparative claim(s) lack a numeric evidence match: "
                "no evidence pointer carries a matched_value that is both stated in the claim "
                "(values or claim text) AND present in the evidence artifact. Generic "
                "attempt_log/summary evidence is not sufficient.",
                "stage-24/claims.json",
            )
        if orphans:
            self.error(
                "claims_orphan_evidence",
                f"{orphans} evidence pointer(s) are missing, outside the run directory, unpinned, or fail digest verification.",
                "stage-24/claims.json",
            )

    def check_citation_support(
        self,
        citations_data: dict[str, Any] | None,
        claims_data: dict[str, Any] | None,
    ) -> None:
        """Citation existence != citation support. Existence is checked by
        stage-23 verification; THIS gate checks that each in-text citation
        instance is mapped to the claim it supports (or declared background)."""
        if not citations_data:
            return
        instances = citations_data.get("instances")
        cited_keys = self.collect_citation_keys()
        if not isinstance(instances, list):
            self.error(
                "citation_support_instances_invalid",
                "stage-24/citations.json must contain an instances array.",
                "stage-24/citations.json",
            )
            return
        if not instances:
            if cited_keys:
                self.error(
                    "citation_support_instances_missing",
                    "The paper contains citation keys, but stage-24/citations.json "
                    "has no citation instances to bind to supported claims.",
                    "stage-24/citations.json",
                )
                return
            # A paper with no citation keys and zero citation instances is
            # suspicious but not a support violation per se.
            self.warning(
                "citation_instances_empty",
                "stage-24/citations.json lists no citation instances.",
                "stage-24/citations.json",
            )
            return
        claim_ids: set[str] = set()
        claim_text_by_id: dict[str, str] = {}
        if claims_data and isinstance(claims_data.get("claims"), list):
            for c in claims_data["claims"]:
                if isinstance(c, dict) and c.get("id"):
                    cid = str(c.get("id"))
                    claim_ids.add(cid)
                    claim_text_by_id[cid] = normalize_for_substr(str(c.get("text", "")))
        # Explicit background whitelist (opt-in escape). Only cite_keys or
        # section names listed here may be background-only; without it,
        # role=background is NOT a blanket escape from claim binding.
        wl = citations_data.get("background_whitelist")
        bg_keys: set[str] = set()
        bg_sections: set[str] = set()
        if isinstance(wl, dict):
            bg_keys = {str(k) for k in (wl.get("cite_keys") or [])}
            bg_sections = {str(s).lower() for s in (wl.get("sections") or [])}
        elif isinstance(wl, list):  # shorthand: list of cite_keys
            bg_keys = {str(k) for k in wl}

        bib_keys = self.collect_bib_keys()
        unmapped = 0
        bad_support = 0
        fabricated_excerpt = 0
        missing_bib = 0
        background_not_whitelisted = 0
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            role = str(inst.get("role", "unmapped"))
            key = str(inst.get("cite_key", ""))
            section = str(inst.get("section", "")).lower()
            if role == "claim_support":
                claim_id = inst.get("supported_claim_id")
                excerpt = str(inst.get("support_excerpt", "")).strip()
                if not claim_id or str(claim_id) not in claim_ids or not excerpt:
                    bad_support += 1
                else:
                    # The excerpt must be a REAL quotation: a normalized
                    # substring of either the citation's local context or the
                    # supported claim's text. A non-empty but fabricated
                    # excerpt must not pass.
                    norm_excerpt = normalize_for_substr(excerpt)
                    context = normalize_for_substr(str(inst.get("context", "")))
                    claim_text = claim_text_by_id.get(str(claim_id), "")
                    if norm_excerpt not in context and norm_excerpt not in claim_text:
                        fabricated_excerpt += 1
            elif role == "background":
                # Background is allowed ONLY when explicitly whitelisted by
                # cite_key or section. Otherwise it must bind a claim like any
                # other instance — background is not a generic escape hatch.
                whitelisted = (key and key in bg_keys) or (
                    section and section in bg_sections
                )
                if not whitelisted:
                    background_not_whitelisted += 1
            else:
                unmapped += 1
            if bib_keys and key and key not in bib_keys:
                missing_bib += 1
        if unmapped:
            self.error(
                "citation_support_unmapped",
                f"{unmapped} citation instance(s) are neither claim_support nor background.",
                "stage-24/citations.json",
            )
        if bad_support:
            self.error(
                "citation_support_invalid",
                f"{bad_support} claim_support instance(s) lack a valid claim id or support excerpt.",
                "stage-24/citations.json",
            )
        if fabricated_excerpt:
            self.error(
                "citation_support_excerpt_fabricated",
                f"{fabricated_excerpt} claim_support instance(s) have a support_excerpt that is not a "
                "normalized substring of the citation context or the supported claim text. "
                "The excerpt must be a real quotation, not merely non-empty.",
                "stage-24/citations.json",
            )
        if background_not_whitelisted:
            self.error(
                "citation_background_not_whitelisted",
                f"{background_not_whitelisted} instance(s) claim role=background without an explicit "
                "background_whitelist entry (cite_key or section). Bind them to a claim, or add an "
                "auditable whitelist. Background is not a blanket escape from claim binding.",
                "stage-24/citations.json",
            )
        if missing_bib:
            self.error(
                "citation_instance_not_in_verified_bib",
                f"{missing_bib} citation instance(s) reference keys absent from the verified bibliography.",
                "stage-24/citations.json",
            )

    def check_digest_invariance(
        self,
        truth: dict[str, Any] | None,
        deai: dict[str, Any] | None,
        claims_data: dict[str, Any] | None,
    ) -> None:
        """Truth-before-prose invariant: the paper frozen by the truth audit
        must be byte-identical (modulo whitespace) after the de-AI audit,
        and the claims digest must match the ledger on disk."""
        if not truth or not deai:
            return
        frozen = str(truth.get("paper_sha256") or "")
        if not frozen:
            self.error(
                "truth_audit_hash_missing",
                "truth_audit.json lacks paper_sha256.",
                "stage-24/truth_audit.json",
            )
            return
        if str(deai.get("truth_audit_sha256") or "") != frozen or str(
            deai.get("paper_sha256") or ""
        ) != frozen:
            self.error(
                "claims_digest_invariance_broken",
                "Paper hash changed between truth audit and de-AI audit. "
                "If prose suggestions were adopted, stages 23/24 must be re-run.",
                "stage-25/deai_audit.json",
            )
        if bool(deai.get("applied")) or deai.get("recommend_only") is False:
            self.error(
                "deai_audit_applied",
                "deai_audit.json indicates edits were applied automatically; the de-AI audit must be recommend-only.",
                "stage-25/deai_audit.json",
            )
        # Recompute paper hash from the canonical artifact. The paper_path
        # MUST be present, inside the run dir, exist, be non-empty, and hash
        # to the frozen value. A missing/empty path is a fail, never a skip —
        # otherwise the whole truth-before-prose invariant is unenforced.
        paper_rel = str(truth.get("paper_path") or "").strip()
        if not paper_rel:
            self.error(
                "truth_audit_paper_path_missing",
                "truth_audit.json lacks paper_path; cannot verify the frozen paper hash.",
                "stage-24/truth_audit.json",
            )
        elif paper_rel.startswith(("/", "..")) or ".." in paper_rel.split("/"):
            self.error(
                "truth_audit_paper_path_outside_run",
                f"truth_audit.paper_path {paper_rel!r} escapes the run directory.",
                "stage-24/truth_audit.json",
            )
        else:
            target = self.run_dir / paper_rel
            if not target.is_file() or target.stat().st_size == 0:
                self.error(
                    "truth_audit_paper_missing",
                    f"truth_audit.paper_path {paper_rel!r} does not exist or is empty in the run directory.",
                    paper_rel,
                )
            else:
                text = self.read_text(paper_rel)
                if paper_hash_of(text) != frozen:
                    self.error(
                        "paper_hash_mismatch",
                        f"Recomputed hash of {paper_rel} differs from the frozen truth-audit hash.",
                        paper_rel,
                    )
        # Recompute claims digest from the ledger on disk.
        recorded_digest = str(truth.get("claims_digest") or "")
        if claims_data and recorded_digest:
            recomputed = claims_digest_of(claims_data.get("claims") or [])
            if recomputed != recorded_digest:
                self.error(
                    "claims_digest_mismatch",
                    "claims.json content does not match the digest frozen in truth_audit.json.",
                    "stage-24/claims.json",
                )

    def check_critique_resolution(self) -> None:
        critique = self.read_json("stage-15/critique.json", required=False)
        if not critique:
            return  # reviewer_isolation already errors on missing critique
        findings = critique.get("findings")
        if not isinstance(findings, list):
            return
        serious = {
            str(f.get("id")): str(f.get("severity", "")).upper()
            for f in findings
            if isinstance(f, dict)
            and str(f.get("severity", "")).upper() in ("P0", "P1")
            and f.get("id")
        }
        if not serious:
            return
        resolution_data = self.read_json("stage-24/critique_resolution.json", required=True)
        resolutions: dict[str, str] = {}
        if resolution_data and isinstance(resolution_data.get("resolutions"), list):
            for r in resolution_data["resolutions"]:
                if isinstance(r, dict) and r.get("finding_id"):
                    resolutions[str(r["finding_id"])] = str(r.get("resolution", ""))
        unresolved = [
            fid
            for fid in serious
            if resolutions.get(fid) not in RESOLUTION_OK
        ]
        if unresolved:
            self.error(
                "critique_findings_unresolved",
                f"{len(unresolved)} P0/P1 Socratic finding(s) lack a resolution "
                f"(fixed/rebutted/accepted-risk): {', '.join(unresolved[:10])}",
                "stage-24/critique_resolution.json",
            )

    def check_cost_log(self) -> None:
        if not (self.run_dir / "cost_log.jsonl").is_file():
            self.warning(
                "cost_log_missing",
                "cost_log.jsonl is missing; per-stage cost accounting is unavailable. "
                "(Deliberately a warning: cost gates must not incentivize skipping audit rounds.)",
                "cost_log.jsonl",
            )

    # ------------------------------------------------------------------
    # Artifact readers
    # ------------------------------------------------------------------

    def collect_citation_keys(self) -> set[str]:
        keys: set[str] = set()
        for rel in ("stage-22/paper_final.md", "stage-23/paper_final_verified.md", "deliverables/paper_final.md"):
            text = self.read_text(rel)
            if text:
                keys.update(extract_markdown_citations(text))
        for rel in ("stage-22/paper.tex", "deliverables/paper.tex"):
            text = self.read_text(rel)
            if text:
                keys.update(extract_latex_citations(text))
        return keys

    def collect_bib_keys(self) -> set[str]:
        keys: set[str] = set()
        for rel in ("stage-23/references_verified.bib", "deliverables/references.bib"):
            text = self.read_text(rel)
            if text:
                keys.update(extract_bib_keys(text))
        return keys

    def read_json(self, rel: str, *, required: bool) -> dict[str, Any] | None:
        path = self.run_dir / rel
        if not path.exists():
            if required:
                self.error("missing_artifact", f"Required artifact is missing: {rel}.", rel)
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.error("invalid_json", f"Could not read JSON artifact {rel}: {exc}.", rel)
            return None
        if not isinstance(data, dict):
            self.error("invalid_json_shape", f"JSON artifact must be an object: {rel}.", rel)
            return None
        data["__release_check_path__"] = rel
        return data

    def read_text(self, rel: str) -> str:
        path = self.run_dir / rel
        if not path.exists() or not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            self.error("unreadable_artifact", f"Could not read artifact: {rel}.", rel)
            return ""

    def find_metadata(self, candidates: tuple[str, ...]) -> dict[str, Any] | None:
        for rel in candidates:
            data = self.read_json(rel, required=False)
            if data is not None:
                return data
        return None

    # ------------------------------------------------------------------
    # Finding helpers
    # ------------------------------------------------------------------

    def error(self, code: str, message: str, path: str = "") -> None:
        self.findings.append(Finding(SEVERITY_ERROR, code, message, path))

    def warning(self, code: str, message: str, path: str = "") -> None:
        self.findings.append(Finding(SEVERITY_WARNING, code, message, path))

    def info(self, code: str, message: str, path: str = "") -> None:
        self.findings.append(Finding(SEVERITY_INFO, code, message, path))

    def exit_code(self) -> int:
        errors = [f for f in self.findings if f.severity == SEVERITY_ERROR]
        if not errors:
            return EXIT_PASS
        degraded_signal_codes = {
            "degradation_signal",
            "stale_degradation_signal",
            "degraded_summary",
        }
        degraded_compatible_codes = degraded_signal_codes | {
            "quality_below_threshold",
        }
        if (
            any(f.code in degraded_signal_codes for f in errors)
            and all(f.code in degraded_compatible_codes for f in errors)
        ):
            return EXIT_DEGRADED
        return EXIT_FAIL

    def report(self) -> dict[str, Any]:
        exit_code = self.exit_code()
        return {
            "run_dir": str(self.run_dir),
            "status": "pass" if exit_code == EXIT_PASS else ("degraded" if exit_code == EXIT_DEGRADED else "fail"),
            "exit_code": exit_code,
            "findings": [
                {
                    "severity": item.severity,
                    "code": item.code,
                    "message": item.message,
                    **({"path": item.path} if item.path else {}),
                }
                for item in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# Digest helpers — MUST stay in sync with researchclaw/pipeline/release_artifacts.py
# (duplicated here so the checker stays runnable without the package installed;
# it prefers the package implementation when importable).
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised implicitly
    from researchclaw.pipeline.release_artifacts import (  # type: ignore
        claims_digest as claims_digest_of,
        paper_sha256 as paper_hash_of,
        extract_numbers as numbers_from_text,
        numbers_match as numbers_close,
        numbers_in_artifact as _numbers_in_artifact_pkg,
    )

    def numbers_in_artifact(path: Path) -> list:
        return _numbers_in_artifact_pkg(path)

except Exception:  # noqa: BLE001

    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    _NUMBER_RE = re.compile(
        r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
        r"|(?<![\w.])[-+]?\d+\.\d+(?:[eE][-+]?\d+)?"
        r"|(?<![\w.])[-+]?\d+(?:[eE][-+]?\d+)?"
    )

    def numbers_from_text(text: str) -> list:
        out = []
        for tok in _NUMBER_RE.findall(text or ""):
            try:
                out.append(float(tok.replace(",", "")))
            except ValueError:
                continue
        return out

    def numbers_close(a: float, b: float, rel_tol: float = 1e-3) -> bool:
        if round(a, 6) == round(b, 6):
            return True
        denom = max(abs(a), abs(b), 1e-12)
        return abs(a - b) / denom <= rel_tol

    def _collect_numeric_leaves(data, depth=0):
        out = []
        if depth > 8 or isinstance(data, bool):
            return out
        if isinstance(data, (int, float)):
            out.append(round(float(data), 4))
        elif isinstance(data, dict):
            for v in data.values():
                out.extend(_collect_numeric_leaves(v, depth + 1))
        elif isinstance(data, list):
            for v in data[:200]:
                out.extend(_collect_numeric_leaves(v, depth + 1))
        return out

    def numbers_in_artifact(path: Path) -> list:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return []
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return numbers_from_text(raw)
        nums = _collect_numeric_leaves(data)
        nums.extend(numbers_from_text(raw))
        return nums

    def paper_hash_of(text: str) -> str:
        return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()

    def claims_digest_of(claims: list) -> str:
        core = []
        for c in sorted(claims, key=lambda c: str(c.get("id", ""))):
            core.append(
                {
                    "id": c.get("id"),
                    "text": _normalize_text(str(c.get("text", ""))),
                    "type": c.get("type"),
                    "status": c.get("status"),
                    "evidence": [
                        {"path": e.get("path"), "sha256": e.get("sha256")}
                        for e in (c.get("evidence") or [])
                        if isinstance(e, dict)
                    ],
                }
            )
        return hashlib.sha256(
            json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()


def normalize_for_substr(text: str) -> str:
    """Whitespace- and case-insensitive normalization for substring checks,
    so a support_excerpt can be matched against context/claim text regardless
    of cosmetic differences — but NOT regardless of content."""
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def is_allowed_claim_evidence_path(rel: str, claim_type: str) -> bool:
    """Return whether a claim evidence pointer names a release-authorized artifact.

    Existence plus sha256 is not enough: otherwise a forged file under stage-24
    could become "evidence" merely by matching its own digest.
    """
    rel = str(rel or "")
    parts = rel.split("/")
    if parts and parts[0].startswith("stage-10"):
        return False
    if claim_type == "citation":
        return rel == _CITATION_EVIDENCE_PATH
    if rel == _CITATION_EVIDENCE_PATH:
        return False
    if rel in {"experiment_summary_best.json", "attempts/attempt_log.jsonl"}:
        return True
    if len(parts) == 2 and _is_stage_dir(parts[0], 14) and parts[1] == "experiment_summary.json":
        return True
    if len(parts) == 2 and _is_stage_dir(parts[0], 13) and parts[1] == "refinement_log.json":
        return True
    return (
        len(parts) == 3
        and _is_stage_dir(parts[0], 12)
        and parts[1] == "runs"
        and parts[2].endswith(".json")
    )


def _is_stage_dir(value: str, stage: int) -> bool:
    return bool(re.fullmatch(rf"stage-{stage}(?:_v\d+)?", value))


def find_experiment_contract(run_dir: Path) -> Path | None:
    direct = run_dir / "stage-09" / "experiment_contract.yaml"
    if direct.is_file():
        return direct
    candidates = [
        path
        for path in run_dir.glob("stage-09_v*/experiment_contract.yaml")
        if path.is_file()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: _stage09_version(p.parent.name), reverse=True)[0]


def _stage09_version(name: str) -> int:
    if "_v" not in name:
        return 0
    try:
        return int(name.rsplit("_v", 1)[1])
    except ValueError:
        return -1


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def sha256_of_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def extract_markdown_citations(text: str) -> set[str]:
    stripped = strip_markdown_non_prose(text)
    keys: set[str] = set()
    key_re = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*\d{4}[A-Za-z0-9_-]*$")
    for bracket in re.findall(r"\[([^\[\]]{4,300})\]", stripped):
        parts = [p.strip() for p in re.split(r"[,;]", bracket)]
        if parts and all(key_re.fullmatch(part) for part in parts if part):
            keys.update(part for part in parts if part)
    keys.update(extract_latex_citations(stripped))
    return keys


def extract_latex_citations(text: str) -> set[str]:
    keys: set[str] = set()
    cite_re = re.compile(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]+)\}")
    for body in cite_re.findall(text):
        for key in body.split(","):
            cleaned = key.strip()
            if cleaned:
                keys.add(cleaned)
    return keys


def extract_bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@\w+\{([^,\s]+),", text))


def strip_markdown_non_prose(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.DOTALL)
    text = re.sub(r"\\\[.*?\\\]", "", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\\)\$.*?(?<!\\)\$", "", text)
    return text


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def float_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def nested_dict(data: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = data.get(key)
    return value if isinstance(value, dict) else None


def first_present(data: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return default


def lower_str(value: Any) -> str:
    return str(value).strip().lower() if value is not None else ""


def metadata_path(data: dict[str, Any]) -> str:
    value = data.get("__release_check_path__")
    return str(value) if value else ""


def format_text_report(report: dict[str, Any]) -> str:
    lines = [
        f"release_check: {report['status']} (exit {report['exit_code']})",
        f"run_dir: {report['run_dir']}",
    ]
    findings = report.get("findings", [])
    if not findings:
        lines.append("findings: none")
        return "\n".join(lines)
    lines.append("findings:")
    for item in findings:
        path = f" [{item['path']}]" if item.get("path") else ""
        lines.append(f"- {item['severity']} {item['code']}{path}: {item['message']}")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a ResearchClaw run directory for release readiness.")
    parser.add_argument("run_dir", type=Path, help="Path to a completed ResearchClaw run directory.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON report.")
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=5.0,
        help="Minimum acceptable stage-20 score_1_to_10 for release checks.",
    )
    parser.add_argument(
        "--allow-suspicious-citations",
        action="store_true",
        help="Do not fail solely on summary.suspicious > 0. Hallucinated citations still fail.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    checker = ReleaseChecker(
        args.run_dir.resolve(),
        quality_threshold=args.quality_threshold,
        allow_suspicious=args.allow_suspicious_citations,
    )
    exit_code = checker.run()
    report = checker.report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text_report(report))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
