#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
# pylint: disable=too-many-lines,too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-branches
"""Machine-checkable acceptance gate for the alpaka3 GEMM project.

Usage:
    python3 validation/check_acceptance.py \
        --requirements validation/gemm-requirements.yaml \
        --run-root /external/evidence/<revision> \
        --head-sha <40-character source SHA> \
        --repo /path/to/source/checkout \
        --json-report /tmp/acceptance-report.json

The checker reads a requirements YAML plus an external evidence tree containing
``acceptance.json`` and any number of ``samples.json`` files. It exits nonzero
for every condition the execution guide makes blocking:

* a required requirement has no acceptance entry (missing entry);
* a required target has no per-executor matrix record;
* a target is marked ``passed`` without evidence, or is marked
  ``unsupported``/``blocked`` without a reason;
* a target is ``not_run`` (an explicit gap, never a pass);
* evidence carries a source SHA that is not an ancestor-or-equal of the tested
  head, predates the last change under ``include/alpaka/**``, or mixes SHAs
  inside one requirement without ``allow_mixed_shas: true``;
* a substitution is declared without approval or is not allowed;
* a required review finding is unresolved, or the review verdict is negative;
* a declared numeric threshold is missing or fails;
* a skip is not explicitly approved;
* a required target/mode pair has neither a passing sample nor an approved skip;
* a performance record lacks samples/rounds/median/host/device/tools/load
  fields, reports a gflops value inconsistent with ``2*M*N*K/median_ns`` by more
  than 1 percent, or was produced inside the terok container;
* a vendor gate (G5/G6) has no matched vendor baseline for a candidate case.

Exit codes let CI classify the failure:

* 2: schema or evidence-integrity error;
* 3: missing evidence;
* 4: stale or unattributed SHA;
* 5: threshold failure;
* 6: unapproved skip.

Evidence is deliberately external. The run root must not live inside this
checkout unless ``--allow-run-root-inside-repo`` is passed.

Evidence schemas (both versioned, tolerant to extra keys):

requirements YAML
    schema_version: 1
    head_sha: "<sha>"                  # optional if --head-sha is given
    required_targets: [cpu-host]       # optional default for each requirement
    allowed_substitutions: []          # optional default
    requirements:
      - id: G1-boundaries
        required: true                 # default true
        required_targets: [cpu-host]   # empty is rejected for a required entry
        modes: [fp32, fp64]
        allowed_substitutions: [cpu-fallback]
        case_ids: [square-256]         # optional sample filter
        allow_mixed_shas: false        # optional, default false
        vendor_baseline: true          # optional; implied for G5/G6 ids
        thresholds:
          - metric: gflops_ratio       # key inside acceptance entry metrics
            op: ">="                   # >=, >, <=, <, ==
            value: 0.90
        review:
          reviewer: someone
          verdict: pass                # pass/accepted are fine
          findings:
            - id: F1
              severity: major
              required: true           # default: true for major/critical/high
              resolved: false

run-root/acceptance.json
    {
      "schema_version": 1,
      "source_sha": "<sha>",
      "entries": {
        "G1-boundaries": {
          "status": "pass",
          "source_sha": "<sha>",
          "metrics": {"gflops_ratio": 0.95},
          "targets": {
            "cpu-host": {
              "status": "passed",
              "source_sha": "<sha>",
              "evidence": ["cpu-host/fp32/samples.json"]
            }
          },
          "substitutions": [{"from": "fp64", "to": "cpu-fallback", "approved": true}],
          "skips": [{"target": "gpu", "mode": "fp64", "reason": "...", "approved": true}],
          "review": {"verdict": "pass", "findings": []},
          "sample_files": ["cpu-host/fp32/samples.json"]
        }
      }
    }

run-root/**/samples.json
    {
      "schema_version": "1.0.0",
      "kind": "alpaka3-gemm-samples",
      "revision": "<sha>",             # alias of source_sha
      "target": "cpu-host",
      "records": [
        {"target": "cpu-host", "mode": "fp32", "status": "ok", "case_id": "square-256"},
        {
          "kind": "performance",
          "target": "cpu-host", "mode": "fp32", "status": "ok",
          "m": 1024, "n": 1024, "k": 1024, "median_ns": 2000000.0,
          "gflops": 1073.741824, "samples": 20, "rounds": 5,
          "host": "hal", "device": "EPYC 7452",
          "tools": {"compiler": "gcc 13", "toolkit": "oneAPI 2026.2", "driver": "n/a"},
          "load": "idle", "governor": "performance", "threads": 28,
          "layout": "row_major", "transpose_a": false, "transpose_b": false
        },
        {
          "kind": "vendor_baseline",
          "target": "cpu-host", "mode": "fp32", "status": "ok",
          "m": 1024, "n": 1024, "k": 1024, "median_ns": 2000000.0,
          "gflops": 1073.741824, "samples": 20, "rounds": 5,
          "host": "hal", "device": "EPYC 7452",
          "tools": {"compiler": "gcc 13", "toolkit": "oneAPI 2026.2", "driver": "n/a"},
          "load": "idle", "governor": "performance", "threads": 28,
          "layout": "row_major", "transpose_a": false, "transpose_b": false
        }
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as error:  # pragma: no cover - environment failure
    raise SystemExit("PyYAML is required: python3 -m pip install pyyaml") from error


CHECKOUT_ROOT = Path(__file__).resolve().parents[1]
PASSING_STATUSES = {"ok", "pass", "passed", "accepted"}
SKIPPED_STATUSES = {"skip", "skipped"}
RESOLVED_STATUSES = {"resolved", "closed", "verified", "done"}
REJECTED_VERDICTS = {"fail", "failed", "reject", "rejected"}

TARGET_STATUSES = {"passed", "failed", "unsupported", "blocked", "vendor_unavailable", "not_run"}
REASON_REQUIRED_TARGET_STATUSES = {"unsupported", "blocked", "vendor_unavailable"}
PERFORMANCE_KINDS = {"performance", "benchmark", "vendor_baseline"}
GFLOPS_TOLERANCE = 0.01

EXIT_SCHEMA = 2
EXIT_MISSING = 3
EXIT_STALE = 4
EXIT_THRESHOLD = 5
EXIT_SKIP = 6

SCHEMA_CATEGORIES = {
    "invalid-requirement",
    "invalid-acceptance",
    "invalid-sample-file",
    "invalid-threshold",
    "invalid-metrics",
    "missing-head-sha",
    "invalid-head-sha",
    "invalid-repository",
    "invalid-target-status",
    "missing-target-reason",
    "invalid-target-matrix",
    "performance-samples",
    "performance-rounds",
    "performance-median",
    "performance-identity",
    "performance-tools",
    "performance-gflops",
    "performance-environment",
    "performance-shape",
    "forbidden-evidence-origin",
}
STALE_CATEGORIES = {
    "stale-sample-sha",
    "stale-acceptance-sha",
    "stale-entry-sha",
    "stale-target-sha",
    "stale-invalidate-sha",
    "not-ancestor-sha",
    "mixed-shas",
}
THRESHOLD_CATEGORIES = {"failing-threshold"}
SKIP_CATEGORIES = {"unapproved-skip"}


class Findings:
    """Collect blocking findings from the acceptance checks."""

    def __init__(self) -> None:
        """Create an empty finding list."""
        self.items: list[dict[str, str]] = []

    def add(self, category: str, message: str) -> None:
        """Append one finding with its category and message."""
        self.items.append({"category": category, "message": message})

    def __bool__(self) -> bool:
        """Return whether any finding was collected."""
        return bool(self.items)

    def extend(self, other: "Findings") -> None:
        """Append every finding of another instance."""
        self.items.extend(other.items)

    def categories(self) -> set[str]:
        """Return the set of categories collected so far."""
        return {item["category"] for item in self.items}


def normalize_sha(value: Any) -> str:
    """Normalize an optional SHA value for comparison."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def load_yaml(path: Path) -> Any:
    """Load a YAML document from disk."""
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def requirement_list(document: Any) -> list[dict[str, Any]]:
    """Return the requirement mappings of a YAML document."""
    if isinstance(document, list):
        return [item for item in document if isinstance(item, dict)]
    if not isinstance(document, dict):
        raise SystemExit("requirements YAML must be a mapping or a list")
    if "requirements" in document:
        entries = document["requirements"]
        if not isinstance(entries, list):
            raise SystemExit("requirements must be a list")
        return [item for item in entries if isinstance(item, dict)]
    return [document]


def is_required(requirement: dict[str, Any]) -> bool:
    """Return whether a requirement is blocking."""
    if requirement.get("required") is False:
        return False
    return requirement.get("status") not in {"not_required", "informational"}


def as_list(value: Any) -> list[Any]:
    """Return the value as a list, or an empty list for non-lists."""
    return value if isinstance(value, list) else []


def approved_skip(entry: dict[str, Any], target: str, mode: str | None) -> bool:
    """Return whether an approved skip covers the target and mode."""
    for skip in as_list(entry.get("skips")):
        if not isinstance(skip, dict):
            continue
        if skip.get("target") not in (None, target):
            continue
        if mode is not None and skip.get("mode") not in (None, mode):
            continue
        if skip.get("approved") is True:
            return True
    return False


def collect_samples(run_root: Path, entry: dict[str, Any]) -> tuple[list[dict[str, Any]], Findings]:
    """Load all sample records declared for an acceptance entry."""
    findings = Findings()
    sample_files: list[Path] = []
    declared = as_list(entry.get("sample_files"))
    if declared:
        for item in declared:
            path = (run_root / str(item)).resolve()
            sample_files.append(path)
    else:
        sample_files = sorted(run_root.rglob("samples.json"))

    records: list[dict[str, Any]] = []
    for path in sample_files:
        if not path.is_file():
            findings.add("missing-sample-file", f"declared sample file does not exist: {path}")
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            findings.add("invalid-sample-file", f"{path}: {error}")
            continue
        if not isinstance(document, dict):
            findings.add("invalid-sample-file", f"{path}: top level is not an object")
            continue
        file_sha = document.get("source_sha", document.get("revision"))
        target = str(document.get("target", ""))
        requirement_id = document.get("requirement_id")
        for record in as_list(document.get("records")):
            if not isinstance(record, dict):
                continue
            merged = dict(record)
            merged.setdefault("target", target)
            merged.setdefault("source_sha", file_sha)
            merged["_file"] = str(path)
            merged["_file_requirement_id"] = requirement_id
            records.append(merged)
    return records, findings


def sample_status(record: dict[str, Any]) -> str:
    """Return the lower-case status of a sample record."""
    return str(record.get("status", "")).lower()


def matches_requirement(record: dict[str, Any], requirement_id: str) -> bool:
    """Return whether a sample record belongs to the requirement."""
    for key in ("requirement_id", "_file_requirement_id"):
        value = record.get(key)
        if value is not None and value != requirement_id:
            return False
    return True


def record_selected(
    record: dict[str, Any],
    requirement_id: str,
    target: str,
    mode: str | None,
    case_ids: set[str],
) -> bool:
    """Return whether a sample record matches the requirement and selection."""
    if not matches_requirement(record, requirement_id):
        return False
    if str(record.get("target", "")) != target:
        return False
    if mode is not None and str(record.get("mode", "")) != mode:
        return False
    if case_ids and str(record.get("case_id", "")) not in case_ids:
        return False
    return True


def split_samples(
    samples: list[dict[str, Any]],
    requirement_id: str,
    target: str,
    mode: str | None,
    case_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition selected samples into passing and skipped records."""
    passing: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for record in samples:
        if not record_selected(record, requirement_id, target, mode, case_ids):
            continue
        status = sample_status(record)
        if status in PASSING_STATUSES:
            passing.append(record)
        elif status in SKIPPED_STATUSES:
            skipped.append(record)
    return passing, skipped


def check_coverage(
    requirement: dict[str, Any],
    entry: dict[str, Any],
    samples: list[dict[str, Any]],
    findings: Findings,
) -> None:
    """Check that every required target and mode has a passing sample."""
    requirement_id = str(requirement.get("id", "<unnamed>"))
    targets = [str(item) for item in as_list(requirement.get("required_targets"))]
    modes: list[str | None] = [str(item) for item in as_list(requirement.get("modes"))]
    case_ids = {str(item) for item in as_list(requirement.get("case_ids"))}

    if not targets:
        findings.add(
            "empty-required-targets",
            f"{requirement_id}: required_targets is empty; an empty set is not accepted",
        )
        return
    if not modes:
        modes = [None]

    for target in targets:
        for mode in modes:
            passing, skipped = split_samples(samples, requirement_id, target, mode, case_ids)
            label = f"{requirement_id}: target {target}" + (f", mode {mode}" if mode else "")
            if passing:
                continue
            if skipped and approved_skip(entry, target, mode):
                continue
            if skipped:
                findings.add("unapproved-skip", f"{label} has only skipped samples without approval")
            else:
                findings.add("missing-sample", f"{label} has no passing sample record")


def check_substitutions(
    requirement: dict[str, Any],
    entry: dict[str, Any],
    global_substitutions: list[Any],
    findings: Findings,
) -> None:
    """Check that every declared substitution is approved and allowed."""
    requirement_id = str(requirement.get("id", "<unnamed>"))
    allowed_raw = requirement.get("allowed_substitutions", [])
    allowed = set()
    for item in as_list(allowed_raw):
        if isinstance(item, dict):
            allowed.add(str(item.get("to", "")))
        else:
            allowed.add(str(item))

    substitutions = as_list(entry.get("substitutions"))
    for item in global_substitutions:
        if isinstance(item, dict) and item.get("requirement_id") in (None, requirement_id):
            substitutions.append(item)

    for index, substitution in enumerate(substitutions):
        if not isinstance(substitution, dict):
            findings.add("unsupported-substitution", f"{requirement_id}: malformed substitution #{index}")
            continue
        target = str(substitution.get("to", "<missing>"))
        if substitution.get("approved") is not True:
            findings.add(
                "unsupported-substitution",
                f"{requirement_id}: substitution to {target} is not approved",
            )
        elif target not in allowed and "*" not in allowed:
            findings.add(
                "unsupported-substitution",
                f"{requirement_id}: substitution to {target} is not in allowed_substitutions",
            )


def check_review(
    requirement: dict[str, Any],
    entry: dict[str, Any],
    findings: Findings,
) -> None:
    """Check reviewer verdicts and unresolved required findings."""
    requirement_id = str(requirement.get("id", "<unnamed>"))
    reviews = [requirement.get("review"), entry.get("review")]
    for review in reviews:
        if not isinstance(review, dict):
            continue
        verdict = str(review.get("verdict", "")).lower()
        if verdict in REJECTED_VERDICTS:
            findings.add("review-verdict", f"{requirement_id}: reviewer verdict is {verdict}")
        for finding in as_list(review.get("findings")):
            if not isinstance(finding, dict):
                continue
            severity = str(finding.get("severity", "")).lower()
            required = finding.get("required")
            if required is None:
                required = severity in {"blocker", "critical", "major", "high"}
            if not required:
                continue
            status = str(finding.get("status", "")).lower()
            resolved = finding.get("resolved") is True or status in RESOLVED_STATUSES
            if not resolved:
                finding_id = finding.get("id", "<unnamed>")
                findings.add(
                    "unresolved-review-finding",
                    f"{requirement_id}: required review finding {finding_id} is unresolved",
                )


def compare_threshold(op: str, actual: float, expected: float) -> bool:
    """Evaluate one numeric comparison operator."""
    if op == ">=":
        return actual >= expected
    if op == ">":
        return actual > expected
    if op == "<=":
        return actual <= expected
    if op == "<":
        return actual < expected
    if op == "==":
        return actual == expected
    raise ValueError(f"unsupported comparison operator {op!r}")


def check_thresholds(
    requirement: dict[str, Any],
    entry: dict[str, Any],
    findings: Findings,
) -> None:
    """Check every declared numeric threshold of a requirement."""
    requirement_id = str(requirement.get("id", "<unnamed>"))
    metrics = entry.get("metrics", {})
    if not isinstance(metrics, dict):
        findings.add("invalid-metrics", f"{requirement_id}: metrics must be a mapping")
        return
    for threshold in as_list(requirement.get("thresholds")):
        if not isinstance(threshold, dict):
            findings.add("invalid-threshold", f"{requirement_id}: malformed threshold")
            continue
        if threshold.get("required") is False:
            continue
        metric = str(threshold.get("metric", ""))
        op = str(threshold.get("op", ">="))
        expected: Any = threshold.get("value")
        actual: Any = metrics.get(metric)
        target = threshold.get("target")
        if target is not None and isinstance(actual, dict):
            actual = actual.get(str(target))
        if actual is None:
            findings.add(
                "missing-threshold-metric",
                f"{requirement_id}: metric {metric!r} is missing from acceptance entry",
            )
            continue
        try:
            passed = compare_threshold(op, float(actual), float(expected))
        except (TypeError, ValueError) as error:
            findings.add(
                "invalid-threshold",
                f"{requirement_id}: cannot evaluate {metric} {op} {expected}: {error}",
            )
            continue
        if not passed:
            findings.add(
                "failing-threshold",
                f"{requirement_id}: {metric}={actual} fails {op} {expected}",
            )


def run_git(repo: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a read-only git command inside the evidence source repository."""
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


@lru_cache(maxsize=4096)
def _is_ancestor(repo: str, ancestor: str, descendant: str) -> bool:
    """Return whether ancestor is an ancestor-or-equal of descendant."""
    if ancestor == descendant:
        return True
    result = run_git(Path(repo), ["merge-base", "--is-ancestor", ancestor, descendant])
    return result.returncode == 0


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    """Cached ancestry test for normalized commit SHAs."""
    return _is_ancestor(str(repo.resolve()), ancestor, descendant)


def sha_exists(repo: Path, sha: str) -> bool:
    """Return whether the repository contains the commit."""
    if not sha:
        return False
    result = run_git(repo, ["cat-file", "-e", f"{sha}^{{commit}}"])
    return result.returncode == 0


def last_include_change(repo: Path, head_sha: str) -> str:
    """Return the newest commit touching include/alpaka/** up to head."""
    result = run_git(repo, ["rev-list", "-1", head_sha, "--", "include/alpaka"])
    if result.returncode != 0:
        return ""
    return normalize_sha(result.stdout.strip())


def check_evidence_sha(
    requirement_id: str,
    label: str,
    sha: Any,
    head_sha: str,
    repo: Path,
    last_include: str,
    findings: Findings,
    stale_category: str,
) -> str:
    """Validate one evidence SHA and return it normalized."""
    normalized = normalize_sha(sha)
    if not normalized:
        findings.add("missing-sample-sha", f"{requirement_id}: {label} has no source SHA")
        return ""
    if not sha_exists(repo, normalized):
        findings.add(
            "not-ancestor-sha",
            f"{requirement_id}: {label} source SHA {normalized} is unknown to the source repository",
        )
        return normalized
    if head_sha and not is_ancestor(repo, normalized, head_sha):
        findings.add(
            stale_category,
            f"{requirement_id}: {label} source SHA {normalized} is not an ancestor of head {head_sha}",
        )
        return normalized
    if last_include and not is_ancestor(repo, last_include, normalized):
        findings.add(
            "stale-invalidate-sha",
            f"{requirement_id}: {label} source SHA {normalized} predates the last include/alpaka change {last_include}",
        )
    return normalized


def target_matrix(entry: dict[str, Any]) -> dict[str, Any]:
    """Return the per-target matrix records of an acceptance entry."""
    raw = entry.get("targets")
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    matrix: dict[str, Any] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "target" in item:
                matrix[str(item["target"])] = item
    return matrix


def target_evidence_paths(record: dict[str, Any]) -> list[str]:
    """Return the evidence paths declared by a target matrix record."""
    paths: list[str] = []
    for key in ("evidence", "evidence_paths"):
        for item in as_list(record.get(key)):
            if isinstance(item, str) and item.strip():
                paths.append(item.strip())
    evidence_path = record.get("evidence_path")
    if isinstance(evidence_path, str) and evidence_path.strip():
        paths.append(evidence_path.strip())
    return paths


def check_target_matrix(
    requirement: dict[str, Any],
    entry: dict[str, Any],
    run_root: Path,
    head_sha: str,
    repo: Path,
    last_include: str,
    findings: Findings,
) -> list[str]:
    """Check the per-target executor matrix and return its evidence SHAs."""
    requirement_id = str(requirement.get("id", "<unnamed>"))
    required_targets = [str(item) for item in as_list(requirement.get("required_targets"))]
    matrix = target_matrix(entry)
    shas: list[str] = []

    for target in required_targets:
        if target not in matrix:
            findings.add("missing-target", f"{requirement_id}: required target {target} has no matrix record")
            continue
        record = matrix[target]
        if not isinstance(record, dict):
            findings.add("invalid-target-matrix", f"{requirement_id}: target {target} record is not a mapping")
            continue
        status = str(record.get("status", "")).lower()
        if status not in TARGET_STATUSES:
            findings.add(
                "invalid-target-status",
                f"{requirement_id}: target {target} has status {status or '<missing>'!r}; "
                f"expected one of {sorted(TARGET_STATUSES)}",
            )
            continue

        sha = check_evidence_sha(
            requirement_id,
            f"target {target}",
            record.get("source_sha"),
            head_sha,
            repo,
            last_include,
            findings,
            "stale-target-sha",
        )
        if sha:
            shas.append(sha)

        paths = target_evidence_paths(record)
        if not paths and status != "not_run":
            findings.add(
                "missing-target-evidence",
                f"{requirement_id}: target {target} is {status} but carries no evidence path",
            )
        for item in paths:
            path = Path(item)
            if not path.is_absolute():
                path = run_root / path
            if not path.is_file():
                findings.add(
                    "missing-target-evidence",
                    f"{requirement_id}: target {target} evidence does not exist: {path}",
                )

        reason = str(record.get("reason", "")).strip()
        if status in REASON_REQUIRED_TARGET_STATUSES and not reason:
            findings.add(
                "missing-target-reason",
                f"{requirement_id}: target {target} is {status} without a non-empty reason",
            )
        if status == "not_run":
            detail = f" (reason: {reason})" if reason else ""
            findings.add(
                "target-not-run",
                f"{requirement_id}: target {target} is not_run and is an explicit acceptance gap{detail}",
            )
        elif status != "passed":
            findings.add(
                "target-not-passed",
                f"{requirement_id}: target {target} is {status} and does not count as a pass",
            )
    return shas


def is_performance_record(record: dict[str, Any]) -> bool:
    """Return whether a sample record is a performance measurement."""
    kind = str(record.get("kind", record.get("record_type", ""))).lower()
    return kind in PERFORMANCE_KINDS or record.get("performance") is True


def is_vendor_baseline(record: dict[str, Any]) -> bool:
    """Return whether a performance record is a vendor baseline."""
    kind = str(record.get("kind", record.get("record_type", ""))).lower()
    role = str(record.get("role", "")).lower()
    return kind == "vendor_baseline" or role == "vendor_baseline" or record.get("baseline") is True


def positive_number(value: Any) -> float | None:
    """Return a positive finite float, or None."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def check_performance_records(
    requirement: dict[str, Any],
    samples: list[dict[str, Any]],
    findings: Findings,
) -> None:
    """Enforce benchmark-evidence integrity for every performance record."""
    requirement_id = str(requirement.get("id", "<unnamed>"))
    for record in samples:
        if not is_performance_record(record):
            continue
        label = f"{requirement_id}: performance record in {record.get('_file')}"
        if str(record.get("host_origin", "")).lower() == "terok_container":
            findings.add(
                "forbidden-evidence-origin",
                f"{label} was produced inside the terok container; use a computeMCP target",
            )

        samples_count = record.get("samples")
        if isinstance(samples_count, bool) or not isinstance(samples_count, int) or samples_count < 10:
            findings.add("performance-samples", f"{label} needs samples >= 10, got {samples_count!r}")
        rounds = record.get("rounds")
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 3:
            findings.add("performance-rounds", f"{label} needs rounds >= 3, got {rounds!r}")
        median_ns = positive_number(record.get("median_ns"))
        if median_ns is None:
            findings.add("performance-median", f"{label} needs median_ns > 0")

        for key in ("host", "device"):
            if not isinstance(record.get(key), str) or not str(record.get(key)).strip():
                findings.add("performance-identity", f"{label} needs a non-empty {key} identity")
        tools = record.get("tools")
        if not isinstance(tools, dict):
            findings.add("performance-tools", f"{label} needs a tools block")
        else:
            for key in ("compiler", "toolkit", "driver"):
                value = tools.get(key)
                if not isinstance(value, str) or not value.strip():
                    findings.add("performance-tools", f"{label} tools.{key} must be a non-empty string")

        load = record.get("load")
        governor = record.get("governor")
        if not isinstance(load, str) or not load.strip():
            findings.add("performance-environment", f"{label} needs a stated load")
        if not isinstance(governor, str) or not governor.strip():
            findings.add("performance-environment", f"{label} needs a stated governor")
        threads = record.get("threads", record.get("thread_count"))
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            findings.add("performance-environment", f"{label} needs a positive thread count")

        dimensions: dict[str, int] = {}
        for key in ("m", "n", "k"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                findings.add("performance-shape", f"{label} needs positive integer {key}")
            else:
                dimensions[key] = value
        if median_ns is None or len(dimensions) != 3:
            continue
        expected_gflops = 2.0 * dimensions["m"] * dimensions["n"] * dimensions["k"] / median_ns
        measured = positive_number(record.get("gflops"))
        if measured is None:
            findings.add("performance-gflops", f"{label} needs a positive gflops value")
        elif abs(measured - expected_gflops) / expected_gflops > GFLOPS_TOLERANCE:
            findings.add(
                "performance-gflops",
                f"{label} gflops={measured} disagrees with 2*M*N*K/median_ns={expected_gflops:.6f}",
            )


def vendor_gate(requirement: dict[str, Any]) -> bool:
    """Return whether a requirement compares against a vendor baseline."""
    requirement_id = str(requirement.get("id", ""))
    return requirement.get("vendor_baseline") is True or requirement_id.startswith(("G5", "G6"))


def baseline_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Return the shape/layout/mode/resource identity used to pair a baseline."""

    def flag(name: str) -> bool:
        return bool(record.get(name, False))

    return (
        record.get("m"),
        record.get("n"),
        record.get("k"),
        str(record.get("layout", "")),
        flag("transpose_a"),
        flag("transpose_b"),
        str(record.get("mode", "")),
        str(record.get("target", "")),
        str(record.get("device", "")).strip().lower(),
    )


def geometric_mean(values: list[float]) -> float:
    """Return the geometric mean of positive values."""
    return math.exp(sum(math.log(value) for value in values) / len(values))


def check_vendor_baseline(
    requirement: dict[str, Any],
    entry: dict[str, Any],
    samples: list[dict[str, Any]],
    findings: Findings,
) -> dict[str, Any]:
    """Pair candidate records with vendor baselines and compute ratios."""
    requirement_id = str(requirement.get("id", "<unnamed>"))
    metrics = dict(entry.get("metrics", {})) if isinstance(entry.get("metrics"), dict) else {}
    if not vendor_gate(requirement):
        return metrics
    for metric in list(metrics):
        if "vendor_ratio" in str(metric):
            metrics.pop(metric)

    candidates = [
        record
        for record in samples
        if is_performance_record(record)
        and not is_vendor_baseline(record)
        and sample_status(record) in PASSING_STATUSES
    ]
    baselines: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in samples:
        if is_vendor_baseline(record) and sample_status(record) in PASSING_STATUSES:
            baselines.setdefault(baseline_key(record), []).append(record)

    ratios: list[float] = []
    for record in candidates:
        key = baseline_key(record)
        matches = baselines.get(key, [])
        case = record.get("case_id", "<unknown>")
        if not matches:
            findings.add(
                "vendor-unavailable",
                f"{requirement_id}: no matched vendor baseline for case {case} on "
                f"target {record.get('target')} mode {record.get('mode')}",
            )
            continue
        baseline = matches[0]
        candidate_gflops = positive_number(record.get("gflops"))
        baseline_gflops = positive_number(baseline.get("gflops"))
        if candidate_gflops is None or baseline_gflops is None:
            findings.add(
                "vendor-unavailable",
                f"{requirement_id}: cannot compute a vendor ratio for case {case} without positive gflops",
            )
            continue
        ratios.append(candidate_gflops / baseline_gflops)

    if not candidates:
        findings.add(
            "vendor-unavailable",
            f"{requirement_id}: vendor gate has no passing performance records to compare",
        )

    if ratios:
        computed = {"geomean": geometric_mean(ratios), "min": min(ratios)}
        for threshold in as_list(requirement.get("thresholds")):
            if not isinstance(threshold, dict):
                continue
            metric = str(threshold.get("metric", ""))
            if "vendor_ratio" not in metric:
                continue
            if "geomean" in metric:
                metrics[metric] = computed["geomean"]
            elif "min" in metric or "case" in metric:
                metrics[metric] = computed["min"]
    return metrics


def check_sample_shas(
    requirement_id: str,
    samples: list[dict[str, Any]],
    head_sha: str,
    repo: Path,
    last_include: str,
    findings: Findings,
) -> list[str]:
    """Check that every sample carries a valid, fresh source SHA."""
    shas: list[str] = []
    for record in samples:
        sha = check_evidence_sha(
            requirement_id,
            f"sample in {record.get('_file')}",
            record.get("source_sha"),
            head_sha,
            repo,
            last_include,
            findings,
            "stale-sample-sha",
        )
        if sha:
            shas.append(sha)
    return shas


def check_mixed_shas(
    requirement: dict[str, Any],
    shas: list[str],
    findings: Findings,
) -> None:
    """Reject multiple evidence SHAs in one requirement unless allowed."""
    unique = {sha for sha in shas if sha}
    if len(unique) <= 1:
        return
    if requirement.get("allow_mixed_shas") is True:
        return
    requirement_id = str(requirement.get("id", "<unnamed>"))
    findings.add(
        "mixed-shas",
        f"{requirement_id}: evidence mixes source SHAs {sorted(unique)}; "
        "set allow_mixed_shas: true only when explicitly justified",
    )


def check_entry_status(requirement_id: str, entry: dict[str, Any], findings: Findings) -> None:
    """Check the entry status and its skip approvals."""
    status = str(entry.get("status", "")).lower()
    if status in SKIPPED_STATUSES:
        if entry.get("approved") is not True:
            findings.add("unapproved-skip", f"{requirement_id}: requirement entry is skipped without approval")
    elif status not in PASSING_STATUSES:
        findings.add(
            "not-passing",
            f"{requirement_id}: acceptance status is {status or '<missing>'}, expected a passing status",
        )

    for skip in as_list(entry.get("skips")):
        if isinstance(skip, dict) and skip.get("approved") is not True:
            findings.add(
                "unapproved-skip",
                f"{requirement_id}: skip for target {skip.get('target')!r} is not approved",
            )


def check_requirement(
    requirement: dict[str, Any],
    acceptance: dict[str, Any],
    run_root: Path,
    head_sha: str,
    repo: Path,
    last_include: str,
    findings: Findings,
) -> None:
    """Run every acceptance check for one required requirement."""
    requirement_id = str(requirement.get("id", ""))
    if not requirement_id:
        findings.add("invalid-requirement", "requirement without an id")
        return
    if not is_required(requirement):
        return

    entry = collect_entries(acceptance).get(requirement_id)
    if entry is None:
        findings.add("missing-entry", f"{requirement_id}: no acceptance entry")
        return

    shas: list[str] = []
    entry_sha = check_evidence_sha(
        requirement_id,
        "acceptance entry",
        entry.get("source_sha"),
        head_sha,
        repo,
        last_include,
        findings,
        "stale-entry-sha",
    )
    if entry_sha:
        shas.append(entry_sha)

    check_entry_status(requirement_id, entry, findings)
    check_substitutions(requirement, entry, as_list(acceptance.get("substitutions")), findings)
    check_review(requirement, entry, findings)

    samples, sample_findings = collect_samples(run_root, entry)
    findings.extend(sample_findings)
    shas.extend(check_sample_shas(requirement_id, samples, head_sha, repo, last_include, findings))
    shas.extend(check_target_matrix(requirement, entry, run_root, head_sha, repo, last_include, findings))
    check_mixed_shas(requirement, shas, findings)

    check_performance_records(requirement, samples, findings)
    effective_entry = dict(entry)
    effective_entry["metrics"] = check_vendor_baseline(requirement, entry, samples, findings)
    check_thresholds(requirement, effective_entry, findings)
    check_coverage(requirement, entry, samples, findings)


def summarize_requirement(requirement: dict[str, Any], findings: Findings) -> dict[str, Any]:
    """Build the per-requirement human and machine-readable summary."""
    categories = findings.categories()
    if not is_required(requirement):
        status = "not_required"
    elif not findings:
        status = "passed"
    elif "vendor-unavailable" in categories:
        status = "vendor_unavailable"
    elif "target-not-run" in categories:
        status = "not_run"
    elif "target-not-passed" in categories or "target-failed" in categories:
        status = "failed"
    elif categories & STALE_CATEGORIES:
        status = "stale"
    elif categories & THRESHOLD_CATEGORIES:
        status = "failed_threshold"
    elif categories & SKIP_CATEGORIES:
        status = "skipped_unapproved"
    elif categories & SCHEMA_CATEGORIES:
        status = "invalid"
    else:
        status = "missing_evidence"

    missing_evidence = [
        item["message"]
        for item in findings.items
        if item["category"] not in SCHEMA_CATEGORIES | STALE_CATEGORIES | THRESHOLD_CATEGORIES | SKIP_CATEGORIES
    ]
    return {
        "id": str(requirement.get("id", "<unnamed>")),
        "status": status,
        "missing_evidence": missing_evidence,
        "reason": "; ".join(item["message"] for item in findings.items),
        "findings": list(findings.items),
    }


def findings_exit_code(findings: Findings) -> int:
    """Map the collected findings to the CI failure class."""
    if not findings:
        return 0
    categories = findings.categories()
    if categories & SCHEMA_CATEGORIES:
        return EXIT_SCHEMA
    if categories & STALE_CATEGORIES:
        return EXIT_STALE
    if categories & THRESHOLD_CATEGORIES:
        return EXIT_THRESHOLD
    if categories & SKIP_CATEGORIES:
        return EXIT_SKIP
    return EXIT_MISSING


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--head-sha", default=None)
    parser.add_argument("--repo", type=Path, default=None, help="source checkout used for SHA ancestry checks")
    parser.add_argument("--acceptance", type=Path, default=None, help="override acceptance.json path")
    parser.add_argument("--json-report", type=Path, default=None)
    parser.add_argument("--allow-run-root-inside-repo", action="store_true")
    return parser.parse_args(argv)


def resolve_run_root(args: argparse.Namespace) -> Path | None:
    """Validate and resolve the external evidence run root."""
    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        print(f"check_acceptance: run root does not exist: {run_root}", file=sys.stderr)
        return None
    if not args.allow_run_root_inside_repo and (run_root == CHECKOUT_ROOT or CHECKOUT_ROOT in run_root.parents):
        print(
            f"check_acceptance: refusing to read evidence from inside the checkout ({CHECKOUT_ROOT}); "
            "pass --allow-run-root-inside-repo only for local debugging",
            file=sys.stderr,
        )
        return None
    return run_root


def resolve_repo(args: argparse.Namespace, findings: Findings) -> Path | None:
    """Resolve the source repository used for ancestry checks."""
    repo = (args.repo or CHECKOUT_ROOT).resolve()
    if not repo.is_dir():
        findings.add("invalid-repository", f"source repository does not exist: {repo}")
        return None
    result = run_git(repo, ["rev-parse", "--git-dir"])
    if result.returncode != 0:
        findings.add("invalid-repository", f"not a git repository: {repo}")
        return None
    return repo


def resolve_head_sha(args: argparse.Namespace, document: Any, findings: Findings) -> str:
    """Determine the head SHA and record format findings."""
    document_sha = document.get("head_sha") if isinstance(document, dict) else None
    head_sha = normalize_sha(args.head_sha or document_sha)
    if not head_sha:
        findings.add("missing-head-sha", "no head SHA given on the command line or in the requirements YAML")
    if head_sha and len(head_sha) < 7:
        findings.add("invalid-head-sha", f"head SHA {head_sha!r} is too short")
    return head_sha


def apply_defaults(document: Any, requirements: list[dict[str, Any]]) -> None:
    """Fill missing per-requirement defaults from the document."""
    default_targets = as_list(document.get("required_targets")) if isinstance(document, dict) else []
    default_substitutions = as_list(document.get("allowed_substitutions")) if isinstance(document, dict) else []
    for requirement in requirements:
        if "required_targets" not in requirement and default_targets:
            requirement["required_targets"] = default_targets
        if "allowed_substitutions" not in requirement and default_substitutions:
            requirement["allowed_substitutions"] = default_substitutions


def load_acceptance(path: Path, findings: Findings) -> dict[str, Any]:
    """Load acceptance.json, recording a finding on failure."""
    acceptance: dict[str, Any] = {}
    if not path.is_file():
        findings.add("missing-acceptance", f"acceptance file does not exist: {path}")
        return acceptance
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            acceptance = loaded
        else:
            findings.add("invalid-acceptance", f"{path}: top level is not an object")
    except (OSError, json.JSONDecodeError) as error:
        findings.add("invalid-acceptance", f"{path}: {error}")
    return acceptance


def collect_entries(acceptance: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index acceptance entries by requirement id."""
    entries_raw = acceptance.get("entries", {})
    entries: dict[str, dict[str, Any]] = {}
    if isinstance(entries_raw, list):
        for entry in entries_raw:
            if isinstance(entry, dict) and "id" in entry:
                entries[str(entry["id"])] = entry
    elif isinstance(entries_raw, dict):
        entries = {str(key): value for key, value in entries_raw.items() if isinstance(value, dict)}
    return entries


def build_report(
    head_sha: str,
    run_root: Path,
    requirements: list[dict[str, Any]],
    findings: Findings,
    requirement_reports: list[dict[str, Any]],
    exit_code: int,
) -> dict[str, Any]:
    """Assemble the JSON-serializable acceptance report."""
    return {
        "schema_version": 1,
        "head_sha": head_sha,
        "run_root": str(run_root),
        "requirements": len(requirements),
        "passed": not findings,
        "exit_code": exit_code,
        "requirement_reports": requirement_reports,
        "findings": findings.items,
    }


def emit_report(report: dict[str, Any], args: argparse.Namespace, findings: Findings) -> int:
    """Write the optional JSON report and print the final verdict."""
    exit_code = int(report["exit_code"])
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if findings:
        print(f"acceptance FAILED with {len(findings.items)} finding(s), exit {exit_code}:", file=sys.stderr)
        for summary in report["requirement_reports"]:
            print(f"  {summary['id']}: {summary['status']}", file=sys.stderr)
            if summary["missing_evidence"]:
                print(f"    missing evidence: {'; '.join(summary['missing_evidence'])}", file=sys.stderr)
            if summary["reason"]:
                print(f"    reason: {summary['reason']}", file=sys.stderr)
        for finding in findings.items:
            print(f"  [{finding['category']}] {finding['message']}", file=sys.stderr)
        return exit_code
    print(f"acceptance PASSED for head {report['head_sha']} ({report['requirements']} requirement(s))")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the acceptance gate and return its exit status."""
    args = parse_arguments(argv)
    findings = Findings()

    run_root = resolve_run_root(args)
    if run_root is None:
        return EXIT_SCHEMA

    if not args.requirements.is_file():
        print(f"check_acceptance: requirements file does not exist: {args.requirements}", file=sys.stderr)
        return EXIT_SCHEMA

    document = load_yaml(args.requirements)
    requirements = requirement_list(document)
    head_sha = resolve_head_sha(args, document, findings)
    apply_defaults(document, requirements)

    repo = resolve_repo(args, findings)
    last_include = ""
    if repo is not None and head_sha:
        if not sha_exists(repo, head_sha):
            findings.add("invalid-head-sha", f"head SHA {head_sha!r} is unknown to {repo}")
        else:
            last_include = last_include_change(repo, head_sha)

    acceptance_path = args.acceptance or (run_root / "acceptance.json")
    acceptance = load_acceptance(acceptance_path, findings)
    if acceptance and head_sha and repo is not None:
        check_evidence_sha(
            "acceptance",
            "acceptance document",
            acceptance.get("source_sha"),
            head_sha,
            repo,
            last_include,
            findings,
            "stale-acceptance-sha",
        )

    requirement_reports: list[dict[str, Any]] = []
    for requirement in requirements:
        requirement_findings = Findings()
        check_requirement(
            requirement,
            acceptance,
            run_root,
            head_sha,
            repo if repo is not None else CHECKOUT_ROOT,
            last_include,
            requirement_findings,
        )
        findings.extend(requirement_findings)
        requirement_reports.append(summarize_requirement(requirement, requirement_findings))

    exit_code = findings_exit_code(findings)
    report = build_report(head_sha, run_root, requirements, findings, requirement_reports, exit_code)
    return emit_report(report, args, findings)


if __name__ == "__main__":
    raise SystemExit(main())
