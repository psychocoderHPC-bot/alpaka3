#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Machine-checkable acceptance gate for the alpaka3 GEMM project.

Usage:
    python3 validation/check_acceptance.py \
        --requirements validation/gemm-requirements.yaml \
        --run-root /external/evidence/<revision> \
        --head-sha <40-character source SHA>

The checker reads a requirements YAML plus an external evidence tree containing
``acceptance.json`` and any number of ``samples.json`` files. It exits nonzero
for every condition the execution guide makes blocking:

* a required requirement has no acceptance entry (missing entry);
* acceptance, sample, or entry source SHA differs from the head SHA;
* a substitution is declared without approval or is not allowed;
* a required review finding is unresolved, or the review verdict is negative;
* a declared numeric threshold is missing or fails;
* a skip is not explicitly approved;
* a required target/mode pair has neither a passing sample nor an approved skip.

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
        {"target": "cpu-host", "mode": "fp32", "status": "ok", "case_id": "square-256"}
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import sys
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


class Findings:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def add(self, category: str, message: str) -> None:
        self.items.append({"category": category, "message": message})

    def __bool__(self) -> bool:
        return bool(self.items)

    def extend(self, other: "Findings") -> None:
        self.items.extend(other.items)


def normalize_sha(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def sha_matches(candidate: Any, head: str) -> bool:
    return bool(head) and normalize_sha(candidate) == head


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def requirement_list(document: Any) -> list[dict[str, Any]]:
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
    if requirement.get("required") is False:
        return False
    return requirement.get("status") not in {"not_required", "informational"}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def approved_skip(entry: dict[str, Any], target: str, mode: str | None) -> bool:
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
    return str(record.get("status", "")).lower()


def matches_requirement(record: dict[str, Any], requirement_id: str) -> bool:
    for key in ("requirement_id", "_file_requirement_id"):
        value = record.get(key)
        if value is not None and value != requirement_id:
            return False
    return True


def check_coverage(
    requirement: dict[str, Any],
    entry: dict[str, Any],
    samples: list[dict[str, Any]],
    findings: Findings,
) -> None:
    requirement_id = str(requirement.get("id", "<unnamed>"))
    targets = [str(item) for item in as_list(requirement.get("required_targets"))]
    modes = [str(item) for item in as_list(requirement.get("modes"))]
    case_ids = {str(item) for item in as_list(requirement.get("case_ids"))}

    if not targets:
        findings.add(
            "empty-required-targets",
            f"{requirement_id}: required_targets is empty; an empty set is not accepted",
        )
        return
    if not modes:
        modes = [None]  # type: ignore[list-item]

    for target in targets:
        for mode in modes:
            passing = []
            skipped = []
            for record in samples:
                if not matches_requirement(record, requirement_id):
                    continue
                if str(record.get("target", "")) != target:
                    continue
                if mode is not None and str(record.get("mode", "")) != mode:
                    continue
                if case_ids and str(record.get("case_id", "")) not in case_ids:
                    continue
                status = sample_status(record)
                if status in PASSING_STATUSES:
                    passing.append(record)
                elif status in SKIPPED_STATUSES:
                    skipped.append(record)
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
        expected = threshold.get("value")
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--head-sha", default=None)
    parser.add_argument("--acceptance", type=Path, default=None, help="override acceptance.json path")
    parser.add_argument("--json-report", type=Path, default=None)
    parser.add_argument("--allow-run-root-inside-repo", action="store_true")
    args = parser.parse_args(argv)

    findings = Findings()

    run_root = args.run_root.resolve()
    if not run_root.is_dir():
        print(f"check_acceptance: run root does not exist: {run_root}", file=sys.stderr)
        return 2
    if not args.allow_run_root_inside_repo and (
        run_root == CHECKOUT_ROOT or CHECKOUT_ROOT in run_root.parents
    ):
        print(
            f"check_acceptance: refusing to read evidence from inside the checkout ({CHECKOUT_ROOT}); "
            "pass --allow-run-root-inside-repo only for local debugging",
            file=sys.stderr,
        )
        return 2

    if not args.requirements.is_file():
        print(f"check_acceptance: requirements file does not exist: {args.requirements}", file=sys.stderr)
        return 2

    document = load_yaml(args.requirements)
    requirements = requirement_list(document)
    head_sha = normalize_sha(args.head_sha or (document.get("head_sha") if isinstance(document, dict) else None))
    if not head_sha:
        findings.add("missing-head-sha", "no head SHA given on the command line or in the requirements YAML")
    if head_sha and len(head_sha) < 7:
        findings.add("invalid-head-sha", f"head SHA {head_sha!r} is too short")

    default_targets = as_list(document.get("required_targets")) if isinstance(document, dict) else []
    default_substitutions = as_list(document.get("allowed_substitutions")) if isinstance(document, dict) else []
    for requirement in requirements:
        if "required_targets" not in requirement and default_targets:
            requirement["required_targets"] = default_targets
        if "allowed_substitutions" not in requirement and default_substitutions:
            requirement["allowed_substitutions"] = default_substitutions

    acceptance_path = args.acceptance or (run_root / "acceptance.json")
    acceptance: dict[str, Any] = {}
    if not acceptance_path.is_file():
        findings.add("missing-acceptance", f"acceptance file does not exist: {acceptance_path}")
    else:
        try:
            loaded = json.loads(acceptance_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                acceptance = loaded
            else:
                findings.add("invalid-acceptance", f"{acceptance_path}: top level is not an object")
        except (OSError, json.JSONDecodeError) as error:
            findings.add("invalid-acceptance", f"{acceptance_path}: {error}")

    if acceptance and head_sha and not sha_matches(acceptance.get("source_sha"), head_sha):
        findings.add(
            "stale-acceptance-sha",
            f"acceptance source_sha {acceptance.get('source_sha')!r} does not match head {head_sha}",
        )

    entries_raw = acceptance.get("entries", {})
    entries: dict[str, dict[str, Any]] = {}
    if isinstance(entries_raw, list):
        for entry in entries_raw:
            if isinstance(entry, dict) and "id" in entry:
                entries[str(entry["id"])] = entry
    elif isinstance(entries_raw, dict):
        entries = {str(key): value for key, value in entries_raw.items() if isinstance(value, dict)}

    global_substitutions: list[Any] = []
    if acceptance:
        global_substitutions = as_list(acceptance.get("substitutions"))
    for requirement in requirements:
        requirement_id = str(requirement.get("id", ""))
        if not requirement_id:
            findings.add("invalid-requirement", "requirement without an id")
            continue
        if not is_required(requirement):
            continue

        entry = entries.get(requirement_id)
        if entry is None:
            findings.add("missing-entry", f"{requirement_id}: no acceptance entry")
            continue

        if head_sha and entry.get("source_sha") is not None and not sha_matches(entry.get("source_sha"), head_sha):
            findings.add(
                "stale-entry-sha",
                f"{requirement_id}: entry source_sha {entry.get('source_sha')!r} does not match head {head_sha}",
            )

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

        check_substitutions(requirement, entry, global_substitutions, findings)
        check_review(requirement, entry, findings)
        check_thresholds(requirement, entry, findings)

        samples, sample_findings = collect_samples(run_root, entry)
        findings.extend(sample_findings)
        for record in samples:
            record_sha = record.get("source_sha")
            if record_sha is None:
                findings.add(
                    "missing-sample-sha",
                    f"{requirement_id}: sample in {record.get('_file')} has no source SHA",
                )
            elif head_sha and not sha_matches(record_sha, head_sha):
                findings.add(
                    "stale-sample-sha",
                    f"{requirement_id}: sample in {record.get('_file')} has stale SHA {record_sha!r}",
                )
        check_coverage(requirement, entry, samples, findings)

    report = {
        "schema_version": 1,
        "head_sha": head_sha,
        "run_root": str(run_root),
        "requirements": len(requirements),
        "passed": not findings,
        "findings": findings.items,
    }
    if args.json_report is not None:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if findings:
        print(f"acceptance FAILED with {len(findings.items)} finding(s):", file=sys.stderr)
        for finding in findings.items:
            print(f"  [{finding['category']}] {finding['message']}", file=sys.stderr)
        return 1
    print(f"acceptance PASSED for head {head_sha} ({len(requirements)} requirement(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
