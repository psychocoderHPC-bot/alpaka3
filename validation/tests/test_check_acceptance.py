# SPDX-License-Identifier: MPL-2.0
"""Fixtures for validation/check_acceptance.py.

Each test writes requirements YAML plus a synthetic external evidence tree to a
temporary directory. No fixture reads evidence from the checkout.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import yaml

VALIDATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATION_DIR))

check_acceptance = importlib.import_module("check_acceptance")

HEAD = "0123456789abcdef0123456789abcdef01234567"
STALE = "fedcba9876543210fedcba9876543210fedcba98"


def base_requirements(**overrides):
    """Return a minimal requirements document for one requirement."""
    requirement = {
        "id": "G1-boundaries",
        "required": True,
        "required_targets": ["cpu-host"],
        "modes": ["fp32"],
        "allowed_substitutions": [],
        "thresholds": [{"metric": "gflops_ratio", "op": ">=", "value": 0.90}],
        "review": {"reviewer": "reviewer", "verdict": "pass", "findings": []},
    }
    requirement.update(overrides)
    return {
        "schema_version": 1,
        "head_sha": HEAD,
        "requirements": [requirement],
    }


def base_acceptance(**entry_overrides):
    """Return a minimal passing acceptance document."""
    entry = {
        "status": "pass",
        "source_sha": HEAD,
        "metrics": {"gflops_ratio": 0.95},
        "substitutions": [],
        "skips": [],
        "review": {"verdict": "pass", "findings": []},
    }
    entry.update(entry_overrides)
    return {
        "schema_version": 1,
        "source_sha": HEAD,
        "entries": {"G1-boundaries": entry},
    }


def base_samples(**record_overrides):
    """Return a minimal passing samples document."""
    record = {
        "target": "cpu-host",
        "mode": "fp32",
        "status": "ok",
        "case_id": "square-256",
        "gflops": 100.0,
    }
    record.update(record_overrides)
    return {
        "schema_version": "1.0.0",
        "kind": "alpaka3-gemm-samples",
        "revision": HEAD,
        "target": "cpu-host",
        "records": [record],
    }


def run_checker(tmp_path, requirements, acceptance, samples, head=HEAD):
    """Write fixtures and run the acceptance checker."""
    run_root = tmp_path / "evidence"
    (run_root / "cpu-host" / "fp32").mkdir(parents=True, exist_ok=True)
    (run_root / "acceptance.json").write_text(json.dumps(acceptance), encoding="utf-8")
    if samples is not None:
        (run_root / "cpu-host" / "fp32" / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    requirements_path = tmp_path / "requirements.yaml"
    requirements_path.write_text(yaml.safe_dump(requirements), encoding="utf-8")
    return check_acceptance.main(
        [
            "--requirements",
            str(requirements_path),
            "--run-root",
            str(run_root),
            "--head-sha",
            head,
        ]
    )


def categories(capsys):
    """Return the finding categories printed to stderr."""
    captured = capsys.readouterr()
    text = captured.err
    found = set()
    for category in (
        "missing-entry",
        "stale-acceptance-sha",
        "stale-sample-sha",
        "unsupported-substitution",
        "unresolved-review-finding",
        "failing-threshold",
        "unapproved-skip",
        "missing-sample",
        "empty-required-targets",
    ):
        if f"[{category}]" in text:
            found.add(category)
    return found


def test_passing_evidence_succeeds(tmp_path, capsys):
    """Passing evidence must be accepted."""
    assert run_checker(tmp_path, base_requirements(), base_acceptance(), base_samples()) == 0
    assert "acceptance PASSED" in capsys.readouterr().out


def test_missing_required_entry_fails(tmp_path, capsys):
    """A missing required entry must fail."""
    acceptance = base_acceptance()
    acceptance["entries"] = {}
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == 1
    assert "missing-entry" in categories(capsys)


def test_stale_acceptance_sha_fails(tmp_path, capsys):
    """A stale acceptance SHA must fail."""
    acceptance = base_acceptance()
    acceptance["source_sha"] = STALE
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == 1
    assert "stale-acceptance-sha" in categories(capsys)


def test_stale_sample_sha_fails(tmp_path, capsys):
    """A stale sample SHA must fail."""
    samples = base_samples()
    samples["revision"] = STALE
    assert run_checker(tmp_path, base_requirements(), base_acceptance(), samples) == 1
    assert "stale-sample-sha" in categories(capsys)


def test_unsupported_substitution_fails(tmp_path, capsys):
    """An unlisted substitution must fail."""
    acceptance = base_acceptance(substitutions=[{"from": "fp64", "to": "cpu-fallback", "approved": True}])
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == 1
    assert "unsupported-substitution" in categories(capsys)


def test_unresolved_required_review_finding_fails(tmp_path, capsys):
    """An unresolved required finding must fail."""
    acceptance = base_acceptance(
        review={
            "verdict": "pass",
            "findings": [{"id": "F1", "severity": "major", "resolved": False}],
        }
    )
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == 1
    assert "unresolved-review-finding" in categories(capsys)


def test_failing_threshold_fails(tmp_path, capsys):
    """A failing numeric threshold must fail."""
    acceptance = base_acceptance(metrics={"gflops_ratio": 0.50})
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == 1
    assert "failing-threshold" in categories(capsys)


def test_unapproved_skip_fails(tmp_path, capsys):
    """An unapproved skip must fail."""
    acceptance = base_acceptance(skips=[{"target": "cpu-host", "mode": "fp32", "reason": "slow", "approved": False}])
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == 1
    assert "unapproved-skip" in categories(capsys)


def test_missing_passing_sample_fails(tmp_path, capsys):
    """A missing passing sample must fail."""
    samples = base_samples(status="skipped")
    assert run_checker(tmp_path, base_requirements(), base_acceptance(), samples) == 1
    found = categories(capsys)
    assert "missing-sample" in found or "unapproved-skip" in found


def test_empty_required_targets_fails(tmp_path, capsys):
    """An empty required target set must fail."""
    requirements = base_requirements(required_targets=[])
    assert run_checker(tmp_path, requirements, base_acceptance(), base_samples()) == 1
    assert "empty-required-targets" in categories(capsys)
