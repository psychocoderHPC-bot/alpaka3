# SPDX-License-Identifier: MPL-2.0
# pylint: disable=too-many-arguments,too-many-positional-arguments,duplicate-code
"""Fixtures for validation/check_acceptance.py.

Each test writes requirements YAML plus a synthetic external evidence tree to a
temporary directory. No fixture reads evidence from the checkout. A module-level
throw-away git repository supplies real ancestry and include-freshness data.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

VALIDATION_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VALIDATION_DIR))

check_acceptance = importlib.import_module("check_acceptance")


def _git(repo: Path, *arguments: str) -> str:
    """Run git in the fixture repository and return stdout."""
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(repo: Path, message: str) -> str:
    """Commit the current index and return the commit SHA."""
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _make_repository() -> tuple[Path, str, str, str, str]:
    """Create a repository with a pre-include, an include, and a head commit."""
    repo = Path(tempfile.mkdtemp(prefix="t24-acceptance-repo-"))
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Fixture")
    _git(repo, "config", "user.email", "fixture@example.invalid")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")
    old = _commit(repo, "pre-include commit")
    (repo / "include" / "alpaka").mkdir(parents=True)
    (repo / "include" / "alpaka" / "gemm.hpp").write_text("// gemm\n", encoding="utf-8")
    include = _commit(repo, "include change")
    (repo / "docs").mkdir()
    (repo / "docs" / "notes.md").write_text("notes\n", encoding="utf-8")
    head = _commit(repo, "head commit")
    _git(repo, "checkout", "-q", "-b", "sibling", old)
    (repo / "sibling.txt").write_text("sibling\n", encoding="utf-8")
    sibling = _commit(repo, "sibling commit")
    _git(repo, "checkout", "-q", "main")
    return repo, head, include, old, sibling


REPO, HEAD, INCLUDE_SHA, OLD_SHA, STALE = _make_repository()


def target_record(status="passed", evidence=None, reason=None, source_sha=HEAD):
    """Return one per-target executor matrix record."""
    if evidence is None:
        evidence = ["cpu-host/fp32/samples.json"]
    record = {"status": status, "source_sha": source_sha, "evidence": evidence}
    if reason is not None:
        record["reason"] = reason
    return record


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
        "targets": {"cpu-host": target_record()},
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


def perf_record(median_ns=2_000_000.0, gflops=None, **overrides):
    """Return a complete performance record for a 1024^3 cube."""
    m = n = k = 1024
    if gflops is None:
        gflops = 2.0 * m * n * k / median_ns if median_ns > 0 else 1.0
    record = {
        "kind": "performance",
        "target": "cpu-host",
        "mode": "fp32",
        "status": "ok",
        "case_id": "square-1024",
        "m": m,
        "n": n,
        "k": k,
        "median_ns": median_ns,
        "gflops": gflops,
        "samples": 20,
        "rounds": 5,
        "host": "hal",
        "device": "EPYC 7452",
        "tools": {"compiler": "gcc 13", "toolkit": "oneAPI 2026.2", "driver": "n/a"},
        "load": "idle",
        "governor": "performance",
        "threads": 28,
        "layout": "row_major",
        "transpose_a": False,
        "transpose_b": False,
    }
    record.update(overrides)
    return record


def baseline_record(median_ns=2_000_000.0, **overrides):
    """Return a vendor baseline record matching perf_record by default."""
    gflops = overrides.pop("gflops", None)
    record = perf_record(median_ns=median_ns, kind="vendor_baseline")
    record.update(overrides)
    if gflops is not None:
        record["gflops"] = gflops
    elif {"m", "n", "k", "median_ns"} & set(overrides):
        m, n, k = record["m"], record["n"], record["k"]
        record["gflops"] = 2.0 * m * n * k / record["median_ns"]
    return record


def samples_with(records, revision=HEAD, **overrides):
    """Return a samples document carrying the supplied records."""
    document = base_samples()
    document["revision"] = revision
    document["records"] = records
    document.update(overrides)
    return document


def run_checker(
    tmp_path,
    requirements,
    acceptance,
    samples,
    head=HEAD,
    json_report=None,
    extra_files=None,
):
    """Write fixtures and run the acceptance checker."""
    run_root = tmp_path / "evidence"
    (run_root / "cpu-host" / "fp32").mkdir(parents=True, exist_ok=True)
    (run_root / "acceptance.json").write_text(json.dumps(acceptance), encoding="utf-8")
    if samples is not None:
        (run_root / "cpu-host" / "fp32" / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
    for relative, content in (extra_files or {}).items():
        path = run_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")
    requirements_path = tmp_path / "requirements.yaml"
    requirements_path.write_text(yaml.safe_dump(requirements), encoding="utf-8")
    arguments = [
        "--requirements",
        str(requirements_path),
        "--run-root",
        str(run_root),
        "--head-sha",
        head,
        "--repo",
        str(REPO),
    ]
    if json_report is not None:
        arguments.extend(["--json-report", str(json_report)])
    return check_acceptance.main(arguments)


def categories(capsys):
    """Return every finding category printed to stderr."""
    text = capsys.readouterr().err
    found = set()
    for category in (
        "missing-entry",
        "stale-acceptance-sha",
        "stale-sample-sha",
        "stale-target-sha",
        "stale-invalidate-sha",
        "not-ancestor-sha",
        "mixed-shas",
        "unsupported-substitution",
        "unresolved-review-finding",
        "failing-threshold",
        "unapproved-skip",
        "missing-sample",
        "empty-required-targets",
        "missing-target",
        "invalid-target-status",
        "missing-target-evidence",
        "missing-target-reason",
        "target-not-run",
        "target-not-passed",
        "performance-samples",
        "performance-rounds",
        "performance-median",
        "performance-identity",
        "performance-tools",
        "performance-gflops",
        "performance-environment",
        "forbidden-evidence-origin",
        "vendor-unavailable",
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
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == check_acceptance.EXIT_MISSING
    assert "missing-entry" in categories(capsys)


def test_stale_acceptance_sha_fails(tmp_path, capsys):
    """A non-ancestor acceptance SHA must fail."""
    acceptance = base_acceptance(source_sha=STALE)
    acceptance["source_sha"] = STALE
    acceptance["entries"]["G1-boundaries"]["targets"] = {"cpu-host": target_record(source_sha=STALE)}
    samples = samples_with([base_samples()["records"][0]], revision=STALE)
    assert run_checker(tmp_path, base_requirements(), acceptance, samples) == check_acceptance.EXIT_STALE
    assert "stale-acceptance-sha" in categories(capsys)


def test_stale_sample_sha_fails(tmp_path, capsys):
    """A sample from a non-ancestor SHA must fail."""
    samples = samples_with([base_samples()["records"][0]], revision=STALE)
    assert run_checker(tmp_path, base_requirements(), base_acceptance(), samples) == check_acceptance.EXIT_STALE
    assert "stale-sample-sha" in categories(capsys)


def test_unsupported_substitution_fails(tmp_path, capsys):
    """An unlisted substitution must fail."""
    acceptance = base_acceptance(substitutions=[{"from": "fp64", "to": "cpu-fallback", "approved": True}])
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == check_acceptance.EXIT_MISSING
    assert "unsupported-substitution" in categories(capsys)


def test_unresolved_required_review_finding_fails(tmp_path, capsys):
    """An unresolved required finding must fail."""
    acceptance = base_acceptance(
        review={
            "verdict": "pass",
            "findings": [{"id": "F1", "severity": "major", "resolved": False}],
        }
    )
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == check_acceptance.EXIT_MISSING
    assert "unresolved-review-finding" in categories(capsys)


def test_failing_threshold_fails(tmp_path, capsys):
    """A failing numeric threshold must fail."""
    acceptance = base_acceptance(metrics={"gflops_ratio": 0.50})
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == check_acceptance.EXIT_THRESHOLD
    assert "failing-threshold" in categories(capsys)


def test_unapproved_skip_fails(tmp_path, capsys):
    """An unapproved skip must fail."""
    acceptance = base_acceptance(skips=[{"target": "cpu-host", "mode": "fp32", "reason": "slow", "approved": False}])
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == check_acceptance.EXIT_SKIP
    assert "unapproved-skip" in categories(capsys)


def test_missing_passing_sample_fails(tmp_path, capsys):
    """A missing passing sample must fail."""
    samples = base_samples(status="skipped")
    assert run_checker(tmp_path, base_requirements(), base_acceptance(), samples) in (
        check_acceptance.EXIT_MISSING,
        check_acceptance.EXIT_SKIP,
    )
    found = categories(capsys)
    assert "missing-sample" in found or "unapproved-skip" in found


def test_empty_required_targets_fails(tmp_path, capsys):
    """An empty required target set must fail."""
    requirements = base_requirements(required_targets=[])
    assert run_checker(tmp_path, requirements, base_acceptance(), base_samples()) == check_acceptance.EXIT_MISSING
    assert "empty-required-targets" in categories(capsys)


def test_target_passed_without_evidence_fails(tmp_path, capsys):
    """A passing target needs an evidence path."""
    acceptance = base_acceptance(targets={"cpu-host": target_record(evidence=[])})
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == check_acceptance.EXIT_MISSING
    assert "missing-target-evidence" in categories(capsys)


def test_target_status_without_reason_fails(tmp_path, capsys):
    """Unsupported and blocked targets need a non-empty reason."""
    acceptance = base_acceptance(targets={"cpu-host": target_record(status="unsupported")})
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == check_acceptance.EXIT_SCHEMA
    found = categories(capsys)
    assert "missing-target-reason" in found
    assert "target-not-passed" in found


def test_unknown_target_status_fails(tmp_path, capsys):
    """Only the declared executor statuses are accepted."""
    acceptance = base_acceptance(targets={"cpu-host": target_record(status="green")})
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == check_acceptance.EXIT_SCHEMA
    assert "invalid-target-status" in categories(capsys)


def test_missing_target_record_fails(tmp_path, capsys):
    """Every required target needs a matrix record."""
    acceptance = base_acceptance(targets={})
    assert run_checker(tmp_path, base_requirements(), acceptance, base_samples()) == check_acceptance.EXIT_MISSING
    assert "missing-target" in categories(capsys)


def test_not_run_target_is_explicit_gap(tmp_path, capsys):
    """A not_run target never passes and appears in the report as a gap."""
    report_path = tmp_path / "report.json"
    acceptance = base_acceptance(
        targets={"cpu-host": target_record(status="not_run", evidence=[], reason="hardware offline")}
    )
    assert (
        run_checker(tmp_path, base_requirements(), acceptance, base_samples(), json_report=report_path)
        == check_acceptance.EXIT_MISSING
    )
    assert "target-not-run" in categories(capsys)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report["requirement_reports"][0]
    assert summary["status"] == "not_run"
    assert summary["missing_evidence"]
    assert "hardware offline" in summary["reason"]


def test_stale_include_evidence_fails(tmp_path, capsys):
    """Evidence older than the last include/alpaka change is stale."""
    acceptance = base_acceptance(source_sha=OLD_SHA)
    acceptance["source_sha"] = OLD_SHA
    acceptance["entries"]["G1-boundaries"]["targets"] = {"cpu-host": target_record(source_sha=OLD_SHA)}
    samples = samples_with([base_samples()["records"][0]], revision=OLD_SHA)
    assert run_checker(tmp_path, base_requirements(), acceptance, samples) == check_acceptance.EXIT_STALE
    assert "stale-invalidate-sha" in categories(capsys)


def test_ancestor_evidence_after_include_change_passes(tmp_path, capsys):
    """An ancestor SHA newer than the last include change is valid evidence."""
    acceptance = base_acceptance(source_sha=INCLUDE_SHA)
    acceptance["source_sha"] = INCLUDE_SHA
    acceptance["entries"]["G1-boundaries"]["targets"] = {"cpu-host": target_record(source_sha=INCLUDE_SHA)}
    samples = samples_with([base_samples()["records"][0]], revision=INCLUDE_SHA)
    assert run_checker(tmp_path, base_requirements(), acceptance, samples) == 0
    assert "acceptance PASSED" in capsys.readouterr().out


def test_mixed_shas_fail_without_permission(tmp_path, capsys):
    """A single requirement must not mix evidence SHAs unless allowed."""
    records = [
        base_samples()["records"][0] | {"source_sha": INCLUDE_SHA},
        base_samples()["records"][0] | {"source_sha": HEAD},
    ]
    samples = samples_with(records)
    assert run_checker(tmp_path, base_requirements(), base_acceptance(), samples) == check_acceptance.EXIT_STALE
    assert "mixed-shas" in categories(capsys)


def test_mixed_shas_pass_when_allowed(tmp_path, capsys):
    """An explicit allow_mixed_shas flag accepts fresh ancestor SHAs."""
    records = [
        base_samples()["records"][0] | {"source_sha": INCLUDE_SHA},
        base_samples()["records"][0] | {"source_sha": HEAD},
    ]
    samples = samples_with(records)
    assert run_checker(tmp_path, base_requirements(allow_mixed_shas=True), base_acceptance(), samples) == 0
    assert "acceptance PASSED" in capsys.readouterr().out


def test_valid_performance_record_passes(tmp_path, capsys):
    """A complete benchmark record with consistent gflops passes."""
    samples = samples_with([perf_record()])
    assert run_checker(tmp_path, base_requirements(), base_acceptance(), samples) == 0
    assert "acceptance PASSED" in capsys.readouterr().out


def test_performance_record_rejections(tmp_path, capsys):
    """Each benchmark-integrity field is enforced with a schema exit code."""
    cases = [
        ("samples", {"samples": 5}, "performance-samples"),
        ("rounds", {"rounds": 2}, "performance-rounds"),
        ("median", {"median_ns": 0}, "performance-median"),
        ("identity", {"host": ""}, "performance-identity"),
        ("tools", {"tools": {"compiler": "gcc", "toolkit": "oneAPI"}}, "performance-tools"),
        ("gflops", {"gflops": 1.0}, "performance-gflops"),
        ("environment", {"load": "", "governor": "", "threads": 0}, "performance-environment"),
    ]
    for name, overrides, expected in cases:
        case_dir = tmp_path / name
        case_dir.mkdir()
        samples = samples_with([perf_record(**overrides)])
        assert run_checker(case_dir, base_requirements(), base_acceptance(), samples) == check_acceptance.EXIT_SCHEMA, (
            name
        )
        assert expected in categories(capsys), name


def test_terok_container_performance_evidence_fails(tmp_path, capsys):
    """Performance evidence produced inside the terok container is rejected."""
    samples = samples_with([perf_record(host_origin="terok_container")])
    assert run_checker(tmp_path, base_requirements(), base_acceptance(), samples) == check_acceptance.EXIT_SCHEMA
    assert "forbidden-evidence-origin" in categories(capsys)


def vendor_requirements():
    """Return a G5-style requirement that needs a vendor baseline."""
    return base_requirements(
        id="G5-large-gemm",
        vendor_baseline=True,
        thresholds=[
            {"metric": "large_square_geomean_vendor_ratio", "op": ">=", "value": 0.90},
            {"metric": "large_square_case_vendor_ratio_min", "op": ">=", "value": 0.80},
        ],
    )


def vendor_acceptance(**overrides):
    """Return a G5 acceptance entry whose producer-supplied ratios are ignored."""
    entry = base_acceptance(
        metrics={
            "large_square_geomean_vendor_ratio": 0.99,
            "large_square_case_vendor_ratio_min": 0.99,
        }
    )
    entry["entries"]["G5-large-gemm"] = entry["entries"].pop("G1-boundaries")
    entry["entries"]["G5-large-gemm"].update(overrides)
    return entry


def test_vendor_baseline_pairing_passes(tmp_path, capsys):
    """A matched baseline yields a checker-computed ratio."""
    samples = samples_with([perf_record(), baseline_record()])
    assert run_checker(tmp_path, vendor_requirements(), vendor_acceptance(), samples) == 0
    assert "acceptance PASSED" in capsys.readouterr().out


def test_vendor_ratio_is_computed_not_supplied(tmp_path, capsys):
    """A producer-supplied passing ratio cannot hide a weak measurement."""
    samples = samples_with([perf_record(median_ns=4_000_000.0), baseline_record()])
    assert run_checker(tmp_path, vendor_requirements(), vendor_acceptance(), samples) == check_acceptance.EXIT_THRESHOLD
    assert "failing-threshold" in categories(capsys)


def test_missing_vendor_baseline_is_vendor_unavailable(tmp_path, capsys):
    """A missing baseline is a reported gap, never a silent pass."""
    report_path = tmp_path / "report.json"
    samples = samples_with([perf_record()])
    assert (
        run_checker(tmp_path, vendor_requirements(), vendor_acceptance(), samples, json_report=report_path)
        == check_acceptance.EXIT_MISSING
    )
    assert "vendor-unavailable" in categories(capsys)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["requirement_reports"][0]["status"] == "vendor_unavailable"
    assert "no matched vendor baseline" in report["requirement_reports"][0]["reason"]


def test_mismatched_vendor_baseline_is_vendor_unavailable(tmp_path, capsys):
    """A baseline with a different shape is not comparable."""
    samples = samples_with([perf_record(), baseline_record(k=2048)])
    assert run_checker(tmp_path, vendor_requirements(), vendor_acceptance(), samples) == check_acceptance.EXIT_MISSING
    assert "vendor-unavailable" in categories(capsys)


def test_json_report_classifies_failures(tmp_path, capsys):
    """The JSON report carries per-requirement status, gaps, reasons, and exit code."""
    report_path = tmp_path / "report.json"
    acceptance = base_acceptance(metrics={"gflops_ratio": 0.10})
    assert (
        run_checker(tmp_path, base_requirements(), acceptance, base_samples(), json_report=report_path)
        == check_acceptance.EXIT_THRESHOLD
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["exit_code"] == check_acceptance.EXIT_THRESHOLD
    assert report["passed"] is False
    summary = report["requirement_reports"][0]
    assert summary["id"] == "G1-boundaries"
    assert summary["status"] == "failed_threshold"
    assert summary["reason"]
    capsys.readouterr()
