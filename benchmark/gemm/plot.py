#!/usr/bin/env python3
# SPDX-License-Identifier: MPL-2.0
"""Render GEMM throughput and ratio plots from samples.json files.

The script reads raw benchmark output and writes figures outside the source
checkout. Measured data, figures, and profiler output never get committed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")


CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


def load_samples(path: Path) -> dict:
    """Load and validate one samples.json document."""
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("kind") != "alpaka3-gemm-samples":
        raise SystemExit(f"{path}: not an alpaka3-gemm-samples document")
    return document


def transpose_code(record: dict) -> str:
    """Return the two-letter transpose code for a record."""

    def code(operation: str) -> str:
        """Translate one transpose flag to a single letter."""
        return "t" if operation == "transpose" else "n"

    return code(record["transpose_a"]) + code(record["transpose_b"])


def label(record: dict) -> str:
    """Build the plot label for one sample record."""
    dims = f"{record['m']}x{record['n']}x{record['k']}"
    return f"{record['case_id']} {dims} {record['mode']} {transpose_code(record)} {record['stride_class']}"


def ok_records(document: dict) -> list[dict]:
    """Return only the successfully timed records of a document."""
    return [record for record in document.get("records", []) if record.get("status") == "ok"]


def ensure_outside_checkout(out_dir: Path, allow_inside: bool) -> None:
    """Refuse to write figures into the source checkout unless allowed."""
    resolved = out_dir.resolve()
    if not allow_inside and (resolved == CHECKOUT_ROOT or CHECKOUT_ROOT in resolved.parents):
        raise SystemExit(
            f"refusing to write plots inside the checkout ({CHECKOUT_ROOT}); "
            f"choose an output directory outside it or pass --allow-inside-repo"
        )


def plot_throughput(document: dict, source: Path, out_dir: Path, index: int) -> Path | None:
    """Write the throughput bar chart for one samples document."""
    records = ok_records(document)
    if not records:
        print(f"{source}: no timed records; skipping throughput plot", file=sys.stderr)
        return None
    records.sort(key=lambda record: record["gflops"])
    labels = [label(record) for record in records]
    values = [record["gflops"] for record in records]
    height = max(4.0, 0.32 * len(records))
    figure, axis = plt.subplots(figsize=(11, height))
    axis.barh(labels, values, color="#3b6ea5")
    axis.set_xlabel("throughput [GFLOP/s], useful 2*M*N*K / median time")
    axis.set_title(f"GEMM throughput, revision {document.get('revision', 'unknown')}")
    axis.grid(axis="x", linestyle=":", alpha=0.5)
    figure.tight_layout()
    target = out_dir / f"throughput-{index:02d}-{source.parent.name or 'run'}.png"
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def ratio_entry(record: dict, reference: dict) -> tuple[str, float, float, float]:
    """Return label, throughput ratio and ratio error bounds for a record."""
    ratio = record["gflops"] / reference["gflops"]
    flops = 2.0 * record["m"] * record["n"] * record["k"]
    record_min = flops / record["max_ns"]
    record_max = flops / record["min_ns"]
    reference_min = flops / reference["max_ns"]
    reference_max = flops / reference["min_ns"]
    return label(record), ratio, record_min / reference_max, record_max / reference_min


def matched_ratios(candidate: dict, baseline: dict) -> list[tuple[str, float, float, float]]:
    """Pair candidate records with baseline records by key."""
    baseline_by_key = {record["key"]: record for record in ok_records(baseline) if "key" in record}
    ratios: list[tuple[str, float, float, float]] = []
    for record in ok_records(candidate):
        reference = baseline_by_key.get(record.get("key"))
        if reference is None or not reference.get("gflops"):
            continue
        ratios.append(ratio_entry(record, reference))
    return ratios


def draw_ratio(ratios: list[tuple[str, float, float, float]], out_dir: Path) -> Path:
    """Render the ratio chart for the given ratios and return its path."""
    ratios.sort(key=lambda item: item[1])
    labels = [item[0] for item in ratios]
    values = [item[1] for item in ratios]
    lower = [max(0.0, value - item[2]) for value, item in zip(values, ratios)]
    upper = [max(0.0, item[3] - value) for value, item in zip(values, ratios)]
    height = max(4.0, 0.32 * len(ratios))
    figure, axis = plt.subplots(figsize=(11, height))
    axis.barh(labels, values, color="#3b6ea5", xerr=[lower, upper], capsize=3)
    axis.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("candidate / baseline median throughput")
    axis.set_title("GEMM throughput ratio (error bars span observed min/max)")
    axis.grid(axis="x", linestyle=":", alpha=0.5)
    figure.tight_layout()
    target = out_dir / "ratio.png"
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def plot_ratio(candidate: dict, baseline: dict, out_dir: Path) -> Path | None:
    """Write the ratio chart if candidate and baseline records match."""
    ratios = matched_ratios(candidate, baseline)
    if not ratios:
        print("no matching timed records between candidate and baseline", file=sys.stderr)
        return None
    return draw_ratio(ratios, out_dir)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, render all requested plots and report the paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", type=Path, help="samples.json files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/alpaka3-gemm-plots"),
        help="plot directory outside the checkout (default: /tmp/alpaka3-gemm-plots)",
    )
    parser.add_argument(
        "--allow-inside-repo",
        action="store_true",
        help="override the guard against writing figures into the checkout",
    )
    args = parser.parse_args(argv)

    ensure_outside_checkout(args.output_dir, args.allow_inside_repo)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    documents = [load_samples(path) for path in args.samples]
    written = []
    for index, (path, document) in enumerate(zip(args.samples, documents), start=1):
        target = plot_throughput(document, path, args.output_dir, index)
        if target is not None:
            written.append(target)
    if len(documents) >= 2:
        target = plot_ratio(documents[0], documents[1], args.output_dir)
        if target is not None:
            written.append(target)

    if not written:
        print("nothing plotted", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
