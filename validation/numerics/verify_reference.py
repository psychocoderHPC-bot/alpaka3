#!/usr/bin/env python3
# Copyright 2026 René Widera
# SPDX-License-Identifier: MPL-2.0
"""Numpy cross-check for validation/numerics/reference_gemm.hpp.

Reads the CSV exported by validation/numerics/verify_reference.cpp, rebuilds
the same float32 operands with the documented LCG, and compares every exported
element against a numpy float64 evaluation of

    C = alpha * op(A) @ op(B) + beta * C_old

The two sides sum in a different order, so the check uses a double-precision
forward bound: 1e-12 * (|alpha| * sum_k |A_ik B_kj| + |beta| * |C_ij|). This
bound is about 200 times the double rounding scale for the tested K and is
meant to catch indexing, layout, transpose and alpha/beta mistakes, not to
re-certify the tolerance model.

Usage:
    python3 validation/numerics/verify_reference.py gemm-reference.csv
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

MULTIPLIER = 6364136223846793005
INCREMENT = 1442695040888963407
MASK = (1 << 64) - 1


class Lcg:
    def __init__(self, seed: int) -> None:
        self.state = seed & MASK

    def next_float(self) -> np.float32:
        self.state = (self.state * MULTIPLIER + INCREMENT) & MASK
        value = (self.state >> 40) / 16777216.0
        return np.float32(value * 2.0 - 1.0)


def make_storage(rows: int, cols: int, layout: str, pad: int, lcg: Lcg) -> np.ndarray:
    """Rebuild the stored logical matrix, consuming the LCG like the C++ tool."""
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols), dtype=np.float32)
    ld = (cols if layout == "row" else rows) + pad
    extent = rows if layout == "row" else cols
    values = np.array([lcg.next_float() for _ in range(ld * extent)], dtype=np.float32)
    if layout == "row":
        return values.reshape(rows, ld)[:, :cols]
    return values.reshape(cols, ld).T[:rows, :cols]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <csv>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"cross-check: file not found: {path}", file=sys.stderr)
        return 2

    with path.open("r", encoding="utf-8") as stream:
        lines = [line for line in stream if not line.startswith("#")]
    reader = csv.DictReader(lines)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in reader:
        grouped.setdefault(row["case_id"], []).append(row)

    if len(grouped) < 6:
        print(f"cross-check: only {len(grouped)} cases with data, need at least 6", file=sys.stderr)
        return 1

    worst_scaled = 0.0
    failures = 0
    for index, (case_id, rows) in enumerate(grouped.items()):
        first = rows[0]
        m = int(first["M"])
        n = int(first["N"])
        k = int(first["K"])
        layout_a = first["layout_a"]
        layout_b = first["layout_b"]
        layout_c = first["layout_c"]
        op_a = first["op_a"]
        op_b = first["op_b"]
        stored_rows_a = k if op_a == "T" else m
        stored_cols_a = m if op_a == "T" else k
        stored_rows_b = n if op_b == "T" else k
        stored_cols_b = k if op_b == "T" else n
        pad_a = int(first["ld_a"]) - (stored_cols_a if layout_a == "row" else stored_rows_a)
        pad_b = int(first["ld_b"]) - (stored_cols_b if layout_b == "row" else stored_rows_b)
        pad_c = int(first["ld_c"]) - (n if layout_c == "row" else m)
        alpha = float(np.float32(first["alpha"]))
        beta = float(np.float32(first["beta"]))

        lcg = Lcg(0x0123456789ABCDEF + 0x9E3779B97F4A7C15 * index)
        stored_a = make_storage(stored_rows_a, stored_cols_a, layout_a, pad_a, lcg)
        stored_b = make_storage(stored_rows_b, stored_cols_b, layout_b, pad_b, lcg)
        stored_c = make_storage(m, n, layout_c, pad_c, lcg)
        used_a = stored_a if op_a == "N" else stored_a.T
        used_b = stored_b if op_b == "N" else stored_b.T
        if used_a.shape != (m, k) or used_b.shape != (k, n) or stored_c.shape != (m, n):
            print(f"{case_id}: reconstructed shape mismatch", file=sys.stderr)
            return 1

        a64 = used_a.astype(np.float64)
        b64 = used_b.astype(np.float64)
        c64 = stored_c.astype(np.float64)
        # Independent summation order: math.fsum is correctly rounded, so the
        # comparison does not inherit numpy's (or the C++ loop's) order.
        expected = np.empty((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                dot = math.fsum(float(a64[i, kk]) * float(b64[kk, j]) for kk in range(k))
                expected[i, j] = alpha * dot + beta * float(c64[i, j])
        scale = abs(alpha) * (np.abs(a64) @ np.abs(b64)) + abs(beta) * np.abs(c64)
        bound = 1e-12 * scale + 1e-300

        exported = np.full((m, n), np.nan, dtype=np.float64)
        for row in rows:
            exported[int(row["i"]), int(row["j"])] = float.fromhex(row["value"])
        if exported.shape != expected.shape:
            print(f"{case_id}: exported shape mismatch", file=sys.stderr)
            return 1

        error = np.abs(exported - expected)
        scaled = np.zeros_like(error)
        np.divide(error, scale, out=scaled, where=scale > 0.0)
        case_scaled = float(np.max(scaled)) if scaled.size else 0.0
        worst_scaled = max(worst_scaled, case_scaled)
        case_failures = int(np.count_nonzero(error > bound))
        failures += case_failures
        status = "ok" if case_failures == 0 else f"FAIL ({case_failures} elements)"
        print(
            f"{case_id:24s} M={m:3d} N={n:3d} K={k:3d} "
            f"max_abs={float(np.max(error)):.3e} max_scaled={case_scaled:.3e} {status}"
        )

    if failures:
        print(f"cross-check FAILED: {failures} element(s) outside the double-precision bound", file=sys.stderr)
        return 1
    print(f"cross-check PASSED for {len(grouped)} cases, worst scaled error {worst_scaled:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
