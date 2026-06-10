#!/usr/bin/env python3
"""The absolute projection gate: PPM within ±20% of measured cycles/iter on
every gate kernel (anchors, FA decision pairs, injection family). The
latency-bound demo is excluded by design and reported with its error — it
is the documented boundary of the issue-rate model's validity (regime
heuristic: occupancy × ILP below the latency-hiding threshold).

Exit 0 iff all gate kernels pass. ADD is reported alongside, never gated.
"""

import collections
import csv
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project  # noqa: E402
from check_sass import loop_mix  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "results", "t5820-2xrtx6000", "proj.csv")
TOL = 0.20

# (measured row_id, binary, kernel regex, mem class, warps, gated)
KERNELS = [
    ("proj.anchor.ffma", "bench/proj/anchors.bin", "ffma_anchor", "none", 8, True),
    ("proj.anchor.stream", "bench/proj/anchors.bin", "stream_anchor", "dram", 8, True),
    ("proj.anchor.smemtile", "bench/proj/anchors.bin", "smemtile_anchor", "none", 8, True),
    ("proj.anchor.capmix", "bench/proj/anchors.bin", "capmix_anchor", "none", 8, True),
    ("proj.anchor.latbound", "bench/proj/anchors.bin", "latbound_demo", "none", 1, False),
    ("proj.fa_mini.base", "bench/proj/fa_mini.bin", "fa_mini_kernelILi0", "l1", 8, True),
    ("proj.fa_mini.dp4a", "bench/proj/fa_mini.bin", "fa_mini_kernelILi1", "l1", 8, True),
    ("proj.fa_mini.staged", "bench/proj/fa_mini.bin", "fa_mini_kernelILi2", "l1", 8, True),
]
INJECT = [("base", 0), ("ffma", 8), ("ffma", 16), ("ffma", 24),
          ("idp4a", 8), ("idp4a", 16), ("idp4a", 24),
          ("lop3", 8), ("lop3", 16), ("lop3", 24),
          ("popc", 4), ("popc", 8), ("popc", 12)]
OPSYM = {"base": "OpNONEELi0", "ffma": "OpFFMAELi", "idp4a": "OpIDP4A_S8ELi",
         "lop3": "OpLOP3ELi", "popc": "OpPOPCELi"}


def measured_medians():
    vals = collections.defaultdict(list)
    with open(RESULTS) as f:
        for row in csv.DictReader(f):
            vals[row["row_id"]].append(float(row["value"]))
    return {k: statistics.median(v) for k, v in vals.items()}


def main():
    rates = project.load_rates()
    meas = measured_medians()
    rows = list(KERNELS)
    for op, k in INJECT:
        sym = OPSYM[op] + (str(k) + "E" if op != "base" else "")
        rows.append((f"proj.inject.{op}.k{k}", "bench/proj/inject.bin",
                     f"inject_kernelINS_\\d*{sym}", "l1", 8, True))

    failures = 0
    print(f"{'kernel':28s} {'measured':>10s} {'PPM':>10s} {'err':>7s}  "
          f"{'ADD':>10s} {'bound':12s}")
    for row_id, binary, regex, mc, warps, gated in rows:
        if row_id not in meas:
            print(f"{row_id:28s} NO MEASUREMENT — run the bench")
            failures += gated
            continue
        mix, n = loop_mix(os.path.join(ROOT, binary), regex)
        if n == 0:
            print(f"{row_id:28s} NO KERNEL MATCH ({regex})")
            failures += gated
            continue
        r = project.project(dict(mix), warps, mc, rates, 5.82)
        m = meas[row_id]
        err = (r["ppm_cycles"] - m) / m
        ok = abs(err) <= TOL
        tag = ("PASS" if ok else "FAIL") if gated else "demo"
        if gated and not ok:
            failures += 1
        print(f"{row_id:28s} {m:10.1f} {r['ppm_cycles']:10.1f} {100*err:+6.1f}%  "
              f"{r['add_cycles']:10.1f} {r['ppm_bound']:12s} {tag}")
    print(f"\nverify_projection: {failures} gate failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
