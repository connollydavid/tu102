#!/usr/bin/env python3
"""Differential-projection analysis for the injection factorial.

Pairs every injected variant against the base, computes measured deltas
(from data/results/<host>/proj.csv) and projected deltas (both models, from
the EMITTED census of bench/proj/inject.bin), and evaluates the three
registered predictions of paper sec. 4.1. Common terms (the base's loads
and balanced work) cancel in every delta — that cancellation is the
mechanism under test.

Pairs whose measured delta falls outside the 5-25% design band are reported
but excluded from the error bound (the band is a design rule of the
registered analysis plan, not a tolerance).

Output: data/proj/differential.csv + verdicts on stdout.
"""

import collections
import csv
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "results", "t5820-2xrtx6000", "proj.csv")
BIN = os.path.join(ROOT, "bench", "proj", "inject.bin")
OUTDIR = os.path.join(ROOT, "data", "proj")
CUOBJDUMP = "/opt/cuda-13.3/bin/cuobjdump"

SYM_RE = re.compile(r"inject_kernelINS_\d*(Op\w+?)ELi(\d+)E")
OPKEY = {"OpNONE": "base", "OpFFMA": "ffma", "OpIDP4A_S8": "idp4a",
         "OpLOP3": "lop3", "OpPOPC": "popc"}
FMA_PIPE = {"ffma", "idp4a"}          # measured pipe map (tu102 a401d03)
ALU_PIPE = {"lop3"}
WARPS = 8
BAND = (0.05, 0.25)


def emitted_censuses():
    sass = subprocess.run([CUOBJDUMP, "-sass", BIN], capture_output=True,
                          text=True, check=True).stdout
    from check_sass import parse_functions, hot_loop
    censuses = {}
    for name, instrs in parse_functions(sass):
        m = SYM_RE.search(name)
        if not m:
            continue
        key = (OPKEY[m.group(1)], int(m.group(2)))
        loop = hot_loop(instrs)
        if loop is None:
            continue
        c = collections.Counter()
        for _, base, text in instrs[loop[0]:loop[1] + 1]:
            full = text.split()[1] if text.startswith("@") else text.split()[0]
            c[full] += 1
        censuses[key] = c
    return censuses


def measured():
    """median cycles/iter per (op, K) across all invocations/GPUs."""
    vals = collections.defaultdict(list)
    with open(RESULTS) as f:
        for row in csv.DictReader(f):
            m = re.match(r"proj\.inject\.(\w+)\.k(\d+)", row["row_id"])
            if m:
                vals[(m.group(1), int(m.group(2)))].append(float(row["value"]))
    return {k: statistics.median(v) for k, v in vals.items()}, \
           {k: len(v) for k, v in vals.items()}


def main():
    rates = project.load_rates()
    censuses = emitted_censuses()
    meas, n_runs = measured()
    if ("base", 0) not in meas:
        print("no base measurement; run bench/proj/inject.bin first")
        return 1
    if min(n_runs.values()) < 2:
        print("WARNING: between-run rule unmet (fewer than 2 invocations)")

    proj_of = {}
    for key, census in censuses.items():
        r = project.project(dict(census), WARPS, "l1", rates, 5.82)
        proj_of[key] = (r["ppm_cycles"], r["add_cycles"], r["ppm_bound"])

    base_m = meas[("base", 0)]
    base_p = proj_of[("base", 0)]
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for key in sorted(meas):
        if key == ("base", 0):
            continue
        if key not in proj_of:
            print(f"NOTE: no census for {key} (stale results row?); skipped")
            continue
        op, k = key
        dm = meas[key] - base_m
        rel = dm / base_m
        dp_ppm = proj_of[key][0] - base_p[0]
        dp_add = proj_of[key][1] - base_p[1]
        in_band = BAND[0] <= rel <= BAND[1]
        err_ppm = abs(dp_ppm - dm) / abs(dm) if dm else float("inf")
        err_add = abs(dp_add - dm) / abs(dm) if dm else float("inf")
        rows.append({
            "op": op, "k": k, "pipe": "fma" if op in FMA_PIPE else
            ("alu" if op in ALU_PIPE else "own"),
            "meas_delta_cyc": f"{dm:.2f}", "rel_delta_pct": f"{100*rel:.1f}",
            "in_band": int(in_band),
            "proj_delta_ppm": f"{dp_ppm:.2f}", "proj_delta_add": f"{dp_add:.2f}",
            "err_ppm_pct": f"{100*err_ppm:.1f}", "err_add_pct": f"{100*err_add:.1f}",
            "ppm_bound_resource": proj_of[key][2],
        })

    out = os.path.join(OUTDIR, "differential.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{out}: {len(rows)} pairs (base {base_m:.1f} cyc/iter; "
          f"PPM {base_p[0]:.1f}, ADD {base_p[1]:.1f}, bound {base_p[2]})")
    for r in rows:
        print(f"  {r['op']:6s} k{r['k']:<3d} pipe={r['pipe']:4s} "
              f"meas {r['meas_delta_cyc']:>8s} ({r['rel_delta_pct']:>5s}%) "
              f"PPM {r['proj_delta_ppm']:>8s} (err {r['err_ppm_pct']:>6s}%) "
              f"ADD {r['proj_delta_add']:>8s} (err {r['err_add_pct']:>6s}%)"
              f"{'' if r['in_band'] else '  [out of band]'}")

    band_rows = [r for r in rows if r["in_band"]]
    if not band_rows:
        print("no pairs in the design band; redesign injection grades")
        return 1

    def errs(rows_, which, pipe=None):
        sel = [float(r[f"err_{which}_pct"]) for r in rows_
               if pipe is None or r["pipe"] == pipe]
        return sel

    # registered predictions (paper sec. 4.1)
    add_fma = statistics.median(errs(band_rows, "add", "fma"))
    add_alu = statistics.median(errs(band_rows, "add", "alu"))
    ppm_fma = statistics.median(errs(band_rows, "ppm", "fma"))
    ppm_alu = statistics.median(errs(band_rows, "ppm", "alu"))
    p_i = add_fma > add_alu
    excess_add = add_fma - add_alu
    excess_ppm = ppm_fma - ppm_alu
    p_ii = excess_ppm < 0.5 * excess_add if excess_add > 0 else None
    all_ppm = sorted(errs(band_rows, "ppm"))
    bound95 = all_ppm[min(len(all_ppm) - 1, int(0.95 * len(all_ppm)))]
    p_iii = bound95 < 100.0 * 0.10 / BAND[0] and bound95 < 50.0  # sane cap

    print(f"\nregistered predictions (in-band pairs only, n={len(band_rows)}):")
    print(f"  (i)   ADD err fma {add_fma:.1f}% vs alu {add_alu:.1f}% -> "
          f"{'CONFIRMED' if p_i else 'REFUTED'}")
    print(f"  (ii)  excess: ADD {excess_add:.1f}pp vs PPM {excess_ppm:.1f}pp -> "
          f"{'CONFIRMED' if p_ii else ('REFUTED' if p_ii is not None else 'N/A (no ADD excess)')}")
    print(f"  (iii) PPM 95th-pct differential error {bound95:.1f}% -> "
          f"{'discriminates >=10% deltas' if p_iii else 'DOES NOT discriminate (kill-criterion input)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
