#!/usr/bin/env python3
"""Project a kernel's loop cost from its SASS op mix and the measured table.

Two registered models (paper sec. 4.1):
  per-pipe max (PPM, binds all gates): per-pipe issue demand, take the max
    across pipes, floored by the 4-inst/SM/cycle issue cap and the memory
    byte budget;
  naive additive (ADD, reported alongside): every term summed.

Input: a sass_census.py --full CSV restricted to one kernel, plus
table/tu102_ops.csv. Output: predicted cycles per loop iteration per warp
at the given warps/SM, for both models, with the binding resource named.

The DRAM byte budget comes from the stride-18 sector row (fetch capability
measured via u8 sector traffic) — deliberately a different method and width
from any streaming kernel this model is asked to judge.
"""

import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLE = os.path.join(ROOT, "table", "tu102_ops.csv")

ISSUE_CAP = 4.0  # instructions per SM per cycle (4 schedulers, single issue)

# mnemonic base -> (pipe, table row for the rate). Pipes follow the measured
# contention-probe map; ops without a measured rate fall back to their pipe's
# class rate (2.0 for fma/alu) with a 'defaulted' marker in the report.
OP_TABLE_ROW = {
    "FFMA": "alu.ffma.tput", "FADD": "alu.fadd.tput", "FMUL": "alu.fmul.tput",
    "IMAD": "alu.imad.tput", "IDP": "alu.idp4a.tput", "HFMA2": "alu.hfma2.tput",
    "LOP3": "alu.lop3.tput", "SHF": "alu.shf.tput", "SEL": "alu.sel.tput",
    "IADD3": "alu.iadd3.tput", "ISETP": "alu.isetp.tput", "PRMT": "alu.prmt.tput",
    "POPC": "alu.popc.tput", "FLO": "alu.flo.tput",
    "DADD": "alu.dadd.tput", "DFMA": "alu.dfma.tput",
    "MUFU": "sfu.mufu.ex2.tput",
    "F2F": "cvt.f2f.tput", "HADD2": "cvt.f2f.tput",
    "F2I": "cvt.i2f_f2i.tput", "I2F": "cvt.i2f_f2i.tput",
    "BAR": None,  # handled by latency, not rate; rare in hot loops
}
PIPE_OF = {
    "FFMA": "fma", "FADD": "fma", "FMUL": "fma", "IMAD": "fma", "IDP": "fma",
    "FSETP": "fma", "FSEL": "alu", "FMNMX": "fma",
    "HADD2": "own", "HMUL2": "own", "HFMA2": "own",
    "LOP3": "alu", "SHF": "alu", "SEL": "alu", "IADD3": "alu", "ISETP": "alu",
    "PRMT": "alu", "LEA": "alu", "MOV": "alu", "BFE": "alu", "BFI": "alu",
    "POPC": "own_xu", "FLO": "own_xu", "MUFU": "own_xu",
    "F2F": "own_xu", "F2I": "own_xu", "I2F": "own_xu",
    "DADD": "own_fp64", "DFMA": "own_fp64", "DMUL": "own_fp64",
}
CONTROL = {"BRA", "NOP", "EXIT", "CS2R", "S2R", "BSSY", "BSYNC", "YIELD"}
LSU = {"LDG", "STG", "LDS", "STS", "LDC", "LDL", "STL", "LDSM"}

# bytes per thread for memory mnemonics by width suffix
WIDTH_BYTES = {"8": 1, "U8": 1, "S8": 1, "16": 2, "U16": 2, "S16": 2,
               "32": 4, "64": 8, "128": 16}


def load_rates():
    rates = {}
    with open(TABLE) as f:
        for row in csv.DictReader(f):
            # sweep rows share a row_id; keep the peak (f32/f64 ceiling)
            if row["row_id"] in rates and \
               float(rates[row["row_id"]]["value"]) >= float(row["value"]):
                continue
            rates[row["row_id"]] = row
    return rates


def l1_budget(rates, defaulted):
    if "mem.l1.bw" in rates:
        return float(rates["mem.l1.bw"]["value"])
    defaulted.append("mem.l1.bw(default32)")
    return 32.0


def mem_bytes(mnemonic_full):
    parts = mnemonic_full.split(".")
    for p in reversed(parts):
        if p in WIDTH_BYTES:
            return WIDTH_BYTES[p]
    return 4  # unsuffixed LDG/LDS default to 32-bit


def lds_inst_rate(bytes_per_thread):
    # measured: 64 B/clk/SM smem ceiling; inst rate = 64 / (32 * width)
    return 64.0 / (32.0 * bytes_per_thread)


def project(census, warps, mem_class, rates, dram_budget):
    """census: dict full-mnemonic -> count (one loop body, one warp).
    Returns dict with both models and the per-resource breakdown."""
    pipe_cycles = collections.defaultdict(float)
    mem_bytes_total = 0.0
    smem_cycles = 0.0
    total_insts = 0
    defaulted = []
    for mn, n in census.items():
        base = mn.split(".")[0]
        if base in CONTROL:
            total_insts += n
            continue
        total_insts += n
        if base in LSU:
            b = mem_bytes(mn) * 32  # per warp
            if base in ("LDS", "STS"):
                smem_cycles += n * (1.0 / lds_inst_rate(mem_bytes(mn)))
            elif base in ("LDL", "STL"):
                defaulted.append(mn)  # local traffic: flagged, not modelled
            else:
                mem_bytes_total += n * b
            continue
        row_id = OP_TABLE_ROW.get(base)
        rate = None
        if row_id and row_id in rates:
            rate = float(rates[row_id]["value"])
            pipe = rates[row_id]["pipe"] or PIPE_OF.get(base, "alu")
        else:
            pipe = PIPE_OF.get(base, "alu")
            rate = 2.0 if pipe in ("fma", "alu") else 0.5
            defaulted.append(mn)
        pipe_cycles[pipe] += n / rate

    per_resource = {f"pipe:{p}": warps * c for p, c in pipe_cycles.items()}
    if smem_cycles:
        per_resource["smem"] = warps * smem_cycles
    if mem_bytes_total:
        budget = dram_budget if mem_class == "dram" else l1_budget(rates, defaulted)
        per_resource["mem"] = warps * mem_bytes_total / budget
    per_resource["issue"] = warps * total_insts / ISSUE_CAP

    ppm_bound = max(per_resource, key=per_resource.get)
    # the additive model has no cap concept: plain sum of the work terms
    add = sum(v for k, v in per_resource.items() if k != "issue")
    return {
        "ppm_cycles": per_resource[ppm_bound],
        "ppm_bound": ppm_bound,
        "add_cycles": add if add > 0 else per_resource["issue"],
        "per_resource": dict(per_resource),
        "defaulted": defaulted,
    }


def census_from_csv(path, kernel_regex):
    import re
    kre = re.compile(kernel_regex)
    census = collections.Counter()
    with open(path) as f:
        for row in csv.DictReader(f):
            if kre.search(row["kernel"]):
                census[row["op"]] += int(row["count"])
    return census


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("census_csv", help="sass_census.py --full output")
    ap.add_argument("--kernel", default=".*")
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--mem-class", choices=["none", "dram", "l1"], default="none")
    ap.add_argument("--dram-budget", type=float, default=5.82,
                    help="B/clk/SM; default from the stride-18 sector row (610 GB/s)")
    args = ap.parse_args()

    rates = load_rates()
    census = census_from_csv(args.census_csv, args.kernel)
    if not census:
        print("no ops matched", file=sys.stderr)
        return 1
    r = project(census, args.warps, args.mem_class, rates, args.dram_budget)
    print(f"PPM: {r['ppm_cycles']:.1f} cycles/iter (bound: {r['ppm_bound']})")
    print(f"ADD: {r['add_cycles']:.1f} cycles/iter")
    for k, v in sorted(r["per_resource"].items(), key=lambda kv: -kv[1]):
        print(f"  {k:12s} {v:10.1f}")
    if r["defaulted"]:
        print(f"  defaulted rates: {' '.join(sorted(set(r['defaulted'])))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
