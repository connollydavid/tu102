#!/usr/bin/env python3
"""Generate table/tu102_ops.csv from data/results/<host>/ + table/priors_t4.csv.

Aggregation per SCHEMA.md:
  - value = median across all qualifying invocations (both GPUs for SM rows)
  - between-run rule: >= 2 independent invocations required; rows where the
    between-run spread exceeds the within-run cv (with a 0.1% floor) are
    flagged in notes
  - GPU agreement: SM-domain medians must agree within combined cv
    (0.3% floor); disagreement -> UNVERIFIED + note
  - tput rows: peak across the warps/SM sweep; variant records the point
  - deviation_pct computed only where a prior exists (prior-applicability
    rule lives in priors_t4.csv itself: no row there, no prior here)

Deterministic output: same inputs -> byte-identical CSV (the idempotency
gate in the Harness verification plan).
"""

import collections
import csv
import os
import statistics
import sys

HOST_DS = "t5820-2xrtx6000"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "data", "results", HOST_DS)
PRIORS = os.path.join(ROOT, "table", "priors_t4.csv")
OUT = os.path.join(ROOT, "table", "tu102_ops.csv")

# row_id prefix -> SASS instruction (as proven by check_sass.py)
INSTRUCTION = {
    "alu.ffma": "FFMA", "alu.fadd": "FADD", "alu.fmul": "FMUL",
    "alu.iadd3_lop3": "IADD3+LOP3", "alu.iadd3": "IADD3", "alu.imad": "IMAD",
    "alu.lop3": "LOP3", "alu.shf": "SHF", "alu.sel": "SEL", "alu.fsel": "SEL",
    "alu.isetp_sel": "ISETP+SEL", "alu.isetp": "ISETP",
    "alu.fsetp_sel": "FSETP+SEL", "alu.fsetp": "FSETP",
    "alu.popc": "POPC", "alu.flo": "FLO", "alu.prmt": "PRMT",
    "alu.idp4a": "IDP.4A", "alu.hfma2": "HFMA2",
    "alu.dadd": "DADD", "alu.dfma": "DFMA",
    "alu.idiv": "(IDIV sequence)",
}


def instruction_for(row_id):
    for prefix in sorted(INSTRUCTION, key=len, reverse=True):
        if row_id.startswith(prefix):
            return INSTRUCTION[prefix]
    return ""


def sweep_base(variant):
    """w8_s8 -> _s8 (sweep family); w8 -> ''. Non-sweep variants pass through."""
    if variant.startswith("w") and variant[1].isdigit():
        rest = variant.lstrip("w0123456789")
        return rest
    return None  # not a sweep point


def main():
    priors = {}
    with open(PRIORS) as f:
        for row in csv.DictReader(f):
            priors[row["row_id"]] = row

    # measurements[(row_id, variant_key)] = list of result dicts
    measurements = collections.defaultdict(list)
    for fname in sorted(os.listdir(RESULTS)):
        if fname == "runs.csv" or not fname.endswith(".csv"):
            continue
        if fname == "proj.csv":
            continue  # diagnostic family (differential experiment); not table rows
        with open(os.path.join(RESULTS, fname)) as f:
            for row in csv.DictReader(f):
                row["family"] = fname[:-4]
                measurements[row["row_id"]].append(row)

    # pipe labels from the contention probes (bench/alu/pipes.cu): rows named
    # alu.<op>.pipe carry "pipe=<label>; ..." in notes; they fill the pipe
    # column and stay out of the table body (diagnostics live in data/)
    pipe_label = {}
    for row_id, rows in list(measurements.items()):
        if row_id.endswith(".pipe"):
            note = rows[0]["notes"]
            if note.startswith("pipe="):
                pipe_label[row_id[:-len(".pipe")]] = note.split(";")[0][len("pipe="):].split(" (")[0]
            del measurements[row_id]

    out_rows = []
    for row_id in sorted(measurements):
        rows = measurements[row_id]
        kind = rows[0]["kind"]
        unit = rows[0]["unit"]
        family = rows[0]["family"]
        bench_src = rows[0]["bench_src"]

        if kind in ("recip_tput", "bandwidth"):
            # w-prefixed variants are occupancy sweeps: peak per suffix.
            # Anything else (stride18, conflict4, broadcast) is its own group.
            by_suffix = collections.defaultdict(list)
            standalone = collections.defaultdict(list)
            for r in rows:
                sb = sweep_base(r["variant"])
                if sb is None:
                    standalone[r["variant"]].append(r)
                else:
                    by_suffix[("sweep", sb)].append(r)
            groups = list(by_suffix.items()) + list(standalone.items())
        else:
            # latency rows: every variant is its own row
            by_variant = collections.defaultdict(list)
            for r in rows:
                by_variant[r["variant"]].append(r)
            groups = list(by_variant.items())

        for variant_key, grp in sorted(groups, key=lambda kv: str(kv[0])):
            if isinstance(variant_key, tuple):
                # per invocation: take the sweep peak
                per_run = collections.defaultdict(list)
                for r in grp:
                    per_run[(r["run_id"], r["gpu_index"])].append(r)
                run_vals, peak_variants, cvs = [], [], []
                gpu_vals = collections.defaultdict(list)
                for (run, gpu), rr in per_run.items():
                    peak = max(rr, key=lambda r: float(r["value"]))
                    run_vals.append(float(peak["value"]))
                    gpu_vals[gpu].append(float(peak["value"]))
                    peak_variants.append(peak["variant"])
                    cvs.append(float(peak["cv_pct"]))
                variant = collections.Counter(peak_variants).most_common(1)[0][0]
            else:
                run_vals = [float(r["value"]) for r in grp]
                gpu_vals = collections.defaultdict(list)
                for r in grp:
                    gpu_vals[r["gpu_index"]].append(float(r["value"]))
                cvs = [float(r["cv_pct"]) for r in grp]
                variant = variant_key

            # provenance binds per published row (this variant group), not
            # per row_id — a later-added variant must not inherit sibling
            # variants' shas or run counts
            grp_shas = sorted({r["git_sha"] for r in grp})
            grp_runs = len({r["run_id"] for r in grp})

            value = statistics.median(run_vals)
            within_cv = statistics.median(cvs) if cvs else 0.0
            flag = "ok"
            extra = []
            grp_notes = [g["notes"] for g in grp if g["notes"]]
            note = grp_notes[-1] if grp_notes else ""  # latest annotation wins

            # cycle-domain rows are deterministic (0.1% floor); wall-clock
            # bandwidth rows carry real DRAM refresh/thermal variation (0.5%);
            # host-domain time rows carry scheduler jitter (5%)
            floor = 5.0 if kind == "time_us" else \
                    0.5 if kind == "bandwidth" and unit == "GB/s" else 0.1
            if len(run_vals) < 2:
                flag = "UNVERIFIED"
                extra.append("between-run rule unmet (single invocation)")
            else:
                spread = 100.0 * (max(run_vals) - min(run_vals)) / value if value else 0.0
                if spread > max(within_cv, floor):
                    extra.append(f"between-run spread {spread:.2f}% exceeds within-run cv {within_cv:.2f}%")
                    flag = "UNVERIFIED"

            if len(gpu_vals) >= 2:
                meds = [statistics.median(v) for v in gpu_vals.values()]
                gpu_diff = 100.0 * abs(meds[0] - meds[1]) / value if value else 0.0
                if gpu_diff > max(2 * within_cv, 0.3):
                    extra.append(f"GPU0-vs-GPU1 medians differ {gpu_diff:.2f}%")
                    flag = "UNVERIFIED"

            # variant-specific priors: <row_id>.<variant> beats <row_id>
            prior = priors.get(f"{row_id}.{variant}") or priors.get(row_id)
            if prior and variant and f"{row_id}.{variant}" not in priors \
               and kind in ("latency_cycles", "latency_ns") \
               and variant not in ("", "l1hit", "broadcast", "conflict1", "derived"):
                prior = None  # a bare-row prior binds only the base variant
            prior_value = prior["prior_value"] if prior else ""
            prior_src = prior["prior_src"] if prior else ""
            deviation = ""
            if prior:
                dev = 100.0 * (value - float(prior["prior_value"])) / float(prior["prior_value"])
                deviation = f"{dev:.1f}"
                if abs(dev) > 25 and flag == "ok":
                    flag = "DEV>25%"

            all_notes = "; ".join([note] * bool(note) + extra)
            out_rows.append({
                "row_id": row_id, "class": family,
                "instruction": instruction_for(row_id), "variant": variant,
                "kind": kind, "value": f"{value:.6g}", "unit": unit,
                "cv_pct": f"{within_cv:.3f}",
                "pipe": next((p for k, p in pipe_label.items()
                              if row_id.startswith(k + ".")), ""),
                "prior_value": prior_value, "prior_src": prior_src,
                "deviation_pct": deviation, "flag": flag,
                "measured_by": f"{bench_src}@{'+'.join(grp_shas)} n_runs={grp_runs}",
                "clock_mhz": "1455", "notes": all_notes,
            })

    # explicit-absence rows: features that do not exist on sm_75 are real
    # rows (kind=na, flag=NA_SM75), never missing keys
    na_path = os.path.join(ROOT, "table", "na_sm75.csv")
    if os.path.exists(na_path):
        with open(na_path) as f:
            for row in csv.DictReader(f):
                out_rows.append({
                    "row_id": row["row_id"], "class": row["row_id"].split(".")[0],
                    "instruction": row["instruction"], "variant": "", "kind": "na",
                    "value": "", "unit": "", "cv_pct": "", "pipe": "",
                    "prior_value": "", "prior_src": row["prior_src"],
                    "deviation_pct": "", "flag": "NA_SM75",
                    "measured_by": "table/na_sm75.csv", "clock_mhz": "",
                    "notes": row["notes"],
                })

    fields = ["row_id", "class", "instruction", "variant", "kind", "value",
              "unit", "cv_pct", "pipe", "prior_value", "prior_src",
              "deviation_pct", "flag", "measured_by", "clock_mhz", "notes"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"{OUT}: {len(out_rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
