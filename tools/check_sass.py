#!/usr/bin/env python3
"""Gate: each bench kernel's timed loop contains exactly the intended SASS op.

Parses `cuobjdump -sass` output for a bench binary, locates the hot loop in
every lat_kernel/tput_kernel instantiation (the innermost backward branch),
and enforces:

  1. the primary op appears >= --min-primary times in the loop body (the
     unrolled chain), and
  2. no instruction outside primary + companions + loop control appears, and
  3. the loop body fits the L0 instruction cache budget (--l0-bytes).

Sequence ops (IDIV) are presence-only: the loop must contain the expected
sequence markers but carries no purity gate (the sequence IS multiple ops).

Exit 0 = all kernels pass; exit 1 = any failure (CI gate).
"""

import argparse
import re
import subprocess
import sys

CUOBJDUMP = "/opt/cuda-13.3/bin/cuobjdump"

# loop-control instructions every timed loop legitimately contains
CONTROL = {"IADD3", "ISETP", "BRA", "NOP"}

# Op struct name (appears in the mangled kernel symbol) -> expectation.
# primary: ops that must dominate the loop. companions: tolerated extras.
# None: presence-only sequence (no purity gate).
EXPECT = {
    "OpFFMA": {"primary": {"FFMA"}},
    "OpFADD": {"primary": {"FADD"}},
    "OpFMUL": {"primary": {"FMUL"}},
    "OpIADD3": {"primary": {"IADD3", "LOP3"}},
    "OpIMAD": {"primary": {"IMAD"}},
    "OpLOP3": {"primary": {"LOP3"}},
    "OpSHF": {"primary": {"SHF"}},
    "OpSEL": {"primary": {"SEL"}, "companions": {"ISETP"}},
    "OpISETPSEL": {"primary": {"ISETP", "SEL"}},
    "OpFSETPSEL": {"primary": {"FSETP", "SEL", "FSEL"}},
    "OpFSEL": {"primary": {"SEL", "FSEL"}, "companions": {"FSETP"}},
    "OpPOPC": {"primary": {"POPC"}},
    "OpFLO": {"primary": {"FLO"}},
    "OpPRMT": {"primary": {"PRMT"}},
    "OpIDP4A_S8": {"primary": {"IDP"}},
    "OpIDP4A_U8": {"primary": {"IDP"}},
    "OpHFMA2": {"primary": {"HFMA2"}},
    "OpDADD": {"primary": {"DADD"}},
    "OpDFMA": {"primary": {"DFMA"}},
    "OpIDIV_U32": None,
}

# Bespoke (non-template) bench kernels, matched by function-name substring.
# Same loop-detection and purity rules as EXPECT.
EXPECT_FN = {
    "smem_chase_kernel": {"primary": {"LDS"}},
    "smem_conflict_kernel": {"primary": {"LDS"}},
    # bandwidth kernels legitimately carry per-load address generation; the
    # LSU stays the bottleneck (LDS 0.5/clk vs 2/clk for the address ops)
    "smem_bw_kernel": {"primary": {"LDS"},
                       "companions": {"FADD", "FFMA", "IMAD", "LOP3", "LEA", "SHF", "MOV"}},
    "l1_chase_kernel": {"primary": {"LDG"}},
    "stride_kernel": {"primary": {"LDG"}, "min": 12,
                      "companions": {"IMAD", "LEA", "SHF", "LOP3", "MOV", "IADD3"}},
    # ptxas lowers the f16->f32 widening to HADD2 (verified): the pair is
    # F2F (narrowing) + HADD2 (widening via the half pipe)
    "cvt_f2f": {"primary": {"F2F", "HADD2"}},
    "cvt_i2f": {"primary": {"F2I", "I2F"}},
    # tput template instantiations carry the pair-fn symbol in their name
    "f2f_pair": {"primary": {"F2F", "HADD2"}},
    "i2f_pair": {"primary": {"F2I", "I2F"}},
    "mufu_ex2": {"primary": {"MUFU"}},
    "bar_kernel": {"primary": {"BAR"}, "min": 8},
}

INSTR_RE = re.compile(r"/\*([0-9a-f]+)\*/\s+((?:@!?P\d+\s+)?[A-Z][A-Z0-9.]*[^;]*)")
FUNC_RE = re.compile(r"^\s*Function : (\S+)")


def parse_functions(sass_text):
    """yield (mangled_name, [(addr_int, mnemonic_base, instr_text)])"""
    name, instrs = None, []
    for line in sass_text.splitlines():
        m = FUNC_RE.match(line)
        if m:
            if name:
                yield name, instrs
            name, instrs = m.group(1), []
            continue
        m = INSTR_RE.search(line)
        if m and name:
            text = m.group(2).strip()
            mnemonic = text.split()[1] if text.startswith("@") else text.split()[0]
            instrs.append((int(m.group(1), 16), mnemonic.split(".")[0], text))
    if name:
        yield name, instrs


def hot_loop(instrs):
    """Largest backward-branch body: (start_idx, end_idx) inclusive, or None."""
    addr_to_idx = {a: i for i, (a, _, _) in enumerate(instrs)}
    best = None
    for i, (addr, base, text) in enumerate(instrs):
        if base != "BRA":
            continue
        m = re.search(r"BRA\s+0x([0-9a-f]+)", text)
        if not m:
            continue
        tgt = int(m.group(1), 16)
        if tgt < addr and tgt in addr_to_idx:
            j = addr_to_idx[tgt]
            if best is None or (i - j) > (best[1] - best[0]):
                best = (j, i)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", help=".bin (cuobjdump is run) or .sass text file")
    ap.add_argument("--min-primary", type=int, default=64,
                    help="minimum primary-op count in the loop body")
    ap.add_argument("--l0-bytes", type=int, default=8192,
                    help="loop-body size budget (L0 i-cache fit)")
    ap.add_argument("--staging-budget", type=int, default=6,
                    help="max non-primary non-control instrs tolerated in the loop")
    args = ap.parse_args()

    if args.binary.endswith(".sass"):
        sass = open(args.binary).read()
    else:
        sass = subprocess.run([CUOBJDUMP, "-sass", args.binary],
                              capture_output=True, text=True, check=True).stdout

    failures = 0
    checked = 0
    for name, raw in parse_functions(sass):
        fn_key = next((k for k in EXPECT_FN if k in name), None)
        if fn_key is not None:
            checked += 1
            instrs = raw
            loop = hot_loop(instrs)
            label = f"fn<{fn_key}>"
            if loop is None:
                print(f"FAIL {label}: no loop found")
                failures += 1
                continue
            body = instrs[loop[0]:loop[1] + 1]
            size = instrs[loop[1]][0] - instrs[loop[0]][0] + 16
            expect = EXPECT_FN[fn_key]
            primary = expect["primary"]
            allowed = primary | expect.get("companions", set()) | CONTROL
            min_primary = expect.get("min", args.min_primary)
            n_primary = sum(1 for _, base, _ in body if base in primary)
            aliens = [base for _, base, _ in body if base not in allowed]
            if size > args.l0_bytes:
                print(f"FAIL {label}: loop body {size} B exceeds L0 budget")
                failures += 1
            elif n_primary < min_primary:
                print(f"FAIL {label}: only {n_primary} primary ops in loop")
                failures += 1
            elif len(aliens) > args.staging_budget:
                print(f"FAIL {label}: {len(aliens)} non-primary ops "
                      f"(budget {args.staging_budget}): {' '.join(sorted(set(aliens)))}")
                failures += 1
            else:
                extra = f" (+{len(aliens)} staging)" if aliens else ""
                print(f"PASS {label}: {n_primary} primary, {len(body)} instrs, {size} B{extra}")
            continue

        kind = next((k for k in ("lat_kernel", "tput_kernel", "pure_kernel",
                                 "mix_kernel") if k in name), None)
        if kind is None:
            continue
        matches = [o for o in EXPECT if o + "E" in name]
        # drop substring shadows (OpSEL inside OpISETPSEL's symbol, ...)
        matches = [o for o in matches
                   if not any(o != p and o in p and p + "E" in name for p in matches)]
        if not matches:
            continue
        if kind == "mix_kernel":
            # interleaved two-op stream: primary is the union, no sequences
            if any(EXPECT[o] is None for o in matches):
                continue
            expect = {"primary": set().union(*(EXPECT[o]["primary"] for o in matches)),
                      "companions": set().union(*(EXPECT[o].get("companions", set())
                                                  for o in matches))}
            op = "+".join(sorted(matches))
        else:
            op = max(matches, key=len)
            expect = EXPECT[op]
        checked += 1
        instrs = raw
        loop = hot_loop(instrs)
        label = f"{kind}<{op}>"
        if loop is None:
            print(f"FAIL {label}: no loop found")
            failures += 1
            continue
        body = instrs[loop[0]:loop[1] + 1]
        size = instrs[loop[1]][0] - instrs[loop[0]][0] + 16
        if size > args.l0_bytes:
            print(f"FAIL {label}: loop body {size} B exceeds L0 budget {args.l0_bytes} B")
            failures += 1
            continue
        if expect is None:
            print(f"PASS {label}: sequence op, {len(body)} instrs, {size} B (no purity gate)")
            continue
        primary = expect["primary"]
        allowed = primary | expect.get("companions", set()) | CONTROL
        n_primary = sum(1 for _, base, _ in body if base in primary)
        aliens = [base for _, base, _ in body if base not in allowed]
        # operand staging (constant-bank rematerialisation, f16x2 repacking)
        # legitimately costs a few per-iteration instructions; a miscompiled
        # chain (strength reduction, uniform-datapath conversion) costs tens
        if n_primary < args.min_primary:
            print(f"FAIL {label}: only {n_primary} primary ops "
                  f"({'/'.join(sorted(primary))}) in loop, need >= {args.min_primary}")
            failures += 1
        elif len(aliens) > args.staging_budget:
            print(f"FAIL {label}: {len(aliens)} non-primary ops in timed loop "
                  f"(budget {args.staging_budget}): {' '.join(sorted(set(aliens)))}")
            failures += 1
        else:
            extra = f" (+{len(aliens)} staging: {' '.join(sorted(set(aliens)))})" if aliens else ""
            print(f"PASS {label}: {n_primary} primary, {len(body)} instrs, {size} B{extra}")

    if checked == 0:
        print("FAIL: no bench kernels found in binary")
        return 1
    print(f"{checked} kernels checked, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
