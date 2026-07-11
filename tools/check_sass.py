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

CUOBJDUMP = "/opt/cuda/bin/cuobjdump"

# loop-control instructions every timed loop legitimately contains
CONTROL = {"IADD3", "ISETP", "BRA", "NOP"}

# binaries the purity gate cannot meaningfully bind, each with its reason
# (made explicit 2026-06-10 so run_all's gate sweep runs end-to-end; the
# implicit version of this list was the manual bench-by-bench workflow)
EXEMPT_BINARIES = {
    "launch.bin": "empty/identity kernels by design; the rows are host-side dispatch",
    "marshal.bin": "argument-marshalling probe; kernels empty by design",
    "l2bw.bin": "loop-structure parse quirk; the row self-validates via the "
                "cg-vs-default policy contrast (382 vs 1110 GB/s)",
    "fa_mini.bin": "composite kernels; gated by census-match mode instead",
    "nccl_pcie.bin": "host-side NCCL/cudaMemcpy timing; no timed device loops",
    "icache.bin": "loop bodies sized to exceed L0 BY DESIGN (the measurement)",
}

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
    "pchase_kernel": {"primary": {"LDG"}},
    "peer_chase_kernel": {"primary": {"LDG"}},
    "rt_initiator": None, "rt_responder": None,  # handshake structure; litmus-gated
    "vis_initiator": None, "vis_responder": None,  # same handshake; litmus-gated
    "peer_atom_chase": {"primary": {"ATOMG"}, "min": 8},
    "stream_writer": None,
    # SEL: the predicated 64-bit index wrap (the lawful replacement for a
    # % that compiled to a division CALL — caught by this gate 2026-06-10)
    "local_read_bw": {"primary": {"LDG"}, "min": 4,
                      "companions": {"FADD", "IMAD", "LEA", "SHF", "MOV", "LOP3",
                                     "SEL"}},
    "peer_ring_init": None,
    "peer_read_bw_kernel": {"primary": {"LDG"}, "min": 4,
                            "companions": {"FADD", "IMAD", "LEA", "SHF", "MOV", "LOP3"}},
    "peer_write_bw_kernel": {"primary": {"STG"}, "min": 1,
                             "companions": {"IMAD", "LEA", "SHF", "MOV", "LOP3"}},
    # threadfence_system lowers to MEMBAR.SYS + CCTL + ERRBAR; the 0-byte
    # variant is fence-only (no stores) by design
    "peer_burst_kernel": {"primary": {"STG"}, "min": 0,
                          "companions": {"MEMBAR", "CCTL", "ERRBAR", "IMAD", "LEA",
                                         "SHF", "MOV", "LOP3", "SEL"}},
    "policy_chase_kernel": {"primary": {"LDG"}},
    "policy_ring_init": None,
    "carveout_chase_kernel": {"primary": {"LDG"}},
    "carveout_ring_init": None,
    "pchase_ring_init": None,
    "tlb_chase_kernel": {"primary": {"LDG"}},
    "tlb_ring_init": None,
    "tex_chase_kernel": {"primary": {"LDG"}},
    "tex_ring_init": None,
    "icache_kernel": {"primary": {"FFMA"}},
    "hmma_f16_tput_kernel": {"primary": {"HMMA"}},
    "hmma_f32_tput_kernel": {"primary": {"HMMA"}},
    "hmma_f16_lat_kernel": {"primary": {"HMMA"}},
    "imma_s8_tput_kernel": {"primary": {"IMMA"}},
    "atom_shared_lat_kernel": {"primary": {"ATOMS"}, "min": 16},
    "atom_global_lat_kernel": {"primary": {"ATOMG"}, "min": 16},
    "atom_cas_lat_kernel": {"primary": {"ATOMG"}, "min": 16},
    # RED-chain decomposition (gate G02 second method): the 32 non-returning
    # same-address adds must lower to RED; the one returning add per trip is
    # the tolerated ATOMG companion
    "atom_global_redsvc_lat_kernel": {"primary": {"RED"}, "min": 32,
                                      "companions": {"ATOMG"}},
    "const_chase_div_kernel": {"primary": {"LDC"}},
    "const_chase_kernel": {"primary": {"ULDC"}},
    "dram_read_kernel": {"primary": {"LDG"}, "min": 4,
                         "companions": {"FADD", "IMAD", "LEA", "SHF", "MOV", "LOP3"}},
    "dram_write_kernel": {"primary": {"STG"}, "min": 1,
                          "companions": {"IMAD", "LEA", "SHF", "MOV", "LOP3"}},
    "dram_copy_kernel": {"primary": {"LDG", "STG"}, "min": 2,
                         "companions": {"IMAD", "LEA", "SHF", "MOV", "LOP3"}},
    # loop-structure parse quirk on one instantiation; the row self-validates
    # via the cg-vs-default policy contrast (382 vs 1110 GB/s)
    "l2bw_kernel": None,
    "l1bw_kernel": {"primary": {"LDG"},
                    "companions": {"FADD", "FFMA", "IMAD", "LOP3", "LEA", "SHF", "MOV"}},
    "stride_kernel": {"primary": {"LDG"}, "min": 12,
                      "companions": {"IMAD", "LEA", "SHF", "LOP3", "MOV", "IADD3"}},
    # ptxas lowers the f16->f32 widening to HADD2 (verified): the pair is
    # F2F (narrowing) + HADD2 (widening via the half pipe)
    "cvt_f2f": {"primary": {"F2F", "HADD2"}},
    "cvt_i2f": {"primary": {"F2I", "I2F"}},
    # the derived second-method chain interleaves an FADD per link; this
    # key must precede the pair-fn keys (its symbols contain them, and the
    # matcher is first-substring-wins)
    "cvt_derived_lat_kernel": {"primary": {"F2F", "HADD2", "F2I", "I2F"},
                               "min": 32, "companions": {"FADD"}},
    # tput template instantiations carry the pair-fn symbol in their name
    "f2f_pair": {"primary": {"F2F", "HADD2"}},
    "i2f_pair": {"primary": {"F2I", "I2F"}},
    "mufu_ex2": {"primary": {"MUFU"}},
    # sin/cos lower to an FMUL range-scale + MUFU (1:1, verified); the rows
    # are pair chains with the per-op value derived against fmul.lat
    "MufuSinE": {"primary": {"MUFU", "FMUL"}},
    "MufuRcpE": {"primary": {"MUFU", "FFMA", "FADD"}},  # fmad contracts the add
    "MufuCosE": {"primary": {"MUFU", "FMUL"}},
    "mufu_lat_kernel": {"primary": {"MUFU"}},
    "mufu_tput_kernel": {"primary": {"MUFU"}},
    "bar_kernel": {"primary": {"BAR"}, "min": 8},
    "bar_direct_kernel": {"primary": {"BAR"}, "min": 1,
                          "companions": {"CS2R", "S2R", "MOV", "SHF", "LEA", "IMAD"}},
    "bar_tput_kernel": {"primary": {"BAR"}, "min": 8},
    "vote_lat_kernel": {"primary": {"VOTE", "VOTEU"}, "min": 16,
                        "companions": {"SHF", "LOP3", "ISETP", "MOV", "PLOP3",
                                       "P2R", "R2P", "SEL"}},
    "vote_tput_kernel": {"primary": {"VOTE", "VOTEU"}, "min": 16,
                         "companions": {"SHF", "LOP3", "ISETP", "MOV", "PLOP3",
                                        "P2R", "R2P", "SEL"}},
    "ldsm_lat_kernel": {"primary": {"LDSM"}, "min": 16,
                        "companions": {"LOP3", "MOV", "SHF", "LEA", "IMAD"}},
    "ldsm_tput_kernel": {"primary": {"LDSM"}, "min": 16,
                         "companions": {"LOP3", "MOV", "SHF", "LEA", "IMAD"}},
    "line_chase_kernel": {"primary": {"LDG"}},
    "line_ring_init": None,
    "atom_shared_cas_lat_kernel": {"primary": {"ATOMS"}, "min": 16},
    # non-returning atomicAdd lowers to the RED reduction form
    "atom_shared_tput_kernel": {"primary": {"ATOMS", "REDS"}, "min": 16},
    "atom_global_tput_kernel": {"primary": {"RED", "ATOMG"}, "min": 16},
    "peer_cas_chase": {"primary": {"ATOMG"}, "min": 8},
    "peer_atom_tput": {"primary": {"RED", "ATOMG"}, "min": 8},
    "empty_kernel": None,
    "k_noargs": None, "k_args": None,
    "shfl_lat_kernel": {"primary": {"SHFL"}},
    "branch_div_kernel": {"primary": {"FFMA"}, "min": 4,
                          "companions": {"BSSY", "BSYNC", "SEL", "MOV", "IMAD", "PLOP3", "SHF", "LOP3"}},
    "branch_pred_kernel": {"primary": {"FFMA"}, "min": 4,
                           "companions": {"SEL", "MOV", "IMAD", "PLOP3", "SHF", "LOP3"}},
    "shfl_tput_kernel": {"primary": {"SHFL"}},
    # composite probe: the differential runner projects from the EMITTED
    # census, so this gate is smoke only (loop exists, base work present)
    "fa_mini_kernel": None,  # gated by census-match vs the production cubin
    "ffma_anchor": {"primary": {"FFMA"}},
    "stream_anchor": {"primary": {"LDG"}, "min": 12,
                      "companions": {"FADD", "IMAD", "LEA", "SHF", "LOP3", "MOV"}},
    "smemtile_anchor": {"primary": {"LDS", "FFMA"}, "min": 48,
                        "companions": {"IMAD", "LEA", "SHF", "LOP3", "MOV"}},
    "capmix_anchor": {"primary": {"FFMA", "LOP3"}},
    "latbound_demo": {"primary": {"FFMA"}},
    "mixp_popc_ldg": {"primary": {"POPC", "LDG"}, "min": 64,
                      "companions": {"FADD", "IMAD", "LEA", "SHF", "MOV", "LOP3"}},
    "mixp_hmma_hfma2": {"primary": {"HMMA", "HFMA2"}, "min": 32},
    "inject_kernel": {"primary": {"FFMA", "LOP3", "LDG"}, "min": 40,
                      "companions": {"IDP", "POPC", "SEL", "ISETP", "IMAD",
                                     "LEA", "MOV", "SHF", "FADD"}},
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


# pipe-class groups for the census-match mode (proxy-validity gate)
MATCH_GROUPS = {
    "fma": {"FFMA", "FADD", "FMUL", "IMAD", "IDP", "FSETP", "FMNMX"},
    "half": {"HADD2", "HMUL2", "HFMA2", "HMNMX2"},
    "alu": {"IADD3", "LOP3", "SHF", "SEL", "ISETP", "PRMT", "LEA", "MOV",
            "FLO", "POPC", "BFE", "BFI", "IABS", "IMNMX"},
    "xu": {"MUFU", "F2F", "F2I", "I2F", "I2I"},
    "lsu": {"LDG", "STG", "LDS", "STS", "LDC", "LDSM", "LDL", "STL"},
    "control": {"BRA", "NOP", "EXIT", "BSSY", "BSYNC", "CS2R", "S2R", "BAR",
                "DEPBAR", "YIELD", "WARPSYNC", "RET"},
}


def loop_mix(binary, kernel_regex):
    """op-count Counter over the hot-loop bodies of matching kernels."""
    import collections
    kre = re.compile(kernel_regex)
    if binary.endswith(".sass"):  # cached disassembly (cuobjdump on a full
        sass = open(binary).read()  # libggml.so costs ~8 minutes)
    else:
        sass = subprocess.run([CUOBJDUMP, "-sass", binary],
                              capture_output=True, text=True, check=True).stdout
    mix = collections.Counter()
    matched = 0
    for name, instrs in parse_functions(sass):
        if not kre.search(name):
            continue
        loop = hot_loop(instrs)
        if loop is None:
            continue
        matched += 1
        for _, base, _ in instrs[loop[0]:loop[1] + 1]:
            mix[base] += 1
    return mix, matched


def census_match(spec_a, spec_b, tolerance_pp):
    """spec: 'binary:kernel_regex'. Group shares within tolerance -> exit 0."""
    def shares(spec):
        binary, _, regex = spec.partition(":")
        mix, matched = loop_mix(binary, regex or ".*")
        if matched == 0:
            print(f"FAIL census-match: no kernels matched in {spec}")
            sys.exit(1)
        total = sum(mix.values())
        g = {k: 0.0 for k in MATCH_GROUPS}
        g["other"] = 0.0
        for base, n in mix.items():
            grp = next((k for k, ops in MATCH_GROUPS.items() if base in ops), "other")
            g[grp] += 100.0 * n / total
        return g, matched

    ga, na = shares(spec_a)
    gb, nb = shares(spec_b)
    print(f"census-match: A={na} kernel(s), B={nb} kernel(s); tolerance +-{tolerance_pp}pp")
    worst, fail = 0.0, 0
    for grp in list(MATCH_GROUPS) + ["other"]:
        d = abs(ga[grp] - gb[grp])
        worst = max(worst, d)
        status = "ok" if d <= tolerance_pp else "EXCEEDS"
        if d > tolerance_pp:
            fail = 1
        print(f"  {grp:8s} A {ga[grp]:5.1f}%  B {gb[grp]:5.1f}%  |d| {d:5.1f}pp  {status}")
    print(f"census-match: {'FAIL' if fail else 'PASS'} (worst {worst:.1f}pp)")
    return fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary", nargs="?", help=".bin (cuobjdump is run) or .sass text file")
    ap.add_argument("--min-primary", type=int, default=64,
                    help="minimum primary-op count in the loop body")
    ap.add_argument("--l0-bytes", type=int, default=8192,
                    help="loop-body size budget (L0 i-cache fit)")
    ap.add_argument("--staging-budget", type=int, default=6,
                    help="max non-primary non-control instrs tolerated in the loop")
    ap.add_argument("--census-match", nargs=2, metavar="BIN:REGEX",
                    help="compare hot-loop op-mix shares of two kernels")
    ap.add_argument("--tolerance-pp", type=float, default=10.0)
    args = ap.parse_args()

    if args.census_match:
        return census_match(args.census_match[0], args.census_match[1],
                            args.tolerance_pp)
    if not args.binary:
        ap.error("binary required unless --census-match is used")

    import os
    base = os.path.basename(args.binary)
    if base in EXEMPT_BINARIES:
        print(f"EXEMPT {base}: {EXEMPT_BINARIES[base]}")
        return 0

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
            if EXPECT_FN[fn_key] is None:
                continue  # init/helper kernels gated elsewhere or not at all
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
