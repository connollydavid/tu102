#!/usr/bin/env python3
"""Unit tests for project.py with hand-computed synthetic censuses.

Run: python3 tools/test_project.py  (CI validate job runs this).
Uses fixed rates rather than the live table so the expectations are exact.
"""

import sys

import project

RATES = {
    "alu.ffma.tput": {"value": "2.0", "pipe": "fma"},
    "alu.lop3.tput": {"value": "2.0", "pipe": "alu"},
    "alu.idp4a.tput": {"value": "2.0", "pipe": "fma"},
    "sfu.mufu.ex2.tput": {"value": "0.5", "pipe": "own_xu"},
}

FAIL = 0


def expect(name, got, want, tol=1e-6):
    global FAIL
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    print(f"{'PASS' if ok else 'FAIL'} {name}: got {got:.3f}, want {want:.3f}")
    if not ok:
        FAIL = 1


# 1. pure FFMA, 1 warp: 128 ops at 2.0/clk -> 64 cycles, fma-bound
r = project.project({"FFMA": 128}, 1, "none", RATES, 5.82)
expect("pure-ffma ppm", r["ppm_cycles"], 64.0)
assert r["ppm_bound"] == "pipe:fma", r["ppm_bound"]

# 2. FFMA+LOP3 50/50, 1 warp: each pipe 32 cycles; issue = 128/4 = 32;
#    PPM = 32 (any), ADD = 64 (sums both pipes)
r = project.project({"FFMA": 64, "LOP3": 64}, 1, "none", RATES, 5.82)
expect("mix ppm", r["ppm_cycles"], 32.0)
expect("mix add", r["add_cycles"], 64.0)

# 3. issue cap: 4 pipes' worth of work cannot beat 4/SM/clk —
#    256 total insts at 1 warp -> issue floor 64 even if pipes say 32 each
r = project.project({"FFMA": 64, "LOP3": 64, "MUFU": 16, "IDP": 64}, 1,
                    "none", RATES, 5.82)
expect("issue floor", r["per_resource"]["issue"], (64 + 64 + 16 + 64) / 4.0)

# 4. dram-bound: 16 LDG.E.U8 per warp = 16*32 B at 5.82 B/clk/SM
#    -> mem = 512/5.82 = 87.97; FFMA 8 ops -> 4 cycles; PPM = mem
r = project.project({"LDG.E.U8": 16, "FFMA": 8}, 1, "dram", RATES, 5.82)
expect("dram ppm", r["ppm_cycles"], 16 * 32 / 5.82)
assert r["ppm_bound"] == "mem", r["ppm_bound"]

# 5. smem: 64 LDS.32 per warp at 0.5 inst/clk -> 128 cycles
r = project.project({"LDS.32": 64}, 1, "none", RATES, 5.82)
expect("smem cycles", r["per_resource"]["smem"], 64 / 0.5)

# 6. warps scale linearly
r1 = project.project({"FFMA": 128}, 1, "none", RATES, 5.82)
r8 = project.project({"FFMA": 128}, 8, "none", RATES, 5.82)
expect("warp scaling", r8["ppm_cycles"], 8 * r1["ppm_cycles"])

print("test_project:", "FAIL" if FAIL else "PASS")
sys.exit(FAIL)
