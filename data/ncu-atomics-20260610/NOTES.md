# ncu corroboration — the atomics chain-method constant

2026-06-10. Corroboration-only (the published rows remain clock64-chain
measurements); Nsight Compute 2026.2.0.0 (build 37790515), driver
610.43.02, CUDA 13.3. Run as root from a disposable copy of the repo so
profiled (replayed) launches cannot append to the reference dataset;
`--clock-control none` keeps the harness's own 1455 MHz SM lock and
gates in force (the ncu "without fixed frequencies" warning refers to
ncu not controlling clocks itself).

## Question

`atomics.shared.add.lat` measures 25.0 cycles uncontended against Jia's
8-cycle T4 prior, with the contention deltas (+2/+4/.../+32 for 2- to
32-way) reproducing the prior exactly. The published interpretation: a
constant ~17-cycle offset attributable to the return-value chain this
method times and the prior's stall-counting method does not. Does a
stall-attribution profile support that placement?

## Command

```bash
sudo /opt/cuda-13.3/bin/ncu --clock-control none \
  --kernel-name 'regex:atom_shared_lat_kernel' --launch-skip 4 --launch-count 2 \
  --metrics sm__cycles_active.avg,smsp__inst_executed.sum,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct,\
smsp__warp_issue_stalled_selected_per_warp_active.pct,\
smsp__warp_issue_stalled_not_selected_per_warp_active.pct \
  ./bench/mem/atomics.bin --dev 0
```

## Result (both profiled launches identical to 0.01%)

| metric | value |
|---|---|
| `sm__cycles_active.avg` | 45,526.7 |
| `smsp__inst_executed.sum` | 134,192 |
| stalled, short scoreboard | 83.57% |
| stalled, wait (fixed-latency dependency) | 12.06% |
| selected (issuing) | 4.09% |
| stalled, long scoreboard | 0% |

## Reading

Shared-memory return values ride the short scoreboard on Turing. 83.6%
of the uncontended chain's active cycles are short-scoreboard stalls —
about 21 of the 25 cycles per atomic — with ~1 cycle issuing and ~3 in
the loop's fixed-latency control chain. The method constant is therefore
a property of the return-value dependency, not of a slower atomic unit:
the chain method waits for the returned old value before it can issue
the next operand, and that wait is exactly where the excess over the
stall-counting prior sits. The contention deltas, which both methods
agree on, ride on top of this constant.
