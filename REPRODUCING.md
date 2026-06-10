# Reproducing the table

A replication run needs a TU102 board (any RTX 6000/8000 Quadro or TITAN
RTX; interconnect rows additionally need two boards and an NVLink
bridge), CUDA 12+ with `nvcc` on the PATH (edit `NVCC` in the Makefile),
the NVML development headers (shipped with the CUDA toolkit), Python 3,
and root for the clock lock. Expect one evening: the full sweep is under
an hour of GPU time; the rest is reading the diffs.

The table is a snapshot of one toolchain/driver pairing. A different
`nvcc` emits different SASS for the same PTX — the purity gate will tell
you when that has happened, and a row that moved under different SASS is
a finding about the toolchain, not a replication failure. Compare like
with like before comparing numbers.

## Preconditions

Every binary re-checks these and exits 2 with instructions if one fails;
setting them up front avoids piecemeal failures.

```bash
# SM clock locked on every GPU (the harness verifies 1455 MHz; TU102
# boards with different bins can edit SM_CLOCK_MHZ in bench/common/harness.cuh)
sudo nvidia-smi -lgc 1455

# CPU governor: host-domain rows (launch family, PCIe) are
# scheduler-sensitive
sudo cpupower frequency-set -g performance

# no other compute or graphics process may touch the GPUs (the harness
# refuses, it does not share); stop desktop sessions and inference
# servers first
nvidia-smi   # the process list must be empty
```

The memory clock cannot be locked on TU102 (`-lmc` is unsupported and
application clocks are deprecated on current drivers); CUDA compute work
always runs in the P2 state, and the harness samples both clock domains
per repetition and rejects contaminated reps. Bandwidth gates in the
tooling are stated against the P2 peak (624 GB/s), not the boost-state
672.

## Run

```bash
git clone https://github.com/connollydavid/tu102.git && cd tu102
make bins                          # ~5 min; edit NVCC in the Makefile if needed
bash bench/common/run_all.sh       # gate sweep + every family, 2 invocations x 2 GPUs
```

`run_all.sh` aborts on the first gate failure. It runs `check_sass.py`
on every binary before any timing (the catalogue of compiler defeats
this gate has caught is in the paper's methodology section), then every
bench twice per GPU — the between-run rule wants two independent process
invocations — and regenerates the table.

Results land in `data/results/<host>/` as append-only per-family CSVs
with a run header carrying driver, toolchain, clock, and ECC state.
Name your host directory after the rig (`t5820-2xrtx6000` is the
reference dataset).

## Compare

```bash
python3 tools/mk_table.py          # regenerate table/tu102_ops.csv from your results
git diff table/tu102_ops.csv
```

Agreement bounds are the published per-domain floors (`table/SCHEMA.md`):
0.1% for cycle-domain rows, 0.5% for bandwidths, 5% for host-domain
times. Cycle-domain rows on a locked clock should land inside the floor;
interconnect and host rows are rig-dependent by design and carry their
configuration in the variant and notes. Rows flagged `UNVERIFIED` in the
published table carry their reason in the notes column — your run may
legitimately resolve or reproduce the flag.

A replication dataset is welcome as a pull request adding
`data/results/<your-host>/` (do not edit the reference dataset) with the
host described in the PR body: board, bus topology, driver, toolchain.
