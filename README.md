# TU102 (sm_75) microarchitecture characterisation

Instruction-level latency/throughput tables, memory-hierarchy measurements, and
interconnect (NVLink/PCIe/NCCL) characterisation for the NVIDIA TU102 GPU
(Turing, compute capability 7.5), measured on 2× Quadro RTX 6000 with an NV2
NVLink bridge. In the style of Agner Fog's x86 instruction tables, with the
methodology of Jia et al.'s Volta/Turing dissections.

**Status: measured; audit in progress.** 264 rows across every family:
ALU with contention-probed pipe bindings, the full memory hierarchy
(smem/L1/L2/DRAM/TLB/constant/instruction caches, carveout, policies,
atomics), tensor cores, SFU, shuffle/branch/barrier, the launch family,
and the NVLink/PCIe/NCCL interconnect — every row gated on locked clocks
and SASS-verified loops, every flag carrying its investigation. All three
pre-registered hypotheses (`paper/` section 4) carry recorded outcomes: a
fired kill criterion, a confirmation reached through two documented gate
remediations (the second supplied by a first-read-after-peer-write
visibility row), and two published refutations. The manifest auditor (`tools/check_complete.py`, CI-gated)
holds the open remainder explicit in `table/COVERAGE.md`. The paper and
v1.0 close the work.

## What this will contain

- `table/tu102_ops.csv` — one row per (op, variant, metric): latency in cycles,
  reciprocal throughput, pipe binding, bandwidth, with per-row provenance
  (`measured_by` → bench + results line), priors from published work
  (`prior_src` → bib locator), and deviation flags. Features absent on sm_75
  (e.g. `cp.async`) appear as explicit `NA_SM75` rows, not missing keys.
- `bench/` — self-contained CUDA microbenchmarks, one family per directory.
  Every binary gates on locked clocks and verifies its own SASS.
- `tools/` — SASS census, table generation, and a projection model
  (op-mix × table → issue-cycle estimate) verified against reference kernels
  to ±20%.
- `data/results/<host>/` — append-only raw run CSVs. The pristine reference
  dataset is `t5820-2xrtx6000` (Xeon W-2140B, both GPUs PCIe 3.0 x16).
- `paper/` — LaTeX write-up (arXiv-ready). `PAPER.md` at the repository root
  is a generated GitHub-readable mirror (`make paper-md`, via pandoc); the
  `.tex` is the source of truth. Prose is British English; code identifiers
  and named APIs stay US English ASCII.

## Hardware

| | |
|---|---|
| GPU | 2× Quadro RTX 6000 (TU102, 72 SMs, 24 GB GDDR6, ~672 GB/s) |
| Interconnect | NVLink NV2 (2 links, ~50 GB/s/dir aggregate), PCIe 3.0 x16 per GPU |
| Clock policy | SM locked at 1455 MHz for all cycle-domain rows; memory clock locked for bandwidth rows; ECC disabled (recorded per run) |
| Toolchain | CUDA 13.3 (`nvcc -O2 -arch=sm_75`), driver 610.43.02 |

Scope: compute path only. RT cores (not reachable from CUDA), NVENC/NVDEC,
and the graphics pipe are out of scope — stated, not silent. The table is a
**snapshot** of this toolchain/driver pairing, not a living document; run
headers carry the exact versions, and the append-only results layout admits
later datasets under newer toolchains.

## Methodology lineage

Format and intent follow Agner Fog's instruction tables [fog2025instruction].
Latency measurement uses dependent-chain timing and fine-grained pointer-chase
[wong2010demystifying, mei2017dissecting]. Per-row priors are taken primarily
from Jia et al.'s T4 dissection [jia2019turing], with the Volta report
[jia2018volta] and the NVIDIA Turing whitepaper as secondary sources. Full
citations in `paper/references.bib`.

## Reproducing

`REPRODUCING.md` is the fresh-clone runbook: lock the clocks, run
`bench/common/run_all.sh`, regenerate the table, compare within the
per-domain floors in `table/SCHEMA.md`. Replication datasets are welcome
as pull requests adding `data/results/<your-host>/`.

## Citing

See `CITATION.cff` (GitHub's "Cite this repository" button). Until the paper
is released, cite the repository directly.

## License

- Code (`bench/`, `tools/`, `Makefile`): MIT — see `LICENSE`.
- Table, data, and paper (`table/`, `data/`, `paper/`): CC BY 4.0 — see
  `LICENSE-CC-BY-4.0`.

## AI assistance

The microbenchmark harness, analysis tooling, and manuscript drafts were
developed with the assistance of Claude (model `claude-fable-5`, Anthropic),
used via Claude Code. All measurements were executed, validated, and
interpreted on the author's hardware under the author's direction; the author
takes sole responsibility for the content.
