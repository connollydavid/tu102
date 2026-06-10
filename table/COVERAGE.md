# Coverage manifest

Architectural completeness checklist. A family is done when every listed row
is present in `tu102_ops.csv` as measured, or as an explicit `NA_SM75` row.
`tools/check_complete.py` diffs this manifest against the CSV before any
release is tagged.

## Core SM — ALU/FP

- [x] `alu.ffma.{lat,tput}` — anchor: 2.0 warpinst/SM/clk
- [x] `alu.fadd.{lat,tput}`, `alu.fmul.{lat,tput}`
- [x] `alu.iadd3.{lat,tput}`, `alu.imad.{lat,tput}` (pipe = fma)
- [x] `alu.lop3.{lat,tput}`, `alu.shf.{lat,tput}`, `alu.sel.{lat,tput}`
- [x] `alu.isetp.{lat,tput}`, `alu.fsetp.{lat,tput}`
- [x] `alu.popc.{lat,tput}`, `alu.flo.{lat,tput}`
- [x] `alu.prmt.{lat,tput}` — byte permute; nibble/byte unpack in dequant
- [x] `alu.idp4a.{lat,tput}` — IDP.4A.{U8,S8} (dp4a); consumer: FA-split dp4a
      variant arbitration, MMVQ q8 dot
- [x] `alu.idiv.u32.tput` — emulated divide sequence cost
- [x] `alu.hfma2.{lat,tput}` (fp16x2)
- [x] `alu.dadd.{lat,tput}`, `alu.dfma.{lat,tput}` (FP64 = 1/32 FFMA expected)
- [x] `alu.regbank.conflict` — **descoped to this note** (the caveat fired):
      ptxas owns register allocation, and the Harness pinning catalogue
      (bench/alu/ops.cuh — constant folding, uniform-datapath conversion,
      IADD3 strength reduction surviving carry pins and cross-coupled
      accumulators) demonstrates PTX-level control over emitted registers
      is not achievable. Operand-bank layout cannot be pinned without a
      SASS-level assembler, so two-method agreement is unreachable from
      this toolchain by construction. Revisit only if a maintained sm_75
      SASS assembler enters the toolchain
- [x] co-issue matrix: pipe binding for every row above

## SFU / conversion

- [x] `sfu.mufu.{rsq,ex2,lg2,sin,cos}.{lat,tput}` + `sfu.mufu.rcp_fadd_pair.{lat,tput}` — 15 cyc set (SIN/COS 21-cyc pairs with the FMUL range scale; RCP derived from the rcp+fadd pair after the approx-algebra deletions), quarter rate throughout
- [x] `cvt.f2f.{lat,tput}` — roundtrip pair (one-way needs an unmatchable operand type), 11.6 cyc per convert; second method (pair+FADD chain minus the FADD row) corroborates at +1 cyc/convert (cross-pipe forwarding rides the derived form). Consumer: delta-net convert storm
- [x] `cvt.i2f_f2i.{lat,tput}` — roundtrip pair 15.0 cyc per convert; second method corroborates at +1 cyc/convert. Consumer: dequant chains

## Tensor cores

- [x] `tensor.hmma.1688.tput` (variants f16acc/f32acc; the half-rate f32acc expectation is GeForce segmentation — this Quadro runs full rate, measured)
- [x] `tensor.imma.8816.tput`
- [x] `tensor.ldsm.{lat,tput}` — x1 chain 30 cyc (link = LDSM + address LOP3, construction named); x4 throughput peaks at 0.125 wi/SM/clk = 512 B/inst = exactly the 64 B/clk/SM unified-datapath ceiling: LDSM is bandwidth-bound on the smem/L1 path

## Memory hierarchy — every cache level present, plus DRAM

TU102 levels, each with latency + bandwidth + geometry rows: register file
(bank conflicts — under ALU above), L0/L1 instruction caches, unified L1
data/texture cache (+ smem carveout configurations), L2, the constant path
(immediate, c[] bank, IDC), L1/L2 TLB, DRAM.

- [x] `mem.icache.lat` — body-size sweep (cycles/op vs unrolled SASS size); cliffs bracket L0 (~32 KiB bodies) and L1I (past 64 KiB). Size/line geometry decomposition open
- [x] `mem.smem.{lat,bw}`; bank-conflict sweep `conflict{2,4,8,16,32}`
- [x] `mem.l1.lat` (l1hit 34 cyc), `mem.l1.bw` (64 B/clk/SM = the smem ceiling), `mem.l1.carveout` (the cliff moves with the 96K split)
- [x] `mem.l1.line` — fill-granularity probe (256 KiB capacity-evicted ring): strides 32/64/128 all cost the full 235-cycle miss while stride 16 costs (miss+hit)/2 and stride 8 (miss+3 hits)/4 exactly — the L1 fills 32-byte sectors with no spatial prefetch
- [x] `mem.tex.lat` — texture-path read (`__ldg`/tex object); same physical
      L1 on Turing — row proves or refutes
- [x] `mem.l2.lat` (161.5 ns; cliff binds between 5 and 8 MiB)
- [x] `mem.l2.bw` — read_cg, grid-wide over a 4 MiB footprint
- [x] `mem.const.{cbank,idc}.lat` + the bonus `mem.const.uldc.lat` (uniform datapath); immediates are decode-embedded (note, not row)
- [x] `mem.tlb.lat` — reach sweep, .cg loads; both coverage cliffs land on the Jia priors (32 MiB, ~8 GiB)
- [x] `mem.dram.bw` (variants read/write/copy) — gate passed: read 608 GB/s ≥530 (85% of the 624 GB/s P2 peak); `mem.dram.lat` 299.9 ns
- [x] `mem.ldg.u8.stride.tput` (variants stride1..128) — the stride-18 sector model reproduces the DRAM peak
- [x] `mem.ldg.policy.lat` (variants default/ldcs/ldcg/ldlu × footprint) — .cs retains in L1; the MMVQ lever is .cg on the streamed side
- [ ] `mem.stg.*` write paths
- [x] `atomics.shared.add.lat` (contention sweep binds Jia's deltas; the +17-cycle chain-method constant is ncu-corroborated as return-value wait — 83.6% short-scoreboard stall, `data/ncu-atomics-20260610/`), `atomics.global.add.lat`, `atomics.global.cas.lat`
- [x] atomics throughput rows + `atomics.shared.cas` — shared CAS 37 cyc (+12 on add); shared add sustains 0.5 warpinst/SM/clk (ATOMS.ADD, no shared RED form); global RED from one SM peaks at one warp (the L2 service path, not warps, binds)
- [x] explicit-absence rows `mem.cpasync`, `sync.asyncbar`, `mem.l2.residency` (NA_SM75 with PTX ISA locators)

## Sync / control / launch

- [x] `sync.bar.lat` warps sweep incl. the 192-thread production point
- [x] `sync.bar.tput` — a co-resident 192-thread CTA per SM leaves block 0's per-barrier cost unchanged (32.34 cyc = the solo w6 row): the barrier unit serves two concurrent CTAs without serialising
- [x] `sync.shfl.{lat,tput}`
- [x] `sync.vote.{lat,tput}` — ballot chain 16.2 cyc (VOTE.ANY; lane-shifted predicate keeps the chain off the uniform datapath); four independent chains sustain 0.96 warpinst/SM/clk
- [x] `branch.divergent` (variants 1/2/4/32way), `branch.predicated` — exactly linear serialisation; predication free
- [x] `launch.empty_kernel.{lat}` (back-to-back, stream)
- [x] `launch.graph_node.replay` — 0.906 µs/node by the 2K−K slope (the stale 736 ns anchor re-baselined: 29.9% launch share at the 16k shape)
- [x] `launch.event.{record,query,sync}`
- [x] uniform-datapath note (U-register ops)

## Interconnect — additive scope

- [x] `x.nvlink.peer_ldg.{lat,bw}` (P-chase + streaming over NVLink)
- [x] `x.nvlink.peer_stg.bw` — read-vs-write asymmetry noted against peer_ldg
- [x] `x.nvlink.peer_atom.add.lat` (541 ns, native over NVLink)
- [x] `x.nvlink.peer_atom.cas` + throughput rows — peer CAS 552-558 ns (+16 on peer add); independent non-returning adds (RED over NVLink) sustain ~800 Mop/s from a single warp, peak at w1 (more warps contend)
- [x] `x.nvlink.msg.oneway` (variants 0b..20480b) — store-burst+fence curve 1.20→1.80 µs
- [x] `x.nvlink.fence_roundtrip` (variants 0b/4096b/20480b) — litmus-checked 3.90/4.91/7.94 µs; visibility-aware composed gate (v3) passes at 4 AND 20 KiB (−4.5/−10.3%); hypothesis #2 confirmed at ≤20 KiB
- [x] `x.nvlink.peer_write_visibility` (variants 4096b/20480b) — single-warp
      first-read-after-peer-write, 0.61/2.59 µs vs the 0.32/1.14 µs
      steady-state consume rows: the formerly named ~2.3 µs residual is now
      mostly a measured row (penalty 0.29/1.45 µs), litmus-gated
- [x] `x.nvlink.contention.local_vs_peer` — scalar at a defined operating
      point: % degradation of local DRAM-read bandwidth while the peer
      streams at full NVLink rate; the full 2D trade surface stays in
      `data/` as a curve. Kernel re-gated 2026-06-10 (a 64-bit % compiled
      to a division CALL — caught by the purity sweep; superseded-revision
      rows purged, row now 509.0 GB/s, −16.4% vs unloaded)
- [ ] `x.nvlink.{sm,ce}.{uni,bi}.bw` size curve 4 KB–256 MB — gate: ±15% of
      50 GB/s/dir
- [x] `x.pcie.bw` (variants h2d/d2h × pinned/pageable × GPU) — both GPUs at x16 rates
- [x] `x.pcie.lat` (variants 4b..64kb pinned) — 4.1 µs floor (logits-D2H consumer)
- [x] `x.nccl.allreduce` (variants 4k..64k steady + cold) — env-pinned floor 21.1–28.3 µs; cold first call 5–10 ms across four fresh-communicator samples (spread-flagged); the 72.9 µs in-situ anchor retired
