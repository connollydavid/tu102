# Coverage manifest

Architectural completeness checklist. A family is done when every listed row
is present in `tu102_ops.csv` as measured, or as an explicit `NA_SM75` row.
`tools/check_complete.py` (M6) diffs this manifest against the CSV.

## Core SM — ALU/FP (M1)

- [ ] `alu.ffma.{lat,tput}` — anchor: 2.0 warpinst/SM/clk
- [ ] `alu.fadd.{lat,tput}`, `alu.fmul.{lat,tput}`
- [ ] `alu.iadd3.{lat,tput}`, `alu.imad.{lat,tput}` (pipe = fma)
- [ ] `alu.lop3.{lat,tput}`, `alu.shf.{lat,tput}`, `alu.sel.{lat,tput}`
- [ ] `alu.isetp.{lat,tput}`, `alu.fsetp.{lat,tput}`
- [ ] `alu.popc.{lat,tput}`, `alu.flo.{lat,tput}`
- [ ] `alu.prmt.{lat,tput}` — byte permute; nibble/byte unpack in dequant
- [ ] `alu.idp4a.{lat,tput}` — IDP.4A.{U8,S8} (dp4a); consumer: FA-split dp4a
      variant arbitration, MMVQ q8 dot
- [ ] `alu.idiv.u32.tput` — emulated divide sequence cost
- [ ] `alu.hfma2.{lat,tput}` (fp16x2)
- [ ] `alu.dadd.{lat,tput}`, `alu.dfma.{lat,tput}` (FP64 = 1/32 FFMA expected)
- [ ] `alu.regbank.conflict` — register-file bank-conflict penalty on 3-src
      ops (operand-layout lever). Caveat: ptxas owns register allocation and
      the operand collector hides small conflicts; if two-method agreement
      is unreachable from PTX-level control, this row is descoped to a
      documented note rather than shipped `UNVERIFIED`
- [ ] co-issue matrix: pipe binding for every row above

## SFU / conversion (M2, M4)

- [ ] `sfu.mufu.{rcp,rsq,ex2,lg2,sin,cos}.{lat,tput}`
- [ ] `cvt.f2f.{f16f32,f32f16}.{lat,tput}` — consumer: delta-net convert storm
- [ ] `cvt.i2f.*`, `cvt.f2i.*` — consumer: dequant chains

## Tensor cores (M4)

- [ ] `tensor.hmma.1688.{f16acc,f32acc}.tput` (f32acc = ½ f16acc expected)
- [ ] `tensor.imma.8816.tput`
- [ ] `tensor.ldsm.{lat,tput}` (LDSM.16 variants)

## Memory hierarchy — every cache level present, plus DRAM (M2, M3)

TU102 levels, each with latency + bandwidth + geometry rows: register file
(bank conflicts — under ALU above), L0/L1 instruction caches, unified L1
data/texture cache (+ smem carveout configurations), L2, the constant path
(immediate, c[] bank, IDC), L1/L2 TLB, DRAM.

- [ ] `mem.icache.{l0,l1}.{size,line,miss_cycles}` — instruction-fetch
      hierarchy; loop-body size cliffs
- [ ] `mem.smem.{lat,bw}`; bank-conflict sweep `conflict{2,4,8,16,32}`
- [ ] `mem.l1.{lat,bw,line}` (l1hit variant); carveout sweep
      `carveout{32k,64k}` — smem/L1 split effect
- [ ] `mem.tex.lat` — texture-path read (`__ldg`/tex object); same physical
      L1 on Turing — row proves or refutes
- [ ] `mem.l2.{lat,bw}`; cliff at 6 MB
- [ ] `mem.const.{imm,cbank,idc}.lat` — constant hierarchy
- [ ] `mem.tlb.{l1,l2}.{pagesize,reach,lat}` (P-chase)
- [ ] `mem.dram.{read,write,copy}.bw` — gate: read ≥570 GB/s (85% of 672);
      `mem.dram.lat`
- [ ] `mem.ldg.u8.stride18.tput` — q4_0 block stride; predict 18 sectors/req
- [ ] `mem.ldg.policy.{ldcs,ldcg,ldlu}` eviction-policy probes — consumer: MMVQ
- [ ] `mem.stg.*` write paths
- [ ] `atomics.{global,shared}.{add,cas}.{lat,tput}` + contention sweeps
- [ ] `NA_SM75`: `cp.async`/LDGSTS, async barriers, L2 residency controls
      (sm_80+)

## Sync / control / launch (M4)

- [ ] `sync.bar.{lat,tput}` warps-per-CTA sweep 32..1024 incl. 192
- [ ] `sync.shfl.{lat,tput}`, `sync.vote.{lat,tput}`
- [ ] `branch.divergent.{2,32}way`, `branch.predicated`
- [ ] `launch.empty_kernel.{lat}` (back-to-back, stream)
- [ ] `launch.graph_node.replay_us` — consumer: launch-bound decode (736 ns gaps)
- [ ] `launch.event.{record,query,sync}.us`
- [ ] uniform-datapath note (U-register ops)

## Interconnect — additive scope (M5)

- [ ] `x.nvlink.peer_ldg.{lat,bw}` (P-chase + streaming over NVLink)
- [ ] `x.nvlink.peer_stg.bw` — read-vs-write asymmetry noted against peer_ldg
- [ ] `x.nvlink.peer_atom.{add,cas}.{lat,tput}` — native over NVLink (NA over
      PCIe); flags for hand-rolled peer exchange
- [ ] `x.nvlink.msg.{8b..4kb}` — sub-4 KB message-size efficiency curve
      (the peer-exchange lever operates at ≤20 KiB)
- [ ] `x.nvlink.fence_roundtrip.us` — peer STG + `__threadfence_system()` +
      system-scope flag poll; the hand-rolled exchange primitive (input to
      the NCCL-floor memo)
- [ ] `x.nvlink.contention.local_vs_peer` — scalar at a defined operating
      point: % degradation of local DRAM-read bandwidth while the peer
      streams at full NVLink rate; the full 2D trade surface stays in
      `data/` as a curve
- [ ] `x.nvlink.{sm,ce}.{uni,bi}.bw` size curve 4 KB–256 MB — gate: ±15% of
      50 GB/s/dir
- [ ] `x.pcie.{h2d,d2h}.{pinned,pageable}.bw` per GPU
- [ ] `x.pcie.lat.{4b..64kb}` — pinned small-transfer latency floor
      (consumer: logits D2H)
- [ ] `x.nccl.allreduce.{4k..64k}.us` 2-rank — consumer: 72.9 µs/call floor
