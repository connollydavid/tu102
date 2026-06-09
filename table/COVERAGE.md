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
- [ ] `alu.hfma2.{lat,tput}` (fp16x2)
- [ ] `alu.dadd.{lat,tput}`, `alu.dfma.{lat,tput}` (FP64 = 1/32 FFMA expected)
- [ ] co-issue matrix: pipe binding for every row above

## SFU / conversion (M2, M4)

- [ ] `sfu.mufu.{rcp,rsq,ex2,lg2,sin,cos}.{lat,tput}`
- [ ] `cvt.f2f.{f16f32,f32f16}.{lat,tput}` — consumer: delta-net convert storm
- [ ] `cvt.i2f.*`, `cvt.f2i.*` — consumer: dequant chains

## Tensor cores (M4)

- [ ] `tensor.hmma.1688.{f16acc,f32acc}.tput` (f32acc = ½ f16acc expected)
- [ ] `tensor.imma.8816.tput`
- [ ] `tensor.ldsm.{lat,tput}` (LDSM.16 variants)

## Memory hierarchy (M2, M3)

- [ ] `mem.smem.{lat,bw}`; bank-conflict sweep `conflict{2,4,8,16,32}`
- [ ] `mem.l1.{lat,bw}` (l1hit variant)
- [ ] `mem.l2.{lat,bw}`; cliff at 6 MB
- [ ] `mem.dram.{lat,bw}` — gate: ≥570 GB/s read (85% of 672)
- [ ] `mem.tlb.{l1,l2}.{pagesize,lat}` (P-chase)
- [ ] `mem.const.{lat,bw}` (immediate vs c[] vs __constant__)
- [ ] `mem.ldg.u8.stride18.tput` — q4_0 block stride; predict 18 sectors/req
- [ ] `mem.ldg.policy.{ldcs,ldcg,ldlu}` eviction-policy probes — consumer: MMVQ
- [ ] `mem.stg.*` write paths
- [ ] `atomics.{global,shared}.{add,cas}.{lat,tput}` + contention sweeps
- [ ] `NA_SM75`: `cp.async`/LDGSTS, `ldgsts.bypass`, async barriers

## Sync / control / launch (M4)

- [ ] `sync.bar.{lat,tput}` warps-per-CTA sweep 32..1024 incl. 192
- [ ] `sync.shfl.{lat,tput}`, `sync.vote.{lat,tput}`
- [ ] `branch.divergent.{2,32}way`, `branch.predicated`
- [ ] `launch.empty_kernel.{lat}` (back-to-back, stream)
- [ ] `launch.graph_node.replay_us` — consumer: launch-bound decode (736 ns gaps)
- [ ] `launch.event.{record,query,sync}.us`
- [ ] uniform-datapath note (U-register ops)

## Interconnect — additive scope (M5)

- [ ] `x.nvlink.peer_ldg.lat` (P-chase over NVLink)
- [ ] `x.nvlink.peer_stg.bw`
- [ ] `x.nvlink.{sm,ce}.{uni,bi}.bw` size curve 4 KB–256 MB — gate: ±15% of 50 GB/s/dir
- [ ] `x.pcie.{h2d,d2h}.{pinned,pageable}.bw` per GPU
- [ ] `x.nccl.allreduce.{4k..64k}.us` 2-rank — consumer: 72.9 µs/call floor
