---
author:
- |
  David Connolly  
  `david@connol.ly`
bibliography: references.bib
date: Draft — 2026-06-10
title: |
  **An Instruction-Level Characterisation of the NVIDIA TU102 GPU  
  Latency, Throughput, Memory Hierarchy, and NVLink Interconnect on
  sm_75**
---

# Introduction

# Related Work

# Methodology

# Registered hypotheses

Three questions in this work are decision-grade: their answers change
what the consuming inference project builds next. Each is registered
here — hypothesis, operationalisation, predictions, decision rule —
before the corresponding data is gathered, with this repository’s commit
history as the receipt. The remaining verification gates in the paper
are ordinary methodology checks and are deliberately not elevated to
this status.

## Issue-coupled costs and differential projection

This hypothesis was formed after the pipe-binding measurements (which
placed `IDP.4A` on the fma pipe, contending with FFMA issue slots) and
is registered before any differential-projection data exists.

*Projected cost deltas between kernel variants that shift work onto a
contended issue pipe carry larger projection error than deltas that
shift the same work onto an uncontended pipe; under a purely additive
cost model the excess swallows the flash-attention variant-arbitration
margin.*

Operationalisation: one synthetic base kernel; graded injections of FFMA
and `IDP.4A` (fma pipe, contended), `LOP3` (alu pipe, uncontended at the
operating point), and `POPC` (separate quarter-rate unit); injection
sizes chosen so true deltas land in the 5–25% band, the arbitration
operating range. Two cost models are compared: *naive additive* (op
count $`\times`$ reciprocal throughput, summed) and *per-pipe max*
(per-pipe issue demand, maximum across pipes, capped at the
4-per-SM-cycle issue rate).

Registered predictions: (i) additive-model differential error is larger
for fma-pipe injections than for alu-pipe injections; (ii) the
per-pipe-max model removes most of that excess; (iii) given (ii), the
per-pipe-max differential error bound discriminates deltas of 10% and
above.

Decision rule: the flash-attention variant arbitration proceeds by
projection only if the measured per-pipe-max differential error bound is
below the projected inter-variant delta; otherwise the projection route
is abandoned for that decision and both variants are implemented and
measured.

## A hand-rolled NVLink exchange against the NCCL call floor

Registered before any interconnect measurement exists.

*A peer-store-plus-fence exchange primitive over NVLink completes a
two-GPU exchange of at most 20 KiB in less than half the steady-state
NCCL all-reduce per-call time at the same size.*

Operationalisation: the primitive is a peer `STG`,
`__threadfence_system()`, and a system-scope flag poll, measured
end-to-end and litmus-tested for correctness (message passing with a
data check). The comparator is a two-rank NCCL all-reduce over 4–64 KiB
with `NCCL_ALGO`, `NCCL_PROTO`, and channel count pinned and recorded,
taken at steady state (cold-channel rows are separate). Before the
comparison is made, a composed prediction of the primitive’s round trip
from its constituent table rows (peer-store latency, fence cost, poll
read) must agree with the end-to-end measurement within a stated
tolerance — the interconnect analogue of the projection gate.

Decision rule: the deterministic peer-exchange mechanism in the
consuming project proceeds only if the validated primitive beats half
the freshly re-measured NCCL per-call floor at 20 KiB and below;
otherwise the mechanism is recorded as not viable on this fabric.

## Dispatch-cost composition and the decode ceiling

Registered before any launch-family measurement exists.

*The launch-family rows (empty-kernel dispatch, graph-node replay, event
costs) compose to predict the per-token host dispatch overhead of a
production-shaped decode graph within $`\pm20\%`$, and the graph-replay
rows predict the decode-rate ceiling that full capture would unlock.*

Operationalisation: per-token dispatch overhead is node count $`\times`$
the relevant per-node row, compared against a profiler baseline measured
freshly on the same host and driver — prior-session anchors are
re-measured, never reused. The ceiling prediction divides measured
compute time by the sum of compute time and predicted captured dispatch
time.

Decision rule: the consuming project’s capture work is prioritised by
the predicted ceiling; a composed prediction outside $`\pm20\%`$ blocks
the use of these rows for that prioritisation and is published as a
model failure.

# SM Pipelines

# Memory Hierarchy

# Interconnect

# Worked Examples

# Conclusion

# Acknowledgements

The microbenchmark harness, analysis tooling, and manuscript drafts were
developed with the assistance of Claude (model `claude-fable-5`,
Anthropic), used via Claude Code; the repository’s commit trailers
record this assistance at commit granularity. All measurements were
executed, validated, and interpreted on the author’s hardware under the
author’s direction; the author takes sole responsibility for the
content.
