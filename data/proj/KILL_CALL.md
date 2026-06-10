# Decision record: FA-variant arbitration by projection (registered rule, paper §4.1)

Date: 2026-06-10. Inputs: factorial differential bound (this directory),
fa_mini decision pairs (4 invocations, 2 GPUs, cv ≤ 0.04%), loop-restricted
censuses, table @ 74 rows (all measured inputs; no defaulted parameters in
binding terms).

## Measured and projected

| pair | measured Δ | PPM Δ | ADD Δ |
|---|---|---|---|
| base → dp4a | +64.8 cyc (+18.9%) | −10.0 cyc (−3.9%) | +27.4 |
| base → staged | +403.1 cyc (+117.5%) | +318.0 cyc (+123.3%) | +441.9 |
| dp4a → staged | +338.3 cyc (+82.9%) | +328.0 cyc (+132.3%) | +414.5 |

Factorial PPM differential error bound (in-band, non-POPC pairs, 95th
percentile): 27.8%. Proxy validity: base passes the binding census gate at
worst 9.6pp vs the production single-warp cubin (dp4a variant 4.5pp; the
staged variant, which has no production analogue, sits at 18.9pp).

## Rule application

The registered rule: projection arbitrates only if the measured PPM
differential error bound is below the projected inter-variant delta.

- **dp4a pair: KILL FIRES.** Bound 27.8% versus a projected delta of 3.9%
  — and the projection's sign is wrong (both variants project issue-bound;
  reality charges fma-pipe coupling for IDP.4A + FFMA epilogue + IMAD
  addressing sharing one pipe). The dp4a-versus-base arbitration proceeds
  by implement-both-and-measure, the registered fallback.
- **staged pair: projection arbitrates.** Bound 27.8% versus a projected
  delta of 123.3%; projection and measurement agree within 5%. At the
  miniature's amortisation ratio (4 staged words per 36 dot reads), the
  staged variant loses decisively. Caveat for the memo: the production
  kernel's K-reuse ratio differs, and the conclusion is stated as
  ratio-dependent; the smem-bound projection mechanism itself is validated.

## Consequences

The Memory-hierarchy span's FA arbitration memo inherits this decision:
dp4a variant cost comes from implementation-and-measurement, not
projection; staged-variant analysis may use the (validated) smem-bound
projection with the ratio caveat. The remaining spans continue as the
publication programme per the kill criterion's registered consequence.
