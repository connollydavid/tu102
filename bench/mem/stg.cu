// mem.stg.* write paths (gate G00). Checkpointing and bank writes in the
// megakernel design are STG-bound; this bench measures the store side the
// ldg rows already cover for loads.
//   mem.stg.dram.bw        coalesced float4 STG stream at 512 MiB (far
//                          beyond the 6 MB L2); the same physical quantity
//                          as mem.dram.bw write measured by independent
//                          code (the gate G01 second method).
//   mem.stg.policy.tput    st.global.{wb,cg,cs,wt} against the plain store
//                          on the same 512 MiB DRAM-bound stream (grid
//                          stride). One inline-PTX store per variant; the
//                          five kernels differ only in the policy token.
//                          The L2-resident (2 MiB) operating point was
//                          probed and rejected: it reproduces mem.l2.bw's
//                          irreducible few-percent between-run spread
//                          (every policy lands ~1.37 TB/s there, i.e. all
//                          store policies still write-allocate in the L2).
// Bandwidth rows are wall-clock quantities (cudaEvent, >=20 ms regions,
// 0.5% between-run floor). The cudaMemsetAsync fill timing at the end is a
// verification instrument for mem.dram.bw write (gate G01), never a row.
#include "../common/harness.cuh"

namespace tu102 {

constexpr size_t STREAM_FLOAT4 = 1ull << 25;  // 512 MiB of float4
constexpr size_t STREAM_BYTES = STREAM_FLOAT4 * 16;
constexpr const char* SRC = "bench/mem/stg.cu";

// block-tiled addressing (each block owns a contiguous tile; a warp writes
// 32 consecutive float4 = two 128 B lines per instruction), deliberately
// different loop structure from drambw.cu's grid-stride write kernel
__global__ void stg_stream_kernel(unsigned iters, float4* buf, float v) {
    size_t per_block = STREAM_FLOAT4 / gridDim.x;
    float4* base = buf + (size_t)blockIdx.x * per_block;
    float4 val = make_float4(v, v, v, v);
    for (unsigned t = 0; t < iters; t++)
        for (size_t i = threadIdx.x; i < per_block; i += blockDim.x)
            base[i] = val;
}

enum Pol { DEF = 0, WB = 1, CG = 2, CS = 3, WT = 4 };

template <int P>
__device__ __forceinline__ void st_pol(float4* p, float x) {
    if (P == WB)
        asm volatile("st.global.wb.v4.f32 [%0], {%1,%1,%1,%1};" ::"l"(p), "f"(x)
                     : "memory");
    else if (P == CG)
        asm volatile("st.global.cg.v4.f32 [%0], {%1,%1,%1,%1};" ::"l"(p), "f"(x)
                     : "memory");
    else if (P == CS)
        asm volatile("st.global.cs.v4.f32 [%0], {%1,%1,%1,%1};" ::"l"(p), "f"(x)
                     : "memory");
    else if (P == WT)
        asm volatile("st.global.wt.v4.f32 [%0], {%1,%1,%1,%1};" ::"l"(p), "f"(x)
                     : "memory");
    else
        asm volatile("st.global.v4.f32 [%0], {%1,%1,%1,%1};" ::"l"(p), "f"(x)
                     : "memory");
}

// grid-stride full-buffer pass per iteration (drambw.cu's write shape, so
// the default variant also re-measures that row's addressing pattern)
template <int P>
__global__ void stg_policy_kernel(unsigned iters, float4* buf, float x) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (unsigned t = 0; t < iters; t++)
        for (size_t i = tid; i < STREAM_FLOAT4; i += total)
            st_pol<P>(buf + i, x);
}

}  // namespace tu102

int main(int argc, char** argv) {
    using namespace tu102;
    Run r = harness_init(argc, argv, "stg");
    float4* d_buf;
    TU102_CUDA_CHECK(cudaMalloc(&d_buf, STREAM_BYTES));
    TU102_CUDA_CHECK(cudaMemset(d_buf, 0x11, STREAM_BYTES));
    const int blocks = 512, threads = 256;

    auto time_ms = [&](auto launch, unsigned it) {
        cudaEvent_t e0, e1;
        TU102_CUDA_CHECK(cudaEventCreate(&e0));
        TU102_CUDA_CHECK(cudaEventCreate(&e1));
        TU102_CUDA_CHECK(cudaEventRecord(e0));
        launch(it);
        TU102_CUDA_CHECK(cudaEventRecord(e1));
        TU102_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return (double)ms;
    };
    auto measure = [&](const char* row, const char* variant,
                       double bytes_per_iter, auto launch, const char* notes) {
        unsigned iters = 1;
        while (time_ms(launch, iters) < 20.0) iters *= 2;
        auto vals = run_reps(r, [&] {
            double ms = time_ms(launch, iters);
            return bytes_per_iter * iters / (ms * 1e-3) / 1e9;
        });
        report_row(r, "mem", row, "bandwidth", variant, median(vals), "GB/s",
                   cv_pct(vals), (int)vals.size(), (int)r.rejected_total, SRC,
                   notes, &vals);
    };

    measure("mem.stg.dram.bw", "", (double)STREAM_BYTES, [&](unsigned it) {
        stg_stream_kernel<<<blocks, threads>>>(it, d_buf, 1.5f);
    }, "coalesced float4 STG stream; 512 MiB footprint; block-tiled code "
       "independent of drambw.cu (second method for mem.dram.bw write)");

    auto pol = [&](const char* variant, auto launch, const char* notes) {
        measure("mem.stg.policy.tput", variant, (double)STREAM_BYTES, launch,
                notes);
    };
    pol("default", [&](unsigned it) {
        stg_policy_kernel<DEF><<<blocks, threads>>>(it, d_buf, 1.5f);
    }, "plain st.global.v4.f32; grid-stride 512 MiB stream; STG.E.128.SYS in SASS");
    pol("wb", [&](unsigned it) {
        stg_policy_kernel<WB><<<blocks, threads>>>(it, d_buf, 1.5f);
    }, "st.global.wb (write-back); STG.E.128.STRONG.CTA in SASS");
    pol("cg", [&](unsigned it) {
        stg_policy_kernel<CG><<<blocks, threads>>>(it, d_buf, 1.5f);
    }, "st.global.cg (cache at L2); STG.E.128.STRONG.GPU in SASS");
    pol("cs", [&](unsigned it) {
        stg_policy_kernel<CS><<<blocks, threads>>>(it, d_buf, 1.5f);
    }, "st.global.cs (streaming / evict-first); STG.E.EF.128.SYS in SASS");
    pol("wt", [&](unsigned it) {
        stg_policy_kernel<WT><<<blocks, threads>>>(it, d_buf, 1.5f);
    }, "st.global.wt (write-through); STG.E.128.STRONG.SYS in SASS");

    // gate G01 verification instrument: the driver's fill path over the
    // same 512 MiB footprint; printed only, never a table row
    auto fill_ms = [&](unsigned n) {
        cudaEvent_t e0, e1;
        TU102_CUDA_CHECK(cudaEventCreate(&e0));
        TU102_CUDA_CHECK(cudaEventCreate(&e1));
        TU102_CUDA_CHECK(cudaEventRecord(e0));
        for (unsigned k = 0; k < n; k++)
            TU102_CUDA_CHECK(cudaMemsetAsync(d_buf, 0x11, STREAM_BYTES));
        TU102_CUDA_CHECK(cudaEventRecord(e1));
        TU102_CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        TU102_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        return (double)ms;
    };
    unsigned fills = 1;
    while (fill_ms(fills) < 20.0) fills *= 2;
    auto fvals = run_reps(r, [&] {
        double ms = fill_ms(fills);
        return (double)STREAM_BYTES * fills / (ms * 1e-3) / 1e9;
    });
    std::fprintf(stderr,
                 "  [instrument] cudaMemsetAsync fill 512 MiB %12.6g GB/s"
                 "           cv=%.2f%% (gate G01 verification; not a table row)\n",
                 median(fvals), cv_pct(fvals));

    std::fprintf(stderr, "stg: done (run %s)\n", r.run_id);
    return 0;
}
